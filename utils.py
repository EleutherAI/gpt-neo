import tensorflow.compat.v1 as tf
from tensorflow.contrib import summary
import re
from urllib.parse import urlparse
from shutil import rmtree
import collections
from absl import logging
import os
import mesh_tensorflow as mtf
from pathlib import Path
import sys

def setup_logging(args):
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('logs/{}.log'.format(os.path.basename(args.model).split(".")[0])),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers
    return logger

def get_batch_size(params):
    return params["{}_batch_size".format(params["mode"])]


def add_mode_to_params(params, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        params["mode"] = "predict"
    elif mode == tf.estimator.ModeKeys.EVAL:
        params["mode"] = "eval"
    elif mode == tf.estimator.ModeKeys.TRAIN:
        params["mode"] = "train"
    else:
        raise ValueError('invalid mode %s' % mode)
    return params


def simd_mesh_setup(params, mesh_shape, layout_rules):
    """constructs SimdMesh function - instructions on how to evenly split tensors across all TPU cores"""

    num_hosts = params['context'].num_hosts
    host_placement_fn = params['context'].tpu_host_placement_function
    device_list = [host_placement_fn(host_id=i) for i in range(num_hosts)]
    tf.logging.info('device_list = {}'.format(device_list))

    # TODO: Better estimation of replica cache size?
    replica_cache_size = 300 * 1000000  # 300M per replica

    # Worker 0 caches all the TPU binaries.
    worker0_mem = replica_cache_size * params['context'].num_replicas
    devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)
    var_placer = mtf.utils.BalancedVariablePlacer(device_list, devices_memory_usage)
    mesh_devices = [''] * mesh_shape.size
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, mesh_devices, params['context'].device_assignment)

    return var_placer, mesh_impl


def remove_batch_from_layout(layout):
    """
    the tf-mesh layout splits across batch size, remove it.
    Useful for prediction steps, when you no longer want large batches.

    :param layout: string describing tf-mesh layout
    :return: layout minus batch dimension
    """
    layout = layout.split(',')
    ret_layout = ""
    for i in layout:
        if "batch" in i:
            pass
        else:
            ret_layout += "{}{}".format(i, ",")
    return ret_layout[:-1]


def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def remove_gs_or_filepath(path):
    parsed_url = urlparse(path)
    if parsed_url.scheme == 'gs':
        os.system('gsutil rm -rf {}'.format(path))
        return
    rmtree(path)


def save_config(params_dict, logdir):
    print('saving config to {}'.format(logdir))
    text = '{\n\n'
    total_params = len(params_dict)
    for count, key in enumerate(params_dict):
        config_value = str(params_dict[key])
        if re.search('[a-zA-Z]', config_value):
            if config_value.lower() != 'true':
                if config_value.lower() != 'false':
                    if config_value[0] != '[':
                        # TODO: making a manual exception for parsing epsilon rn since it's the only number in
                        #       scientific notation. Should fix this.
                        if key != "epsilon":
                            config_value = '"{}"'.format(config_value)
        if count == total_params - 1:
            text += '"{}"'.format(str(key)) + ' : ' + config_value + '\n\n'
        else:
            text += '"{}"'.format(str(key))  + ' : ' + config_value + ',\n\n'
    text += '\n\n}'
    sess = tf.InteractiveSession()
    summary_op = tf.summary.text('run_config', tf.convert_to_tensor(text))
    summary_writer = tf.summary.FileWriter("{}/config".format(logdir), sess.graph)
    text = sess.run(summary_op)
    summary_writer.add_summary(text, 0)
    summary_writer.flush()
    summary_writer.close()
    tf.reset_default_graph()
    print('Done!')


def expand_attention_types_params(params_list):
    newlist = []
    for item in params_list:
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


def get_n_trainable_vars(graph):
    """
    gets number of trainable vars in a MTF model.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    # Getting total number of trainable vars
    print('\n')
    total_parameters = 0
    for variable in graph.trainable_variables:
      shape = variable.shape.dims
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.size
      total_parameters += variable_parameters
    print("N TRAINABLE VARS:")
    print('{:,}'.format(total_parameters))
    print('\n')


def print_dim_names(graph):
    """

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    all_dim_names = []
    for variable in graph.all_variables:
        names = variable.shape.dimension_names
        all_dim_names.append(names)

    # print all dim names in graph & write to file
    all_dim_names = [item for sublist in all_dim_names for item in sublist] # flatten all dims
    unique_dims = list(set(all_dim_names))
    print("ALL DIM NAMES:")
    for dim_name in unique_dims:
        print(dim_name)
    print('\n')


def get_graph_info(graph):
    """
    wrapper fn that calculates number of trainable vars in an MTF graph & prints all dim_names to file
    TODO: how to get un-trainable dim-names too, batch etc.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    get_n_trainable_vars(graph)
    print_dim_names(graph)


def loss_denominator(targets, num_microbatches):
    """Denominator applied to losses.

    This is usually the size of the targets tensor (omitting ensemble
    dimensions).  Alternatively, it is an override value passed to the
    class constructor.

    Args:
      targets: a mtf.Tensor
      num_microbatches: an integer - greater than one if the step has been
        serialized into multiple microbatches to save memory.
    Returns:
      a float
    """
    ret = float(targets.shape.size) * num_microbatches
    return float(ret)


"""Provide a helper class for using summaries on TPU via a host call.

TPUEstimator does not support writing TF summaries out of the box and TPUs can't
perform operations that write files to disk. To monitor tensor values during
training you can copy the tensors back to the CPU of the host machine via
a host call function. This small library provides a convenient API to do this.

Example:
from compare_gan.tpu import tpu_summaries
def model_fn(features, labels, params, mode):
  summary = tpu_summries.TpuSummaries(my_model_dir)

  summary.scalar("my_scalar_summary", tensor1)
  summary.scalar("my_counter", tensor2, reduce_fn=tf.math.reduce_sum)

  return TPUEstimatorSpec(
      host_call=summary.get_host_call(),
      ...)

Warning: The host call function will run every step. Writing large tensors to
summaries can slow down your training. High ranking outfeed operations in your
XProf profile can be an indication for this.
"""

TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn")


class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, save_summary_steps=500):
    self._log_dir = log_dir
    self._scalar_entries = []
    # While False no summary entries will be added. On TPU we unroll the graph
    # and don't want to add multiple summaries per step.
    self.record = True
    self._save_summary_steps = save_summary_steps
    #assert TpuSummaries.inst is None
    TpuSummaries.inst = self

  def has(self, name):
    for entry in self._scalar_entries:
      if entry.name == name:
        return True
    return False

  def scalar(self, name, tensor, reduce_fn=tf.math.reduce_mean):
    """Add a summary for a scalar tensor."""
    if not self.record:
      return
    if self.has(name):
      logging.info("TpuSummaries.scalar: skipping duplicate %s", name)
    else:
      tensor = tf.convert_to_tensor(tensor)
      if tensor.shape.ndims == 0:
        tensor = tf.expand_dims(tensor, 0)
      self._scalar_entries.append(
          TpuSummaryEntry(summary.scalar, name, tensor, reduce_fn))

  def get_host_call(self):
    """Returns the tuple (host_call_fn, host_call_args) for TPUEstimatorSpec."""
    # All host_call_args must be tensors with batch dimension.
    # All tensors are streamed to the host machine (mind the band width).
    global_step = tf.train.get_or_create_global_step()
    host_call_args = [tf.expand_dims(global_step, 0)]
    host_call_args.extend([e.tensor for e in self._scalar_entries])
    logging.info("host_call_args: %s", host_call_args)
    return (self._host_call_fn, host_call_args)

  def _host_call_fn(self, step, *args):
    """Function that will run on the host machine."""
    # Host call receives values from all tensor cores (concatenate on the
    # batch dimension). Step is the same for all cores.
    step = step[0]
    logging.info("host_call_fn: args=%s", args)
    ops = []

    # log scalars
    with summary.create_file_writer(os.path.join(self._log_dir, 'scalars')).as_default():
      offset = 0
      with summary.record_summaries_every_n_global_steps(
            self._save_summary_steps, step):
        for i, e in enumerate(self._scalar_entries):
          value = e.reduce_fn(args[i + offset])
          e.summary_fn(e.name, value, step=step)
      offset += len(self._scalar_entries)
      ops.append(summary.all_summary_ops())
    return tf.group(ops)


TpuSummaries.inst = None
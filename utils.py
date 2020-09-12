from tensorflow.contrib import summary
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import mesh_tensorflow as mtf
import re
from urllib.parse import urlparse
from shutil import rmtree


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
    with open('all_dim_names.txt', 'w') as f:
        for dim_name in unique_dims:
            f.write("%s\n" % dim_name)
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

import collections

from absl import logging
import os

TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn")


class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, save_summary_steps=10):
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

def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.

  Borrowed from t2t.
  TODO(noam): remove this code once there is a better way to get summaries on
  TPU.

  Args:
    model_dir: String containing path to train

  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  # a list of (name, lowered tensor) tuples
  summaries = graph.get_collection(mtf.utils.SCALAR_SUMMARIES_COLLECTION_KEY)

  def maybe_cast(tensor):
    assert tensor.shape.is_compatible_with([]), tensor.name
    if tensor.dtype == tf.int64:
      return tf.to_int32(tensor)
    if tensor.dtype == tf.bfloat16:
      return tf.cast(tensor, tf.float32)
    return tensor

  reshaped_tensors = [tf.reshape(maybe_cast(t), [1]) for _, t in summaries]

  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not reshaped_tensors:
    return None

  def host_call_fn(global_step, *args):
    """Training host call. Creates scalar summaries for training metrics."""
    # This function is executed on the CPU and should not directly reference
    # any Tensors in the rest of the `model_fn`. To pass Tensors from the
    # model to the `model_fn`, provide as part of the `host_call`.
    global_step = tf.cast(global_step[0], tf.int64)
    with tf2.summary.create_file_writer(model_dir).as_default():
      # We cannot directly use any tensor from summaries, because each
      # tensor here must be a concat of multiple tensors from all shards.
      # Therefore, we rely on the assumption that args wil have the same
      # length as summaries, and all tensors in args will have the same
      # order of self._tup_summaries.
      assert len(args) == len(summaries)
      for i, tensor in enumerate(args):
        name = summaries[i][0]
        tf2.summary.scalar(
            name, tf.reduce_mean(tensor), step=global_step)
      return tf.summary.all_v2_summary_ops()

  global_step_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  return host_call_fn, [global_step_t] + reshaped_tensors
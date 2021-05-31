import re
from urllib.parse import urlparse
from shutil import rmtree
import logging
import os
from pathlib import Path
import sys
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import mesh_tensorflow as mtf
import mesh_tensorflow.auto_mtf
from data.encoders import fetch_encoder
import re

def setup_logging(args):
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    tf.get_logger().propagate = False  # Remove double log on console
    name = os.path.splitext(os.path.basename(args.model))[0]
    handlers = [
        logging.FileHandler(f"logs/{name}.log"),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger("tensorflow")
    logger.handlers = handlers
    return logger


def get_batch_size(params):
    return params[f"{params['mode']}_batch_size"]


def add_mode_to_params(params, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        params["mode"] = "predict"
    elif mode == tf.estimator.ModeKeys.EVAL:
        params["mode"] = "eval"
    elif mode == tf.estimator.ModeKeys.TRAIN:
        params["mode"] = "train"
    else:
        raise ValueError(f"Invalid mode {mode}")
    return params


def simd_mesh_setup(params, mesh_shape, layout_rules):
    """Constructs SimdMesh function - instructions on how to evenly split tensors across all TPU cores"""

    num_hosts = params["context"].num_hosts
    host_placement_fn = params["context"].tpu_host_placement_function
    device_list = [host_placement_fn(host_id=i) for i in range(num_hosts)]
    tf.logging.info(f"device_list = {device_list}")

    # TODO: Better estimation of replica cache size?
    replica_cache_size = 300 * 1000000  # 300M per replica

    # Worker 0 caches all the TPU binaries
    worker0_mem = replica_cache_size * params["context"].num_replicas
    devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)
    var_placer = mtf.utils.BalancedVariablePlacer(device_list, devices_memory_usage)
    mesh_devices = [""] * mesh_shape.size
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, mesh_devices, params["context"].device_assignment)

    return var_placer, mesh_impl


def remove_batch_from_layout(layout):
    """
    The tf-mesh layout splits across batch size, remove it.
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
            ret_layout += f"{i},"
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
    if parsed_url.scheme == "gs":
        os.system(f"gsutil rm -rf {path}")
        return
    rmtree(path)


def save_config(params_dict, logdir):
    print(f"Saving config to {logdir}")
    text = "{\n\n"
    total_params = len(params_dict)
    for count, key in enumerate(params_dict):
        config_value = str(params_dict[key])
        if re.search('[a-zA-Z]', config_value):
            if config_value.lower() != 'true':
                if config_value.lower() != 'false':
                    if config_value[0] != '[':
                        # TODO: Making a manual exception for parsing epsilon right now since it's the only number in
                        #       scientific notation. Should fix this.
                        if key != "epsilon":
                            config_value = f'"{config_value}"'
        if count == total_params - 1:
            text += f'"{str(key)}"' + ' : ' + config_value + '\n\n'
        else:
            text += f'"{str(key)}"'  + ' : ' + config_value + ',\n\n'
    text += '\n\n}'
    sess = tf.InteractiveSession()
    summary_op = tf.summary.text("run_config", tf.convert_to_tensor(text))
    summary_writer = tf.summary.FileWriter(f"{logdir}/config", sess.graph)
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
    Gets number of trainable vars in a MTF model.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    total_parameters = 0
    for variable in graph.trainable_variables:
      shape = variable.shape.dims
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.size
      total_parameters += variable_parameters
    print(f"\n\nN TRAINABLE VARS:\n{total_parameters:,}\n\n")


def print_dim_names(graph):
    """
    Print names of all Dimensions
    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    all_dim_names = []
    for variable in graph.all_variables:
        names = variable.shape.dimension_names
        all_dim_names.append(names)

    # Print all dim names in graph & write to file
    all_dim_names = [item for sublist in all_dim_names for item in sublist] # Flatten all dims
    unique_dims = list(set(all_dim_names))
    print("ALL DIM NAMES:")
    for dim_name in unique_dims:
        print(dim_name)
    print('\n')


def get_graph_info(graph):
    """
    Wrapper fn that calculates number of trainable vars in an MTF graph & prints all dim_names to file
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

def check_dataset(input_fn, params, global_step=None):
    tf.enable_eager_execution()
    if global_step is not None:
        dataset = input_fn(params, global_step=global_step)
    else:
        dataset = input_fn(params)
    dataset_iter = dataset.make_one_shot_iterator()
    tensor, _ = next(dataset_iter)
    enc = fetch_encoder(params)

    for p in tensor[:1]:
        txt = enc.decode(p)

    print('-' * 50)
    print(txt[:500], '\n\n...\n\n', txt[-500:])
    print('-' * 50)
    exit()

def auto_layout(graph, mesh_shape, logits, loss):
    layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, [logits, loss])
    print(f"Auto-selected layout:\n{layout_rules}\nRe-initialize graph with selected layout")
    quit() 

def auto_layout_and_mesh_shape(graph, num_cores, logits, loss):
    layout_rules, mesh_shape = mtf.auto_mtf.layout_and_mesh_shape(graph, num_cores,
                                                                    [logits, loss], max_mesh_shape_dimensions=4)
    print(f"Num cores:\n{num_cores}\nAuto-selected layout:\n{layout_rules}\nAuto-selected mesh shape:\n{mesh_shape}" \
            f"\nRe-initialize graph with selected layout & mesh shape")
    quit() 

def create_host_call(model_dir):
    """Construct a host_call writing scalar summaries.

    Borrowed from t2t.
    
    Args:
        model_dir: String containing path to train
    Returns:
        (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    graph = tf.get_default_graph()
    # A list of (name, lowered tensor) tuples
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
                tf2.summary.scalar(name, tf.reduce_mean(tensor), step=global_step)
        return tf.summary.all_v2_summary_ops()

    global_step_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
    return host_call_fn, [global_step_t] + reshaped_tensors


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

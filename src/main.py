"""
"Sub Main" that contains one function to start the training loop.
"""

from functools import partial
import argparse
import json
import re

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import tpu
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
#from tensorflow.contrib.tpu import device_assignment
from tensorflow.python.tpu.device_assignment import device_assignment
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.summary_ops_v2 import create_file_writer, always_record_summaries, flush
from tensorflow import summary
from tensorflow.python.framework import ops
from tensorflow.python.tpu.ops import tpu_ops
from .monitored_session import _MonitoredSession, ChiefSessionCreator, WorkerSessionCreator, MonitoredSession, MonitoredTrainingSession
import numpy as np

from .dataclass import ModelParameter
from .inputs import dataset, gpt_neo_input
from .train import get_model_fn, _CkptLoaderHook
from .utils_mtf import SimdMeshImplInputReader, _host_id_to_tf_device
from .eval import gen_sample


def main(args: argparse.Namespace) -> None:
    """
    Given previously captured arguments, this function runs the following steps (in order):
    * Load given config
    * initialize data loader
    * create model graph
    * start training loop.
    :param args: argparse arguments from the parent main function
    :return: None
    """
    # Setup logging
    model_path = args.model if args.model.endswith(".json") else f"configs/{args.model}.json"
    with open(model_path) as f:
        params = json.load(f)
    params = ModelParameter(params)
    # Read params of model

    # Fetch appropriate input functions

    if params.model_mode == 'jannet':
        input_fn = partial(dataset, params=params, step=0, train=args.run_mode == 'train')
    elif params.model_mode == 'gpt':
        input_fn = partial(gpt_neo_input, params=params, step=0, eval=False)

        # Set params for text only GPT mode.
        params.use_language = True
        params.use_video = False

    else:
        raise ValueError(f"model_mode need to be 'jannet' or 'gpt' {params.model_mode}, "
                         "is a not supported option.")

    # Add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    params.num_cores = mesh_shape.size
    params.use_tpu = True if not args.tpu is None else False
    params.gpu_ids = args.gpu_ids
    # Expand attention types param
    params.predict = args.predict

    if args.dry:
        inp = {'token_x': tf.zeros([1]),
               'token_y': tf.zeros([1]),
               'frame': tf.zeros([1]),
               'vid_msk': tf.zeros([1]),
               'tkn_msk': tf.zeros([1])}

        inp = params.align_tensor_op(inp)
        get_model_fn(params, args.run_mode)(inp)
        return

    mtf_mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    params.layout_rules = mtf.convert_to_layout_rules(params.layout)

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    tpu_cluster_spec = tpu_cluster_resolver.cluster_spec()

    if tpu_cluster_spec:
        config.cluster_def.CopyFrom(tpu_cluster_spec.as_cluster_def())

    with tf.Graph().as_default():

        with tf.Session(target=tpu_cluster_resolver.master(), config=config) as sess:
            tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

            all_devices = sess.list_devices()

            cpus = []
            for d in all_devices:
                if d.device_type == 'CPU':
                    cpus += [re.sub('device:CPU', 'cpu', d.name)]

            cpu_devices = []
            for c in cpus:
                m = re.match('/job:(.*)/replica:(.*)/task:(.*)/.*', c)
                cpu_devices.append((m.group(1), int(m.group(2)), int(m.group(3)), c))

            cpu_devices = [_[3] for _ in sorted(cpu_devices)]
            params.cpu_devices = [n for n in cpu_devices if 'coordinator' not in n]

            topology = sess.run(tpu.initialize_system())
            topo_object = Topology(serialized=topology)

            num_cores = int(np.prod(topo_object.mesh_shape))
            params.num_hosts = int(topo_object.num_tasks)
            num_cores_per_host = int(num_cores // params.num_hosts)
            assert num_cores_per_host == int(topo_object.num_tpus_per_task)

            params.d_assignment = device_assignment(topology,
                                                    computation_shape=[1, ] * mtf.utils.topology_rank(topology),
                                                    num_replicas=num_cores)

            params.mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(mtf_mesh_shape,
                                                               params.layout_rules,
                                                               None, params.d_assignment)

        summary_writer = create_file_writer(params.model_path)
        with summary_writer.as_default(), (always_record_summaries()):

            model_fn, captured_hooks, captured_output_dtypes_shapes = get_model_fn(params, args.run_mode)

            simd_input_reader = SimdMeshImplInputReader(params.mesh_impl, input_fn,
                                                                     params.input_pipeline_shape,
                                                                     external_worker=True,
                                                                     is_eval_mode=False)

            train_computation = tpu.replicate(computation=model_fn,
                                              inputs=[[]] * num_cores,
                                              infeed_queue=simd_input_reader.infeed_queue,
                                              device_assignment=params.d_assignment)

            master_to_slice_hook, slice_to_master_hook = captured_hooks.get()
            ckpt_loader_hook = _CkptLoaderHook(params.model_path)
            step_counter_hook = tf.train.StepCounterHook(every_n_steps=10)
            all_hooks = [ckpt_loader_hook, master_to_slice_hook, slice_to_master_hook, step_counter_hook]

            # if params.write_summary:
            flush_summary = flush()

            with MonitoredTrainingSession(master=tpu_cluster_resolver.master(), hooks=all_hooks, config=config) as sess:
                print(sess)

                summary.initialize(session=sess)
                simd_input_reader.start_infeed_thread(sess)

                current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))
                #while current_step < params.train_steps:
                while current_step < 10:
                    sess.run(train_computation)
                    sess.run(flush_summary)

                    print('current_step:', current_step)

                    tf.logging.info('train steps: {}'.format(current_step))

                    current_step += 1

                tf.logging.info('finished.')

    with tf.Session(target=tpu_cluster_resolver.get_master(), config=config) as sess:
        sess.run(tpu.shutdown_system())
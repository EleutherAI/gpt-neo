"""
Contains functions to create a training loop and log its outputs to tensorboard
"""
import collections
import threading
import time

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import tpu
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow_estimator.python.estimator import estimator as estimator_lib

from .dataclass import ModelParameter
from .model import build
from .optimizers import get_optimizer

tf.config.optimizer.set_experimental_options({"layout_optimizer":              True,
                                              "constant_folding":              True,
                                              "shape_optimization":            True,
                                              "remapping":                     True,
                                              "arithmetic_optimization":       True,
                                              "dependency_optimization":       True,
                                              "loop_optimization":             True,
                                              "function_optimization":         True,
                                              "debug_stripper":                True,
                                              "disable_model_pruning":         False,
                                              "scoped_allocator_optimization": True,
                                              "pin_to_host_optimization":      False,
                                              "implementation_selector":       True,
                                              "auto_mixed_precision":          True,
                                              "disable_meta_optimizer":        False,
                                              "min_graph_nodes":               0
                                              })


class CapturedObject(object):
    """A placeholder to capture an object.
    This is useful when we need to capture a Python object in the Tensorflow
    control flow body function and use it outside the control flow.
    """

    def __init__(self):
        self._object = None
        self._captured = False

    def capture(self, o):
        if self._captured:
            raise RuntimeError(
                    'InternalError: Object can capture only once. Please file bug.')

        self._captured = True
        self._object = o

    def get(self):
        if not self._captured:
            raise RuntimeError(
                    'InternalError: Object is not captured properly before `get`. '
                    'Please file bug.')
        return self._object


_NONE_PNUM = None
_NO_DATA = None


def _host_id_to_tf_device(host_id, external_worker):
    if not isinstance(host_id, int):
        raise ValueError
    return f"{'/job:worker' * external_worker}/task:{host_id}/device:CPU:0"


class SubBatchSlicer(object):
    """Reads and distributes a sub-batch on a host."""

    def __init__(self, sub_batch_ds_creator, host_id, all_sub_batch_pnums,
                 simd_mesh_impl, mtf_input_shapes, external_worker, global_batch):
        self._host_id = host_id
        self._all_sub_batch_pnums = all_sub_batch_pnums
        self._simd_mesh_impl = simd_mesh_impl
        self._mtf_input_shapes = mtf_input_shapes
        self._external_worker = external_worker
        self._global_batch = global_batch

        self._validate_args()

        with ops.device(_host_id_to_tf_device(self._host_id, self._external_worker)):
            self._ds_iterator = sub_batch_ds_creator().make_initializable_iterator()

    @property
    def initializer(self):
        return self._ds_iterator.initializer

    def get_slices(self):
        """Yields sliced tensors and which remote pnums they should go to.
        Yields:
          tf_tensor: The sliced tensor.
          pnum: Which process number the tf_tensor should to go.
          input_i: The input ordinal of the tf_tensor.
        """
        with ops.device(_host_id_to_tf_device(self._host_id, self._external_worker)):

            all_input_tensors = self._ds_iterator.get_next()
            if isinstance(all_input_tensors, tf.Tensor):
                all_input_tensors = [all_input_tensors]
            assert len(all_input_tensors) == len(self._all_sub_batch_pnums)

            for input_i in range(len(all_input_tensors)):
                input_tensor = all_input_tensors[input_i]
                sub_batch_pnums = self._all_sub_batch_pnums[input_i]
                mtf_input_shape = self._mtf_input_shapes[input_i]

                # Initialize the cache for each input_i
                self._init_slice_cache()

                for pnum in sub_batch_pnums:
                    input_slice = self._slice_tensor(input_tensor, mtf_input_shape, pnum)
                    yield input_slice, pnum, input_i

    def _validate_args(self):
        if not isinstance(self._all_sub_batch_pnums, list):
            raise ValueError
        if not isinstance(self._mtf_input_shapes, list):
            raise ValueError
        if not self._all_sub_batch_pnums:
            raise ValueError
        if not self._mtf_input_shapes:
            raise ValueError
        if len(self._all_sub_batch_pnums) != len(self._mtf_input_shapes):
            raise ValueError

    def _init_slice_cache(self):
        # Cache for tensor slices
        self._slice_dict = collections.defaultdict(list)

    def _slice_tensor(self, input_tensor, mtf_input_shape, pnum):
        """Slice input_tensor according to mtf_input_shape and pnum."""
        s_begin = self._simd_mesh_impl.slice_begin(mtf_input_shape, pnum)
        if not self._global_batch:
            # Always slice from 0 in the first dimension (batch dimension), since
            # input_tensor a sub-batch tensor.
            s_begin[0] = 0
        if tuple(s_begin) in self._slice_dict:
            return self._slice_dict[tuple(s_begin)]

        s_shape = self._simd_mesh_impl.slice_shape(mtf_input_shape)
        input_slice = tf.slice(input_tensor, s_begin, s_shape)

        self._slice_dict[tuple(s_begin)] = input_slice
        return input_slice


class SimdMeshImplInputReader(object):
    """Handles input pipeline for SimdMeshImpl."""

    def __init__(self,
                 simd_mesh_impl,
                 ds_creator,
                 mtf_input_shapes,
                 ds_prefetch_size=tf.data.experimental.AUTOTUNE,
                 external_worker=True,
                 is_eval_mode=False):
        """Input pipeline for the SIMD implementation of MeshTensorflow.
        Args:
          simd_mesh_impl: A mtf.simd_mesh_impl.SimdMeshImpl object.
          ds_creator: A function that creates a dataset.
          mtf_input_shapes: A list of mtf.Shape. Then length of it must be equal
            to the number of elements generated by the ds_creator. NOTE, we assume:
              1. The 0-th dimension is the batch dimension.
              2. The batch dimension is consistent across all input shapes in
                 mtf_input_shapes.
          ds_prefetch_size: The buffer size for prefetching
            (default tf.data.experimental.AUTOTUNE).
          external_worker: Whether you have an external tpu_worker or not. Set it to
            False if you run the program locally, for example, during local unit
            test.
          is_eval_mode: In evaluation mode, only one dataset object will be created,
            as opposed to one dataset for each sub-batch. Default is False. Set it
            to True during evaluation, to ensure that one evaluation instance will
            be used once and only once.
        Note:
          1. The efficiency is optimized according to the shape of the 0-th tensor:
             mtf_input_shapes[0]. We recommand you to put the largest tensor as the
             0-th input.
          2. You need to call start_infeed_thread() before your train ops.
        Example:
            simd_mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(...)
            # ds_creator is function that creates a tf.data.Dataset.
            # This Dataset must return single examples (no batch dimension).
            def ds_creator():
              return tf.data.Dataset.from_tensors(x)
            # mtf_input_shapes is a list of Shapes of all input tensors given by the
            # dataset. All shapes must begin with the same batch dimension.
            simd_input_reader = SimdMeshImplInputReader(simd_mesh_impl,
                                                        ds_creator,
                                                        mtf_input_shapes)
            batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
            io_dim = mtf.Dimension('io', FLAGS.io_size)
            mtf_input_shapes = [mtf.Shape([batch_dim, io_dim])]
            infeed_queue = simd_input_reader.infeed_queue
            tpu_train_computation = tpu.replicate(
                computation=model_fn,
                inputs=[[]] * num_cores,
                infeed_queue=infeed_queue, ...)
            with tf.Session() as sess:
              simd_input_reader.start_infeed_thread(sess,
                                                    number_steps=num_training_steps)
              for _ in range(num_training_steps):
                sess.run(tpu_train_computation)
        """
        super().__init__()
        assert mtf_input_shapes
        assert isinstance(mtf_input_shapes, list)

        # TODO(lehou): Support nested structures for ds_creator, mtf_input_shapes.
        self._simd_mesh_impl = simd_mesh_impl
        self.num_cores = simd_mesh_impl.device_assignment.num_replicas

        self.ordered_ordinals = []
        self.ordered_hosts = []
        self.ordered_host_ids = []
        d_assignment = simd_mesh_impl.device_assignment

        for pnum in range(self.num_cores):
            physical_pnum = simd_mesh_impl.l2p(pnum)
            # For MTF, there's always 1 core per replica. So logical_core=0.
            self.ordered_ordinals.append(d_assignment.tpu_ordinal(replica=physical_pnum, logical_core=0))
            host_device = d_assignment.host_device(replica=physical_pnum)
            host_id = int(host_device.lower().split("/task:")[1].split("/device:")[0])
            self.ordered_hosts.append(host_device)
            self.ordered_host_ids.append(host_id)

        self.num_hosts = len(set(self.ordered_hosts))
        self.num_cores_per_host = self.num_cores // self.num_hosts
        if self.num_cores != self.num_hosts * self.num_cores_per_host:
            raise ValueError
        self._ds_creator = ds_creator
        self._mtf_input_shapes = mtf_input_shapes
        self._ds_prefetch_size = ds_prefetch_size
        self._external_worker = external_worker
        self._is_eval_mode = is_eval_mode

        self._gen_infeed_queue()

    @property
    def infeed_queue(self):
        return self._infeed_queue

    def start_infeed_thread(self, sess, number_steps=-1, initial_wait_sec=0.5):
        """Start running enqueue ops in a thread.
        Args:
          sess: A tf.Session.
          number_steps: Number of times to call sess.run(enqueue_ops).
            default is -1 (forever).
          initial_wait_sec: Number of seconds to wait before starting the enqueue
            loop. Default is 0.5.
        """

        def _thread_fn():
            time.sleep(initial_wait_sec)
            if number_steps > 0:
                for _ in range(number_steps):
                    sess.run(self._enqueue_ops)
            else:
                while True:
                    sess.run(self._enqueue_ops)

        sess.run(self._input_initializers)
        self._infeed_thread = threading.Thread(target=_thread_fn)
        self._infeed_thread.start()

    def _gen_infeed_queue(self):
        """Generates _infeed_queue, _enqueue_ops, _input_initializers."""
        pnum_maps = []
        batch_size = self._mtf_input_shapes[0].to_integer_list[0]
        for mtf_shape in self._mtf_input_shapes:
            # Make sure that the batch size is the same across all input tensors.
            assert batch_size == mtf_shape.to_integer_list[0]
            pnum_maps.append(self._get_pnum_map(mtf_shape))

        # For each sub-batch, we need to know which host should read it.
        if self._is_eval_mode:
            # There should be just one dataset-holding host. Make the last host do it.
            hosts_to_hold_ds = [self.num_hosts - 1]
        else:
            hosts_to_hold_ds = self._get_hosts_to_hold_ds(pnum_maps[0])
        sub_batch_size = batch_size // len(hosts_to_hold_ds)
        tf.logging.info("MTF sub_batch_size: {}".format(sub_batch_size))
        assert sub_batch_size * len(hosts_to_hold_ds) == batch_size

        def sub_batch_ds_creator():
            return self._ds_creator().batch(
                    sub_batch_size, drop_remainder=True).prefetch(
                    self._ds_prefetch_size)

        # For each sub-batch, create a SubBatchSlicer object.
        # Get the list of pnums for each input.
        sub_batch_slicer_list = [SubBatchSlicer(sub_batch_ds_creator,
                                                host_id,
                                                [pnum_map.flatten().tolist() if self._is_eval_mode else
                                                 pnum_map[sub_batch_i, ...].flatten().tolist()
                                                 for pnum_map in pnum_maps],
                                                self._simd_mesh_impl,
                                                self._mtf_input_shapes,
                                                self._external_worker,
                                                global_batch=not self._is_eval_mode)
                                 for sub_batch_i, host_id in enumerate(hosts_to_hold_ds)]

        # Slots for all laidout tensors.
        all_laidout_tensors = [[_NO_DATA] * len(self._mtf_input_shapes) for _ in range(self.num_cores)]

        # Read tf_tensors, put them in slots.
        for sub_batch_slicer in sub_batch_slicer_list:
            for tf_tensor, pnum, input_i in sub_batch_slicer.get_slices():
                all_laidout_tensors[pnum][input_i] = tf_tensor

        # Make sure that there are no Nones in all_laidout_tensors.
        for laidout_tensors in all_laidout_tensors:
            assert _NO_DATA not in laidout_tensors

        with ops.device(_host_id_to_tf_device(hosts_to_hold_ds[0],
                                              self._external_worker)):
            self._infeed_queue, self._enqueue_ops = self._enqueue_laidout_tensors(
                    all_laidout_tensors)

        self._input_initializers = [s.initializer for s in sub_batch_slicer_list]

    def _get_pnum_map(self, mtf_shape):
        """Returns the pnum_map according to mtf_shape.
        Args:
          mtf_shape: A mtf.Shape object.
        Returns:
          A numpy array pnum_map. For the i-th sub-batch, pnum_map[i] is a numpy
          array containing all pnums that tensor slices of the i-th sub-batch
          will be send to.
        """
        s_shape = self._simd_mesh_impl.slice_shape(mtf_shape)
        shape_list = [dim_size // s_dim_size for dim_size, s_dim_size in zip(
                mtf_shape.to_integer_list, s_shape)]

        pnum_map_shape = shape_list + [
                self.num_cores // np.prod(shape_list)]
        assert np.prod(pnum_map_shape) == self.num_cores

        # Initialize the pnum_map to _NONE_PNUM.
        pnum_map = np.empty(pnum_map_shape, dtype=object)
        pnum_map[:] = _NONE_PNUM

        for pnum in range(self.num_cores):
            s_begin = self._simd_mesh_impl.slice_begin(mtf_shape, pnum)
            coord = [dim_size // s_dim_size for dim_size, s_dim_size in zip(
                    s_begin, s_shape)]
            # put pnum in pnum_map[coord]
            pnum_array_ref = pnum_map[tuple(coord)]
            for idx, value in enumerate(pnum_array_ref):
                if value is _NONE_PNUM:
                    pnum_array_ref[idx] = pnum
                    break

        tf.logging.info("MTF pnum_map: {}".format(pnum_map))
        assert _NONE_PNUM not in pnum_map
        return pnum_map

    def _get_hosts_to_hold_ds(self, pnum_map):
        """Finds which host should read which sub-batch."""
        assert _NONE_PNUM not in pnum_map

        # This records how many datasets (ds) are already stored on each host.
        num_dss_per_host = [0] * self.num_hosts

        # A list of host_ids that holds datasets (ds).
        hosts_to_hold_ds = []

        def _get_num_pnums_per_host(sub_batch_pnum_map):
            num_pnums_per_host = [0] * self.num_hosts
            for pnum in sub_batch_pnum_map.flatten():
                num_pnums_per_host[self.ordered_host_ids[pnum]] += 1
            return num_pnums_per_host

        def _find_host_id_with_most_pnums_and_least_ds(num_pnums_per_host,
                                                       num_dss_per_host):
            host_metics = [(
                    host_id, num_pnums_per_host[host_id],
                    num_dss_per_host[host_id]) \
                    for host_id in range(self.num_hosts)]
            # Major max key: num_pnums
            # Minor max key: -num_dss. We need to find a relatively spare host.
            host_id, _, _ = max(host_metics, key=lambda keys: (keys[1], -keys[2]))
            return host_id

        for sub_batch_pnum_map in pnum_map:
            num_pnums_per_host = _get_num_pnums_per_host(sub_batch_pnum_map)
            host_id = _find_host_id_with_most_pnums_and_least_ds(num_pnums_per_host,
                                                                 num_dss_per_host)
            num_dss_per_host[host_id] += 1
            hosts_to_hold_ds.append(host_id)

        return hosts_to_hold_ds

    def _enqueue_laidout_tensors(self, all_laidout_tensors):
        """Generate enqueue ops to enqueue all_laidout_tensors."""

        def _tpu_ordinal_function_impl(pnum):
            return self.ordered_ordinals[pnum]

        def _placement_function_impl(pnum):
            return self.ordered_hosts[pnum]

        laidout_tensors0 = all_laidout_tensors[0]
        infeed_queue = tpu_feed.InfeedQueue(
                number_of_tuple_elements=len(laidout_tensors0),
                tuple_types=[x.dtype for x in laidout_tensors0],
                tuple_shapes=[x.shape for x in laidout_tensors0])
        enqueue_ops = infeed_queue.generate_enqueue_ops(
                all_laidout_tensors,
                tpu_ordinal_function=_tpu_ordinal_function_impl,
                placement_function=_placement_function_impl)

        return infeed_queue, enqueue_ops


class CheckpointLoaderHook(tf.estimator.SessionRunHook):
    """Load checkpoint right after the session started."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def after_create_session(self, session, coord):
        # pylint: disable=protected-access
        saver_collection = tf.get_collection(tf.GraphKeys.SAVERS)
        if saver_collection:
            saver = saver_collection[0]
            check_point = tf.train.latest_checkpoint(self.checkpoint_dir)
            if check_point:
                saver.restore(session, check_point)


def computation_func(params: ModelParameter, input_fn, session_config, tpu_cluster_resolver, run_mode: str):
    captured_hooks = CapturedObject()
    captured_output_dtypes_shapes = CapturedObject()

    def model_fn(*args):
        """
        Create model partitioned graph given example input tensor
        :param features: inputs and targets in dict
        :param mode: training mode
        :param params: serialized dict of ModelParameters instance
        :return: tpu estimator spec
        """

        def _add_summary(tf_loss, value, global_step):
            """Add all summaries."""

            def _host_loss_summary(tf_loss, value, global_step):
                """Add summary.scalar in host side."""
                gs = tf.cast(global_step, tf.int64)

                sum_ops = []

                for key in value.keys():
                    sum_ops.append(summary.scalar(key, value[key], step=gs))
                with tf.control_dependencies(sum_ops):
                    return tf.identity(tf_loss)

            # Cast the global step to tf.int32, since
            # outside_compilation does not support tf.int64.
            tf_loss = tpu.outside_compilation(_host_loss_summary, tf_loss, value, tf.cast(global_step, tf.int32))

            return tf_loss

        # Get global step
        global_step = tf.train.get_or_create_global_step()

        # Construct mtf graph + mesh from params
        graph = mtf.Graph()
        mesh_shape = mtf.convert_to_shape(params.mesh_shape)
        layout_rules = mtf.convert_to_layout_rules(params.layout)

        # Mesh setup
        replica_cache_size = 300 * 1024 * 1024  # 300M per replica.
        worker0_mem = replica_cache_size * 8 * params.num_hosts
        devices_memory_usage = [worker0_mem] + [0] * (params.num_hosts - 1)
        var_placer = mtf.utils.BalancedVariablePlacer(params.cpu_devices, devices_memory_usage)
        mesh_devices = [""] * mesh_shape.size
        # mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        #        mesh_shape, layout_rules, mesh_devices, params.context.device_assignment)

        # Build mtf mesh object
        mesh = mtf.Mesh(graph, "mesh", var_placer)
        params.mesh = mesh

        # Build mtf_features & seq length dict for getting number of microbatches
        # We need to pack inputs into a dict to pass into serialize_training_step
        # params.mode = mode

        frame_input = None
        token_x_input = None
        token_y_input = None
        frame_mask = None
        token_mask = None

        if params.use_video:
            frame_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[0]]),
                                                     params.frame_input_shape, "frame_input")

            if params.use_language:
                token_x_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[1]]),
                                                           params.token_dim_shape, "tkn_src")
                token_y_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[2]]),
                                                           params.token_dim_shape, "tkn_tgt")

                frame_mask = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[3]]),
                                                        params.frame_mask_shape, "vid_msk")
                token_mask = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[4]]),
                                                        params.token_dim_shape, "tkn_msk")

        elif params.use_language:

            token_x_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[0]]),
                                                       params.token_dim_shape, "tkn_src")
            token_y_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[1]]),
                                                       params.token_dim_shape, "tkn_tgt")


        if run_mode == 'sample' and params.use_autoregressive_sampling:
            sequence_dim = mtf.Dimension("sequence", params.time_patch_size)

            def cond_fn(position):
                is_done = mtf.greater_equal(position, sequence_dim.size)
                is_done = mtf.logical_or(is_done, mtf.greater_equal(position - params.initial_autoregressive_position,
                                                                    sequence_dim))
                is_done = mtf.reduce_sum(is_done)

                return mtf.logical_not(is_done)

            def body_fn(position, frame_input, token_x_input, token_y_input, frame_mask, token_mask, *states):
                with tf.variable_scope('jannet'):
                    if token_mask is None:
                        token_mask = mtf.ones(params.mesh, [], params.dtype)
                    else:
                        token_mask = mtf.cast(token_mask, params.dtype)
                    if frame_mask is None:
                        frame_mask = mtf.ones(params.mesh, [], params.dtype)
                    else:
                        frame_mask = mtf.cast(frame_mask, params.dtype)
                    video_loss, _, frame_out, token_out = build(params,
                                                                frame_input,
                                                                token_x_input,
                                                                token_y_input,
                                                                frame_mask,
                                                                token_mask)

                language_token_per_frame_dim = mtf.Dimension("language_token_per_frame",
                                                             params.language_token_per_frame)

                # (batch, sequence_dim, language_token_patch, token_patch_size, vocab_size) ->
                # (batch, sequence_dim, language_token_per_frame, vocab_size)
                token_out = mtf.reshape(token_out, new_shape=mtf.Shape([params.batch_dim,
                                                                        sequence_dim,
                                                                        language_token_per_frame_dim,
                                                                        params.vocab_dim]))

                # (batch, sequence_dim, language_token_per_frame, vocab_size) ->
                # (batch, sequence_dim, language_token_per_frame)
                token_out: mtf.Tensor = mtf.argmax(token_out, reduced_dim=params.vocab_dim)

                # (language_token_per_frame_dim)
                token_mask_out_range = mtf.range(mesh, language_token_per_frame_dim, dtype=tf.int32)
                # (language_token_per_frame_dim) -> (batch, sequence_dim, language_token_per_frame, vocab_size)
                token_mask_out_range = mtf.broadcast(token_mask_out_range, new_shape=token_out.shape)

                # (batch, sequence_dim, language_token_per_frame) -> (batch, sequence_dim)
                token_mask_out_argmin = mtf.argmax(mtf.negative(token_out), reduced_dim=language_token_per_frame_dim)

                # (batch, sequence_dim) -> (batch, sequence_dim, language_token_per_frame, vocab_size)
                token_mask_out_argmin = mtf.broadcast(token_mask_out_argmin, new_shape=token_out.shape)

                token_mask_out = mtf.less_equal(token_mask_out_range, token_mask_out_argmin)

                # (batch, sequence_dim, language_token_per_frame, vocab_size) ->
                # (batch_dim, sequence_dim, language_token_patch, token_patch_size)
                token_out = mtf.reshape(token_out, new_shape=params.token_dim_shape)
                token_mask_out = mtf.reshape(token_mask_out, new_shape=params.token_dim_shape)

                # (sequence_dim)
                one_hot_sequence = mtf.one_hot(position, sequence_dim, dtype=tf.int32)
                neg_one_hot_sequence = (1 - one_hot_sequence)

                # frame_input = mtf.pad(anonymize(frame_out, sequence_dim),[1, 0], anonymize_dim(sequence_dim)).name * mtf.cast(one_hot_sequence, tf.float32) + frame_input * mtf.cast(neg_one_hot_sequence, tf.float32)
                token_x_input = token_out * one_hot_sequence + token_x_input * neg_one_hot_sequence
                token_mask = token_mask_out * one_hot_sequence + token_mask * neg_one_hot_sequence

                position_out = position + 1

                return [position_out, frame_input, token_x_input, token_y_input, frame_mask, token_mask, video_loss]

            while_loop_inputs = [params.initial_autoregressive_position, frame_input,
                                 token_x_input, token_y_input, frame_mask, token_mask]

            _, frame_out, token_out, _, _, _, loss = mtf.while_loop(cond_fn=cond_fn,
                                                                    body_fn=body_fn,
                                                                    inputs=while_loop_inputs)
        else:
            with mtf.utils.outside_all_rewrites(), tf.variable_scope('jannet'):
                if token_mask is None:
                    token_mask = mtf.ones(params.mesh, [], tf.float32)
                else:
                    token_mask = mtf.cast(token_mask, tf.float32)
                if frame_mask is None:
                    frame_mask = mtf.ones(params.mesh, [], tf.float32)
                else:
                    frame_mask = mtf.cast(frame_mask, tf.float32)
                if frame_input is not None:
                    frame_input = mtf.cast(frame_input, tf.float32)
                video_loss, token_loss, frame_out, token_out = build(params,
                                                                     frame_input,
                                                                     token_x_input,
                                                                     token_y_input,
                                                                     frame_mask,
                                                                     token_mask)
                loss = video_loss + token_loss
                video_loss = video_loss * frame_mask.size / mtf.reduce_sum(frame_mask)
                token_loss = token_loss * token_mask.size / mtf.reduce_sum(token_mask)

        if run_mode == 'train':
            update_ops = get_optimizer(mesh, loss, params)
        else:  # run_mode == 'sample'

            if params.use_video:
                frame_out = mtf.anonymize(frame_out)

            if params.use_language:
                token_out = mtf.anonymize(token_out)

        total_parameters = 0
        for variable in graph.trainable_variables:
            shape = variable.shape.dims
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim.size
            total_parameters += variable_parameters

        print(f"\n\nN TRAINABLE VARS:\n{total_parameters:,}\n\n")
        all_dim_names = []

        for variable in graph.all_variables:
            names = variable.shape.dimension_names
            all_dim_names.append(names)

        # Print all dim names in graph & write to file
        all_dim_names = [item for sublist in all_dim_names for item in sublist]  # Flatten all dims
        unique_dims = list(set(all_dim_names))
        print("ALL DIM NAMES:")
        for dim_name in unique_dims:
            print(dim_name)
        print('\n')

        lowering = mtf.Lowering(graph, {mesh: params.mesh_impl}, autostack=True)

        tf_loss = lowering.export_to_tf_tensor(loss)
        tf_loss = tf.cast(tf_loss, tf.float32)

        log_dict = {}

        if run_mode == 'train':
            if params.use_video:
                video_loss = lowering.export_to_tf_tensor(video_loss)
                video_loss = tf.cast(video_loss, tf.float32)
                log_dict.update({'video_loss': video_loss})

            if params.use_language:
                token_loss = lowering.export_to_tf_tensor(token_loss)
                token_loss = tf.cast(token_loss, tf.float32)
                log_dict.update({'token_loss': token_loss})

            tf_loss = _add_summary(tf_loss=tf_loss, value=log_dict, global_step=global_step)

        else:  # run_mode == 'sample'
            predictions = {}
            if params.use_video:
                predictions.update({'frame_out': lowering.export_to_tf_tensor(frame_out)})
                predictions.update({'frame_inp': args[0]})

            if params.use_language:
                predictions.update({'token_out': lowering.export_to_tf_tensor(token_out)})
                predictions.update({'token_inp': args[2]})

        if run_mode == 'train':

            # Creates train_op
            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
            tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
            tf.logging.info(f"tf_update_ops: {tf_update_ops}")

            with mtf.utils.outside_all_rewrites():

                hooks = [mtf.MtfRestoreHook(lowering)]
                if params.use_checkpointing:
                    saver = tf.train.Saver(
                            tf.global_variables(),
                            sharded=True,
                            max_to_keep=10,
                            keep_checkpoint_every_n_hours=2,
                            defer_build=False,
                            save_relative_paths=True)
                    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

                    hooks.append(tf.train.CheckpointSaverHook(
                            params.model_path,
                            save_steps=params.steps_per_checkpoint,
                            saver=saver,
                            listeners=[mtf.MtfCheckpointSaverListener(lowering)]))

                captured_hooks.capture(hooks)

                return tf.group([tf_loss] + tf_update_ops)

        else:  # run_mode == 'sample'

            predictions = [tf.cast(predictions[key], tf.float32) for key in predictions.keys()]
            predictions_dtypes = [pred.dtype for pred in predictions]
            predictions_shapes = [pred.shape for pred in predictions]
            captured_hooks.capture([mtf.MtfRestoreHook(lowering), None])
            captured_output_dtypes_shapes.capture([predictions_dtypes, predictions_shapes])

            return tpu_ops.outfeed_enqueue_tuple(predictions)

    simd_input_reader = SimdMeshImplInputReader(params.mesh_impl, input_fn,
                                                params.input_pipeline_shape,
                                                external_worker=True,
                                                is_eval_mode=run_mode == 'sample')

    computation = tpu.replicate(computation=model_fn,
                                inputs=[[]] * params.num_cores,
                                infeed_queue=simd_input_reader.infeed_queue,
                                device_assignment=params.d_assignment)

    if run_mode == 'sample':
        output_dtypes, output_shapes = captured_output_dtypes_shapes.get()
        outfeed_dequeue_ops = []

        # Create outfeed_dequeue_ops.
        for host_id in range(params.num_hosts):
            # pylint: disable=protected-access
            with ops.device(_host_id_to_tf_device(host_id, external_worker=True)):
                for device_ordinal in range(params.num_cores_per_host):
                    outfeed_dequeue_op = tpu_ops.outfeed_dequeue_tuple(
                            dtypes=output_dtypes,
                            shapes=output_shapes,
                            device_ordinal=device_ordinal)

                    # We don't need output other than from core 0.
                    if outfeed_dequeue_ops:
                        outfeed_dequeue_ops.append([tf.reduce_mean(x) for x in outfeed_dequeue_op])
                    else:
                        outfeed_dequeue_ops.append(outfeed_dequeue_op)

    if run_mode == 'train':

        slice_hook = [hook for hook in captured_hooks.get()]
        ckpt_loader_hook = CheckpointLoaderHook(params.model_path)
        step_counter_hook = tf.train.StepCounterHook(every_n_steps=10)
        all_hooks = [ckpt_loader_hook, step_counter_hook] + slice_hook

        # if params.write_summary:
        flush_summary = summary.flush()

        with tf.train.MonitoredTrainingSession(master=tpu_cluster_resolver.master(),
                                               hooks=all_hooks, config=session_config) as sess:
            simd_input_reader.start_infeed_thread(sess)
            summary.initialize(session=sess)

            current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))
            while current_step < params.train_steps:
                sess.run(computation)
                sess.run(flush_summary)

                tf.logging.info('train steps: {}'.format(current_step))

                _current_step = current_step
                current_step += 1

                yield _current_step

    else:  # run_mode == 'sample'

        slice_hook = [hook for hook in captured_hooks.get()]
        ckpt_loader_hook = CheckpointLoaderHook(params.model_path)
        all_hooks = [ckpt_loader_hook, slice_hook[0]]

        with tf.train.MonitoredSession(
                session_creator=tf.train.ChiefSessionCreator(master=tpu_cluster_resolver.master(),
                                                             config=session_config),
                hooks=all_hooks) as sess:

            simd_input_reader.start_infeed_thread(sess)

            while True:
                sess.run(computation)
                yield sess.run(outfeed_dequeue_ops)[0]

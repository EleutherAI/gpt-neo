"""
Contains functions to create a training loop and log its outputs to tensorboard
"""
import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.tpu import tpu_estimator

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


def create_host_call(model_dir: str) -> typing.Optional[typing.Tuple[typing.Callable, typing.List[tf.Tensor]]]:
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

    def maybe_cast(tensor: tf.Tensor) -> tf.Tensor:
        if tensor.shape.is_compatible_with([]):
            tensor = tf.reshape(tensor, [1])
        if tensor.dtype == tf.int64:
            return tf.cast(tensor, tf.int32)
        if tensor.dtype == tf.bfloat16 or tensor.dtype == tf.float16:
            return tf.cast(tensor, tf.float32)
        return tensor

    reshaped_tensors = [maybe_cast(t) for _, t in summaries]

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

    global_step_t = tf.reshape(tf.cast(tf.train.get_global_step(), tf.int32), [1])
    return host_call_fn, [global_step_t] + reshaped_tensors


class _CapturedObject(object):
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


class _CkptLoaderHook(tf.estimator.SessionRunHook):
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


def get_model_fn(params: ModelParameter):
    captured_hooks = _CapturedObject()
    captured_output_dtypes_shapes = _CapturedObject()

    def model_fn(frame) -> tpu_estimator.TPUEstimatorSpec:
        """
        Create model partitioned graph given example input tensor
        :param features: inputs and targets in dict
        :param mode: training mode
        :param params: serialized dict of ModelParameters instance
        :return: tpu estimator spec
        """
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
        #mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        #        mesh_shape, layout_rules, mesh_devices, params.context.device_assignment)

        # Build mtf mesh object
        mesh = mtf.Mesh(graph, "mesh", var_placer)
        params.mesh = mesh

        # Build mtf_features & seq length dict for getting number of microbatches
        # We need to pack inputs into a dict to pass into serialize_training_step
        #params.mode = mode

        frame_input = None
        token_x_input = None
        token_y_input = None
        frame_mask = None
        token_mask = None

        if params.use_video:

            frame_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([frame]),
                                                     params.frame_input_shape, "frame_input")

            if params.use_language:
                token_x_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['token_x']),
                                                           params.token_dim_shape, "tkn_src")
                token_y_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['token_y']),
                                                           params.token_dim_shape, "tkn_tgt")

                frame_mask = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['vid_msk']),
                                                        params.frame_mask_shape, "vid_msk")
                token_mask = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['tkn_msk']),
                                                        params.token_dim_shape, "tkn_msk")

        elif params.use_language and False:

            token_x_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['token_x']),
                                                       params.token_dim_shape, "tkn_src")
            token_y_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor(features['token_y']),
                                                       params.token_dim_shape, "tkn_tgt")

        else:
            raise ValueError("use_video and use_language is both False.")

        if False and params.use_autoregressive_sampling:
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

                language_token_per_frame_dim = mtf.Dimension("language_token_per_frame", params.language_token_per_frame)

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

                frame_input = frame_out * one_hot_sequence + frame_input * neg_one_hot_sequence
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

        if False:
            if params.use_video:
                mtf.scalar_summary("video_loss", video_loss)

            if params.use_language:
                mtf.scalar_summary("token_loss", token_loss)

        update_ops = get_optimizer(mesh, loss, params)
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

        if False:
            predictions = {}
            if params.use_video:
                predictions.update({'frame_out': lowering.export_to_tf_tensor(mtf.anonymize(frame_out))})
                predictions.update({'frame_inp': features['frame']})

            if params.use_language:
                predictions.update({'token_out': lowering.export_to_tf_tensor(mtf.anonymize(token_out))})
                predictions.update({'token_inp': features['token_y']})
        else:
            predictions = None
        # Use our patched version until mtf updates theirs
        host_call = create_host_call(params.model_path)
        mtf.utils.remove_summaries()



        # Creates train_op
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
        tf.logging.info(f"tf_update_ops: {tf_update_ops}")

        master_to_slice_hook = mtf.MtfRestoreHook(lowering)

        with mtf.utils.outside_all_rewrites():

            saver = tf.train.Saver(tf.global_variables(), save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            slice_to_master_hook = tf.train.CheckpointSaverHook(
                params.model_path,
                save_steps=params.steps_per_checkpoint,
                saver=saver, listeners=[saver_listener])

            captured_hooks.capture([master_to_slice_hook, slice_to_master_hook])

            return tf.group([tf_loss] + tf_update_ops)


    return model_fn, captured_hooks, captured_output_dtypes_shapes
"""
Contains functions to create a training loop and log its outputs to tensorboard
"""
import typing
import six

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.compat.v1 import tpu
from tensorflow.python.tpu.ops import tpu_ops
#from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.ops.summary_ops_v2 import scalar, create_file_writer

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


def get_model_fn(params: ModelParameter, run_mode):
    captured_hooks = _CapturedObject()
    captured_output_dtypes_shapes = _CapturedObject()
    host_call = []

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
                    sum_ops.append(scalar(key, value[key], step=gs))

                with tf.control_dependencies(sum_ops):
                    return tf.identity(tf_loss)

            # Cast the global step to tf.int32, since
            # outside_compilation does not support tf.int64.
            tf_loss = tpu.outside_compilation(_host_loss_summary, tf_loss, value,tf.cast(global_step, tf.int32))

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

        elif params.use_language and False:

            token_x_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[0]]),
                                                       params.token_dim_shape, "tkn_src")
            token_y_input = mtf.import_laid_out_tensor(mesh, params.mesh_impl.LaidOutTensor([args[1]]),
                                                       params.token_dim_shape, "tkn_tgt")

        else:
            raise ValueError("use_video and use_language is both False.")

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

                #frame_input = mtf.pad(anonymize(frame_out, sequence_dim),[1, 0], anonymize_dim(sequence_dim)).name * mtf.cast(one_hot_sequence, tf.float32) + frame_input * mtf.cast(neg_one_hot_sequence, tf.float32)
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

        if run_mode == 'train':
            update_ops = get_optimizer(mesh, loss, params)
        else: #run_mode == 'sample'

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

        else: #run_mode == 'sample'
            predictions = {}
            if params.use_video:
                predictions.update({'frame_out': lowering.export_to_tf_tensor(frame_out)})
                predictions.update({'frame_inp': args[0]})

            if params.use_language:
                predictions.update({'token_out': lowering.export_to_tf_tensor(token_out)})
                predictions.update({'token_inp': args[2]})

        master_to_slice_hook = mtf.MtfRestoreHook(lowering)

        if run_mode == 'train':

            # Creates train_op
            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
            tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
            tf.logging.info(f"tf_update_ops: {tf_update_ops}")

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

        else: #run_mode == 'sample'

            predictions = [tf.cast(predictions[key], tf.float32) for key in predictions.keys()]
            predictions_dtypes = [pred.dtype for pred in predictions]
            predictions_shapes = [pred.shape for pred in predictions]
            captured_hooks.capture([master_to_slice_hook, None])
            captured_output_dtypes_shapes.capture([predictions_dtypes, predictions_shapes])

            return tpu_ops.outfeed_enqueue_tuple(predictions)

    return model_fn, captured_hooks, captured_output_dtypes_shapes
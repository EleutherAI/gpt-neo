import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.tpu import tpu_estimator

from .dataclass import ModelParameter
from .optimizers import get_optimizer


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
        if tensor.shape.is_compatible_with([]):
            tensor = tf.reshape(tensor, [1])
        if tensor.dtype == tf.int64:
            return tf.to_int32(tensor)
        if tensor.dtype == tf.bfloat16:
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

    global_step_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
    return host_call_fn, [global_step_t] + reshaped_tensors


def model_fn(features: tf.Tensor, mode: str, params: dict):

    # Get global step
    params = ModelParameter(params)
    global_step = tf.train.get_global_step()

    # Construct mtf graph + mesh from params
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(params.layout)

    # Mesh setup
    num_hosts = params.context.num_hosts
    host_placement_fn = params.context.tpu_host_placement_function
    device_list = [host_placement_fn(host_id=i) for i in range(num_hosts)]
    tf.logging.info(f"device_list = {device_list}")

    var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                  [300 * 1000000 * params.context.num_replicas] + [0] * (num_hosts - 1))
    mesh_devices = [""] * mesh_shape.size
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, layout_rules, mesh_devices, params.context.device_assignment)

    # Trainable variable precision
    # Store to checkpoints in master type, train in slice type, compute in activation type
    variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float32)

    # Build mtf mesh object
    mesh = mtf.Mesh(graph, "the_mesh", var_placer)
    params.mesh = mesh

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    params.mode = mode
    batch_dim = mtf.Dimension("batch", params.the_batch_size)
    sequence_plus = mtf.Dimension("sequence", params.time_patch_size + 1)
    sequence = mtf.Dimension("sequence", params.time_patch_size)
    height = mtf.Dimension("height", params.frame_height_patch)
    language_token = mtf.Dimension("height", params.language_token_per_frame)
    length_dim = mtf.Dimension("color_channels", params.channel_color_size)

    batch_dims = [batch_dim, sequence_plus, height]

    if params.three_axes:
        batch_dims.append(mtf.Dimension("width", params.frame_width_patch))

    batch_dims = batch_dims + [length_dim]
    #frame_input = mtf.import_fully_replicated(mesh, features['frame'], mtf.Shape(batch_dims), "frame_input")
    frame_input = mtf.import_fully_replicated(mesh, features, mtf.Shape(batch_dims), "frame_input")

    with mtf.utils.outside_all_rewrites():
        with tf.variable_scope('jannet'):
            logits, loss = params.build(frame_input)

    _, update_ops, var_grads = get_optimizer(mesh, loss, params, variable_dtype=variable_dtype, inp_var_grads=None)

    mtf.scalar_summary("loss", loss)

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

    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.cast(tf_loss, tf.float32)

    # Use our patched version until mtf updates theirs
    host_call = create_host_call(params.model_path)
    mtf.utils.remove_summaries()

    # Creates train_op
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
    tf.logging.info(f"tf_update_ops: {tf_update_ops}")
    train_op = tf.group(tf_update_ops)

    with mtf.utils.outside_all_rewrites():
        print(params.attribute_accesses())
        restore_hook = mtf.MtfRestoreHook(lowering)

        saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=2,
                defer_build=False,
                save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
                params.model_path,
                save_steps=params.steps_per_checkpoint,
                saver=saver,
                listeners=[saver_listener])

        return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.TRAIN,
                loss=tf_loss,
                host_call=host_call,
                training_hooks=[restore_hook, saver_hook],
                train_op=train_op)

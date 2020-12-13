import mesh_tensorflow as mtf
import mesh_tensorflow.transformer as mtf_transformer
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_estimator

from .dataclass import ModelParameter
from .model import model
from .optimizers import get_optimizer
from .utils import add_mode_to_params, create_host_call, get_graph_info, \
    remove_batch_from_layout, simd_mesh_setup


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str, params: dict):
    # Get global step
    params = ModelParameter(params)
    global_step = tf.train.get_global_step()

    # Construct mtf graph + mesh from params
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        params.layout = remove_batch_from_layout(params.layout)
    layout_rules = mtf.convert_to_layout_rules(params.layout)

    # Mesh setup
    var_placer, mesh_impl = simd_mesh_setup(params, mesh_shape, layout_rules)

    # Trainable variable precision
    # Store to checkpoints in master type, train in slice type, compute in activation type
    variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float32)

    # Build mtf mesh object
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    features_dict = {"inputs": features, "labels": labels}
    sequence_length_dict = {"inputs": params.n_ctx, "labels": params.n_ctx}

    params = add_mode_to_params(params, mode)

    model_input = next(iter(features_dict.values()))
    model_input_shape = model_input.get_shape().as_list()
    batch_dim = mtf.Dimension("batch", model_input_shape[0])
    sequence = mtf.Dimension("sequence", model_input_shape[1])
    width = mtf.Dimension("width", model_input_shape[2])
    height = mtf.Dimension("height", model_input_shape[3])
    batch_dims = [batch_dim, sequence, width, height]
    length_dim = mtf.Dimension("color_channels", model_input_shape[4])

    mtf_features = {}
    for key, x in features_dict.items():
        if x is not None:
            feature_shape = mtf.Shape(batch_dims + [length_dim])
            mtf_features[key] = mtf.import_fully_replicated(
                    mesh, features_dict[key], feature_shape, name=key)

    # Instantiate dict for dimensions, bias, etc that can be calculated here once then passed into model
    other_features = {}
    memory_length_dim = mtf.Dimension("memory_length", length_dim.size)

    # Define other Dimensions that we'll need inside the model
    embd_dim = mtf.Dimension("embd", params.n_embd)
    vocab_dim = mtf.Dimension("vocab", params.n_vocab)

    # We need this because gathering when both the args have the same dimension in them breaks things
    # This dim is specifically for the weights
    # This prevents the "Einsum has lhs dimension without corresponding rhs or output dimension." error
    embed_sequence_dim = mtf.Dimension("embed_sequence", params.n_ctx)

    other_features["embd_dim"] = embd_dim
    other_features["vocab_dim"] = vocab_dim
    other_features["embed_sequence_dim"] = embed_sequence_dim
    other_features["memory_length_dim"] = memory_length_dim

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    # Gets number of microbatches per batch for serialized training
    # if param tokens_per_mb_per_replica = None, this defaults to 1 and no microbatching is performed
    num_microbatches = int(mtf_transformer.utils.serialize_num_microbatches(batch_dim=batch_dim,
                                                                            sequence_length=sequence_length_dict,
                                                                            mesh_shape=mesh_shape,
                                                                            layout_rules=layout_rules,
                                                                            tokens_per_microbatch_per_replica=params.tokens_per_mb_per_replica))

    # Add num microbatches to params

    if num_microbatches > 1:
        # For serialize_training_step we need to modify the model to output results in a dict
        def serialized_fn(mtf_features):
            with tf.variable_scope('gpt2'):
                logits, loss = model(mtf_features, other_features, params, mesh,
                                     variable_dtype=variable_dtype)
            return {"logits": logits, "loss": loss}

        # Serialize the training step - Gradients are accumulated locally and reduced once.
        var_grads, output_dict = mtf.serialize_training_step(mtf_features, serialized_fn, batch_dim, num_microbatches)
        loss = output_dict["loss"]
        logits = output_dict["logits"]

    else:
        with mtf.utils.outside_all_rewrites():
            with tf.variable_scope('gpt2'):
                logits, loss = model(mtf_features, other_features, params, mesh,
                                     variable_dtype=variable_dtype)

    # In TRAIN mode, get optimizer
    _, update_ops, var_grads = get_optimizer(mesh, loss, params, variable_dtype=variable_dtype,
                                             inp_var_grads=None if num_microbatches == 1 else var_grads)

    # Log summaries to tensorboard
    mtf.scalar_summary("loss", loss)

    # Gets & prints info about no. trainable vars in the model & dimension names
    get_graph_info(graph)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.cast(tf_loss, tf.float32)

    # Use our patched version until mtf updates theirs
    host_call = create_host_call(params.model_path, labels)
    mtf.utils.remove_summaries()

    # Creates train_op
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
    tf.logging.info(f"tf_update_ops: {tf_update_ops}")
    train_op = tf.group(tf_update_ops)

    with mtf.utils.outside_all_rewrites():

        # Copy master variables to slices. Must be called first.
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
                train_op=train_op,
                training_hooks=[restore_hook, saver_hook])

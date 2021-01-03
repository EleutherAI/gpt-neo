from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


def get_optimizer(mesh, loss, params, variable_dtype, inp_var_grads=None):
    """Creates and returns an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=params.lr, shape=[], dtype=variable_dtype.slice_dtype)
    clip_value = mtf.constant(mesh, params.gradient_clipping, dtype=variable_dtype.slice_dtype)

    if inp_var_grads is None:
        var_grads = mtf.gradients([loss], [v.outputs[0] for v in mesh.graph.trainable_variables])
    else:
        var_grads = inp_var_grads

    # Cast to full precision
    var_grads_fp = [mtf.cast(v, variable_dtype.slice_dtype) for v in var_grads]

    learning_rate = tf.train.cosine_decay(
        learning_rate,
        global_step,
        params.train_steps,
        alpha=0.1  # Alpha is min lr value as a fraction of init lr.
    )

    if params.warmup_steps > 0:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(params.warmup_steps, dtype=tf.int32)

        dtype = variable_dtype.slice_dtype

        global_steps_float = tf.cast(global_steps_int, dtype)
        warmup_steps_float = tf.cast(warmup_steps_int, dtype)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, dtype)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    learning_rate = mtf.import_fully_replicated(mesh, learning_rate, mtf.Shape([]), name="learning_rate")
    mtf.scalar_summary("lr", learning_rate)

    optimizer = mtf.optimize.AdamWeightDecayOptimizer(learning_rate,
                                                      params.weight_decay,
                                                      params.beta1,
                                                      params.beta2,
                                                      params.epsilon,
                                                      [norm", "bias"],
                                                      variable_dtype
                                                     )

    if params.gradient_clipping is not None:
        global_norm = mtf.sqrt(mtf.add_n([mtf.reduce_sum(mtf.square(t)) for t in var_grads_fp if t is not None]))
        multiplier = clip_value / mtf.maximum(global_norm, clip_value)
        var_grads_fp = [None if t is None else t * multiplier for t in var_grads_fp]

    update_ops = optimizer.apply_grads(var_grads_fp, mesh.graph.trainable_variables)
    return learning_rate, update_ops, var_grads_fp

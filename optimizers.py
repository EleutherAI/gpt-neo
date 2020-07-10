import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf


def get_update_ops(loss, params, graph):
    lr = params["lr"]
    var_grads = mtf.gradients([loss], [v.outputs[0] for v in graph.trainable_variables])
    if "warmup_steps" in params.keys():
        #TODO: warmup won't work in tfm
        raise Exception('Warmup not yet implemented')
        lr = cosine_decay_with_warmup(tf.train.get_global_step(), lr,
                                        params["max_steps"], warmup_steps=params["warmup_steps"])
    optimizer = mtf.optimize.AdamWeightDecayOptimizer(
        learning_rate=lr,
        weight_decay_rate=lr * params["weight_decay"],
        beta_1=params["beta1"],
        beta_2=params["beta2"],
        epsilon=params["epsilon"])
    return optimizer.apply_grads(var_grads, graph.trainable_variables)
    #TODO: add adafactor optimizer


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             name="learning_rate"):
    #TODO: convert to mtf code
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                        'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                                learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                        'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name=name)

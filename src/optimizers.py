from __future__ import absolute_import, division, print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


def get_optimizer(mesh, loss, params, inp_var_grads=None):
    """Creates and returns an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=params.lr, shape=[], dtype=tf.float32)
    clip_value = mtf.constant(mesh, params.gradient_clipping, dtype=tf.float32)

    if inp_var_grads is None:
        var_grads = mtf.gradients([loss], [v.outputs[0] for v in mesh.graph.trainable_variables])
    else:
        var_grads = inp_var_grads

    # Cast to full precision
    var_grads_fp = [mtf.cast(v, tf.float32) for v in var_grads]

    learning_rate = tf.train.cosine_decay(
            learning_rate,
            global_step,
            params.train_steps,
            alpha=0.1  # Alpha is min lr value as a fraction of init lr.
            )

    dtype = tf.float32
    global_steps_int = tf.cast(global_step, tf.int32)
    global_steps_float = tf.cast(global_steps_int, dtype)

    if params.warmup_steps > 0:
        warmup_steps_int = tf.constant(params.warmup_steps, dtype=tf.int32)

        warmup_steps_float = tf.cast(warmup_steps_int, dtype)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, dtype)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    learning_rate = mtf.import_fully_replicated(mesh, learning_rate, mtf.Shape([]), name="learning_rate")
    mtf.scalar_summary("lr", learning_rate)

    optimizer = Ranger(learning_rate,
                       params.weight_decay,
                       0.9,
                       0.95,
                       global_steps_float=global_steps_float
                       )

    if params.gradient_clipping is not None:
        global_norm = mtf.sqrt(mtf.add_n([mtf.reduce_sum(mtf.square(t)) for t in var_grads_fp if t is not None]))
        multiplier = clip_value / mtf.maximum(global_norm, clip_value)
        var_grads_fp = [None if t is None else t * multiplier for t in var_grads_fp]

    update_ops = optimizer.apply_grads(var_grads_fp, mesh.graph.trainable_variables)
    return learning_rate, update_ops, var_grads_fp


class Ranger(mtf.optimize.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-5,
                 N_sma_threshhold=5,
                 alpha=0.5,
                 k=6,
                 use_gc=True,
                 gc_loc=True,
                 global_steps_float=0.):
        """Constructs a AdamWeightDecayOptimizer."""

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta1 = beta_1
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.beta2 = beta_2
        self.epsilon = epsilon
        self.global_steps_float = global_steps_float

    def apply_grad(self, grad, var: mtf.Variable):
        """See base class."""
        if grad is None:
            tf.logging.warning("Gradient is None for variable %s" % var.name)
            return []
        grad = mtf.to_float(grad)
        var_ptr = var
        var = var.value
        exp_avg = exp_avg_ptr = mtf.get_variable(
                var.mesh, var.name + "/ranger/exp_avg", var.shape,
                initializer=tf.zeros_initializer(), trainable=False)

        exp_avg_sq = exp_avg_sq_ptr = mtf.get_variable(
                var.mesh, var.name + "/ranger/exp_avg_sq", var.shape,
                initializer=tf.zeros_initializer(), trainable=False)
        slow_buffer = slow_buffer_ptr = mtf.get_variable(
                var.mesh, var.name + "/ranger/slow_buffer", var.shape,
                initializer=tf.zeros_initializer(), trainable=False)
        slow_buffer = slow_buffer + var * float(self.global_steps_float == 0)

        if self.use_gc and self.gc_loc and var.shape.ndims > 1:
            var -= mtf.reduce_mean(var, output_shape=[var.shape[0]])

        exp_avg_sq = exp_avg_sq * self.beta2 + mtf.square(grad) * (1 - self.beta2)
        exp_avg = exp_avg * self.beta1 + grad * (1 - self.beta1)

        beta2_t = self.beta2 ** self.global_steps_float
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = N_sma_max - 2 * self.global_steps_float * beta2_t / (1 - beta2_t)
        thres = float(N_sma > self.N_sma_threshhold)
        step_size = (mtf.sqrt((1 - beta2_t) * (N_sma - 4) /
                              (N_sma_max - 4) * (N_sma - 2) /
                              N_sma * N_sma_max /
                              (N_sma_max - 2)) / (1 - self.beta1 ** self.global_steps_float) * thres
                     + (1.0 / (1 - self.beta1 ** self.global_steps_float)) * (1 - thres))
        G_grad = exp_avg / ((mtf.sqrt(exp_avg_sq) + self.epsilon) * thres + (1 - thres))

        G_grad += var * self.weight_decay_rate

        if self.use_gc and not self.gc_loc and var.shape.ndims > 1:
            G_grad -= mtf.reduce_mean(G_grad, output_shape=[var.shape[0]])

        var = var - G_grad * step_size * self.learning_rate

        look_ahead = float((self.global_steps_float % self.k) == (self.global_steps_float - 1))
        slow_buffer = slow_buffer + (var - look_ahead) * self.alpha
        var = slow_buffer * look_ahead + var * (1 - look_ahead)

        return [mtf.assign(var_ptr, var),
                mtf.assign(exp_avg_ptr, exp_avg),
                mtf.assign(exp_avg_sq_ptr, exp_avg_sq),
                mtf.assign(slow_buffer_ptr, slow_buffer)]

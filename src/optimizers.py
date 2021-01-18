"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
"""
from __future__ import absolute_import, division, print_function

import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter
from .utils import default


def get_optimizer(mesh: mtf.Mesh, loss: mtf.Tensor, params: ModelParameter
                  ) -> typing.Tuple[mtf.Tensor, typing.List[mtf.Assign], typing.List[mtf.Tensor]]:
    """
    Creates optimizing and update/training operations.
    :param mesh: Mesh Tensorflow mesh
    :param loss: Final scalar loss of the model
    :param params: ModelParameter instance
    :return: scalar learning rate, update operations, gradients
    """
    global_step = tf.train.get_or_create_global_step()
    dtype = tf.float32
    learning_rate = tf.constant(value=params.learning_reate, shape=[], dtype=tf.float32)
    global_steps_float = tf.cast(global_step, tf.float32)
    # Cast to full precision
    learning_rate = tf.train.cosine_decay(learning_rate, global_step, params.train_steps, alpha=0.1)
    # Alpha is min learning_reate value as a fraction of init learning_reate.
    if params.warmup_steps > 0:
        warmup_steps_float = tf.constant(params.warmup_steps, dtype=tf.float32)
        is_warmup = tf.cast(global_steps_float < warmup_steps_float, tf.float32)
        learning_rate = (learning_rate * (is_warmup * global_steps_float / warmup_steps_float + (1 - is_warmup)))

    def _import_constant(name, x):
        return mtf.import_fully_replicated(mesh,
                                           tf.constant(x, default(dtype, dtype), []),
                                           mtf.Shape([]),
                                           name=name)

    learning_rate = mtf.import_fully_replicated(mesh, tf.cast(learning_rate, dtype), [], "learning_reate")
    # global_steps_float = mtf.import_fully_replicated(mesh, tf.cast(global_step, dtype), [], "global_steps_float")
    beta1 = _import_constant("beta1", 0.9)
    beta2 = _import_constant("beta2", 0.95)
    mtf.scalar_summary("learning_reate", learning_rate)

    optimizer = mtf.optimize.AdamWeightDecayOptimizer(learning_rate, params.weight_decay, beta1, beta2)

    clip_value = mtf.constant(mesh, params.gradient_clipping, dtype=dtype)
    var_grads = [None if t is None else mtf.minimum(mtf.maximum(mtf.cast(t, dtype), -clip_value), clip_value)
                 for t in mtf.gradients([loss], [v.outputs[0] for v in mesh.graph.trainable_variables])
                 if t is not None]
    update_ops = optimizer.apply_grads(var_grads, mesh.graph.trainable_variables)

    return learning_rate, update_ops, var_grads


class Ranger(mtf.optimize.Optimizer):
    """WIP Ranger - Highly unstable"""

    def __init__(self,
                 learning_rate: mtf.Tensor,
                 weight_decay_rate: mtf.Tensor,
                 beta_1: mtf.Tensor,
                 beta_2: mtf.Tensor,
                 global_steps_float: mtf.Tensor,
                 epsilon=1e-5,
                 N_sma_threshhold=5,
                 alpha=0.5,
                 k=6):
        """Constructs a AdamWeightDecayOptimizer."""

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta1 = beta_1
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        self.beta2 = beta_2
        self.epsilon = epsilon
        self.global_steps_float = global_steps_float

    def apply_grad(self, grad: mtf.Tensor, var: mtf.Variable):
        """
        See base class.
        Applies Ranger optimizier to gradient/variable pairs.
        :param grad: Gradient for variable
        :param var: Variable to be updates
        :return: Update operations for variable and buffers
        """
        if grad is None:
            tf.logging.warning("Gradient is None for variable %s" % var.name)
            return []
        grad = mtf.cast(grad, self.learning_rate.dtype)
        var_ptr = var
        exp_avg = exp_avg_ptr = mtf.get_variable(var.mesh, var.name + "/ranger/exp_avg", var.shape,
                                                 initializer=tf.zeros_initializer(), trainable=False)

        exp_avg_sq = exp_avg_sq_ptr = mtf.get_variable(var.mesh, var.name + "/ranger/exp_avg_sq", var.shape,
                                                       initializer=tf.zeros_initializer(), trainable=False)
        slow_buffer = slow_buffer_ptr = mtf.get_variable(var.mesh, var.name + "/ranger/slow_buffer", var.shape,
                                                         initializer=tf.zeros_initializer(), trainable=False)
        var = var.value
        slow_buffer = slow_buffer + var * mtf.cast(mtf.equal(self.global_steps_float, 0), var.dtype)

        if var.shape.ndims > 1:
            var -= mtf.reduce_mean(var, output_shape=[var.shape[0]])

        exp_avg_sq = exp_avg_sq * mtf.cast(self.beta2 + mtf.square(grad) * (1 - self.beta2), var.dtype)
        exp_avg = exp_avg * mtf.cast(self.beta1 + grad * (1 - self.beta1), var.dtype)

        beta2_t = mtf.pow(self.beta2, self.global_steps_float)
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = -self.global_steps_float * 2 * beta2_t / (1 - beta2_t) + N_sma_max
        thres = mtf.greater(N_sma, self.N_sma_threshhold)
        thres_var = mtf.cast(thres, var.dtype)
        thres_fp64 = mtf.cast(thres, self.learning_rate.dtype)

        var -= (exp_avg / (mtf.sqrt(exp_avg_sq) + self.epsilon) * thres_var + (1 - thres_var)
                * mtf.cast((mtf.sqrt((1 - beta2_t)
                                     * N_sma_max / N_sma
                                     * (N_sma - 2) / (N_sma_max - 2)
                                     * (N_sma - 4) / (N_sma_max - 4))
                            * thres_fp64 + (1 - thres_fp64))
                           / (1 - mtf.pow(self.beta1, self.global_steps_float))
                           * self.learning_rate, var.dtype)
                + var * self.weight_decay_rate)

        look_ahead = mtf.cast(mtf.equal(mtf.mod(self.global_steps_float, self.k), (self.global_steps_float - 1)),
                              var.dtype)
        slow_buffer = slow_buffer + (var - slow_buffer) * look_ahead * self.alpha
        var = slow_buffer * look_ahead + var * (1 - look_ahead)

        return [mtf.assign(var_ptr, var),
                mtf.assign(exp_avg_ptr, exp_avg),
                mtf.assign(exp_avg_sq_ptr, exp_avg_sq),
                mtf.assign(slow_buffer_ptr, slow_buffer)]

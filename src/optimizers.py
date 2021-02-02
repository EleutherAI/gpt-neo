"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""
from __future__ import absolute_import, division, print_function

import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter
from .utils_core import default


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
    dtype = params.dtype
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
    global_steps_float = mtf.import_fully_replicated(mesh, tf.cast(global_step, dtype), [], "global_steps_float")
    beta1 = _import_constant("beta1", 0.9)
    beta2 = _import_constant("beta2", 0.95)
    mtf.scalar_summary("learning_reate", learning_rate)

    if params.optimizer.lower() == 'novograd':
        optimizer = NovoGrad(learning_rate, params.weight_decay, beta1, beta2)
    elif params.optimizer.lower() == 'adam':
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(learning_rate, params.weight_decay, beta1, beta2)
    elif params.optimizer.lower() == 'factorized_adam':
        optimizer = FactorizedAdam(params.dtype)
    elif params.optimizer.lower() == 'sm3':
        optimizer = SM3(learning_rate, params.weight_decay)
    else:
        raise ValueError(f"{params.optimizer} is not the name of a supported optimizer.")

    clip_value = mtf.constant(mesh, params.gradient_clipping, dtype=dtype)
    var_grads = [None if t is None else mtf.minimum(mtf.maximum(mtf.cast(t, dtype), -clip_value), clip_value)
                 for t in mtf.gradients([loss], [v.outputs[0] for v in mesh.graph.trainable_variables])
                 if t is not None]
    update_ops = optimizer.apply_grads(var_grads, mesh.graph.trainable_variables)

    return update_ops


def weighted_add(left, right, alpha):
    return left * alpha + right * (1 - alpha)


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

        if var.shape.ndims > 1:
            var -= mtf.reduce_mean(var, output_shape=[var.shape[0]])

        exp_avg_sq = weighted_add(exp_avg_sq, mtf.square(grad), self.beta2)
        exp_avg = weighted_add(exp_avg, grad, self.beta2)

        beta2_t = mtf.pow(self.beta2, self.global_steps_float)
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = N_sma_max - self.global_steps_float * 2 * beta2_t / (1 - beta2_t)

        var -= (exp_avg
                * mtf.rsqrt(exp_avg_sq + self.epsilon)
                * mtf.sqrt((1 - beta2_t)
                           * N_sma_max / N_sma
                           * (N_sma - 2) / (N_sma_max - 2)
                           * (N_sma - 4) / (N_sma_max - 4))
                / (1 - mtf.pow(self.beta1, self.global_steps_float))
                * self.learning_rate
                + var * self.weight_decay_rate)

        slow_buffer = slow_buffer + (var - slow_buffer) * (self.alpha / self.k)
        var = slow_buffer * ((self.k - 1) / self.k) + var / self.k

        return [mtf.assign(var_ptr, var),
                mtf.assign(exp_avg_ptr, exp_avg),
                mtf.assign(exp_avg_sq_ptr, exp_avg_sq),
                mtf.assign(slow_buffer_ptr, slow_buffer)]


class NovoGrad(mtf.optimize.Optimizer):
    """WIP Ranger - Highly unstable"""

    def __init__(self,
                 learning_rate: mtf.Tensor,
                 weight_decay_rate: mtf.Tensor,
                 beta_1: mtf.Tensor,
                 beta_2: mtf.Tensor,
                 epsilon=1e-5):
        """Constructs a AdamWeightDecayOptimizer."""

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.epsilon = epsilon

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
        exp_avg = exp_avg_ptr = mtf.get_variable(var.mesh, var.name + "/novograd/exp_avg", var.shape,
                                                 initializer=tf.zeros_initializer(), trainable=False, dtype=var.dtype)
        exp_avg_sq = exp_avg_sq_ptr = mtf.get_variable(var.mesh, var.name + "/novograd/exp_avg_sq", [],
                                                       initializer=tf.zeros_initializer(), trainable=False,
                                                       dtype=var.dtype)

        exp_avg_sq = weighted_add(exp_avg_sq, mtf.reduce_sum(mtf.square(grad)), self.beta2)
        exp_avg = self.beta1 * exp_avg
        rsqrt = mtf.rsqrt(exp_avg_sq + self.epsilon)

        if var.shape.ndims > 1:
            center = mtf.reduce_mean(var.value)
        else:
            center = 0

        if var.size < 2:  # Adam if scalar value
            exp_avg = exp_avg + (1.0 - self.beta1) * grad
            update = exp_avg * rsqrt
        else:  # Novograd if not
            update = exp_avg = exp_avg + grad * rsqrt

        return [mtf.assign_sub(var_ptr, update * self.learning_rate + center + self.weight_decay_rate * var.value),
                mtf.assign(exp_avg_ptr, exp_avg),
                mtf.assign(exp_avg_sq_ptr, exp_avg_sq)]


class FactorizedAdam(mtf.optimize.Optimizer):
    def __init__(self, dtype):
        """Construct a new FactorizedAdam optimizer.
        See class comment.
        Raises:
          ValueError: if absolute_update_scale and relative_update_scale_fn are both
            present or both absent.
        """
        self._learning_rate = tf.cast(tf.minimum(tf.math.rsqrt(tf.cast(tf.train.get_or_create_global_step(),
                                                                       tf.float32) + 1.0), 0.01), dtype)
        self._decay_rate = tf.cast(mtf.optimize.adafactor_decay_rate_pow(0.8), dtype)

    def apply_grad(self, grad, var):
        if grad is None:
            tf.logging.warning("Gradient is None for variable %s" % var.name)
            return []

        with tf.variable_scope(var.name + "/adafactor"):
            updates = []
            grad_factors = []

            for idx, dim in enumerate(var.shape.dims if var.shape.ndims else [None]):
                dim = [dim] if dim else []
                p1_ptr = mtf.get_variable(var.mesh, var.name + f"_dim{idx}_p1", dim,
                                          initializer=tf.zeros_initializer(), trainable=False,
                                          dtype=var.dtype)
                p2_ptr = mtf.get_variable(var.mesh, var.name + f"_dim{idx}_p2", dim,
                                          initializer=tf.zeros_initializer(), trainable=False,
                                          dtype=var.dtype)
                p1 = weighted_add(p1_ptr, mtf.reduce_mean(grad, output_shape=dim), self._decay_rate)
                p2 = weighted_add(p2_ptr, mtf.reduce_mean(mtf.square(grad), output_shape=dim), self._decay_rate)
                updates.extend([mtf.assign(p1_ptr, p1), mtf.assign(p2_ptr, p2)])
                grad_factors.append(p1 * mtf.rsqrt(p2 + 1e-6))

            updates.append(mtf.assign_sub(var,
                                          mtf.add_n(grad_factors)
                                          * mtf.maximum(mtf.optimize.reduce_rms(var.value), 1e-3)
                                          * self._learning_rate
                                          / len(grad_factors)))
            return updates


class SM3(mtf.optimize.Optimizer):
    """SM3 https://arxiv.org/abs/1901.11150"""

    def __init__(self,
                 learning_rate: mtf.Tensor,
                 weight_decay_rate: mtf.Tensor,
                 epsilon=1e-5):

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.epsilon = epsilon

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
        buffer = []
        rank = var.shape.ndims

        if rank == 0:
            buffer.append(mtf.get_variable(var.mesh, var.name + "/sm3/0", var.shape,
                                           initializer=tf.zeros_initializer(), trainable=False, dtype=var.dtype))
        for i in range(rank):
            buffer.append(mtf.get_variable(var.mesh, var.name + f"/sm3/{i}", var.shape,
                                           initializer=tf.zeros_initializer(), trainable=False, dtype=var.dtype))

        update = buffer[0]
        for buf in buffer[1:]:
            update = mtf.minimum(update, buf)
        update += mtf.square(grad)

        return ([mtf.assign_sub(var_ptr, grad * mtf.rsqrt(update + self.epsilon) * self.learning_rate
                                + (0 if rank == 0 else mtf.reduce_mean(var.value))
                                + self.weight_decay_rate * var.value)] +
                [mtf.assign(buf_ptr, mtf.reduce_max(update, output_shape=[dim]))
                 for buf_ptr, dim in zip(buffer, update.shape.dims)])

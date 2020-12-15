from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

def clip_by_global_norm(grads, clip_norm):
    """Clip the grads by global norm."""
    global_norm = mtf.sqrt(mtf.add_n([mtf.reduce_sum(mtf.square(t)) for t in grads if t is not None]))
    multiplier = clip_norm / mtf.maximum(global_norm, clip_norm)
    clipped_grads = [None if t is None else t * multiplier for t in grads]
    return clipped_grads, global_norm

def get_optimizer(mesh, loss, params, variable_dtype, inp_var_grads=None):
    """Creates and returns an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=params["lr"], shape=[], dtype=variable_dtype.slice_dtype)
    clip_value = mtf.constant(mesh, params["gradient_clipping"], dtype=variable_dtype.slice_dtype)

    if inp_var_grads is None:
        var_grads = mtf.gradients([loss], [v.outputs[0] for v in mesh.graph.trainable_variables])
    else:
        var_grads = inp_var_grads

    # Cast to full precision
    var_grads_fp = [mtf.cast(v, variable_dtype.slice_dtype) for v in var_grads]

    # decrease LR to final lr (lr*0.1) by this step - defaults to train_steps
    end_step = params.get("lr_decay_end", params["train_steps"]) 

    if params["lr_decay"] == "linear":
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            end_step,
            end_learning_rate=params["lr"]*0.1, # Decrease to 10% of initial LR according to GPT-3 paper
            power=1.0,
            cycle=False)
    elif params["lr_decay"] == "cosine":
        learning_rate = tf.train.cosine_decay(
            learning_rate,
            global_step,
            end_step,
            alpha=0.1  # Alpha is min lr value as a fraction of init lr.
        )

    if params["warmup_steps"] > 0:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(params["warmup_steps"], dtype=tf.int32)

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

    if params["opt_name"].lower() == "adam":
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=params["weight_decay"],
            beta_1=params["beta1"],
            beta_2=params["beta2"],
            epsilon=params["epsilon"],
            exclude_from_weight_decay=["norm", "bias"],
            variable_dtype=variable_dtype
        )
    else:
        optimizer = mtf.optimize.AdafactorOptimizer(
            learning_rate=params["lr"],
            decay_rate=params["weight_decay"],
            beta1=params["beta1"],
            epsilon1=params["ada_epsilon1"],
            epsilon2=params["ada_epsilon2"]
        )

    if params["gradient_clipping"] is not None:
        (var_grads_fp, _) = clip_by_global_norm(var_grads_fp, clip_norm=clip_value)

    update_ops = optimizer.apply_grads(var_grads_fp, mesh.graph.trainable_variables)
    return learning_rate, update_ops, var_grads_fp


class AdamWeightDecayOptimizer(mtf.optimize.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               variable_dtype=None):
    """Constructs a AdamWeightDecayOptimizer."""

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.variable_dtype = variable_dtype

  def apply_grad(self, grad, var):
    """See base class."""
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []
    
    grad = mtf.to_float(grad)

    assignments = []

    m = mtf.get_variable(
        var.mesh, var.name + "/adam_m", var.shape,
        initializer=tf.zeros_initializer(), 
        # master_dtype=self.variable_dtype.master_dtype, 
        # slice_dtype=self.variable_dtype.slice_dtype, 
        # activation_dtype=self.variable_dtype.activation_dtype, 
        trainable=False)

    v = mtf.get_variable(
        var.mesh, var.name + "/adam_v", var.shape,
        initializer=tf.zeros_initializer(), 
        # master_dtype=self.variable_dtype.master_dtype, 
        # slice_dtype=self.variable_dtype.slice_dtype, 
        # activation_dtype=self.variable_dtype.activation_dtype, 
        trainable=False)

    # Standard Adam update.
    next_m = self.beta_1 * m + (1.0 - self.beta_1) * grad
    next_v = self.beta_2 * v + (1.0 - self.beta_2) * mtf.square(grad)

    update = next_m / (mtf.sqrt(next_v) + self.epsilon)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want to decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.
    if self._do_use_weight_decay(var.name):
      update += mtf.to_float(var.value) * self.weight_decay_rate 

    update_with_lr = self.learning_rate * update

    var_update = mtf.assign_sub(var, update_with_lr)

    assignments.extend(
        [var_update,
         mtf.assign(m, next_m),
         mtf.assign(v, next_v)])
    return assignments

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True
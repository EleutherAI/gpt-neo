# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate schedules for training the transformer.

All learning rate schedule functions must take the scalar named argument `step`
and the numeric argument `total_train_steps`. They must output a tf.Scalar which
is the learning rate for the step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import gin
import tensorflow.compat.v1 as tf


@gin.configurable
def product_learning_rate(step,
                          total_train_steps,
                          factors=gin.REQUIRED,
                          offset=0):
  """Learning rate is the product of one or more factors.

  Takes a list of factors which are either numbers or learning-rate functions
  each taking step and total_train_step arguments.

  If `offset` is nonzero, then subtract offset from the step and from
  total_train_steps before computing the learning rate.

  Args:
    step: a tf.Scalar
    total_train_steps: a number
    factors: a list of numbers and/or functions
    offset: an optional float

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  ret = 1.0
  for f in factors:
    ret *= f(step - offset, total_train_steps - offset) if callable(f) else f
  return ret


@gin.configurable
def linear_decay(step,
                 total_train_steps,
                 steps_or_fraction=0.1):
  """Linearly decay the learning rate to 0.

  If steps_or_fraction > 1 , it is the absolute number of final steps
  over which to decay.  If it is <=1, then it is a fraction of the total number
  of training steps.

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    steps_or_fraction: a number


  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  decay_steps = steps_or_fraction
  if steps_or_fraction <= 1:
    decay_steps *= total_train_steps
  step = tf.cast(step, tf.float32)
  return tf.minimum(1.0, (total_train_steps - step) / decay_steps)


@gin.configurable
def linear_warmup(step,
                  total_train_steps,
                  steps_or_fraction=10000):
  """Linearly warm up the learning rate from 0.

  If steps_or_fraction > 1 , it is the absolute number of initial steps over
  which to warm up.  If it is <=1, then it is a fraction of the total number of
  training steps.

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    steps_or_fraction: a number


  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  warmup_steps = steps_or_fraction
  if steps_or_fraction <= 1:
    warmup_steps *= total_train_steps
  step = tf.cast(step, tf.float32)
  return tf.minimum(1.0, step / warmup_steps)


@gin.configurable
def truncated_rsqrt(step,
                    total_train_steps,
                    warmup_steps=10000):
  """Noam's favorite learning-rate schedule.

  rsqrt(max(step_num, warmup_steps)

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    warmup_steps: a number

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  del total_train_steps
  step_num = tf.cast(step, tf.float32)
  return tf.math.rsqrt(tf.maximum(step_num, warmup_steps))


@gin.configurable
def constant(step, total_train_steps, value=1.0):
  """Constant learning rate (multiplier).

  Args:
    step: a tf.Scalar
    total_train_steps: a number
    value: a number or tf.Scalar

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  del step, total_train_steps
  return value


@gin.configurable
def constant_learning_rate(step, total_train_steps, learning_rate=gin.REQUIRED):
  """Learning rate independent of step.

  DEPRECATED: use constant() or pass a float directly to utils.run.learning_rate

  Args:
    step: a tf.Scalar
    total_train_steps: a number
    learning_rate: a number or tf.Scalar

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  del step, total_train_steps
  return tf.cast(learning_rate, tf.float32)


@gin.configurable
def linear_decay_learning_rate(step,
                               total_train_steps,
                               initial_lr=0.1,
                               offset=0):
  """Linearly decay the learning rate to 0.

  DEPRECATED - use product_learning_rate instead with factors:

  [<initial_lr>,
   @learning_rate_schedules.linear_decay]
  learning_rate_schedules.linear.decay.steps_or_fraction = 1.0


  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    initial_lr: initial learning rate. Decays from here.
    offset: a number used for finetuning. Starts the learning-rate decay
      schedule from this step forwards.

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  offset = tf.cast(offset, tf.float32)
  step = tf.cast(step, tf.float32)

  return initial_lr * tf.minimum(1.0, (total_train_steps - step) /
                                 (total_train_steps - offset))


@gin.configurable
def learning_rate_schedule_noam(step,
                                total_train_steps,
                                warmup_steps=10000,
                                linear_decay_fraction=0.1,
                                multiplier=1.0,
                                offset=0):
  """Noam's favorite learning-rate schedule.

  DEPRECATED - use product_learning_rate instead with factors:

  [<multiplier>,
   @learning_rate_schedules.truncated_rsqrt,
   @learning_rate_schedules.linear_decay]

  (rsqrt(max(step_num, warmup_steps))
   * multiplier
   * min(1.0, (train_steps-step_num)/(train_steps*linear_decay_fraction)))

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    warmup_steps: a number
    linear_decay_fraction: a number
    multiplier: a number
    offset: a number used for finetuning. Starts the learning-rate decay
      schedule from this step forwards. Prior to this step, the learning rate is
      the same as if it were a warmup step.

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  train_steps = float(total_train_steps)
  step_num = tf.cast(step, tf.float32) - offset
  learning_rate = tf.math.rsqrt(tf.maximum(step_num, warmup_steps))
  learning_rate *= multiplier
  if linear_decay_fraction > 0:
    learning_rate *= tf.minimum(1.0, (train_steps - step_num) /
                                (train_steps * linear_decay_fraction))
  return learning_rate


@gin.configurable
def slanted_triangular(step,
                       total_train_steps,
                       cut_fraction=0.1,
                       ratio=32,
                       max_learning_rate=0.01,
                       start_step=0):
  """Triangular learning rate with short increase and long decay.

  TODO(noam): add minimum_value arguments to linear_decay() and linear_warmup()
  so that this function can be replaced.

  Taken from "Universal Language Model Fine-tuning for Text Classification",
  see https://arxiv.org/abs/1801.06146. Default parameters are those specified
  in the paper.

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: a number, the total number of training steps.
    cut_fraction: a number between 0 and 1, fraction of iterations for which we
      are increasing the learning rate.
    ratio: a number greater than 1, the ratio from the smallest learning rate to
      the max learning rate.
    max_learning_rate: a number, the highest learning rate reached during
      training.
    start_step: a number, the step training starts at. Useful when fine-tuning
      from a checkpoint that hasn't had its global step reset.

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  train_steps = float(total_train_steps)
  start_step = float(start_step)
  step_num = tf.cast(step, tf.float32) - start_step
  cut = math.floor(train_steps * cut_fraction)
  p = tf.cond(
      step_num < cut,
      lambda: step_num / cut,
      lambda: 1 - (step_num - cut) / (cut * (1 / cut_fraction - 1)),
  )
  return max_learning_rate * (1 + p * (ratio - 1)) / ratio

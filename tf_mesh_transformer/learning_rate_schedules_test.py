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

# Lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import dataclasses
from mesh_tensorflow.transformer import learning_rate_schedules
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


@dataclasses.dataclass
class LearningRateSpec(object):
  step: tf.Tensor
  total_train_steps: int
  initial_lr: float
  offset: int


def _get_linear_decay_lr(spec):
  return learning_rate_schedules.linear_decay_learning_rate(
      spec.step, spec.total_train_steps, spec.initial_lr, spec.offset)


class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  def testLinearDecayLearningRate(self):
    with self.test_session() as sess:

      # At step 0 (no offset), the learning rate should be initial_lr.
      spec = LearningRateSpec(
          step=tf.constant(0, tf.int32),
          total_train_steps=100,
          initial_lr=0.001,
          offset=0)
      self.assertAlmostEqual(0.001, sess.run(_get_linear_decay_lr(spec)))

      # Halfway, the learning rate should be initial_lr / 2.
      spec.step = tf.constant(50, tf.int32)
      self.assertAlmostEqual(0.0005, sess.run(_get_linear_decay_lr(spec)))

      # At the end of training it should be 0.
      spec.step = 100
      self.assertAlmostEqual(0, sess.run(_get_linear_decay_lr(spec)))

      # If the 0 > step > offset, then lr should be initial_lr.
      spec.offset = 50
      spec.step = 40
      self.assertAlmostEqual(0.001, sess.run(_get_linear_decay_lr(spec)))


if __name__ == "__main__":
  tf.test.main()

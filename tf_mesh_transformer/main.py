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

r"""Transformer using Mesh-TensorFlow.

Training/Eval/Inference of a transformer machine-translation model.

Data comes from TensorFlow Datasets.

The core transformer model code is in the mesh_tensorflow/transformer/
directory of this repository.

Instructions for running this on cloud TPU are in the README .
TODO(noam): instructions are obsolete and need updating.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys
from mesh_tensorflow.transformer import utils
import tensorflow.compat.v1 as tf

tf.flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from"
    " default tpu_worker.")
tf.flags.DEFINE_string(
    "model_dir", "/tmp/transformer_standalone", "Estimator model_dir")


tf.flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

tf.flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

# TFDS Module Import
tf.flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this when your DatasetBuilder is defined outside "
    "of tensorflow_datasets so that it is registered.")

FLAGS = tf.flags.FLAGS


def main(_):
  if FLAGS.module_import:
    for module in FLAGS.module_import:
      importlib.import_module(module)

  tf.io.gfile.makedirs(FLAGS.model_dir)
  suffix = 0
  command_filename = os.path.join(FLAGS.model_dir, "command")
  while tf.io.gfile.exists(command_filename):
    suffix += 1
    command_filename = os.path.join(
        FLAGS.model_dir, "command.{}".format(suffix))
  with tf.io.gfile.GFile(command_filename, "w") as f:
    f.write(" ".join(sys.argv))

  utils.parse_gin_defaults_and_flags()
  utils.run(
      tpu_job_name=FLAGS.tpu_job_name,
      tpu=FLAGS.tpu,
      gcp_project=FLAGS.gcp_project,
      tpu_zone=FLAGS.tpu_zone,
      model_dir=FLAGS.model_dir)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

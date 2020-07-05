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

r"""Dataset utilities for Transformer example.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow_datasets as tfds


class Vocabulary(object):
  """Abstract class for encoding strings as lists of integers.

  We will subclass this and wrap multiple implementations of text encoders.
  We follow the convention that ids 0=PAD and 1=EOS are reserved.
  """

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD and 1=EOS).

    Returns:
      an integer
    """
    raise NotImplementedError("Not implemented.")

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    raise NotImplementedError("Not implemented.")

  def decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    raise NotImplementedError("Not implemented.")

  def encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    raise NotImplementedError("Not implemented.")

  def decode_tf(self, ids):
    """Decode in TensorFlow.

    I don't know when we will use this, but it seems logical to
    have if we can.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    raise NotImplementedError("Not implemented.")


class TFDSVocabulary(Vocabulary):
  """Wrapper for tensorflow_datasets encoders.

  In the TFDS encoders, ID=0 is reserved for padding.
  We want to also reserve ID=1 for EOS, so we shift all IDs up by 1.
  """

  def __init__(self, tfds_encoder):
    self._tfds_encoder = tfds_encoder

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD and 1=EOS).

    Returns:
      an integer
    """
    return self._tfds_encoder.vocab_size + 1

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    # shift IDs up by 1 to make room for EOS=1 (see class docstring)
    return [i + 1 for i in self._tfds_encoder.encode(s)]

  def decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    return self._tfds_encoder.decode([i - 1 for i in ids])


@gin.configurable
def get_tfds_vocabulary(dataset_name=gin.REQUIRED):
  info = tfds.builder(dataset_name).info
  # this assumes that either there are no inputs, or that the
  # inputs and targets have the same vocabulary.
  return TFDSVocabulary(info.features[info.supervised_keys[1]].encoder)

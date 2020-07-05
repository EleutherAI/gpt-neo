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

r"""Wrapper around vocabulary from the Tensor2Tensor library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin

from mesh_tensorflow.transformer import vocabulary


class T2tVocabulary(vocabulary.Vocabulary):
  """Wrapper around tensor2tensor SubwordTextEncoder.

  1 is already reserved for EOS - no need to shift.
  """

  def __init__(self, filepath):
    """Create a T2tVocabulary.

    Args:
      filepath: a string
    """
    # Only import tensor2tensor if necessary.
    from tensor2tensor.data_generators import text_encoder   # pylint: disable=g-import-not-at-top
    from tensor2tensor.data_generators.ops import subword_text_encoder_ops   # pylint: disable=g-import-not-at-top

    self._filepath = filepath
    self._subword_text_encoder = text_encoder.SubwordTextEncoder(filepath)
    self._subword_text_encoder_encode = (
        subword_text_encoder_ops.subword_text_encoder_encode)

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD and 1=EOS).

    Returns:
      an integer
    """
    return self._subword_text_encoder.vocab_size

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    return self._subword_text_encoder.encode(s)

  def decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    return self._subword_text_encoder.decode(ids)

  def encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    ids = self._subword_text_encoder_encode(s, self._filepath)
    # the c++ op apppends 1=EOS - drop it.
    return ids[:-1]


@gin.configurable
def get_t2t_vocabulary(data_dir=gin.REQUIRED,
                       vocabulary_filename=gin.REQUIRED):
  return T2tVocabulary(os.path.join(data_dir, vocabulary_filename))

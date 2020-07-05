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

During training/eval, transformer gets input from a tf.data.Dataset.  The
utilities in this file are for loading such tf.data.Datasets from various data
sources.

Format:

The tf.data.Dataset outputs a dictionary of features, each of which is
an integer tensor with fixed shape [batch_size, sequence_length].

The keys of the dictionary (some of which are optional) are:
{
  "inputs"
  "inputs_segmentation"
  "inputs_position"
  "targets"
  "targets_segmentation"
  "targets_position"
}

We follow the convention that ID=0 represents padding and ID=1 represents EOS.
All sequences are terminated by EOS.  There is no BOS token included.

"inputs" represents the input sequences in a sequence-to-sequence problem.  A
language-modeling problem has no "inputs" feature.

"targets" represents the target sequences of a sequence-to-sequence problem or
the sequences in a language-modeling problem.

A dataset may be "packed", in which case each row in the tensors represents
multiple training examples concatenated together (each terminated by EOS=1).
In this case, the output dictionary will contain additional features:

"inputs_segmentation" (if "inputs" is present)
"targets_segmentation"
"inputs_position" (if "inputs" is present)
"targets_position"

"inputs_segmentation" and "inputs_position" are both aligned with "inputs".

"inputs_segmentation" specifies which of the original examples a particular
token belongs to.  "inputs_position" specifies the position of this token in the
original sequence.  "targets_segmentation" and "targets_position" are similarly
defined.

Example:

Two original sequence-pairs are packed together to form the first combined
example in the batch:

The original sequence-pairs are:
  {"inputs": [8, 7, 1=EOS], "targets": [4, 1=EOS]}
  {"inputs": [2, 3, 4, 1=EOS], "targets": [5, 6, 1=EOS]}

The output dictionary looks like this:
{
               "inputs": [[8, 7, 1, 2, 3, 4, 1, 0, 0, 0], ...]
  "inputs_segmentation": [[1, 1, 1, 2, 2, 2, 2, 0, 0, 0], ...]
      "inputs_position": [[0, 1, 2, 0, 1, 2, 3, 0, 0, 0], ...]
              "targets": [[4, 1, 5, 6, 1, 0, 0, 0, 0, 0], ...]
 "targets_segmentation": [[1, 1, 2, 2, 2, 0, 0, 0, 0, 0], ...]
     "targets_position": [[0, 1, 0, 1, 2, 0, 0, 0, 0, 0], ...]
}

The "_segmentation" tensors have 1s and 2s to demacrate these two original
examples, and 0s for padding.  The "_position" tensors contain the positions
within the two original examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import gin
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


@gin.configurable
def pack_or_pad(
    dataset, length, pack=True, feature_keys=None, ensure_eos=False):
  """Creates a 'packed' version of a dataset or pads examples with zeros.

  If pack=True, then multiple examples concatenated to form one combined
  example with the given length.

  If pack=False, then examples are padded with zeros to 'length'.

  Args:
    dataset: a tf.data.Dataset
    length: an integer or a dict from feature-key to integer
    pack: a boolean, whether to pack (True) or pad (False).
    feature_keys: (optional) list of strings, the feature names to limit
      packing or padding to. Packing will filter out other features whereas
      padding will pass them through unchanged. Defaults to all features.
    ensure_eos: a boolean, whether to replace the final token with EOS=1 if it
      is not PAD=0.
  Returns:
    a tf.data.Dataset where all features have fixed shape [length].
  """
  feature_keys = feature_keys or list(tf.data.get_output_shapes(dataset).keys())
  if pack:
    dataset = pack_dataset(dataset, length=length, keys=feature_keys)
  # Pad/trim length of each example to length.
  dataset = trim_and_pad_dataset(
      dataset, length=length, feature_keys=feature_keys)
  if ensure_eos:
    dataset = ensure_dataset_eos(dataset, feature_keys)
  return dataset


def ensure_dataset_eos(dataset, feature_keys=None):
  """Replaces the final token of features with EOS=1 if it is not PAD=0.

  Args:
    dataset: a tf.data.Dataset
    feature_keys: (optional) list of strings, the feature names to ensure end
      with EOS or padding. Defaults to all features.
  Returns:
    a tf.data.Dataset where all specified features end with PAD=0 or EOS=1.
  """
  feature_keys = feature_keys or tf.data.get_output_shapes(dataset).keys()
  def _ensure_eos(k, v):
    if k not in feature_keys:
      return v
    return tf.concat([v[0:-1], tf.clip_by_value(v[-1:], 0, 1)], axis=0)
  return dataset.map(
      lambda ex: {k: _ensure_eos(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


def encode_dataset(dataset, vocabulary):
  """Encode from strings to token ids.

  Args:
    dataset: a tf.data.Dataset with string values.
    vocabulary: a mesh_tensorflow.transformer.Vocabulary
  Returns:
    a tf.data.Dataset with integer-vector values ending in EOS=1
  """
  def encode(features):
    return {k: vocabulary.encode_tf(v) for k, v in features.items()}
  return dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable
def pretokenized_tfds_dataset(dataset_name=gin.REQUIRED,
                              text2self=gin.REQUIRED,
                              tfds_data_dir=gin.REQUIRED,
                              dataset_split=gin.REQUIRED,
                              batch_size=None,
                              sequence_length=gin.REQUIRED,
                              vocabulary=None):
  """Reads a tensorflow_datasets dataset.

  Args:
    dataset_name: a string
    text2self: a boolean
    tfds_data_dir: a boolean
    dataset_split: a string
    batch_size: an integer, DEPRECATED
    sequence_length: an integer
    vocabulary: ignored
  Returns:
    a tf.data.Dataset of batches
  """
  del batch_size
  del vocabulary
  dataset = tfds.load(
      dataset_name,
      split=dataset_split,
      as_supervised=True,
      data_dir=tfds_data_dir,
      shuffle_files=dataset_split == "train")
  if dataset_split == "train":
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000)
  def shift_and_append_eos(t):
    # tfds encoder does not reserve an EOS token, so we need to shift
    # in order to do so.  We also append EOS=1.
    return tf.concat([t + 1, [1]], 0)
  def feature_map(inputs, targets):
    if text2self:
      return {"targets": shift_and_append_eos(targets)}
    else:
      return {"inputs": shift_and_append_eos(inputs),
              "targets": shift_and_append_eos(targets)}
  dataset = dataset.map(feature_map,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return pack_or_pad(dataset, sequence_length)


@gin.configurable
def sample_from_text_line_datasets(glob_weight_list,
                                   shuffle_buffer_size=100000,
                                   prefetch=1000):  # pylint: disable=missing-docstring
  globs, weights = zip(*glob_weight_list)
  datasets = [
      tf.data.TextLineDataset(tf.gfile.Glob(g)).repeat().shuffle(
          shuffle_buffer_size).prefetch(prefetch) for g in globs
  ]
  return tf.data.experimental.sample_from_datasets(
      datasets=datasets, weights=weights)


@gin.configurable
def make_text_line_dataset(glob=gin.REQUIRED):
  return sample_from_text_line_datasets([(glob, 1.0)])


@gin.configurable
def simple_text_line_dataset(glob=gin.REQUIRED, shuffle_buffer_size=100000):
  return tf.data.TextLineDataset(
      tf.gfile.Glob(glob)).shuffle(shuffle_buffer_size)


@gin.configurable
def packed_parallel_tsv_dataset(dataset=gin.REQUIRED,
                                dataset_split=gin.REQUIRED,
                                batch_size=None,
                                sequence_length=gin.REQUIRED,
                                vocabulary=gin.REQUIRED,
                                append_eos=True,
                                eos_id=1,
                                max_encoded_len=0):
  """Reads parallel tab-separated text file. One example per line."""
  del batch_size
  del dataset_split

  def _parse_fn(record):  # pylint: disable=missing-docstring
    tokens = tf.decode_csv(
        record,
        record_defaults=[""] * 2,
        field_delim="\t",
        use_quote_delim=False)
    return {"inputs": tokens[0], "targets": tokens[1]}

  def _encode_fn(features):  # pylint: disable=missing-docstring
    inputs_vocabulary = vocabulary[0] if isinstance(vocabulary,
                                                    tuple) else vocabulary
    targets_vocabulary = vocabulary[1] if isinstance(vocabulary,
                                                     tuple) else vocabulary
    inputs_enc = inputs_vocabulary.encode_tf(features["inputs"])
    targets_enc = targets_vocabulary.encode_tf(features["targets"])
    if append_eos:
      inputs_enc = tf.concat([tf.cast(inputs_enc, tf.int64), [eos_id]], 0)
      targets_enc = tf.concat([tf.cast(targets_enc, tf.int64), [eos_id]], 0)
    return {"inputs": inputs_enc, "targets": targets_enc}

  dataset = dataset.map(
      _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      _encode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _filter_fn(features):  # pylint: disable=missing-docstring
    return tf.less_equal(
        tf.reduce_max(
            tf.stack([tf.size(v) for v in features.values()], axis=0)),
        max_encoded_len)

  if max_encoded_len:
    tf.logging.info("Filtering encoded examples longer than %d" %
                    max_encoded_len)
    dataset = dataset.filter(_filter_fn)

  return pack_or_pad(dataset, sequence_length)


@gin.configurable
def untokenized_tfds_dataset(dataset_name=gin.REQUIRED,
                             text2self=gin.REQUIRED,
                             tfds_data_dir=gin.REQUIRED,
                             dataset_split=gin.REQUIRED,
                             batch_size=None,
                             sequence_length=gin.REQUIRED,
                             vocabulary=gin.REQUIRED,
                             pack=gin.REQUIRED):
  """Reads a tensorflow_datasets dataset.

  Returns a tf.data.Dataset containing single tokenized examples where each
  feature ends in EOS=1.

  Args:
    dataset_name: a string
    text2self: a boolean, if true, run unsupervised LM-style training. if false,
      the dataset must support supervised mode.
    tfds_data_dir: a boolean
    dataset_split: a string
    batch_size: an integer
    sequence_length: an integer
    vocabulary: a vocabulary.Vocabulary
    pack: if True, multiple examples emitted by load_internal() are concatenated
        to form one combined example.
  Returns:
    a tf.data.Dataset of batches
  """
  del batch_size
  dataset = tfds.load(
      dataset_name, split=dataset_split,
      as_supervised=not text2self, data_dir=tfds_data_dir)
  if dataset_split == "train":
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000)
  if not text2self:
    dataset = supervised_to_dict(dataset, text2self)
  dataset = encode_all_features(dataset, vocabulary)
  return pack_or_pad(dataset, sequence_length, pack)


def supervised_to_dict(dataset, text2self):
  """Turns a supervised dataset into a dataset with a feature dictionary.

  if text2self, then the features dictionary contains a "targets" key.
  else, the features dictionary contains "inputs" and "targets" keys.

  Args:
    dataset: a tf.data.Dataset
    text2self: a boolean
  Returns:
    a tf.data.Dataset
  """
  def my_fn(inputs, targets):
    if text2self:
      return {"targets": targets}
    else:
      return {"inputs": inputs, "targets": targets}
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def encode_all_features(dataset, vocabulary):
  """Encode all features.

  Args:
    dataset: a tf.data.Dataset
    vocabulary: a vocabulary.Vocabulary
  Returns:
    a tf.data.Dataset
  """
  def my_fn(features):
    """Encode all features that are strings and return a dictionary.

    Args:
      features: a dictionary
    Returns:
      a dictionary
    """
    ret = {}
    for k, v in features.items():
      if v.dtype == tf.string:
        v = vocabulary.encode_tf(v)
        v = tf.concat([tf.cast(v, tf.int64), [1]], 0)
        ret[k] = v
      else:
        tf.logging.info(
            "encode_all_features: skipping non-string feature %s:%s", k, v)
    return ret
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def pretokenized_tfrecord_dataset(filenames,
                                  text2self,
                                  eos_included,
                                  repeat,
                                  batch_size,
                                  sequence_length,
                                  vocab_shift=0):
  """Reads tensor2tensor-style data files.

  The dataset is defined by sets of TFRecord files of TFExample protos.
  There should be a "targets" feature (a 1d tensor of integers)
  If not text2self, there should also be an "inputs" feature.
  Other features get ignored.

  eos_included specifies whether the inputs and targets were written with an
  EOS token, as in tensor2tensor

  Args:
    filenames: a list of strings
    text2self: a boolean
    eos_included: a boolean
    repeat: a boolean
    batch_size: an integer, DEPRECATED
    sequence_length: an integer
    vocab_shift: an optional integer - add this value to all ids
  Returns:
    A tf.data.Dataset of batches
  """
  del batch_size
  dataset = tf.data.TFRecordDataset(filenames, buffer_size=64 * 1024 * 1024)
  if repeat:
    dataset = dataset.repeat()
  keys = ["targets"] if text2self else ["inputs", "targets"]
  def decode_example(serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""
    decoded = tf.io.parse_example(
        serialized=[serialized_example],
        features={k: tf.VarLenFeature(tf.int64) for k in keys})
    decoded = {k: v.values for k, v in decoded.items()}
    if vocab_shift:
      decoded = {k: v + vocab_shift for k, v in decoded.items()}
    if not eos_included:
      decoded = {k: tf.concat([v, [1]], 0) for k, v in decoded.items()}
    return decoded
  dataset = dataset.map(decode_example,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return pack_or_pad(dataset, sequence_length)


@gin.configurable
def pretokenized_t2t_dataset(dataset_name=gin.REQUIRED,
                             text2self=False,
                             data_dir=gin.REQUIRED,
                             dataset_split="train",
                             batch_size=None,
                             sequence_length=gin.REQUIRED,
                             vocabulary=None,
                             eos_included=True,
                             vocab_shift=0):
  """Loads the Tensor2tensor dataset specified by dataset_name.

  Args:
    dataset_name: TensorFlow Datasets dataset name.
    text2self: a boolean
    data_dir: string, data_dir for TensorFlow Datasets
    dataset_split: a string - "train" or "dev"
    batch_size: an integer, DEPRECATED
    sequence_length: an integer
    vocabulary: ignored
    eos_included: a boolean
    vocab_shift: an optional integer - add this value to all ids read

  Returns:
    A tf.data.Dataset of batches
  """
  del vocabulary
  filepattern = os.path.join(
      data_dir, dataset_name + "-" + dataset_split + "-*")
  filenames = tf.gfile.Glob(filepattern)
  tf.logging.info("Found %s files matching %s" % (len(filenames), filepattern))
  if not filenames:
    raise ValueError("No matching files found")
  dataset = pretokenized_tfrecord_dataset(
      filenames=filenames,
      text2self=text2self,
      eos_included=eos_included,
      repeat=dataset_split == "train",
      batch_size=batch_size,
      sequence_length=sequence_length,
      vocab_shift=vocab_shift)
  if dataset_split == "train":
    dataset = dataset.shuffle(1000)
  return dataset


@gin.configurable
def pack_dataset(dataset, length, keys=None, use_custom_ops=False):
  """Creates a 'packed' version of a dataset on-the-fly.

  Borrowed from the tensor2tensor library.
  TODO(noam): make this faster

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.

  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }

  0 represents padding in both the inputs and the outputs.

  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])
    use_custom_ops: a boolean - custom ops are faster but require a custom-built
      binary, which is not currently possible on cloud-tpu.

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.data.get_output_shapes(dataset)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError("Key %s not found in dataset.  Available keys are %s"
                       % (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError("Tensors to be packed must be one-dimensional.")
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  length_dict = {}
  for k in keys:
    for suffix in ["", "_segmentation", "_position"]:
      length_dict[k + suffix] = length if isinstance(length, int) else length[k]
  length = length_dict

  # trim to length
  dataset = dataset.map(lambda x: {k: x[k][:length[k]] for k in keys},
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  if use_custom_ops and len(keys) <= 2:
    dataset = _pack_with_custom_ops(dataset, keys, length)
  else:
    dataset = _pack_with_tf_ops(dataset, keys, length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [length[k]]) for k, v in x.items()}
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _pack_with_tf_ops(dataset, keys, length):
  """Helper-function for packing a dataset which has already been batched.

  See pack_dataset()

  Uses tf.while_loop.  Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + "_position"] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k],
                 [[0, length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example
    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length[k]])
      outputs[k + "_position"] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length[k]])
    def cond_fn(i, partial, outputs):
      del partial, outputs
      return i < dynamic_batch_size
    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray
      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), length[k]))
      def false_fn():
        return write_packed_example(partial, outputs)
      def true_fn():
        return partial, outputs
      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_position"] = tf.concat(
            [partial[k + "_position"],
             tf.range(new_seq_len, dtype=tf.int32)], 0)
      partial = new_partial
      return i+1, partial, outputs

    i, partial, outputs = tf.while_loop(
        cond_fn, body_fn, (i, partial, outputs),
        back_prop=False,
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
            ))
    partial, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segmentation"] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + "_position"], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed
  dataset = dataset.map(map_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.unbatch()


def _pack_with_custom_ops(dataset, keys, length):
  """Helper-function for packing a dataset which has already been batched.

  See pack_dataset()

  Relies on custom ops which require a custom compiled binary.
  Faster than _pack_with_tf_ops(), and denser packing.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings (must have length 1 or 2)
    length: a dictionary from key to integer

  Returns:
    a dataset.
  """
  from tensor2tensor.data_generators.ops import pack_sequences_ops  # pylint: disable=g-import-not-at-top
  # faster and better packing but requires custom-built binary.
  if len(keys) == 1:
    k1, = keys
    k2 = k1
  elif len(keys) == 2:
    k1, k2 = keys
  else:
    raise ValueError("must have 1 or 2 keys")
  def map_fn_custom(x):
    """Map-function."""
    (k1_packed, k1_segmengation, k1_position,
     k2_packed, k2_segmentation, k2_position) = (
         pack_sequences_ops.pack_sequences2(
             x[k1], x[k2], length[k1], length[k2]))
    packed = {
        k1: k1_packed,
        k1 + "_segmentation": k1_segmengation,
        k1 + "_position": k1_position,
    }
    if len(keys) == 2:
      packed.update({
          k2: k2_packed,
          k2 + "_segmentation": k2_segmentation,
          k2 + "_position": k2_position,
      })
    return packed
  dataset = dataset.map(map_fn_custom,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset


def trim_and_pad_dataset(dataset, length, feature_keys=None):
  """Trim and pad first dimension of features to size `length`.

  Args:
    dataset: tf.data.Dataset, the dataset to trimp/pad examples in.
    length: int, or a dict from feature-key to int
    feature_keys: (optional) list of strings, the feature names to limit
      trimming/padding to. Defaults to all features.
  Returns:
    Trimmed/padded tf.data.Dataset.
  """
  def _trim_and_pad(k, t):
    """Trim/pad to the first axis of `t` to be of size `length`."""
    if feature_keys and k not in feature_keys:
      return t
    length_k = length if isinstance(length, int) else length[k]
    t = t[:length_k]
    pad_amt = length_k - tf.shape(t)[0]
    padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1))
    padded_t.set_shape([length_k] + t.shape.as_list()[1:])
    return padded_t

  return dataset.map(
      lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


EvalDataset = collections.namedtuple(
    "EvalDataset",
    [
        "name",  # string, the task name
        "dataset_fn",  # function which returns a tf.data.Dataset
        "postprocess_fn",  # function which converts decodes to evalable strs
        "metric_fns",  # list of metric_fn(targets, predictions) returning dicts
    ]
)


def pad_dataset_with_zeroed_out_examples(ds):
  def _zero_out(x):
    return {k: tf.zeros_like(v) for k, v in x.items()}
  return ds.concatenate(ds.map(_zero_out).repeat())

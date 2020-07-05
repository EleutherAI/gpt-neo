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

r"""Utilities for running training and inference.

The `run` function for training the Transformer model is defined in this file.

TODO(katherinelee): add details about gin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import random
import re

import gin
import gin.tf

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer import learning_rate_schedules
from mesh_tensorflow.transformer import transformer
import numpy as np
import pkg_resources
import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from tensorflow.python.ops import resources  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import

tf.flags.DEFINE_multi_string("gin_file", None, "Path to a Gin file.")
tf.flags.DEFINE_multi_string("gin_param", None, "Gin parameter binding.")
tf.flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")

FLAGS = tf.flags.FLAGS

_DEFAULT_CONFIG_FILE = "./gin/defaults.gin"

# List of features used by model.
_MODEL_FEATURES = [
    "inputs", "inputs_position", "inputs_segmentation", "targets",
    "targets_position", "targets_segmentation", "targets_subsegmentation"
]


def _filter_features(ex):
  """Filters example features, keeping only valid model features."""
  return {k: v for k, v in ex.items() if k in _MODEL_FEATURES}


def parse_gin_defaults_and_flags():
  """Parses all default gin files and those provided via flags."""
  # Register .gin file search paths with gin
  for gin_file_path in FLAGS.gin_location_prefix:
    gin.add_config_file_search_path(gin_file_path)
  # Set up the default values for the configurable parameters. These values will
  # be overridden by any user provided gin files/parameters.
  gin.parse_config_file(
      pkg_resources.resource_filename(__name__, _DEFAULT_CONFIG_FILE))
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)


# TODO(noam): maybe add gin-config to mtf.get_variable so we can delete
#  this stupid VariableDtype class and stop passing it all over creation.
@gin.configurable
def get_variable_dtype(
    master_dtype=tf.bfloat16,
    slice_dtype=tf.float32,
    activation_dtype=tf.float32):
  """Datatypes to use for the run.

  Args:
    master_dtype: string, datatype for checkpoints
      keep this the same between training and eval/inference
    slice_dtype: string, datatype for variables in memory
      must be tf.float32 for training
    activation_dtype: string, datatype for activations
      less memory usage if tf.bfloat16 but possible numerical issues
  Returns:
    a mtf.VariableDtype
  """
  return mtf.VariableDType(
      master_dtype=tf.as_dtype(master_dtype),
      slice_dtype=tf.as_dtype(slice_dtype),
      activation_dtype=tf.as_dtype(activation_dtype))


def inputs_vocabulary(vocabulary):
  """Get the inputs vocabulary.

  Args:
    vocabulary: Vocabulary or (inputs_vocabulary, targets_vocabulary) tuple.

  Returns:
    a Vocabulary
  """
  if isinstance(vocabulary, tuple):
    vocabulary = vocabulary[0]
  return vocabulary


def targets_vocabulary(vocabulary):
  """Get the targets vocabulary.

  Args:
    vocabulary: Vocabulary or (inputs_vocabulary, targets_vocabulary) tuple.

  Returns:
    a Vocabulary
  """
  if isinstance(vocabulary, tuple):
    vocabulary = vocabulary[1]
  return vocabulary


@gin.configurable
def separate_vocabularies(inputs=gin.REQUIRED, targets=gin.REQUIRED):
  """Gin-configurable helper function to generate a tuple of vocabularies."""
  return (inputs, targets)


# TODO(katherinelee): Update layout_rules string when noam updates the
# definition in run
def build_model(model_type="bitransformer",
                input_vocab_size=gin.REQUIRED,
                output_vocab_size=gin.REQUIRED,
                layout_rules=None,
                mesh_shape=None):
  """Build a transformer model.

  Currently, four types of models are supported:

  "bitransformer": The traditional encoder-decoder architecture from
     "Attention is All You Need".  Requires a non-text2self dataset.

  "lm": an autoregressive language model (one layer stack).  Effectively the
     decoder of the bitransformer. There is no attention over the encoder, since
     there is no encoder.  Requires a text2self dataset, with targets, but no
     inputs.

  "delimited_lm": an autoregressive language model trained on a text2text
     dataset.  Each training example is expressed as
     [<input_tokens>, EOS, <target_tokens>, EOS].  Model checkpoints are
     compatible with "lm" models.  One strategy is to pretrain as "lm"
     then fine-tune as "delimited_lm".

  "aligned": a non-autoregressive single-stack model (like BERT).  Requires
     a non-text2self dataset with inputs and targets.  The targets and inputs
     have the same length and each entry in the inputs is aligned to the
     corresponding entry in targets, eg:
      "inputs": "The X sat on X X."
      'targets": "The cat sat on the mat."
      (except, inputs are token ID sequences, not strings)

  "bi_teacher_student": a teacher-student model where both the student and
    teacher are bitransformers. Requires a non-text2self dataset.

  A text2self dataset has targets that are offset of the inputs. Non-text2self
  datasets have targets that differ from their inputs, like:
    input: 'hello'
    target: 'bonjour'

  Args:
    model_type: a string, one of "bitransformer", "lm", "delimited_lm",
      "aligned", or "bi_teacher_student"
    input_vocab_size: an integer
    output_vocab_size: an integer
    layout_rules: optional, input to mtf.convert_to_layout_rules
    mesh_shape: optional, an input to mtf.convert_to_shape()
  Returns:
    a Unitransformer or Bitransformer
  """
  if model_type == "bitransformer":
    return transformer.make_bitransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        mesh_shape=mesh_shape,
        layout=layout_rules)
  elif model_type == "bi_student_teacher":
    return transformer.make_bi_student_teacher(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        mesh_shape=mesh_shape,
        layout=layout_rules)
  elif model_type in ["lm", "delimited_lm", "aligned"]:
    return transformer.Unitransformer(
        autoregressive=model_type in ["lm", "delimited_lm"],
        layer_stack=transformer.make_layer_stack(),
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        mesh_shape=mesh_shape,
        layout=layout_rules)
  else:
    raise ValueError("unknown model_type")


@gin.configurable
def tpu_mesh_shape(tpu_topology=gin.REQUIRED,
                   model_parallelism=gin.REQUIRED,
                   ensemble_parallelism=None):
  """Create a mesh_shape for data-parallelism and model-parallelism on TPU.

  Example: tpu_mesh_shape("4x4", 8) -> mtf.Shape(("batch", 4), ("model", 8))
  Since there are 4x4x2=32 total cores, and we want 8-way model paralleism.

  This function is passed through gin to the argument `mesh_shape` inside the
  function `run`.

  Alternatively, for model_parallelism, pass a mesh_spec (see simd_mesh_impl.py)
  TODO(noam): describe

  Args:
    tpu_topology: a string - e.g. "2x2" or "v3-8"
    model_parallelism: an integer - the number of cores per model replica
      alternatively a list that can be passed to
      simd_mesh_impl.HierarchicalTiling
    ensemble_parallelism: an optional integer - if present then create an
      "ensemble" mesh-dimension as well, for splitting the models in an
      ensemble.
  Returns:
    a mtf.Shape
  """
  if tpu_topology.startswith("v"):
    num_cores = int(tpu_topology.split("-")[-1])
  else:
    x, y = tpu_topology.split("x")
    num_cores = int(x) * int(y) * 2
  if isinstance(model_parallelism, list):
    # model_parallelism is actually a spec used to
    # construct a simd_mesh_impl.HierarchicalTiling object
    return mtf.simd_mesh_impl.HierarchicalTiling.spec_to_mesh_shape(
        model_parallelism, num_cores)
  data_parallelism = num_cores // model_parallelism
  if ensemble_parallelism:
    data_parallelism //= ensemble_parallelism
  dims = []
  if ensemble_parallelism and ensemble_parallelism > 1:
    dims.append(mtf.Dimension("ensemble", ensemble_parallelism))
  if data_parallelism > 1:
    dims.append(mtf.Dimension("batch", data_parallelism))
  if model_parallelism > 1:
    dims.append(mtf.Dimension("model", model_parallelism))
  return mtf.Shape(dims)


@gin.configurable
def variable_filter_max_size(v, max_size=1e7):
  return v.size <= max_size


@gin.configurable
def tpu_estimator_model_fn(model_type,
                           transformer_model,
                           vocabulary,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           batch_size,
                           sequence_length,
                           autostack,
                           keep_checkpoint_max,
                           save_checkpoints_steps,
                           learning_rate_schedule=None,
                           optimizer=None,
                           outer_batch_size=1,
                           tpu_summaries=False,
                           predict_fn=None,
                           score_in_predict_mode=False,
                           variable_filter=None,
                           init_checkpoint=None,
                           ensemble_inputs=None,
                           mesh_devices=None,
                           model_info_file=None,
                           hierarchical_tiling_spec=None):
  """Create a TPUEstimator model function.

  Args:
    model_type: a string. One of "bitransformer", "lm", "delimited_lm",
      "aligned", or "bi_teacher_student"
    transformer_model: a transformer.Unitransformer or transformer.Bitransformer
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple. Used for decoding in predict mode.
    model_dir: a string, directory to save the model to.
    use_tpu: a boolean
    mesh_shape: a mtf.Shape
    layout_rules: a mtf.LayoutRules
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    autostack: a boolean
    keep_checkpoint_max: an integer, maximum number of checkpoints to keep
    save_checkpoints_steps: an integer, save a checkpoint every this number of
      steps
    learning_rate_schedule: a constant or a function from step to learning rate
    optimizer: a class extending optimize.Optimizer, required for training
    outer_batch_size: outer batch dimension that could be used to enable the mix
      of data-parallel and model-parallel training of Mixture of Experts (MoE)
      models
    tpu_summaries: a boolean, use rewrites to make summaries work on TPU.  This
      may be slow, since it uses a host call hack.
    predict_fn: an optional function, see docs for `run` for more information.
    score_in_predict_mode: compute log-likelihood scores instead of predictions
    variable_filter: controls which variables are trained.
      If None (default), train all trainable variables.
      If a string regex, train all variables that match this regex.
      If a function (mtf.Variable -> boolean), then train variables for which
        the function returns True.
    init_checkpoint: a string, if not None then read in variables from this
      checkpoint path when initializing variables. Will only initialize
      variables that appear both in the current graph and the checkpoint.
    ensemble_inputs: an optional integer - pass the size of the ensemble to
      train an ensemble where each model gets different inputs.
      You also need to configure Unitransformer.ensemble  to the right size.
      If None, then all models are trained on the same inputs.
    mesh_devices: a list of strings, the device names to use for each mesh
      slice. Only required for GPU.
    model_info_file: an optional string, information about variables and
      operations will be logged to this file during the TRAIN mode.
    hierarchical_tiling_spec: an optional list that can be passed as the
      spec argument to simd_mesh_impl.HierarchicalTiling
  Returns:
    a function to be passed to TPUEstimator
  """
  mesh_devices = mesh_devices or [""] * mesh_shape.size

  def my_model_fn(features, labels, mode, params=None, config=None):
    """Estimator model function.

    Args:
      features: dictionary where keys are strings like "inputs" and "targets"
        and the values are the actual values of "inputs". See TPUEstimator's
        docs for more information
      labels: ignored argument
      mode: a tf.estimator.ModeKeys
      params: dictionary containing the key "context"
      config: ignored argument

    Returns:
      a TPUEstimatorSpec
    """
    del labels, config
    if mode == tf.estimator.ModeKeys.PREDICT and score_in_predict_mode:
      mode = "score"
    global_step = tf.train.get_global_step()
    if use_tpu and "context" in params:
      ctx = params["context"]
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      # TODO(ylc): Better estimation of replica cache size?
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      physical_shape = [int(i) for i in
                        params["context"].device_assignment.topology.mesh_shape]
      if len(physical_shape) == 4:
        physical_shape = (
            mtf.simd_mesh_impl.physical_shape_3d_from_topology_proto_4d(
                physical_shape))
      if hierarchical_tiling_spec is not None:
        logical_to_physical = mtf.simd_mesh_impl.HierarchicalTiling(
            hierarchical_tiling_spec,
            physical_shape).logical_to_physical
      else:
        logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
            mesh_shape.to_integer_list, physical_shape)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment,
          logical_to_physical=logical_to_physical)
    else:
      var_placer = None
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    if (outer_batch_size and
        mode not in [tf.estimator.ModeKeys.PREDICT, "score"]):
      outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
      batch_dim = mtf.Dimension("batch", batch_size // outer_batch_size)
      batch_dims = [outer_batch_dim, batch_dim]
    else:
      batch_dim = mtf.Dimension("batch", batch_size)
      batch_dims = [batch_dim]
    ensemble_dims = ([mtf.Dimension("ensemble", ensemble_inputs)]
                     if ensemble_inputs else [])

    mtf_features = {}
    for key, x in features.items():
      # Some auxiliary features may have been generated in packing.
      # The names of these new features are of the form
      #   "<original_feature_name>_<suffix>", e.g. "inputs_segmentation".
      #   We look up the lengths based on the original feature name, without
      #   the "_<suffix>".
      feature_length = sequence_length[key.split("_")[0]]
      length_dim = mtf.Dimension("length", feature_length)
      feature_shape = mtf.Shape(
          ensemble_dims + batch_dims + [length_dim])
      x = tf.cast(features[key], tf.int32)
      x = tf.reshape(x, feature_shape.to_integer_list)
      if not use_tpu:
        tf.logging.info("feature %s : %s" % (key, x))
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=10)
      mtf_features[key] = mtf.import_fully_replicated(
          mesh, x, feature_shape, name=key)

    def _verify_feature_exists(feature_name, should_exist):
      if should_exist != (feature_name in mtf_features):
        message = (
            "mode=%s model_type=%s should%s have feature %s" %
            (mode, model_type, "" if should_exist else " not", feature_name))
        if "lm" in model_type:
          message += (
              "\nA common mistake is that model_type=\"lm\" should be used "
              "with tasks that produce inputs and targets, while "
              "model_type=\"delimited_lm\" should be used with tasks that "
              "produce targets only.")
        raise ValueError(message)

    # Verify that the right features exist, and transform them if necessary
    if mode == tf.estimator.ModeKeys.PREDICT:
      _verify_feature_exists("inputs", True)
      # "targets" may or may not exist depending on whether we are doing
      # evaluation or open-ended inference.
    else:
      _verify_feature_exists("targets", True)
      _verify_feature_exists("inputs", model_type != "lm")
      if model_type == "delimited_lm":
        mtf_features = _dynamic_text2self(mtf_features)

    if mode == "score":
      # compute log-likelihoods per sequence
      if predict_fn:
        # predict_fn contains a custom scoring function
        # this code-path has not been tested
        scores = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      targets = mtf_features["targets"]
      if isinstance(transformer_model, transformer.Unitransformer):
        length_dim = targets.shape.dims[-1]
        inputs = mtf.shift(mtf_features["targets"], offset=1,
                           dim=length_dim, wrap=False)
      elif isinstance(transformer_model,
                      (transformer.Bitransformer,
                       transformer.StudentTeacher)):
        inputs = mtf_features["inputs"]
        weights = None
      else:
        raise ValueError("unrecognized class")
      logits, _ = transformer_model.call_simple(
          inputs=inputs,
          targets=targets,
          compute_loss=False,
          mode=mode,
          variable_dtype=get_variable_dtype())
      batch_dim, length_dim, vocab_dim = logits.shape.dims
      cross_entropy = mtf.layers.softmax_cross_entropy_with_logits(
          logits, mtf_features["targets"], vocab_dim)
      cross_entropy *= mtf.cast(
          mtf.not_equal(targets, 0), cross_entropy.dtype)
      if mode == "delimited_lm":
        cross_entropy *= mtf.cast(mtf.logical_not(
            transformer.delimited_lm_inputs_mask(targets)), cross_entropy.dtype)
      scores = -mtf.reduce_sum(cross_entropy, reduced_dim=length_dim)
      scores = mtf.anonymize(scores)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      predictions = {
          "scores": lowering.export_to_tf_tensor(scores)
      }
    elif mode == tf.estimator.ModeKeys.PREDICT:
      inputs = mtf_features["inputs"]
      if predict_fn:
        mtf_samples = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model, transformer.Unitransformer):
        # pad so that there is enough room for the targets
        inputs = mtf.pad(
            inputs, [0, sequence_length["targets"]], length_dim.name)
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=get_variable_dtype(),
            remove_partial_sequences=True)
      elif isinstance(
          transformer_model,
          (transformer.Bitransformer, transformer.StudentTeacher)):
        mtf_samples = transformer_model.decode(
            inputs, variable_dtype=get_variable_dtype())
      else:
        raise ValueError("unrecognized class")
      mtf_samples = mtf.anonymize(mtf_samples)
      inputs = mtf.anonymize(inputs)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      inputs = clean_decodes(lowering.export_to_tf_tensor(inputs))
      outputs = clean_decodes(lowering.export_to_tf_tensor(mtf_samples))

      # Detokenize in the graph if supported by vocabulary and accelerator.
      def _maybe_detokenize(ids, vocab):
        if not use_tpu and hasattr(vocab, "decode_tf"):
          return vocab.decode_tf(ids)
        return ids

      inputs = _maybe_detokenize(inputs, inputs_vocabulary(vocabulary))
      outputs = _maybe_detokenize(outputs, targets_vocabulary(vocabulary))

      predictions = {
          "inputs": inputs,
          "outputs": outputs}

    if mode in ["score", tf.estimator.ModeKeys.PREDICT]:
      # When exporting a model, we need to communicate to TF-Serving that
      # master variables need to be copied to their slave slice variables.
      # Estimator uses a Scaffold's "local_init_op" for this purpose, so we
      # augment the default "local_init_op" here.
      #
      # The "ready_op" is also constructed here to ensure the variables
      # initialized by "local_init_op" are the same ones checked by "ready_op".
      #
      # WARNING: Any variables created outside of this model_fn()
      # (e.g. tpu_estimator/iterations_per_loop) will NOT be initialized nor
      # checked by these ops.
      def scaffold_fn():
        return tf.train.Scaffold(
            local_init_op=tf.group(
                tf.train.Scaffold.default_local_init_op(),
                lowering.copy_masters_to_slices(),
                name="mtf_local_init_op"),
            ready_op=tf.concat(
                [tf.report_uninitialized_variables(),
                 resources.report_uninitialized_resources()],
                axis=0,
                name="mtf_ready_op"))

      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          scaffold_fn=scaffold_fn,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    assert (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL)

    def logits_and_loss(mtf_features, num_microbatches=1):
      """Compute logits and loss.

      Args:
        mtf_features: a dictionary
        num_microbatches: integer
      Returns:
        logits: a mtf.Tensor
        loss: a mtf.Tensor
      """
      if model_type in ["lm", "delimited_lm"]:
        _, _, length_dim = mtf_features["targets"].shape
        inputs = mtf.shift(mtf_features["targets"], offset=1,
                           dim=length_dim, wrap=False)
      else:
        inputs = mtf_features["inputs"]

      if isinstance(transformer_model, transformer.Unitransformer):
        position_kwargs = dict(
            sequence_id=mtf_features.get("targets_segmentation", None),
            position=mtf_features.get("targets_position", None),
        )
      elif isinstance(
          transformer_model,
          transformer.Bitransformer) or model_type == "bi_student_teacher":
        position_kwargs = dict(
            encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
            decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                 None),
            decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                    None),
            encoder_position=mtf_features.get("inputs_position", None),
            decoder_position=mtf_features.get("targets_position", None),
        )
      else:
        raise ValueError("unrecognized class")

      return transformer_model.call_simple(
          inputs=inputs,
          targets=mtf_features["targets"],
          compute_loss=True,
          mode=mode,
          variable_dtype=get_variable_dtype(),
          num_microbatches=num_microbatches,
          **position_kwargs)

    if mode == tf.estimator.ModeKeys.TRAIN:
      num_microbatches = serialize_num_microbatches(batch_dim,
                                                    sequence_length,
                                                    mesh_shape,
                                                    layout_rules)
      if num_microbatches > 1:
        def serialized_fn(mtf_features):
          return {"loss": logits_and_loss(mtf_features, num_microbatches)[1]}
        var_grads, loss_dict = mtf.serialize_training_step(
            mtf_features, serialized_fn, batch_dim, num_microbatches)
        loss = loss_dict["loss"]
      else:
        loss = logits_and_loss(mtf_features)[1]
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])

      if tpu_summaries:
        mtf.scalar_summary("loss", loss)

      if callable(learning_rate_schedule):
        # the following happens on CPU since TPU can't handle summaries.
        with mtf.utils.outside_all_rewrites():
          learning_rate = learning_rate_schedule(
              step=tf.train.get_global_step())
          tf.summary.scalar("learning_rate", learning_rate)
      else:
        learning_rate = learning_rate_schedule

      if isinstance(variable_filter, str):
        pattern = re.compile(variable_filter)
        variable_filter_fn = lambda v: pattern.search(v.name)
      elif variable_filter is None:
        variable_filter_fn = lambda v: True
      elif callable(variable_filter):
        variable_filter_fn = variable_filter
      else:
        raise ValueError(
            "variable_filter must be None, a string, or a callable function")
      trainable_vars = [
          v for v in graph.trainable_variables if variable_filter_fn(v)]
      trainable_var_grads = [
          g for g, v in zip(var_grads, graph.trainable_variables)
          if variable_filter_fn(v)]
      if len(trainable_vars) != len(graph.trainable_variables):
        tf.logging.info("Variables being trained:")
        tf.logging.info([v.name for v in trainable_vars])
        tf.logging.info("Variables not being trained:")
        tf.logging.info([v.name for v in graph.trainable_variables
                         if not variable_filter_fn(v)])

      update_ops = optimizer(learning_rate=learning_rate).apply_grads(
          trainable_var_grads, trainable_vars
      )

      lowering = mtf.Lowering(
          graph, {mesh: mesh_impl},
          autostack=autostack,
          log_file=model_info_file)

      tf_loss = lowering.export_to_tf_tensor(loss)
      tf_loss = tf.cast(tf_loss, tf.float32)
      if not use_tpu:
        tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                           "step, tf_loss")

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      train_op = tf.group(tf_update_ops)

      if hasattr(transformer_model, "initialize"):
        with mtf.utils.outside_all_rewrites():
          transformer_model.initialize()

      if tpu_summaries:
        # has to be outside of
        # with mtf.utils.outside_all_rewrites()
        host_call = mtf.utils.create_host_call(model_dir)
        mtf.utils.remove_summaries()
      else:
        host_call = None

      with mtf.utils.outside_all_rewrites():

        if init_checkpoint:
          ckpt_vars = {v for v, _ in tf.train.list_variables(init_checkpoint)}
          global_vars = {v.op.name for v in tf.global_variables()}
          restore_vars = ckpt_vars.intersection(global_vars)
          tf.logging.info("Initializing variables from %s:", init_checkpoint)
          tf.logging.debug("\n".join(sorted(restore_vars)))
          tf.logging.info("Variables in %s but not in graph:", init_checkpoint)
          tf.logging.info("\n".join(sorted(ckpt_vars - global_vars)))
          tf.logging.info("Variables in graph but not in %s:", init_checkpoint)
          tf.logging.info("\n".join(sorted(global_vars - ckpt_vars)))
          tf.train.init_from_checkpoint(
              init_checkpoint, {v: v for v in restore_vars}
          )

        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=2,
            defer_build=False,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            model_dir,
            save_steps=save_checkpoints_steps,
            saver=saver,
            listeners=[saver_listener])
        gin_config_saver_hook = gin.tf.GinConfigSaverHook(
            model_dir, summarize_config=True, include_step_in_filename=False)

        if use_tpu:
          return tpu_estimator.TPUEstimatorSpec(
              mode=tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              host_call=host_call,
              training_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
    elif mode == tf.estimator.ModeKeys.EVAL:
      # perplexity eval
      logits, loss = logits_and_loss(mtf_features)
      # compute cross-entropy while still on TPU to avoid having to outfeed the
      # logits, which might be big.
      logits = mtf.cast(logits, tf.float32)
      vocab_dim = logits.shape.dims[-1]
      targets = mtf_features["targets"]
      cross_entropy = mtf.layers.softmax_cross_entropy_with_logits(
          logits, targets, vocab_dim)
      anon_cross_entropy = mtf.anonymize(cross_entropy)
      predictions = mtf.cast(mtf.argmax(logits, vocab_dim), targets.dtype)
      anon_predictions = mtf.anonymize(predictions)
      anon_targets = mtf.anonymize(targets)
      anon_weights = mtf.layers.weights_nonzero(anon_targets, dtype=tf.float32)
      if model_type == "delimited_lm":
        anon_weights *= mtf.cast(
            mtf.logical_not(transformer.delimited_lm_inputs_mask(anon_targets)),
            dtype=tf.float32)

      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
      tf_loss = tf.cast(tf_loss, tf.float32)
      tf_predictions = lowering.export_to_tf_tensor(anon_predictions)
      tf_cross_entropy = lowering.export_to_tf_tensor(anon_cross_entropy)

      def simple_metrics(xent, predictions, labels, weights):
        """Simple metrics for teacher-forced eval."""
        token_correct = tf.cast(
            tf.equal(predictions, labels), tf.float32) * weights
        sequence_correct = tf.cast(
            tf.equal(tf.reduce_sum(token_correct, -1),
                     tf.reduce_sum(weights, -1)),
            tf.float32)
        sequence_weights = tf.cast(
            tf.not_equal(tf.reduce_sum(weights, -1), 0),
            tf.float32)
        # the purpose of "mean_label" is as a checksum to ensure that
        # models were evaluated on the same data.
        return {"neg_log_perplexity": tf.metrics.mean(-xent, weights),
                "token_accuracy": tf.metrics.mean(token_correct, weights),
                "sequence_accuracy": tf.metrics.mean(
                    sequence_correct, sequence_weights),
                "mean_label": tf.metrics.mean(tf.cast(labels, tf.float32)),
                "num_eval_tokens": metric_sum(weights, name="num_eval_tokens"),
                "max_targets_length": metric_max(tf.reduce_sum(
                    weights, axis=-1), name="max_targets_length"),
               }

      labels = lowering.export_to_tf_tensor(anon_targets)
      weights = lowering.export_to_tf_tensor(anon_weights)
      eval_metrics = (simple_metrics, [
          tf_cross_entropy, tf_predictions, labels, weights])
      with mtf.utils.outside_all_rewrites():
        restore_hook = mtf.MtfRestoreHook(lowering)
      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=tf_loss,
          eval_metrics=eval_metrics)

  return my_model_fn


def metric_sum(values, name=None, **kwargs):
  del kwargs
  with tf.variable_scope(name, "metric_sum", [values]):
    accum = tf.get_variable(
        "accum", shape=[], dtype=tf.float32, trainable=False,
        collections=[tf.GraphKeys.LOCAL_VARIABLES],
        initializer=tf.zeros_initializer())
    update_op = tf.assign_add(accum, tf.reduce_sum(tf.cast(values, tf.float32)))
    return accum, update_op


def metric_max(values, name=None, **kwargs):
  del kwargs
  with tf.variable_scope(name, "metric_max", [values]):
    accum = tf.get_variable(
        "accum", shape=[], dtype=tf.float32, trainable=False,
        collections=[tf.GraphKeys.LOCAL_VARIABLES],
        initializer=tf.zeros_initializer())
    update_op = tf.assign(
        accum, tf.maximum(accum, tf.reduce_max(tf.cast(values, tf.float32))))
    return accum, update_op


def _dynamic_text2self(mtf_features):
  """Convert a packed feature dictionary from text2text into text2self.

  This conversion is used when training a "delimited_lm" model.

  This allows us to train a text2self model on data that has been tokenized and
  packed in text2text format.

  Inputs and targets for each example get concatenated into the new targets.
  Length doubles.

  Args:
    mtf_features: a feature dictionary containing
       "inputs", "inputs_segmentation", "inputs_position",
       "targets", "targets_segmentation", "targets_position"
  Returns:
    a feature dictionary containing
      "targets", "targets_segmentation", "targets_position"
  """
  tf.logging.info(
      "_dynamic_text2self: Converting text2text problem to text2self")
  inputs = mtf_features["inputs"]
  targets = mtf_features["targets"]
  inputs_segmentation = mtf_features["inputs_segmentation"]
  targets_segmentation = mtf_features["targets_segmentation"]
  inputs_position = mtf_features["inputs_position"]
  targets_position = mtf_features["targets_position"]
  inputs_length_dim = inputs.shape.dims[-1]
  targets_length_dim = targets.shape.dims[-1]
  # compute lengths of inputs and targets portions of each segment
  # segments_dim must be larger than the maximum number of segments.
  segments_dim = mtf.Dimension("segments", targets_length_dim.size)
  inputs_segment_length = mtf.reduce_sum(
      mtf.one_hot(inputs_segmentation, segments_dim, dtype=tf.int32),
      reduced_dim=inputs_length_dim)
  targets_segment_length = mtf.reduce_sum(
      mtf.one_hot(targets_segmentation, segments_dim, dtype=tf.int32),
      reduced_dim=targets_length_dim)
  # segment 0 means padding.  Zero out the segment lengths for segment 0.
  segments_range = mtf.range(targets.mesh, segments_dim, dtype=tf.int32)
  nonzero_segment = mtf.to_int32(mtf.not_equal(segments_range, 0))
  inputs_segment_length *= nonzero_segment
  targets_segment_length *= nonzero_segment
  combined_segment_length = inputs_segment_length + targets_segment_length
  # for targets, position in sequence increases by inputs_segment_length
  targets_position += mtf.gather(
      inputs_segment_length, targets_segmentation, segments_dim)
  # this is the new length dimension
  new_length_dim = mtf.Dimension(
      "new_length", inputs_length_dim.size + targets_length_dim.size)
  new_length_range = mtf.range(
      targets.mesh, new_length_dim, dtype=tf.int32)
  # compute permutation tensors mapping from the old length dimension to the
  # new length dimension
  combined_segment_length_cumulative = mtf.cumsum(
      combined_segment_length, segments_dim, exclusive=True)
  # segment 0 is padding - this causes it to get mapped out of range.
  combined_segment_length_cumulative += new_length_dim.size * mtf.to_int32(
      mtf.equal(segments_range, 0))
  inputs_destination = inputs_position + mtf.gather(
      combined_segment_length_cumulative, inputs_segmentation, segments_dim)
  inputs_permutation = mtf.to_int32(mtf.equal(
      new_length_range, inputs_destination))
  targets_destination = targets_position + mtf.gather(
      combined_segment_length_cumulative, targets_segmentation, segments_dim)
  targets_permutation = mtf.to_int32(mtf.equal(
      new_length_range, targets_destination))
  # map from the old length dimension to the new length dimension
  def _convert(t, perm):
    return mtf.rename_dimension(
        mtf.einsum([t, perm],
                   output_shape=inputs.shape.dims[:-1] + [new_length_dim]),
        "new_length", "length")
  targets = (
      _convert(inputs, inputs_permutation) +
      _convert(targets, targets_permutation))
  targets_segmentation = (
      _convert(inputs_segmentation, inputs_permutation) +
      _convert(targets_segmentation, targets_permutation))
  targets_position = (
      _convert(inputs_position, inputs_permutation) +
      _convert(targets_position, targets_permutation))
  mtf_features = {
      "targets": targets,
      "targets_segmentation": targets_segmentation,
      "targets_position": targets_position,
  }
  return mtf_features


def get_inputs_from_file(input_filename, ignore_comments=False):
  """Read data from file and strip new lines."""
  inputs = [line.rstrip() for line in tf.io.gfile.GFile(input_filename)]

  # Strip the last empty line.
  if not inputs[-1]:
    inputs.pop()

  if ignore_comments:
    inputs = [l for l in inputs if not l.startswith("#")]

  return inputs


def encode_inputs(inputs,
                  vocabulary,
                  model_type,
                  batch_size,
                  sequence_length,
                  eos_id=1):
  """Encode string inputs for inference/scoring.

  Args:
    inputs: list of strings
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    eos_id: EOS id

  Returns:
    all_input_ids: encoded inputs
  """
  n = len(inputs)
  all_input_ids = []
  for line in inputs:
    ids = inputs_vocabulary(vocabulary).encode(line.strip())
    if model_type != "lm":
      # for text2self problems, the inputs represent a partial sequence
      # to be continued, and should not be terminated by EOS.
      # for sequence-to-sequence problems, the input needs to be EOS-terminated
      ids += [eos_id]
    if len(ids) > sequence_length:
      ids = ids[:sequence_length]
    else:
      ids.extend([0] * (sequence_length - len(ids)))
    all_input_ids.append(ids)
  # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % batch_size))
  all_input_ids = np.array(all_input_ids, dtype=np.int32)

  return all_input_ids


def encode_delimited_lm(inputs,
                        targets,
                        vocabulary,
                        batch_size,
                        sequence_length,
                        eos_id=1,
                        include_final_eos=True):
  """Encode inputs and targets for scoring a delimited langauge model.

  Args:
    inputs: list of strings
    targets: list of strings
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    eos_id: EOS id
    include_final_eos: a boolean

  Returns:
    all_ids: encoded inputs
  """
  n = len(inputs)
  all_ids = []
  for inp, tgt in zip(inputs, targets):
    input_ids = inputs_vocabulary(vocabulary).encode(inp.strip()) + [eos_id]
    target_ids = targets_vocabulary(vocabulary).encode(tgt.strip())
    if include_final_eos:
      target_ids.append(eos_id)
    ids = input_ids + target_ids
    if len(ids) > sequence_length:
      ids = ids[:sequence_length]
    else:
      ids.extend([0] * (sequence_length - len(ids)))
    all_ids.append(ids)
  # pad to make an integral number of batches
  all_ids.extend([all_ids[0]] * (-n % batch_size))
  all_ids = np.array(all_ids, dtype=np.int32)
  return all_ids


@gin.configurable
def decode(estimator,
           input_fn,
           vocabulary,
           checkpoint_path=None):
  """Decode from an input_fn.

  Args:
    estimator: a TPUEstimator
    input_fn: function that returns a tf.Dataset
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    checkpoint_path: an optional string

  Returns:
    list of decoded strings
  """
  result_iter = estimator.predict(
      input_fn, checkpoint_path=checkpoint_path)

  def _maybe_detokenize(value, vocab):
    if isinstance(value, six.binary_type):
      return value
    return vocab.decode([int(x) for x in value])

  decodes = []
  for i, result in enumerate(result_iter):
    input_string = _maybe_detokenize(
        result["inputs"], inputs_vocabulary(vocabulary))
    output_string = _maybe_detokenize(
        result["outputs"], targets_vocabulary(vocabulary))
    decodes.append(output_string)
    if i & (i - 1) == 0:
      # LOG every power of 2.
      tf.logging.info("decoded {}: {}".format(i, input_string))
      tf.logging.info("            -> {}".format(output_string))
  return decodes


@gin.configurable
def compute_log_likelihoods(estimator,
                            input_fn,
                            checkpoint_path=None):
  """Decode from an input_fn.

  Args:
    estimator: a TPUEstimator
    input_fn: function that returns a tf.Dataset
    checkpoint_path: an optional string

  Returns:
    list of floats
  """
  result_iter = estimator.predict(
      input_fn, checkpoint_path=checkpoint_path)
  return [float(f) for f in result_iter]


def write_lines_to_file(lines, filename):
  """Write each line to a filename, replacing the file if it exists.

  Args:
    lines: list of str, lines to write out.
    filename: str, path to filename.
  """
  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)
  with tf.io.gfile.GFile(filename, "w") as output_file:
    for line in lines:
      output_file.write("{}\n".format(line))


def get_step_from_checkpoint_path(checkpoint_path):
  """Returns the global step for the checkpoint at `checkpoint_path`.

  Assumes `checkpoint_path` corresponds to a file which contains the substring
  model.ckpt-{global_step}

  Args:
    checkpoint_path: str of path to a checkpoint file.

  Returns:
    int of the global step corresponding to the checkpoint file.

  Raises:
    ValueError if checkpoint_path does not correspond to a model checkpoint file
    which contains the global_step in its filename.
  """
  match = re.match(r".*model\.ckpt\-(\d+).*", checkpoint_path)
  if match is None:
    raise ValueError("Invalid checkpoint path {}".format(checkpoint_path))
  return int(match.group(1))


# TODO(noam): include more descriptive definitions
@gin.configurable
def decode_from_file(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path=None,
                     input_filename=gin.REQUIRED,
                     output_filename=gin.REQUIRED,
                     eos_id=1,
                     repeats=1):
  """Decode from a text file and write to output_filename.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    checkpoint_path: an optional string
    input_filename: a string
    output_filename: a string
    eos_id: EOS id
    repeats: an integer, the number of times to repeat each input.
  """
  inputs = get_inputs_from_file(input_filename)

  all_input_ids = encode_inputs(inputs, vocabulary, model_type, batch_size,
                                sequence_length["inputs"], eos_id=eos_id)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(repeats))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  checkpoint_step = get_step_from_checkpoint_path(checkpoint_path)
  decodes = decode(
      estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path)
  # Remove any padded examples
  dataset_size = len(inputs) * repeats
  decodes = decodes[:dataset_size]
  output_filename = "{}-{}".format(output_filename, checkpoint_step)
  write_lines_to_file(decodes, output_filename)


@gin.configurable
def clean_decodes(ids, eos_id=1, pad_id=0, length_axis=-1):
  """Replaces everything after EOS with PAD (along last axis).

  Args:
    ids: a d Tensor of type int.
    eos_id: int, EOS id.
    pad_id: int, PAD id.
    length_axis: an integer.

  Returns:
    a Tensor of type int of ids.
  """
  eos_and_after = tf.cumsum(tf.cast(tf.equal(ids, eos_id), tf.int32),
                            exclusive=True, axis=length_axis)
  valid_ids = tf.equal(eos_and_after, 0)
  return tf.where_v2(valid_ids, ids, pad_id)


def _score_with_estimator(estimator, input_fn, eval_checkpoint_step, model_dir,
                          scores_filename, num_examples=None):
  """For each example returned by input_fn, compute log likelihood.

  Args:
    estimator: a TPUEstimator
    input_fn: a function that that returns a tf.data.Dataset with examples
      containing the string field 'targets' and optionally the field 'inputs'
    eval_checkpoint_step: int, list of ints, or None, see `eval_model`
      docstring.
    model_dir: string, estimator model_dir
    scores_filename: a string (path of file to write scores to)
    num_examples: int, the total # of examples being scored, None if unknown

  Returns:
    a list of floats
  """
  checkpoint_path, = get_checkpoint_iterator(eval_checkpoint_step, model_dir)

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
  scores = [m["scores"] for m in result_iter]
  # Remove any padding examples
  scores = scores[:num_examples]
  if scores_filename is not None:
    write_lines_to_file(["%f" % f for f in scores], scores_filename)
  return scores


@gin.configurable
def score_from_strings(estimator, vocabulary, model_type, batch_size,
                       sequence_length, model_dir, eval_checkpoint_step,
                       inputs=gin.REQUIRED, targets=gin.REQUIRED,
                       scores_filename=gin.REQUIRED, eos_id=1, score_eos=True):
  """Compute log likelihoods per example and write to a text file.

  inputs & targets must either be the same length (in lines) or have inputs
  evenly divide targets N times, where each input has N decodes sequentially
  in targets.

  The function returns a list of floats represnenting the log-liekelihood of the
  target given the input.  If `scores_filename` is present, then these are also
  written out as a text file, one per line.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    model_dir: string, estimator model_dir
    eval_checkpoint_step: int, list of ints, or None, see `eval_model`
      docstring.
    inputs: optional - a list of strings (inputs) the same length as targets
      alternatively, a string filepath for a text file (one string per line)
    targets: a list of strings (targets)
      alternatively, a string filepath for a text file (one string per line)
    scores_filename: a string (path of file to write)
    eos_id: EOS id
    score_eos: a boolean - whether to score the final eos token of each line
      If this is set to false, the scores can be interpreted as prefix
      log-likelihoods
  Returns:
    a list of floats
  """
  if isinstance(inputs, str):
    inputs = get_inputs_from_file(inputs)
  if isinstance(targets, str):
    targets = get_inputs_from_file(targets)
  has_inputs = inputs is not None
  if has_inputs:
    if len(inputs) < len(targets):
      # We assume that the targets file contains n targets for each input.
      # So we repeat each input n times.
      if len(targets) % len(inputs):
        raise ValueError("len(inputs) must divide len(targets), got %d and %d"
                         % (len(inputs), len(targets)))
      repeats = len(targets) // len(inputs)
      inputs = [inputs[i // repeats] for i in range(len(targets))]
    elif len(targets) < len(inputs):
      # `targets` is a list of one string.  Use it as a target for all inputs.
      if len(targets) != 1:
        raise ValueError("Expected only one target string")
      targets = targets * len(inputs)
  if model_type == "delimited_lm":
    all_target_ids = encode_delimited_lm(
        inputs,
        targets,
        vocabulary,
        batch_size,
        sequence_length["targets"],
        eos_id=eos_id)
    has_inputs = False
  else:
    if has_inputs:
      all_input_ids = encode_inputs(inputs, vocabulary, model_type, batch_size,
                                    sequence_length["inputs"], eos_id=eos_id)
    all_target_ids = encode_inputs(
        targets, vocabulary, model_type, batch_size,
        sequence_length["targets"], eos_id=eos_id if score_eos else 0)

  def input_fn(params):
    del params
    m = ({"inputs": all_input_ids, "targets": all_target_ids} if has_inputs
         else {"targets": all_target_ids})
    dataset = tf.data.Dataset.from_tensor_slices(m)
    dataset = dataset.flat_map(tf.data.Dataset.from_tensors)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return _score_with_estimator(estimator, input_fn, eval_checkpoint_step,
                               model_dir, scores_filename, len(targets))


@gin.configurable
def score_from_dataset(estimator, vocabulary, batch_size, sequence_length,
                       model_dir, eval_checkpoint_step, dataset_split,
                       score_dataset_fn=None, scores_filename=gin.REQUIRED):
  """Compute log likelihoods per example and write to a text file.



  The function returns a list of floats represnenting the log-liekelihood of the
  target given the input.  If `scores_filename` is present, then these are also
  written out as a text file, one per line.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    model_dir: string, estimator model_dir
    eval_checkpoint_step: int, list of ints, or None, see `eval_model`
      docstring.
        dataset_split: a string
    score_dataset_fn: A function returning a list of dataset.EvalDataset tuples.
      See `eval_dataset_fn` argument to `eval_model` for details.
    scores_filename: a string (path of file to write)

  Returns:
    a list of floats
  """
  scoring_datasets = score_dataset_fn(
      sequence_length=sequence_length,
      vocabulary=vocabulary,
      dataset_split=dataset_split)
  if len(scoring_datasets) != 1:
    raise ValueError("Only scoring from a single dataset supported.")
  scoring_dataset = scoring_datasets[0]

  def input_fn(params):
    """Eval input function for estimator."""
    del params
    dataset = scoring_dataset.dataset_fn()
    dataset = dataset.map(_filter_features)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    # Pad the final batch.
    dataset = transformer_dataset.trim_and_pad_dataset(
        dataset, length=batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  # TODO(dei): Since we pass in the num_examples as None, scores for the
  # padding examples will get written to the output file. Should fix this.
  return _score_with_estimator(estimator, input_fn, eval_checkpoint_step,
                               model_dir, scores_filename, None)


def get_estimator(model_type, vocabulary, mesh_shape,
                  layout_rules, model_dir, batch_size, sequence_length,
                  autostack, learning_rate_schedule, keep_checkpoint_max,
                  save_checkpoints_steps, optimizer, predict_fn,
                  variable_filter, ensemble_inputs, use_tpu, tpu_job_name,
                  iterations_per_loop, cluster, init_checkpoint=None,
                  mesh_devices=None):
  """Create TPU estimator for the transfomer Mesh-TF model.

  Args:
    model_type: a string - either "bitransformer", "bi_student_teacher", lm" or
      "aligned"
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    mesh_shape: a function passed in through gin that returns a mtf.Shape
    layout_rules: an input to mtf.convert_to_layout_rules()
    model_dir: a string, model directory path.
    batch_size: an integer, global batch size.
    sequence_length: a dict, see `train_model` docstring for details.
    autostack: boolean, internally combine variables
    learning_rate_schedule: an optional function taking the scalar name argument
      `step` and the numeric argument `total_train_steps` and return the scalar
      learning rate
    keep_checkpoint_max: an integer, maximum number of checkpoints to keep
    save_checkpoints_steps: integer, steps per checkpoint
    optimizer: a class extending optimize.Optimizer, required for training
    predict_fn: an optional function that can be used to override the default
      transformer prediction behavior. Must return a tensor of shape [batch_dim,
      length_dim] that will be the prediction for each example. Must accept the
      following arguments:
        - model: a Unitransformer or Bitransformer
        - features: a dict representing an example. Every value will be an
          mtf.Tensor with shape [batch_dim, length_dim].
        - variable_dtype: an mtf.VariableDType
    variable_filter: a string, a variable will only be trained if its name
      matches this regex. If None (default), train all trainable variables.
    ensemble_inputs: an integer, see `train_model` docstring for details.
    use_tpu: string, the Cloud TPU to use for training
    tpu_job_name: string, name of TPU worker binary
    iterations_per_loop: integer, steps per train loop
    cluster: a TPUClsuterResolver object
    init_checkpoint: a string, if not None then read in variables from this
      checkpoint path when initializing variables. Will only initialize
      variables that appear both in the current graph and the checkpoint.
    mesh_devices: a list of strings, the device names to use for each mesh
      slice. Only required for GPU.
  Returns:
    an Estimator object.
  """
  my_tpu_config = tpu_config.TPUConfig(
      tpu_job_name=tpu_job_name,
      iterations_per_loop=iterations_per_loop,
      num_cores_per_replica=1,
      per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST,
  )

  run_config = tpu_config.RunConfig(
      cluster=cluster,
      model_dir=model_dir,
      tpu_config=my_tpu_config,
      # We use a saver hook, so disable checkpoints here to prevent double
      # saving.
      save_checkpoints_steps=None,
      save_checkpoints_secs=None)

  transformer_model = build_model(
      model_type=model_type,
      input_vocab_size=inputs_vocabulary(vocabulary).vocab_size,
      output_vocab_size=targets_vocabulary(vocabulary).vocab_size,
      layout_rules=layout_rules,
      mesh_shape=mesh_shape)

  model_fn = tpu_estimator_model_fn(
      model_type=model_type,
      transformer_model=transformer_model,
      vocabulary=vocabulary,
      model_dir=model_dir,
      use_tpu=use_tpu,
      mesh_shape=mesh_shape,
      layout_rules=layout_rules,
      batch_size=batch_size,
      sequence_length=sequence_length,
      autostack=autostack,
      learning_rate_schedule=learning_rate_schedule,
      keep_checkpoint_max=keep_checkpoint_max,
      save_checkpoints_steps=save_checkpoints_steps,
      optimizer=optimizer,
      predict_fn=predict_fn,
      variable_filter=variable_filter,
      ensemble_inputs=ensemble_inputs,
      init_checkpoint=init_checkpoint,
      mesh_devices=mesh_devices)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      use_tpu=use_tpu,
      export_to_tpu=False,
      params={})

  return estimator


def train_model(estimator, vocabulary, sequence_length, batch_size,
                train_dataset_fn, train_steps, ensemble_inputs,
                dataset_split="train"):
  """Train a Mesh-TF model.

  Args:
    estimator: Estimator object, created with the appropriate model_fn.
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    sequence_length: a dict from feature-key to integer the (packed)
      sequence length, e.g. {"inputs": 512, "targets": 128}
    batch_size: an integer, global batch size
    train_dataset_fn: A function returning a tf.data.Dataset. Should accept the
     following arguments:
      - sequence_length: an integer or a dict from feature-key to integer
        the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
      - vocabulary: Vocabulary instance to use for encoding.
      - dataset_split: str, which dataset split to load.
    train_steps: an integer, number of steps for training.
    ensemble_inputs: an optional integer - pass the size of the ensemble to
      train an ensemble where each model gets different inputs. You also need to
      configure Unitransformer.ensemble  to the right size. If None, then all
      models are trained on the same inputs.
    dataset_split: str, which dataset split to train on.
  """

  def input_fn(params):
    del params
    dataset = train_dataset_fn(
        sequence_length=sequence_length,
        vocabulary=vocabulary,
        dataset_split=dataset_split)
    dataset = dataset.repeat().batch(
        batch_size * (ensemble_inputs or 1), drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  estimator.train(input_fn=input_fn, max_steps=train_steps)


@gin.configurable
def infer_model(estimator,
                vocabulary,
                sequence_length,
                batch_size,
                model_type,
                model_dir,
                eval_checkpoint_step,
                input_filename=None,
                output_filename=None,
                checkpoint_paths=None,
                decode_from_file_fn=decode_from_file):
  """Infer a Mesh-TF model.

  Args:
    estimator: Estimator object, created with the appropriate model_fn.
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    sequence_length: a dict from feature-key to integer the (packed)
      sequence length, e.g. {"inputs": 512, "targets": 128}
    batch_size: an integer, global batch size
    model_type: a string - either "bitransformer", "bi_student_teacher", lm" or
      "aligned"
    model_dir: string, estimator model_dir
    eval_checkpoint_step: int, list of ints, or None, see `eval_model`
      docstring.
    input_filename: a string, input file with examples
    output_filename: a string, output file to save decodes
    checkpoint_paths: optional list of checkpoints to run inference for
    decode_from_file_fn: decoding function, defaults to decode_from_file
  """
  if checkpoint_paths is None:
    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)

  for checkpoint_path in checkpoint_paths:
    decode_from_file_fn(
        estimator,
        vocabulary=vocabulary,
        model_type=model_type,
        batch_size=batch_size,
        sequence_length=sequence_length,
        checkpoint_path=checkpoint_path,
        input_filename=input_filename,
        output_filename=output_filename)


def eval_model(estimator, vocabulary, sequence_length, batch_size,
               dataset_split, model_dir, eval_dataset_fn, eval_summary_dir,
               eval_checkpoint_step):
  """Eval a Mesh-TF model.

  Args:
    estimator: Estimator object, created with the appropriate model_fn.
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    sequence_length: a dict from feature-key to integer the (packed)
      sequence length, e.g. {"inputs": 512, "targets": 128}
    batch_size: an integer, global batch size
    dataset_split: a string
    model_dir: a string, directory with the model.
    eval_dataset_fn: A function returning a list of dataset.EvalDataset tuples.
      Must be provided for mode="eval". Should accept the following arguments:
        - sequence_length: an integer or a dict from feature-key to integer
          the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
        - vocabulary: Vocabulary instance to use for encoding.
        - dataset_split: str, which dataset split to load.
      dataset.EvalDataset tuples are namedtuples with the following fields:
        - name: string, the task name
        - dataset_fn: function which returns a tf.data.Dataset of tokenized and
          padded examples. Must not require any arguments and must include the
          feature keys 'inputs' and 'targets_plaintext'.
        - postprocess_fn: function which converts plaintext targets to values
          that can be processed by a `metric_fn`.
        - list_of_metric_fns: list of metric functions with the call signature
          `metric_fn(targets, predictions)` which returns a dict mapping
          submetric names to scalar values. TensorBoard summaries and other tags
          will be written out using the submetric names.
    eval_summary_dir: str, path to write TensorBoard events file summaries for
      eval. If None, use model_dir/eval_{split}.
    eval_checkpoint_step: int, list of ints, or None. If an int or list of ints,
      evaluation or inference will be run on the checkpoint files in `model_dir`
      whose global steps are closest to the global steps provided. If None and
      mode="eval", run eval continuously waiting for new checkpoints via
      `tf.train.checkpoints_iterator`.
  """
  if eval_dataset_fn is None:
    raise ValueError("Must provide eval_dataset_fn through gin for eval.")

  eval_datasets = eval_dataset_fn(
      sequence_length=sequence_length,
      vocabulary=vocabulary,
      dataset_split=dataset_split,
  )

  valid_eval_datasets = []
  for eval_dataset in eval_datasets:
    if not eval_dataset.metric_fns:
      tf.logging.info("Skipping %s because metric_fns is empty",
                      eval_dataset.name)
      continue
    # Convert to EvalDataset tuple in case eval_dataset_fn returns raw tuples
    valid_eval_datasets.append(transformer_dataset.EvalDataset(*eval_dataset))
  eval_datasets = valid_eval_datasets

  if not eval_datasets:
    tf.logging.info(
        "All provided EvalDatasets have metric_fns=[]; eval is not possible.")
    return

  eval_summary_dir = eval_summary_dir or os.path.join(
      model_dir, "{}_eval".format(dataset_split))
  summary_writer = tf.summary.FileWriter(eval_summary_dir)

  # Pre-load in all of the targets once before entering continuous eval loop
  cached_targets = {}
  cached_examples = {}
  # Need to create a separate graph for loading in plaintext targets
  # or else TF will complain that we modified the graph
  with tf.Graph().as_default():
    for eval_dataset in eval_datasets:
      if eval_dataset.metric_fns:
        ds = eval_dataset.dataset_fn()
        # Create list of postprocessed text targets
        examples = [ex for ex in tfds.as_numpy(ds)]
        targets = [
            eval_dataset.postprocess_fn(  # pylint:disable=g-complex-comprehension
                tf.compat.as_text(ex["targets_plaintext"]),
                example=ex, is_target=True)
            for ex in examples
        ]
        targets_filename = os.path.join(
            eval_summary_dir,
            "{}_targets".format(eval_dataset.name),
        )
        write_lines_to_file(targets, targets_filename)

        inputs_filename = os.path.join(
            eval_summary_dir,
            "{}_inputs".format(eval_dataset.name))
        inputs = [ex["inputs_plaintext"] for ex in examples]
        write_lines_to_file(inputs, inputs_filename)

        cached_targets[eval_dataset.name] = targets
        cached_examples[eval_dataset.name] = examples

  def input_fn(params):
    """Eval input function for estimator."""
    del params
    # Concatenate all dataset inputs to only have to do one decode loop
    combined_ds = None
    for eval_dataset in eval_datasets:
      # Only cache targets for those tasks with eval functions provides
      if eval_dataset.metric_fns:
        ds = eval_dataset.dataset_fn()
        ds = ds.map(_filter_features)
        combined_ds = ds if not combined_ds else combined_ds.concatenate(ds)
    combined_ds = combined_ds.batch(batch_size, drop_remainder=False)
    # Pad the final batch.
    combined_ds = transformer_dataset.trim_and_pad_dataset(
        combined_ds, length=batch_size)
    combined_ds = combined_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return combined_ds

  checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)
  for checkpoint_path in checkpoint_paths:
    tf.logging.info("Checkpoint path %s" % checkpoint_path)
    global_step = int(get_step_from_checkpoint_path(checkpoint_path))
    decodes = decode(estimator, input_fn, vocabulary, checkpoint_path)
    for eval_dataset in eval_datasets:
      # Extract the portion of decodes corresponding to this dataset
      examples = cached_examples[eval_dataset.name]
      dataset_size = len(examples)
      predictions = [
          eval_dataset.postprocess_fn(tf.compat.as_text(d), example=ex)
          for d, ex in zip(decodes[:dataset_size], examples)
      ]
      # Remove the used decodes.
      del decodes[:dataset_size]

      global_step = int(get_step_from_checkpoint_path(checkpoint_path))

      predictions_filename = os.path.join(
          eval_summary_dir,
          "{}_{}_predictions".format(eval_dataset.name, global_step),
      )
      write_lines_to_file(predictions, predictions_filename)

      for metric_fn in eval_dataset.metric_fns:
        summary = tf.Summary()
        targets = cached_targets[eval_dataset.name]
        metric_result = metric_fn(targets, predictions)
        for metric_name, metric_value in metric_result.items():
          tag = "eval/{}/{}".format(eval_dataset.name, metric_name)
          tf.logging.info("%s at step %d: %.3f", tag, global_step, metric_value)
          summary.value.add(tag=tag, simple_value=metric_value)
          summary_writer.add_summary(summary, global_step)
      summary_writer.flush()

    # Only padding should remain.
    expected_pad = -sum(len(t) for t in cached_targets.values()) % batch_size
    if len(decodes) != expected_pad:
      raise ValueError("{} padded decodes, {} expected.".format(
          len(decodes), expected_pad))


def export_model(estimator, export_dir, vocabulary, sequence_length,
                 batch_size=1, checkpoint_path=None):
  """Export a model in TF SavedModel format to be used for inference on CPUs.

  Args:
    estimator: Estimator object, estimator created with the appropriate
      model_fn.
    export_dir: str, a directory in which to create timestamped subdirectories
      containing exported SavedModels.
    vocabulary: sentencepiece vocab, vocabulary instance to use for encoding.
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    batch_size: int, number of sequences per batch. Should match estimator.
    checkpoint_path: str, path to checkpoint. If None (default), use the most
      recent in the model directory.

  Returns:
    The string path to the exported directory.
  """

  def serving_input_fn():
    """Constructs input portion of Graph in serving.

    Input is a batch of strings.

    Returns:
      a ServingInputReceiver
    """
    inputs = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name="inputs")

    padded_inputs = tf.pad(inputs, [(0, tf.mod(-tf.size(inputs), batch_size))])

    dataset = tf.data.Dataset.from_tensor_slices(padded_inputs)
    dataset = dataset.map(lambda x: {"inputs": x})
    dataset = transformer_dataset.encode_all_features(dataset, vocabulary)
    dataset = transformer_dataset.pack_or_pad(
        dataset=dataset,
        length=sequence_length,
        pack=False,
        feature_keys=["inputs"]
    )

    dataset = dataset.batch(batch_size)

    features = tf.data.experimental.get_single_element(dataset)
    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=inputs)

  return estimator.export_saved_model(
      export_dir, serving_input_fn, checkpoint_path=checkpoint_path)


def compute_batch_size(sequence_length,
                       mesh_shape,
                       layout_rules,
                       method_and_value):
  """Compute the total batch size in sequences.

  method_and_value is a (string, int) pair.
  The method string is one of the following four options:

  "sequences_per_batch"
  "tokens_per_batch"
  "sequences_per_replica"
  "tokens_per_replica"

  According to the method string, the value represents either a number of
  sequences or a number of tokens, and represents either the size of the total
  batch or the fraction of the batch assigned to each model replica.

  For example ("tokens_per_replica", 2048) means that the batch size should be
  set so that the number of tokens per model replica is 2048.  So if the
  sequence length is 1024 and there is 16-way data-parallelism, then the number
  of sequences per batch would be 2048 * 16 / 1024 = 32.

  The "per_batch" versions are useful for ensuring indentical overall batch
  sizes across different mesh shapes/layouts.  The "per_replica" versions are
  useful for scaling up the total batch size relative to the degree of
  data-parallelism

  Args:
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    method_and_value: a pair
  Returns:
    an integer - the number of sequences per batch
  """
  sequence_length = max(sequence_length.values())
  def checkdiv(a, b):
    if a % b:
      raise ValueError("%d is not divisible by %d" % (a, b))
    return a // b
  num_replicas = (
      mtf.tensor_dim_to_mesh_dim_size(
          layout_rules, mesh_shape, mtf.Dimension("batch", 0)) *
      mtf.tensor_dim_to_mesh_dim_size(
          layout_rules, mesh_shape, mtf.Dimension("outer_batch", 0)))
  method, value = method_and_value
  if method == "sequences_per_batch":
    return value
  elif method == "tokens_per_batch":
    return checkdiv(value, sequence_length)
  elif method == "sequences_per_replica":
    return value * num_replicas
  elif method == "tokens_per_replica":
    return checkdiv(value, sequence_length) * num_replicas
  else:
    raise ValueError("unknown method %s" % method,)


@gin.configurable
def serialize_num_microbatches(batch_dim,
                               sequence_length,
                               mesh_shape,
                               layout_rules,
                               tokens_per_microbatch_per_replica=None):
  """Number of microbatches per batch for serialized training.

  We want to split each training step into multiple sequential steps
  to limit memory usage.  Gradients are accumulated locally and reduced once.

  This function determines the number of microbatches per batch.
  If tokens_per_microbatch_per_replica=None, then the batch is not split.

  Args:
    batch_dim: a mtf.Dimension
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    tokens_per_microbatch_per_replica: an optional integer, e.g. 2048
  Returns:
    an integer
  """
  if not tokens_per_microbatch_per_replica:
    return 1
  batch_per_replica = mtf.tensor_dim_to_size_per_split(
      layout_rules, mesh_shape, batch_dim)
  # number of sequences per microbatch
  microbatch_size = max(
      1, tokens_per_microbatch_per_replica // max(sequence_length.values()))
  # decrease microbatch_size until it is a divisor of batch_per_replica
  # This is guaranteed to stop at microbatch_size=1 if not earlier.
  while batch_per_replica % microbatch_size:
    microbatch_size -= 1
  num_microbatches = batch_per_replica // microbatch_size
  tf.logging.info(
      "serialize_num_microbatches: "
      "tokens_per_microbatch_per_replica=%d "
      "batch_dim=%s "
      "sequence_length=%s "
      "batch_per_replica=%d "
      "num_microbatches=%d",
      tokens_per_microbatch_per_replica,
      batch_dim,
      sequence_length,
      batch_per_replica,
      num_microbatches)
  return num_microbatches


@gin.configurable
def auto_train_steps(batch_size,
                     sequence_length,
                     train_tokens=2 ** 36):
  """Automatically compute number of training steps.

  Since the batch size and sequence length can vary across experiments, we
  specify the amount of training in terms of (non-unique) input tokens processed
  over the course of training the model.  The number of steps is computed as

    train_steps = train_tokens // (batch_size * sequence_length)

  Args:
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    train_tokens: an integer (train_steps * batch_size * sequence_length)
  Returns:
    an integer
  """
  return train_tokens // (batch_size * max(sequence_length.values()))


@gin.configurable
def get_checkpoint_iterator(checkpoint_step, model_dir, skip_until=0):
  """Get an iterable of checkpoint paths from a provided checkpoint step(s).

  Args:
    checkpoint_step: If checkpoint_step is an int, find the checkpoint with the
      closest global step and return a singleton list. If checkpoint_step is a
      list of ints, replace each int with the path to the checkpoint with the
      closest global step. If checkpoint_step == "all", return the path of every
      checkpoint in model_dir, starting from the earliest checkpoint. If
      checkpoint_step is None, return `tf.train.checkpoints_iterator`
      for `model_dir`.
    model_dir: str, directory to look for checkpoints in.
    skip_until: an integer - for "all" or "None" behavior, filter out
      checkpoint numbers that are <= skip_until.

  Returns:
    An iterable which yields checkpoint paths.
  """

  def _get_closest_checkpoint(target_checkpoint):
    """Returns checkpoint with closest global step to `target_checkpoint`."""
    checkpoints = set()
    for f in tf.io.gfile.listdir(model_dir):
      try:
        checkpoints.add(int(get_step_from_checkpoint_path(f)))
      except ValueError:
        continue
    if not checkpoints:
      raise ValueError("No checkpoint files found in {}".format(model_dir))
    closest = float("inf")
    for c in checkpoints:
      if abs(target_checkpoint - c) < abs(target_checkpoint - closest):
        closest = c
    if closest != target_checkpoint:
      tf.logging.info(
          "Using checkpoint at step %d which is closest to requested step %d",
          closest,
          target_checkpoint,
      )
    return closest

  def _get_checkpoint_path(step):
    return os.path.join(model_dir, "model.ckpt-{}".format(step))

  def _filter_fn(p):
    return get_step_from_checkpoint_path(p) > skip_until

  if checkpoint_step == "all":
    ckpt_paths = tf.gfile.Glob(os.path.join(model_dir, "model.ckpt*"))
    # Use set for deduplication; glob will find multiple files for each ckpt
    ckpt_steps = {get_step_from_checkpoint_path(p) for p in ckpt_paths}
    return filter(_filter_fn,
                  [_get_checkpoint_path(s) for s in sorted(list(ckpt_steps))])
  elif checkpoint_step is None:
    return filter(_filter_fn, tf.train.checkpoints_iterator(model_dir))
  elif isinstance(checkpoint_step, int):
    return [_get_checkpoint_path(_get_closest_checkpoint(checkpoint_step))]
  else:
    closests = np.unique([_get_closest_checkpoint(c) for c in checkpoint_step])
    return [_get_checkpoint_path(closest) for closest in closests]


# TODO(noam): provide a more informative string for layout_rules:
# example: "d_ff:model,heads:model,vocab:model"
@gin.configurable
def run(tpu_job_name,
        tpu, gcp_project, tpu_zone,
        model_dir,
        model_type="bitransformer",
        vocabulary=None,
        train_dataset_fn=None,
        eval_dataset_fn=None,
        dataset_split="train",
        autostack=True,
        eval_checkpoint_step=None,
        export_checkpoint_step=None,
        export_path="",
        mode="train",
        iterations_per_loop=100,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=None,
        eval_summary_dir=None,
        batch_size=("tokens_per_replica", 2048),
        train_steps=auto_train_steps,
        sequence_length=gin.REQUIRED,
        mesh_shape=gin.REQUIRED,
        mesh_devices=None,
        layout_rules=gin.REQUIRED,
        learning_rate_schedule=None,
        optimizer=None,
        predict_fn=None,
        variable_filter=None,
        perplexity_eval_steps=100,
        init_checkpoint=None,
        ensemble_inputs=None,
        train_model_fn=train_model):
  """Run training, eval, or inference depending on `mode`.

  Args:
    tpu_job_name: string, name of TPU worker binary
    tpu: string, the Cloud TPU to use for training
    gcp_project: string, project name for the Cloud TPU-enabled project
    tpu_zone: string, GCE zone where the Cloud TPU is located in
    model_dir: string, estimator model_dir
    model_type: a string, set `get_estimator` docstring for details.
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple.
    train_dataset_fn: A function returning a tf.data.Dataset, see `train_model`
      docstring for details.
    eval_dataset_fn: A function returning a list of dataset.EvalDataset tuples.
      See `eval_model` docstring for details.
    dataset_split: a string
    autostack: boolean, see `get_estimator` docstring for details.
    eval_checkpoint_step: int, list of ints, or None, see `eval_model` doc
      string for details.
    export_checkpoint_step: int or None, see `export_model` doc string for
      details.
    export_path: a string, path to export the saved model
    mode: string, train/eval/perplexity_eval/infer
      perplexity_eval computes the perplexity of the dev set.
    iterations_per_loop: integer, steps per train loop
    save_checkpoints_steps: integer, see `get_estimator` docstring.
    keep_checkpoint_max: an integer, see `get_estimator` docstring.
    eval_summary_dir: str, see `eval_model` docstring for details.
    batch_size: An integer or a (method, value) pair to pass to
      compute_batch_size(). Note that this is the global batch size and not the
      per-shard batch size.
    train_steps: An integer or a function with the same signature as
      auto_train_steps().  Total number of training steps.
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    mesh_shape: an input to mtf.convert_to_shape()
    mesh_devices: a list of strings, see `get_estimator` docstring.
    layout_rules: an input to mtf.convert_to_layout_rules()
    learning_rate_schedule: a function which takes the scalar name argument
      `step` and the numeric argument `total_train_steps` and returns the scalar
      learning rate.  Alternatively a float.  Alternatively, a list of
      such factos to be multiplied together.
    optimizer: a class extending optimize.Optimizer, required for training
    predict_fn: an optional function, see `get_estimator` docstring for details.
    variable_filter: a string, see `get_estimator` docstring for details.
    perplexity_eval_steps: an integer - number of steps for perplexity eval
    init_checkpoint: a string, see `get_estimator` docstring for details.
    ensemble_inputs: an integer, see `train_model` docstring for details.
    train_model_fn: an optional train function, is `train_model` by default.
  """
  if isinstance(sequence_length, int):
    sequence_length = {"inputs": sequence_length,
                       "targets": sequence_length}

  if not isinstance(batch_size, int):
    batch_size = compute_batch_size(
        sequence_length, mesh_shape, layout_rules, batch_size)

  if not isinstance(train_steps, int):
    train_steps = train_steps(batch_size, sequence_length)

  if isinstance(learning_rate_schedule, list):
    learning_rate_schedule = functools.partial(
        learning_rate_schedules.product_learning_rate,
        factors=learning_rate_schedule)

  if callable(learning_rate_schedule):
    learning_rate_schedule = functools.partial(
        learning_rate_schedule, total_train_steps=train_steps)

  tf.logging.info("model_type=%s" % model_type,)
  tf.logging.info("mode=%s" % mode,)
  tf.logging.info("sequence_length=%s" % sequence_length,)
  tf.logging.info("batch_size=%s" % batch_size,)
  tf.logging.info("train_steps=%s" % train_steps,)
  tf.logging.info("mesh_shape=%s" % mesh_shape,)
  tf.logging.info("layout_rules=%s" % layout_rules,)

  if mode == "train" and dataset_split != "train":
    raise ValueError("mode==\"train\" requires dataset_split==\"train\"")

  if mode != "train":
    ensemble_inputs = None

  mesh_shape = mtf.convert_to_shape(mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(layout_rules)

  cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu, zone=tpu_zone, project=gcp_project) if tpu else None

  tf.logging.info(
      "Building TPUConfig with tpu_job_name={}".format(tpu_job_name)
  )

  estimator = get_estimator(
      model_type=model_type,
      vocabulary=vocabulary,
      layout_rules=layout_rules,
      mesh_shape=mesh_shape,
      model_dir=model_dir,
      batch_size=batch_size,
      sequence_length=sequence_length,
      autostack=autostack,
      learning_rate_schedule=learning_rate_schedule,
      keep_checkpoint_max=keep_checkpoint_max,
      save_checkpoints_steps=save_checkpoints_steps,
      optimizer=optimizer,
      predict_fn=predict_fn,
      variable_filter=variable_filter,
      init_checkpoint=init_checkpoint,
      ensemble_inputs=ensemble_inputs,
      use_tpu=tpu,
      tpu_job_name=tpu_job_name,
      iterations_per_loop=iterations_per_loop,
      cluster=cluster,
      mesh_devices=mesh_devices)

  if mode == "train":
    # train_dataset_fn could be None if train_model_fn is not equal to
    # train_model
    if train_dataset_fn is None:
      raise ValueError("Must provide train_dataset_fn through gin")
    train_model_fn(estimator, vocabulary, sequence_length, batch_size,
                   train_dataset_fn, train_steps, ensemble_inputs)
  elif mode == "perplexity_eval":
    if eval_dataset_fn is None:
      if train_dataset_fn is not None:
        tf.logging.warning("Using train_dataset_fn for perplexity eval")
        eval_datasets = [transformer_dataset.EvalDataset(
            name="eval",
            dataset_fn=functools.partial(train_dataset_fn,
                                         sequence_length=sequence_length,
                                         vocabulary=vocabulary,
                                         dataset_split=dataset_split),
            postprocess_fn=None,
            metric_fns=None)]
      else:
        raise ValueError(
            "for perplexity_eval, "
            "must provide one of eval_dataset_fn and train_dataset_fn")
    else:
      eval_datasets = eval_dataset_fn(
          sequence_length=sequence_length,
          vocabulary=vocabulary,
          dataset_split=dataset_split,
      )
    def _input_fn(params, eval_dataset):
      del params
      ds = eval_dataset.dataset_fn().map(_filter_features)
      ds = transformer_dataset.pad_dataset_with_zeroed_out_examples(ds)
      ds = (ds.batch(batch_size * (ensemble_inputs or 1), drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
      return ds
    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)
    for checkpoint_path in checkpoint_paths:
      for eval_dataset in eval_datasets:
        tf.random.set_random_seed(12345)
        random.seed(12345)
        num_examples = batch_size * perplexity_eval_steps
        # include the number of examples in the evaluation name so as to
        # make sure we are comparing apples to apples.
        name = "%s_%d" % (eval_dataset.name, num_examples)
        _ = estimator.evaluate(
            input_fn=functools.partial(_input_fn, eval_dataset=eval_dataset),
            steps=perplexity_eval_steps,
            checkpoint_path=checkpoint_path,
            name=name)
  elif mode == "eval":
    eval_model(estimator, vocabulary, sequence_length, batch_size,
               dataset_split, model_dir, eval_dataset_fn, eval_summary_dir,
               eval_checkpoint_step)
  elif mode == "infer":
    infer_model(estimator, vocabulary, sequence_length, batch_size, model_type,
                model_dir, eval_checkpoint_step)
  elif mode == "score_from_strings":
    score_from_strings(estimator, vocabulary, model_type, batch_size,
                       sequence_length, model_dir, eval_checkpoint_step,
                       dataset_split)
  elif mode == "score_from_dataset":
    score_from_dataset(estimator, vocabulary, batch_size, sequence_length,
                       model_dir, eval_checkpoint_step, dataset_split)
  elif mode == "export":
    if export_checkpoint_step:
      checkpoint_path = get_checkpoint_iterator(
          export_checkpoint_step, model_dir)
      if isinstance(checkpoint_path, list):
        checkpoint_path = checkpoint_path[0]
      else:
        checkpoint_path = next(checkpoint_path)
    else:
      # Use the latest checkpoint in the model directory.
      checkpoint_path = None
    export_model(estimator, export_path, vocabulary, sequence_length,
                 batch_size, checkpoint_path)

  else:
    raise ValueError(
        "unknown mode %s - must be train/perplexity_eval/eval/infer/export"
        % mode)

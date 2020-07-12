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

"""A toy model using Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import numpy
import tensorflow.compat.v1 as tf
import os, json

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib

FLAGS = flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 64, 'Training batch size.')
tf.flags.DEFINE_integer('sequence_size', 128, 'Sequence Len')
tf.flags.DEFINE_integer('hidden_size', 16, 'Size of each hidden layer.')
tf.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of layers.')
tf.flags.DEFINE_string('master_dtype', 'float32', 'dtype for master vars.')
tf.flags.DEFINE_string('slice_dtype', 'float32', 'dtype for slice vars.')
tf.flags.DEFINE_string('activation_dtype', 'float32', 'dtype for activations.')
tf.flags.DEFINE_string('optimizer', 'Adafactor', 'optimizer (SGD or Adafactor).')
tf.flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
tf.flags.DEFINE_string('mesh_shape', 'all:8', 'mesh shape')
tf.flags.DEFINE_string('layout', 'hidden_odd:all', 'layout rules')
tf.flags.DEFINE_integer('iterations', 500,
                        'Number of iterations per training loop.')
tf.flags.DEFINE_integer('train_steps', 10000, 'max steps')
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')
tf.flags.DEFINE_string(
    'model_dir',
    default='gs://datasets_storage_1/models/GPTNeo_prettybig',
    help='The directory where the model will be stored.')
tf.flags.DEFINE_string(
    'data_path',
    default='gs://datasets_storage_1/datasets/bundestag',
    help='The directory where the data is stored.')
tf.flags.DEFINE_string('datasets', default='bundestag_*.tfrecords","",10,"random_sample",1.0', help="dataset details")

# need flags for: batch_size, iterations, n_ctx, datasets, data_path
tf.flags.DEFINE_integer('n_ctx', 128, ' ')

# Optimizer settings
tf.flags.DEFINE_float('weight_decay', 0.01, 'weight decay setting for Adam optimizer')  # beta1, beta2, epsilon
tf.flags.DEFINE_float('beta1', 0.9, 'beta1 setting for Adam optimizer')
tf.flags.DEFINE_float('beta2', 0.98, 'beta2 setting for Adam optimizer')
tf.flags.DEFINE_float('epsilon', 1e-9, 'epsilon setting for Adam optimizer')

tf.flags.DEFINE_bool('use_tpu', True, 'use TPU')
tf.flags.DEFINE_string('model_params', 'configs/GPT_NEO_TEST.json', help="path to model config")
# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')


def text_dataset(files, stitch, datatype, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))

    if "sample" in datatype:
        def _parse_function(example_proto):
            features = {
                "hash": tf.VarLenFeature(tf.string),
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"], parsed_features["text"].dense_shape[0]
    else:
        def _parse_function(example_proto):
            features = {
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"]  # Assuming the text is not sparse

    dataset = dataset.map(_parse_function, num_parallel_calls=1)

    # Subsample method
    if "sample" in datatype:
        # Since samples can be less than the correct length, and TPUs don't like variable lengths, this function stitches together enough samples
        # to have a text at least 1024 tokens long. For this to work the stitch parameter must be correctly tuned so that
        # stitch * min(characters_in_text) >= amount
        def _stitch_text(x, y):
            x = tf.sparse.to_dense(x)

            def _get_x(i):
                return tf.gather(x[i], tf.range(y[i]))

            out = _get_x(0)
            for i in range(1, stitch):
                out = tf.concat([out, [50256], _get_x(i)], axis=0)  # text1<|endoftext|>text2

            return out

        # Hack-y way to stitch together multiple texts
        dataset = dataset.shuffle(1000 * stitch).batch(stitch, drop_remainder=True).map(_stitch_text,
                                                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Sample 1024(+1) tokens from the stitched together text
        if datatype == "random_sample":
            def _sample_text(x):
                s = tf.size(x)
                r = tf.random.uniform([], maxval=s - (FLAGS.n_ctx + 1), dtype=tf.dtypes.int32)
                r1 = tf.range(r, r + FLAGS.n_ctx)
                r2 = tf.range(r + 1, (r + 1) + FLAGS.n_ctx)
                r1 = tf.reshape(r1, [FLAGS.n_ctx])  # Somehow, this makes the compiler happy
                r2 = tf.reshape(r2, [
                    FLAGS.n_ctx])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
                vals1 = tf.gather(x, r1)
                vals2 = tf.gather(x, r2)

                vals1 = tf.reshape(vals1, [FLAGS.n_ctx])
                vals2 = tf.reshape(vals2, [FLAGS.n_ctx])
                return vals1, vals2

        else:
            def _sample_text(x):
                vals1 = x[:FLAGS.n_ctx]
                vals2 = x[1:FLAGS.n_ctx + 1]

                vals1 = tf.reshape(vals1, [FLAGS.n_ctx])
                vals2 = tf.reshape(vals2, [FLAGS.n_ctx])
                return vals1, vals2

        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(FLAGS.iterations * 2)

    dataset = dataset.repeat()

    return dataset


def generic_text(eval=False, dsets=[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]):
    # params["datasets"] = [(train glob, eval_glob, stitch, ["random_sample", "sample", "chunk"] weight)]
    i = 0 if not eval else 1
    datasets = [text_dataset(tf.io.gfile.glob(os.path.join(FLAGS.data_path, dataset[i])), stitch=dataset[2],
                             datatype=dataset[3], batch=False)
                for dataset in dsets]
    weights = [dataset[4] for dataset in dsets]

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(FLAGS.iterations * 2)

    return dataset


class TextInput(object):

    def __init__(self):
        self.dsets = [["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]

    def __call__(self, params):
        dset = generic_text(dsets=self.dsets)
        return dset


def toy_model(features, params, mesh):
    """A toy model implemented by mesh tensorlfow."""
    print('input details:')
    print(features.shape)

    batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
    sequence_dim = mtf.Dimension('sequence', FLAGS.sequence_size)

    master_dtype = tf.as_dtype(FLAGS.master_dtype)
    slice_dtype = tf.as_dtype(FLAGS.slice_dtype)
    activation_dtype = tf.as_dtype(FLAGS.activation_dtype)

    x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, sequence_dim]))
    x = mtf.cast(x, activation_dtype)
    h = x
    for lnum in range(1, FLAGS.num_hidden_layers + 2):
        if lnum + 1 == FLAGS.num_hidden_layers + 2:
            # output layer
            dim = io_dim
        elif lnum % 2 == 0:
            dim = mtf.Dimension('hidden_even', FLAGS.hidden_size)
        else:
            dim = mtf.Dimension('hidden_odd', FLAGS.hidden_size)
        h = mtf.layers.dense(
            h, dim,
            use_bias=False,
            master_dtype=master_dtype,
            slice_dtype=slice_dtype,
            name='layer_%d' % lnum)
    y = h
    g = tf.train.get_global_step()
    loss = mtf.reduce_mean(mtf.square(y - x))
    return y, loss


def model_fn(features, labels, mode, params):
    """A model is called by TpuEstimator."""
    del labels
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
    print('PARAMS:')
    print(params)
    print('Loading other params from file')
    # Read params of model
    with open(FLAGS.model_params, "r") as f:
        new_params = json.load(f)
    params.update(new_params)
    print('PARAMS AFTER LOAD FROM FILE:')
    print(params)
    if FLAGS.use_tpu:
        ctx = params['context']
        num_hosts = ctx.num_hosts
        host_placement_fn = ctx.tpu_host_placement_function
        device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
        tf.logging.info('device_list = %s' % device_list, )
        # TODO(ylc): Better estimation of replica cache size?
        replica_cache_size = 300 * 1000000  # 300M per replica
        # Worker 0 caches all the TPU binaries.
        worker0_mem = replica_cache_size * ctx.num_replicas
        devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
        var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                      devices_memeory_usage)
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
    else:
        var_placer = None
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, mesh_devices)
    mesh = mtf.Mesh(graph, 'my_mesh', var_placer)

    with mtf.utils.outside_all_rewrites():
        logits, loss = toy_model(features, params, mesh)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = mtf.gradients([loss],
                                  [v.outputs[0] for v in graph.trainable_variables])
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(
            learning_rate=FLAGS.lr,
            weight_decay_rate=FLAGS.lr * FLAGS.weight_decay,
            beta_1=FLAGS.beta1,
            beta_2=FLAGS.beta2,
            epsilon=FLAGS.epsilon)
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
    else:
        # for now, we can only export fully-replicated tensors.
        fully_replicated_logits = mtf.anonymize(logits)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    tf_loss = tf.to_float(lowering.export_to_tf_tensor(loss))

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))
        tf.logging.info('tf_update_ops: {}'.format(tf_update_ops))
        train_op = tf.group(tf_update_ops)
    else:
        tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)

    with mtf.utils.outside_all_rewrites():
        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        if mode == tf.estimator.ModeKeys.TRAIN:
            saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=2,
                defer_build=False,
                save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            saver_hook = tf.train.CheckpointSaverHook(
                FLAGS.model_dir,
                save_steps=1000,
                saver=saver,
                listeners=[saver_listener])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.TRAIN,
                loss=tf_loss,
                train_op=train_op,
                training_hooks=[restore_hook, saver_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(tf_logits):
                mean_logits = tf.metrics.mean(tf_logits)
                return {'mean_logits': mean_logits}

            eval_metrics = (metric_fn, [tf_logits])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                evaluation_hooks=[restore_hook],
                loss=tf_loss,
                eval_metrics=eval_metrics)


def run_toy_model_tpu():
    """Run a toy model on TPU."""
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    iterations_per_loop = FLAGS.iterations
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=iterations_per_loop,
        save_summary_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=iterations_per_loop,
            num_cores_per_replica=1,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
    classifier = tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size)
    current_step = estimator_lib._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    logging.info('Current step %d', current_step)
    if FLAGS.steps_per_checkpoint == 0:
        classifier.train(input_fn=TextInput(), max_steps=FLAGS.train_steps)
        return
    while current_step < FLAGS.train_steps:
        next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                              FLAGS.train_steps)
        classifier.train(input_fn=TextInput(), max_steps=next_checkpoint)
        current_step = next_checkpoint
        logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(
            input_fn=TextInput(),
            steps=156)  # since we have 10000 examples and batch_size = 64 per host
        logging.info('Eval results: %s', eval_results)


def main(_):
    run_toy_model_tpu()


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

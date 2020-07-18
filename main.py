"""GPT-like model in Mesh-Tensorflow"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import json
from functools import partial

from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from inputs import generic_text
from model_fns import model_fn

FLAGS = flags.FLAGS
tf.flags.DEFINE_string('model_params', 'configs/colab.json', help="path to model config")
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')

# Optimizer settings
tf.flags.DEFINE_bool('use_tpu', True, 'use TPU')

#Auto layout
tf.flags.DEFINE_bool('auto_layout', False, 'set layout rules automatically')
tf.flags.DEFINE_bool('auto_layout_and_mesh_shape', False, 'set layout rules automatically')

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


def run_model_tpu():
    """Run a GPT model on TPU."""
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # Read params of model
    with open(FLAGS.model_params, "r") as f:
        params = json.load(f)

    iterations_per_loop = params["iterations"]
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])

    # add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    params["auto_layout"] = FLAGS.auto_layout
    params["auto_layout_and_mesh_shape"] = FLAGS.auto_layout_and_mesh_shape
    params["use_tpu"] = FLAGS.use_tpu
    params["num_cores"] = mesh_shape.size
    tf.logging.info('params = %s' % params, )

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params["model_path"],
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
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["train_batch_size"],
        params=params)
    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(
        params["model_path"]))
    logging.info('Current step %d', current_step)
    if FLAGS.steps_per_checkpoint == 0:
        classifier.train(input_fn=partial(generic_text, eval=False), max_steps=params["train_batch_size"])
        return
    while current_step < params["train_steps"]:
        next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                              params["train_steps"])
        classifier.train(input_fn=partial(generic_text, eval=False), max_steps=next_checkpoint)
        current_step = next_checkpoint
        # logging.info('Starting to evaluate.')
        # eval_results = classifier.evaluate(
        #     input_fn=TextInput(),
        #     steps=156)  # since we have 10000 examples and batch_size = 64 per host
        # logging.info('Eval results: %s', eval_results)


def main(_):
    run_model_tpu()


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

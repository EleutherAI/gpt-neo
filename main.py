"""GPT-like model in Mesh-Tensorflow"""
import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from utils import save_config, expand_attention_types_params
from inputs import generic_text
from model_fns import model_fn
import pprint


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str) # JSON file that contains model parameters
    parser.add_argument('--steps_per_checkpoint', type=int, default=5000)
    parser.add_argument('--autostack', action="store_false")
    parser.add_argument('--auto_layout', action="store_true")
    parser.add_argument('--auto_layout_and_mesh_shape', action="store_true")
    args = parser.parse_args()

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('logs/{}.log'.format(os.path.basename(args.model).split(".")[0])),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers

    # Read params of model
    with open(args.model, "r") as f:
        params = json.load(f)

    # saves config to logdir for experiment management
    save_config(pprint.pformat(params), params["model_path"])

    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])

    # add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    params["auto_layout"] = args.auto_layout
    params["auto_layout_and_mesh_shape"] = args.auto_layout_and_mesh_shape
    params["autostack"] = args.autostack
    params["use_tpu"] = True if not args.tpu is None else False
    params["num_cores"] = mesh_shape.size
    params["steps_per_checkpoint"] = args.steps_per_checkpoint
    params["num_microbatches"] = 1
    # expand attention types param
    params["attention_types"] = expand_attention_types_params(params["attention_types"])
    assert len(params["attention_types"]) == params["n_layers"]  # assert that the length of expanded list = num layers
    logger.info('params = {}'.format(params))

    # Set up TPUs and Estimator
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params["model_path"],
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=params["iterations"],
        save_summary_steps=params["iterations"],
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=params["iterations"],
            num_cores_per_replica=1,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
    estimator = tpu_estimator.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=config,
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["train_batch_size"],
        params=params)

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info('Current step {}'.format(current_step))

    if args.steps_per_checkpoint == 0:
        estimator.train(input_fn=partial(generic_text, eval=False), max_steps=params["train_batch_size"])
        return

    if params["eval_steps"] > 0:
        # If eval is on - stop and eval every ckpt
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])
            estimator.train(input_fn=partial(generic_text, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint
            logger.info('Starting to evaluate.')
            eval_results = estimator.evaluate(
                input_fn=partial(generic_text, eval=False),
                steps=params["eval_steps"])
            logger.info('Eval results: %s', eval_results)

    else:
        while current_step < params["train_steps"]:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(generic_text, eval=False), max_steps=params["train_steps"])


if __name__ == '__main__':
    tf.disable_v2_behavior()
    main()

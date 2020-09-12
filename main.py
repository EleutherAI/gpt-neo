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
from utils import save_config, expand_attention_types_params, yes_or_no, remove_gs_or_filepath
from inputs import generic_text, pred_input, test_generic_text, test_pred_input, handle_pred_output, test_handle_pred_output
from model_fns import model_fn
from encoders import fetch_encoder
from configs import fetch_model_params
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tasks import task_descriptors

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str, default=None) # JSON file that contains model parameters
    parser.add_argument('--steps_per_checkpoint', type=int, default=5000)
    parser.add_argument('--auto_layout', action="store_true")
    parser.add_argument('--auto_layout_and_mesh_shape', action="store_true")
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--check_dataset', action='store_true')
    args = parser.parse_args()

    # rewire to use testing related functions if --test is on
    if args.test:
        args.model = 'test'

    input_fn = generic_text if not args.test else test_generic_text
    pred_input_fn = pred_input if not args.test else test_pred_input
    handle_pred_output_fn = handle_pred_output if not args.test else test_handle_pred_output
    assert args.model is not None, 'Model must be set'

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
    params = fetch_model_params(args.model)

    # Fetch encoder per params

    encoder = fetch_encoder(params)
    pred_input_fn = partial(pred_input_fn, enc = encoder)

    # Sample from Dataset if check dataset flag is on
    if args.check_dataset:
        tf.enable_eager_execution()
        dataset = input_fn(params)
        dataset_iter = dataset.make_one_shot_iterator()
        tensor, _ = next(dataset_iter)
        enc = fetch_encoder(params)

        for p in tensor[:1]:
            txt = enc.decode(p)
        #txt = enc.decode(tensor)
        max_id = tf.reduce_max(tensor)
        min_id = tf.reduce_min(tensor)

        print(tensor)
        print(tensor.shape)
        print('-' * 50)
        print(txt[:500], '\n...\n', txt[-500:])
        print('-' * 50)
        print('min token id: ', min_id)
        print('max token id: ', max_id)
        exit()


    # confirm deletion of checkpoint files if --new flag
    if args.new:
        path = params["model_path"]
        if yes_or_no("Are you sure you want to remove '{}' to start afresh?".format(path)):
            remove_gs_or_filepath(path)
        else:
            exit()

    # saves config to logdir for experiment management
    # save_config(pprint.pformat(params), params["model_path"])
    save_config(params, params["model_path"])

    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])

    # add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    params["auto_layout"] = args.auto_layout
    params["auto_layout_and_mesh_shape"] = args.auto_layout_and_mesh_shape
    params["use_tpu"] = True if not args.tpu is None else False
    params["num_cores"] = mesh_shape.size
    params["steps_per_checkpoint"] = args.steps_per_checkpoint
    # expand attention types param
    params["attention_types"] = expand_attention_types_params(params["attention_types"])
    assert len(params["attention_types"]) == params["n_layer"]  # assert that the length of expanded list = num layers
    logger.info('params = {}'.format(params))

    #TODO: we would like this to be as small as possible,
    # but if we're splitting by batch, a value < the dimensions batch is divided over will error.
    # can we change the mesh layout so batch will not be split at prediction time?
    params["predict_batch_size"] = params.get("predict_batch_size", 1) # Default to 1
    params["predict"] = args.predict

    eval_tasks = params.get('eval_tasks', [])
    has_predict_or_eval_steps_or_eval_tasks = params['predict_steps'] > 0 or params['eval_steps'] > 0 or len(eval_tasks) > 0

    for t in eval_tasks:
        assert t in task_descriptors, f'Eval task {t} is not known'
        task_descriptors[t]['init_fn'](params)

    # Set up TPUs and Estimator
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu) if params["use_tpu"] else None

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
        predict_batch_size=params["predict_batch_size"],
        params=params)

    def _make_task_estimator(task):
        task_params = params.copy()
        task_params['eval_task'] = task
        return tpu_estimator.TPUEstimator(
            use_tpu=params["use_tpu"],
            model_fn=model_fn,
            config=config,
            train_batch_size=params["train_batch_size"],
            eval_batch_size=params["train_batch_size"],
            predict_batch_size=params["predict_batch_size"],
            params=task_params)

    eval_task_estimators = {
        task: _make_task_estimator(task)
        for task in eval_tasks
    }

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info('Current step {}'.format(current_step))

    if args.predict:
        predictions = estimator.predict(input_fn=pred_input_fn)
        logger.info('Predictions generated')
        enc = fetch_encoder(params)
        handle_pred_output_fn(predictions, logger, enc, out_name=f"predictions_{current_step}")
        return
    elif has_predict_or_eval_steps_or_eval_tasks:
        # If predict and/or eval is on - stop and predict and/or eval every ckpt
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint
            if params['predict_steps'] > 0:
                logger.info('Starting to run predictions.')
                predictions = estimator.predict(input_fn=pred_input_fn)
                enc = fetch_encoder(params)
                handle_pred_output_fn(predictions, logger, enc, out_name=f"predictions_{current_step}")
            if params['eval_steps'] > 0:
                logger.info('Starting to evaluate.')
                eval_results = estimator.evaluate(
                    input_fn=partial(input_fn, eval=True),
                    steps=params["eval_steps"])
                logger.info('Eval results: %s', eval_results)
            for task in params['eval_tasks']:
                logger.info(f'Starting evaluation task {task}.')
                task_info = task_descriptors[task]['get_task_info_fn'](params)
                task_estimator = eval_task_estimators[task]
                task_input_fn = task_descriptors[task]['input_fn']
                eval_results = task_estimator.evaluate(
                    input_fn=task_input_fn,
                    steps=task_info['n_steps'],
                    name=task)
                logger.info(f'Eval task {task} results: {eval_results}')
        return
    else:
        while current_step < params["train_steps"]:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=params["train_steps"])


if __name__ == '__main__':
    tf.disable_v2_behavior()
    main()


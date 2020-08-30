"""GPT-like model in Mesh-Tensorflow"""
import argparse
import collections
import json
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

from typing import Any, Dict

import mesh_tensorflow as mtf
import tensorflow as tf
from absl.flags import argparse_flags
from absl import app

from tensorflow.compat import v1
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)

from encoders import fetch_encoder
from inputs import (generic_text, handle_pred_output, pred_input,
                    test_generic_text, test_handle_pred_output,
                    test_pred_input)
from model_fns import model_fn
from utils import (expand_attention_types_params, remove_gs_or_filepath,
                   save_config, yes_or_no)

import dataclasses

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass


@dataclass
class DatasetConfig:
    src: str

@dataclass
class ClusterConfig:
    num_cores: int
    use_tpu: bool

@dataclass
class InfeedConfig:
    batch_size: int
    dataset: DatasetConfig

    def __getitem__(self, value):
        return 42

@dataclass
class ModelConfig:
    activation_function: str

@dataclass
class TrainerConfig:
    cluster: ClusterConfig
    infeed: InfeedConfig
    model: Dict
    trainer: Dict
    other: Any
    regularization: Dict

def setup_logging(args):
    # Setup logging
    # Path("logs").mkdir(exist_ok=True)
    logging.set_verbosity(logging.INFO)
    # handlers = [
    #     logging.FileHandler('logs/{}.log'.format(os.path.basename(args.model).split(".")[0])),
    #     logging.StreamHandler(sys.stdout)
    # ]
    # logger = logging.getLogger('tensorflow')
    # logger.handlers = handlers
    #return logger

def parse_args(args):
    # Parse command line arguments
    parser = argparse_flags.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--config', required=True, default='test', type=str) # JSON file that contains model parameters
    # parser.add_argument('--steps_per_checkpoint', type=int, default=5000)
    # parser.add_argument('--auto_layout', action="store_true")
    # parser.add_argument('--auto_layout_and_mesh_shape', action="store_true")
    # parser.add_argument('--new', action='store_true')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--predict', action='store_true', default=False)
    #parser.add_argument('--check-input', action='store_true', default=True)
    parser.add_argument('--dry_run', action='store_true', 
                                     default=True, 
                                     help="load the configuration and everything but run for zero steps")
    return parser.parse_args(args[1:])

Trainer = collections.namedtuple('Trainer', 
                                    ['name',
                                    'input_fn', 
                                    'config',
                                    'model_fn',
                                    # 'predict_fn', 
                                    'handle_prediction_output_fn'])

def load_trainer_config(location):

    with tf.io.gfile.GFile(location) as fd: 
        params = json.loads(fd.read())

    n_vocab = params['n_vocab']
    params['datasets'] = []
    datasets = params.get('datasets', [])
    
    for d in datasets:
        with tf.io.gfile.GFile(d) as fd: 
            dataset_config = json.load(fd.read())
            params['datasets'].append(dataset_config)

    return params

def load_trainer(args) -> Trainer:
    with tf.io.gfile.GFile(args.config) as fd:
        params = json.load(fd)

    cfg = TrainerConfig(**params)
    
    json.dump(params, sys.stdout, indent=2)

    if args.test:
        # rewire to use testing related functions if --test is on
        return Trainer(
            name='test',
            config=cfg,
            model_fn=lambda *args: None,
            input_fn=test_generic_text,
            # pred_input_fn=test_pred_input,
            handle_prediction_output_fn=test_handle_pred_output
        )
    
    if args.model == '':
        raise ValueError('Model must be set')
    
    # params = load_trainer_config(args.model)

    # Fetch encoder per params
    encoder = fetch_encoder(params)
   
    # model.pred_input_fn = partial(pred_input_fn, enc = encoder)

    return Trainer(
        name=args.model,
        input_fn=generic_text,
        config=cfg,
        # pred_input_fn=pred_input,
        handle_prediction_output_fn=handle_pred_output,
    )

def check_dataset(trainer, args):
    sample_size = 10
    sampled_files = random.choices(trainer.config.infeed.dataset.src, k=sample_size)
    with v1.Session(graph=tf.Graph()) as sess:
        ds = trainer.input_fn(trainer.config.infeed)

        it = ds.make_one_shot_iterator()
        example = it.get_next()
        
        for _ in range(42):
            try:
                result = sess.run(example) #, max_id_tf, min_id_tf])
                # pt = PreProcessedTextLine(
                #     id = result['id'],
                #     content=result['content'],
                #     target=result['target'],
                #     offset_start=result['offset_start'],
                #     offset_end=result['offset_end'],
                # )

                # ids = tokenizer.decode(result['target'])

                # logging.info('gold text:    %r', pt.content.decode('utf-8'))
                # logging.info('decoded:       %r', ids),
                # logging.info('tokenization: %s', [pt.content.decode('utf-8')[slice(int(start), int(end))] for start,end in zip(pt.offset_start, pt.offset_end)])
                # logging.info('-' * 10)
                print(result)
            except tf.errors.OutOfRangeError:
                break

def load_model_config(params, args):
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
    logging.info('params = {}', params)

    #TODO: we would like this to be as small as possible,
    # but if we're splitting by batch, a value < the dimensions batch is divided over will error.
    # can we change the mesh layout so batch will not be split at prediction time?
    params["predict_batch_size"] = params.get("predict_batch_size", 1) # Default to 1
    params["predict"] = args.predict
    return params

def main(args):
    logger = setup_logging(args)
    
    trainer = load_trainer(args)
    
    # Sample from Dataset if check dataset flag is on
    if args.dry_run:
        check_dataset(trainer, args)
        return

    # confirm deletion of checkpoint files if --new flag
    # if args.new:
    #     path = params["model_path"]
    #     if yes_or_no("Are you sure you want to remove '{}' to start afresh?".format(path)):
    #         remove_gs_or_filepath(path)
    #     else:
    #         exit()

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

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info('Current step {}', current_step)

    if args.predict:
        predictions = estimator.predict(input_fn=pred_input_fn)
        enc = fetch_encoder(params)
        handle_pred_output_fn(predictions, logger, enc, out_name=f"predictions_{current_step}")
        return
    elif params["predict_steps"] > 0:
        # If both predict & eval are on - stop and eval / predict every ckpt
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint
            logger.info('Starting to run predictions.')
            predictions = estimator.predict(input_fn=pred_input_fn)
            enc = fetch_encoder(params)
            handle_pred_output_fn(predictions, logger, enc, out_name=f"predictions_{current_step}")
    elif params["predict_steps"] > 0 and params["eval_steps"] > 0:
        # If predict is on - stop and predict every ckpt
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint
            logger.info('Starting to run predictions.')
            predictions = estimator.predict(input_fn=pred_input_fn)
            enc = fetch_encoder(params)
            handle_pred_output_fn(predictions, logger, enc, out_name=f"predictions_{current_step}")
            logger.info('Starting to evaluate.')
            eval_results = estimator.evaluate(
                input_fn=partial(input_fn, eval=True),
                steps=params["eval_steps"])
            logger.info('Eval results: %s', eval_results)
        return
    elif params["eval_steps"] > 0:
        # If eval is on - stop and eval every ckpt
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint
            logger.info('Starting to evaluate.')
            eval_results = estimator.evaluate(
                input_fn=partial(input_fn, eval=True),
                steps=params["eval_steps"])
            logger.info('Eval results: %s', eval_results)
        return
    else:
        while current_step < params["train_steps"]:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=params["train_steps"])


if __name__ == '__main__':
    tf.disable_v2_behavior()
    app.run(main, flags_parser=parse_args)
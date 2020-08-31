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
from typing import Any, Dict, Optional


import mesh_tensorflow as mtf
import tensorflow as tf
from absl.flags import argparse_flags
from absl import app

from tensorflow.compat import v1
from tensorflow.python.platform import tf_logging as logging

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)

from encoders import fetch_encoder
from inputs import (generic_text, handle_pred_output, pred_input,
                    RandomTokenGenerator, RandomTokenGeneratorConfig, test_handle_pred_output,
                    test_pred_input)
from model_fns import model_fn
from utils import (remove_gs_or_filepath, save_config, yes_or_no)



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

# def add_config_cmd(subparsers):

#     config_opt = subparsers.add_parser(
#         'config', help='Configuration management',
#         # Do not make absl FLAGS available after the subcommand `roll_dice`.
#         inherited_absl_flags=None)
#     config_opt.add_argument('input_file', type=str)

    # shuffle_parser = subparsers.add_parser('shuffle', help='Shuffle inputs.')
    # shuffle_parser.add_argument(
    #     'inputs', metavar='I', nargs='+', help='Inputs to shuffle.')

def parse_args(args):
    # Parse command line arguments
    parser = argparse_flags.ArgumentParser()
    subparsers = parser.add_subparsers(help='Available commands', dest='subparser')
    for name, cmd in SUBCOMMANDS.items(): 
        cmd_parser = subparsers.add_parser(
            name, help=cmd.__doc__,
            # Do not make absl FLAGS available after the subcommand `roll_dice`.
            inherited_absl_flags=False)
        cmd.parse_args(args, cmd_parser)

    # parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    # # parser.add_argument('--config', required=True, default='test', type=str) # JSON file that contains model parameters
    # # parser.add_argument('--steps_per_checkpoint', type=int, default=5000)
    # # parser.add_argument('--auto_layout', action="store_true")
    # # parser.add_argument('--auto_layout_and_mesh_shape', action="store_true")
    # # parser.add_argument('--new', action='store_true')

    # parser.add_argument('--train', action='store_true', default=False)
    # parser.add_argument('--eval', action='store_true', default=False)
    # parser.add_argument('--predict', action='store_true', default=False)
    # #parser.add_argument('--check-input', action='store_true', default=True)
    # parser.add_argument('--dry_run', action='store_true', 
    #                                  default=True, 
    #                                  help="load the configuration and everything but run for zero steps")
    return parser.parse_args(args[1:])



import importlib

SUBCOMMANDS = {}

def register_subcommand(module_name):
    m = importlib.import_module(module_name)
    SUBCOMMANDS[module_name] = m


def main(args):
    logger = setup_logging(args)

    cmd = SUBCOMMANDS.get(args.subparser, None)
    if cmd is None:
        raise ValueError('invalid command %s', args.subparser)
    return cmd.main(args)


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
    register_subcommand('configure')
    register_subcommand('train')
    register_subcommand('eval')
    register_subcommand('interactive')
    # register_subcommand('predict')
    app.run(main, flags_parser=parse_args)
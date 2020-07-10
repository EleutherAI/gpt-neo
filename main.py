import argparse
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path

import tensorflow as tf

from inputs import generic_text
from model_fns import *
from predict_fns import *


# This program was designed to function with multiple kinds of models, but currently only GPT2 is supported
# The first element in the tupel is the model function, the second is the function called when predicting
models = {
    "GPT2": (gpt2_model, gpt2_predict),
    "GPT2_mesh": (gpt2_model_mesh, gpt2_predict) #TODO: mesh predict fn
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str, default="sparse") # Name of TPU to train on, if any
    parser.add_argument('--model', type=str, default="configs/117M.json") # JSON file that contains model parameters
    parser.add_argument("--predict_file", type=str) # File to take as input for predict
    parser.add_argument("--predict_text", type=str) # Take string directly from args
    parser.add_argument("--top_k", type=int) # Top K truncation parameter for text generation
    args = parser.parse_args()


    # Read params of model
    with open(args.model, "r") as f:
        params = json.load(f)

    params["use_tpu"] = True if not args.tpu is None else False

    # Get prediction text
    predict_mode = False
    if not params["use_tpu"]:
      if args.predict_file is not None:
          predict_mode = True
          with open(args.predict_file) as f:
              text = f.read()
      elif args.predict_text is not None:
          predict_mode = True
          text = args.predict_text
      elif args.predict_file is not None and args.predict_text is not None:
          print("ERROR: Specify exactly one of --predict_file and --predict_text!")
          sys.exit()


    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('logs/{}.log'.format(args.model.split("/")[-1])),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers

    if args.top_k is not None:
        params["top_k"] = args.top_k 

    if not "precision" in params.keys():
        params["precision"] = "float32" # Doesn't actually do anything since float32 is the default anyways. Only recognized other dtype is "bfloat16"

    if not "iterations" in params.keys():
        params["iterations"] = 1 # Because this controls how many samples are prefetched

    logger.info(params)

    model_fn = models[params["model"]][0]
    predict_fn = models[params["model"]][1]

    if params["use_tpu"] and not predict_mode:
        # Resolve TPU cluster and runconfig
        if args.tpu == 'colab_tpu':
          tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver()
        else:
          tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=params["model_path"],
            cluster=tpu_cluster_resolver,
            save_checkpoints_secs=60*30,
            session_config=tf.ConfigProto(
                # allow_soft_placement=True,
                # log_device_placement=True
                ),
                tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=params["iterations"])
        )

        # Set up network
        network = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                use_tpu=True,
                train_batch_size=params["train_batch_size"], # These are the global sizes, must be divisible by replicas
                eval_batch_size=params["eval_batch_size"],
                predict_batch_size=params["predict_batch_size"],
                config=run_config,
                params=params)

    else:
        # Non TPU setup
        if not predict_mode:
            params["batch_size"] = params["train_batch_size"]
        else:
            params["batch_size"] = params["predict_batch_size"]

            from models.gpt2 import encoder
            enc = encoder.get_encoder(params["encoder_path"])
            tokens = enc.encode(text)
            params["text_len"] = len(tokens)
            if params["text_len"] > 1024:
                params["text_len"] = 1024

        run_config = tf.estimator.RunConfig(
            model_dir=params["model_path"],
            session_config=tf.ConfigProto(
                # log_device_placement=True,
                # allow_soft_placement=True
            ),
        )

        network = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params)

    if predict_mode:
        logger.info("Generating predictions...")
        predict_fn(network, text, params)
        sys.exit()

    # Train eval loop
    while True:
        start = time.time()

        network.train(
                input_fn=partial(generic_text, eval=False),
                steps=params["train_steps"])


        end = time.time()
        logger.info("\nTrain loop took {:.2f}s\n".format(end-start))

        eval_result = network.evaluate(
           input_fn=partial(generic_text, eval=True),
           steps=params["eval_steps"])

        logger.info("\nEval Results: {}\n".format(str(eval_result)))

        if network.get_variable_value("global_step") > params["max_steps"]:
            logger.info("Done!")
            break

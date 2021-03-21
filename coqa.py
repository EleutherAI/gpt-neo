"""GPT-like model in Mesh-Tensorflow"""

from functools import partial
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from utils import expand_attention_types_params, setup_logging
from model_fns import model_fn
from data.encoders import fetch_encoder
from configs import fetch_model_params
import argparse
from tasks import coqa_input, coqa_init
from inputs import handle_pred_output
import json
import os

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.")
    parser.add_argument("-p", "--results_path", type=str, help="path to save coqa results to", default='coqa_results.json')
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=["device:GPU:0"],
                        help="If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--entmax_sampling", action="store_true", help="(experimental) use entmax sampling")

    args = parser.parse_args()
    assert args.model is not None, "Model must be set"
    return args


def main(args):
    # Setup logging
    logger = setup_logging(args)

    # Read params of model
    params = fetch_model_params(args.model)

    # get current step
    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info(f"Current step {current_step}")

    # Fetch encoder per params
    encoder = fetch_encoder(params)

    # Add to params:
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    params["num_cores"] = mesh_shape.size
    params["use_tpu"] = True if not args.tpu is None else False
    params["gpu_ids"] = args.gpu_ids
    # Expand attention types param
    params["attention_types"] = expand_attention_types_params(params["attention_types"])
    assert len(params["attention_types"]) == params["n_layer"]  # Assert that the length of expanded list = num layers
    params["predict_batch_size"] = params.get("predict_batch_size", 1)  # Default to 1
    params['model'] = params.get("model", "GPT") # Default model selection to GPT since it's the only option for now
    # Set sampling parameters
    params["sampling_use_entmax"] = args.entmax_sampling
    # Sample quality of MoE models suffers when using the faster sampling method, so default to slow_sampling if
    # moe layers are present
    params["slow_sampling"] = True if params["moe_layers"] is not None else False
    params['remove_partial_sequences'] = True
    logger.info(f"params = {params}")

    # Set up TPUs and Estimator
    if args.tpu == "colab":
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() if params["use_tpu"] else None
    else:
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



    coqa_init(params)
    input_fn, prompts, keys, steps = coqa_input(params)
    coqa_out = []

    estimator = tpu_estimator.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=config,
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["train_batch_size"],
        predict_batch_size=params["predict_batch_size"],
        params=params)

    # Predict
    predictions = estimator.predict(input_fn=partial(input_fn, prompts=prompts))
    logger.info("Predictions generated")
    predictions = handle_pred_output(predictions, logger, encoder, params)
    for p, _k in zip(predictions, keys):
        _id, turn_id = _k.split('_')
        p = p.split('.')[0].replace('\n', '').replace('A:', '').strip() # only take the first sentence
        coqa_out.append({'id': _id, 'turn_id': int(turn_id), 'answer': p})
        with open(args.results_path, 'w') as f:
            json.dump(coqa_out, f)

    COQA_DEV_SET = "coqa-dev-v1.0.json"
    if not os.path.isfile(COQA_DEV_SET):
        os.system(f'wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O {COQA_DEV_SET}')
    os.system(f'python3 evaluate_coqa.py --data-file {COQA_DEV_SET} --pred-file {args.results_path}')

if __name__ == "__main__":
    tf.disable_v2_behavior()
    args = parse_args()
    main(args)

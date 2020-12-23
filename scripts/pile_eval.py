import tensorflow.compat.v1 as tf
from copy import deepcopy
import json
import argparse
import subprocess
import logging
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("--tpu_name", type=str)
parser.add_argument("--batch_split", type=int, default=-1)
parser.add_argument("--models", nargs="+", type=str,
                    default=["gpt3_XL_256_SmallPileAblation_CC100en",
                             "gpt3_XL_256_SmallPileAblation_CC_raw",
                             "gpt3_XL_256_SmallPileAblation_Pile"])
parser.add_argument("--path_to_test_sets", type=str, default="gs://neo-datasets/pile_test_sets/*")
args = parser.parse_args()

dataset_config_template = {
    "n_vocab": 50257,
    "path": "gs://neo-datasets/pile/pile_*.tfrecords",
    "eval_path": "",
    "tokenizer_is_pretrained": True,
    "tokenizer_path": "gpt2",
    "eos_id": 50256,
    "padding_id": 50257
}

all_eval_sets = tf.io.gfile.glob(args.path_to_test_sets)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f"logs/pile_eval.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

results = {}
for model in args.models:
    results[model] = {}
    for eval_set in all_eval_sets:
        eval_set_name = eval_set.split("/")[-1]

        eval_path = f"{eval_set}/*.tfrecords"
        model_path = f"configs/{model}.json"
        tmp_model_path = "configs/tmp.json"
        tmp_dataset_path = "configs/dataset_configs/tmp.json"
        with open(model_path) as f:
            params = json.load(f)
        if args.batch_split != -1:
            params["mesh_shape"] = re.sub("x:\d+,", f"x:{args.batch_split},", params["mesh_shape"])
        dataset_config = deepcopy(dataset_config_template)
        dataset_config["eval_path"] = eval_path
        params.pop("eval_tasks")
        params["datasets"] = ["tmp"]
        with open(tmp_dataset_path, 'w') as f:
            json.dump(dataset_config, f)
        with open(tmp_model_path, 'w') as f:
            json.dump(params, f)
        command = f"python3 main.py --tpu {args.tpu_name} --model tmp --eval"
        logging.info("PARAMS: ")
        logging.info(params)
        logging.info("COMMAND: ")
        logging.info(command)
        logs = subprocess.check_output(command, shell=True)
        m = re.search('Eval results: \{(.+?)}', logs.decode())
        if m:
            eval_results = m.group(1)
        else:
            eval_results = "none"

        result_str = f"Eval results {model} / {eval_set}: \n {eval_results}"
        print(result_str)
        logging.info(result_str)
        results[model][eval_set] = eval_results

        print('-'*10)
        print(results)
        print('-'*10)


with open("pile_eval_results.json", 'w') as f:
    json.dump(results, f, indent=4)

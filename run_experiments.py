import subprocess
import argparse
import hashlib
import time
import json
import os


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('base_config', type=str,
                        help="The path to the .json config with will be use as the bases for this run.")
    parser.add_argument('tpu_start_id', type=int, help="The tpu ID at with the TPU IDs will start.")
    parser.add_argument('--run_config', default='', type=str, help="The path to the .json config that continents "
                                                             "the Hyperparameters to be run. The config must contain a"
                                                             " dict in with each entry is a list, the list continents "
                                                             "the different Hyperparameters for variable")
    parser.add_argument('--run_name_prefix', type=str, default='gs://text-datasets/video-transformer/')
    parser.add_argument('--number_of_repetitions', type=int, default=1, help="The number of times the same "
                                                                            "parameters will get tested.")
    parser.add_argument('--repetition_start_idx', type=int, default=0)
    parser.add_argument('--tpu_name_subfix', type=str, default='', help="A string that will be added "
                                                                       "at the and of all TPU names.")
    parser.add_argument('--use_preemptible', type=str, default='true')
    parser.add_argument('--tpu_type', type=str, default='v3-8')
    parser.add_argument('--start_up_sleep', type=int, default=0)

    args = parser.parse_args()

    tpu_type = args.tpu_type
    tpu_type_str = '"' + tpu_type + '"'

    with open(args.base_config) as f:
        base_config = json.load(f)

    if args.run_config != "":
        with open(args.run_config) as f:
            run_config = json.load(f)
    else:
        run_config = {}

    if not os.path.exists("buffer_configs/"):
        os.makedirs("buffer_configs/")

    tpu_id = args.tpu_start_id
    run_config_key = run_config.keys()
    run_param_pos = [0] * len(run_config_key)
    param_pos = 0

    run = True

    while run:

        copy_base_config = base_config.copy()

        for idx, key in enumerate(run_config_key):
            copy_base_config[key] = run_config[key][run_param_pos[idx]]
            run_param_pos[idx] = run_param_pos[idx] + 1
            if run_param_pos[idx] >= len(run_config[key]):
                run_param_pos[idx] = 0
                param_pos = param_pos + 1

        for repetition_idx in range(args.repetition_start_idx, args.number_of_repetitions):
            tpu_name = f"tpu-{tpu_type}-euw4a-{tpu_id}" + args.tpu_name_subfix

            run_name = f"-run={repetition_idx}"
            run_name = "-".join([f"{key}={copy_base_config[key]}" for key in run_config_key]) + run_name
            run_name = run_name.replace(' ', '_').replace("'", '').replace(":", '=').replace(",", '-')

            copy_base_config['model_path'] = args.run_name_prefix + run_name

            with open(f"buffer_configs/{tpu_id}.json", 'w+') as w:
                w.write(json.dumps(copy_base_config))

            experiment_command = f"python3 main.py --model buffer_configs/{tpu_id}.json --tpu {tpu_name}"
            delete_command = f"pu delete {tpu_name} --yes"
            tpu_creat_command = f"gcloud compute tpus create {tpu_name} --zone europe-west4-a " \
                                f"--range 10.48.{tpu_id}.0/29 --network tpu-euw4a --version 1.15.4 " \
                                f"--accelerator-type {tpu_type_str}"

            if str2bool(args.use_preemptible):
                tpu_creat_command = tpu_creat_command + " --preemptible"

            if len(run_name) > 66:
                run_name = hashlib.sha256(run_name.encode('utf-8')).hexdigest()

            prosses_name = f"tpu_id:{tpu_id}--{run_name}"

            subprocess.run(['screen', '-dmS', prosses_name, 'bash', '-c',
                            f"({tpu_creat_command} && {experiment_command}) ; {delete_command}"])

            tpu_id = tpu_id + 1

            print(f"Creating {prosses_name}")
            time.sleep(args.start_up_sleep)

        if param_pos >= len(run_config_key):
            run = False
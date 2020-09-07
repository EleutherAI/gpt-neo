import atexit
import sacred
import argparse
import time
import math
import subprocess
import shutil
import os
import json
import threading
import requests
import glob
from configs import fetch_model_params

parser = argparse.ArgumentParser()
parser.add_argument('--tpu', type=str, required=True) # Name of TPU to train on, if any
parser.add_argument('--model', type=str, required=True) # JSON file that contains model parameters
parser.add_argument('--experiment_name', type=str, required=True) # name of experiment (will show up in omniboard)
parser.add_argument('--steps_per_checkpoint', type=int, default=5000)
parser.add_argument('--autostack', action="store_false")
parser.add_argument('--auto_layout', action="store_true")
parser.add_argument('--auto_layout_and_mesh_shape', action="store_true")
parser.add_argument('--new', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--no_delete_tpu', action='store_true')
args = parser.parse_args()

params = fetch_model_params(args.model)

ex = sacred.Experiment(args.experiment_name)
ex.observers.append(sacred.observers.QueuedMongoObserver(url='127.0.0.1:27017', db_name='db', username='user', password='password'))

import socket
def get_open_port(lo=8000, hi=8100):
    for i in range(lo, hi):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', i)) != 0:
                return i


def train_thread(tpu, id):
    print('starting training on', tpu)

    # pass binary flags through
    opts = ''
    for flag in ['auto_layout', 'auto_layout_and_mesh_shape', 'new', 'test', 'predict', ]:
        if args.__getattribute__(flag):
            opts += ' --' + flag
    opts = ''
    for flag in ['autostack', ]:
        if not args.__getattribute__(flag):
            opts += ' --' + flag

    cmd = "python3 main.py --tpu {tpu} --model run_configs/config_{id}.json --steps_per_checkpoint {steps_per_checkpoint} {opts}".format(tpu=tpu, id=id, steps_per_checkpoint=args.steps_per_checkpoint, opts=opts)
    print('Running:', cmd)
    os.system(cmd)
    print('exited training!')
    
    if args.no_delete_tpu:
        print('recreate done, exiting train_thread - not killing tpu!')
        return
    print("Recreating {} in 60sec...".format(tpu))
    time.sleep(60)
    os.system("pu recreate {} --yes --retry 3600 --retry-randomness 1.5".format(tpu))
    print('recreate done, exiting train_thread')


def get_json(uri, params=None, timeout=15):
    resp = requests.get(uri, params, timeout)
    resp.raise_for_status()
    return resp.json()

def get_tag_sets(base_uri):
    j = get_json(f'{base_uri}/data/plugin/scalars/tags', {'experiment': ''})
    assert isinstance(j, dict)
    return {
        run: j[run].keys()
        for run in j.keys()
    }

def get_scalar_data(base_uri, run, tag):
    j = get_json(f'{base_uri}/data/plugin/scalars/scalars', {'experiment': '', 'run': run, 'tag': tag})
    assert isinstance(j, list)
    return j

def get_run_data(port):
    base_uri = f'http://localhost:{port}/'
    r = {}
    try:
        tag_sets = get_tag_sets(base_uri)
        runs = tag_sets.keys()
        if '.' in runs:
            if 'loss' in tag_sets['.']:
                r['loss'] = get_scalar_data(base_uri, '.', 'loss')
        if 'eval_lambada' in runs:
            if 'lambada_acc' in tag_sets['eval_lambada']:
                r['lambada_acc'] = get_scalar_data(base_uri, 'eval_lambada', 'lambada_acc')
            if 'lambada_log_ppl' in tag_sets['eval_lambada']:
                r['lambada_ppl'] = [
                    [t, s, math.exp(lp)]
                    for [t, s, lp] in get_scalar_data(base_uri, 'eval_lambada', 'lambada_log_ppl')
                ]
    except:
        import traceback
        traceback.print_exc()
    return r


@ex.main
def main(_run):
    print('Starting run', _run._id)
    print('WARNING: please remember to remove old metric log files from the model directory.')

    os.makedirs('run_configs', exist_ok=True)
    shutil.copy(args.model if args.model.endswith('.json') else 'configs/{}.json'.format(args.model), 'run_configs/config_{}.json'.format(_run._id))

    tensorboard_port = get_open_port()
    print('Tensorboard at port:', tensorboard_port)
    print('Tensorboard url: ', 'http://eleutherai.bmk.sh:'+ str(tensorboard_port))
    os.system("screen -S tensorboard_{} -d -m bash -c 'tensorboard --logdir {} --port {} --bind_all --reload_multifile=true || tensorboard --logdir {} --port {} --reload_multifile=true'".format(_run._id, params["model_path"], tensorboard_port,params["model_path"], tensorboard_port,))
    atexit.register(goodbye, _run._id)

    curr_step = {}

    while True:
        trainthd = threading.Thread(target=train_thread, args=(args.tpu, _run._id))
        trainthd.start()
        while trainthd.is_alive():
            time.sleep(60)
            print('Polling tensorboard for metrics...')
            data = get_run_data(tensorboard_port)
            for k in data.keys():
                for ts, step, val in data[k]:
                    if step <= curr_step.get(k, -1):
                        continue
                    _run.log_scalar(k, val, step)
                    if k == 'loss':
                        _run.log_scalar('tb_ts', ts, step)
                        print('Logged to sacred: step={},loss={},tb_ts={}'.format(step, val, ts))
                    curr_step[k] = step

        if args.no_delete_tpu:
            break


def goodbye(id):
    print("You are now leaving the Python sector.")
    print("Sie verlassen den pythonischen Sektor.")

    os.system("screen -S tensorboard_{} -X quit".format(id))

        
if __name__ == '__main__':
    for file in glob.glob("**/*"):
        if file.split('.')[-1] in ['py']:    
            print('Adding', file, 'to sacred')
            ex.add_source_file(file)

    ex.add_config({
        'tpu_name': args.tpu,
        **params
    })

    ex.run()

import atexit
import sacred
import argparse
import main
import time
import subprocess
import shutil
import os
import json
import threading
import requests

ex = sacred.Experiment('eleutherai-gpt3')
ex.observers.append(sacred.observers.QueuedMongoObserver(url='127.0.0.1:27017', db_name='db', username='user', password='password'))

import socket
def get_open_port(lo=8000, hi=8100):
    for i in range(lo, hi):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', i)) != 0:
                return i


def train_thread(tpu, id):
    print('starting training on', tpu)
    os.system("python3 main.py --tpu {tpu} --model run_configs/config_{id}.json".format(tpu=tpu, id=id))
    print('exited training!')

    print("Recreating {} in 60sec...".format(tpu))
    time.sleep(60)
    os.system("pu recreate {} --yes".format(tpu))
    print('recreate done, exiting train_thread')


def get_run_data(port):
    url_stem = 'http://localhost:{}'.format(port)
    run_stem = ''
    request_timeout = 15
    url = '{}/data/plugin/scalars/scalars'.format(url_stem)
    try:
        resp = requests.get(url, params={
            'tag': 'loss',
            'run': '{}.'.format(run_stem),
            'experiment': '',
        }, timeout=request_timeout)
        resp.raise_for_status()
        return resp.text
    except:
        import traceback
        traceback.print_exc()
        print(url_stem, run_stem)
        return None


@ex.main
def main(_run):
    print('Starting run', _run._id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str, required=True) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str, required=True) # JSON file that contains model parameters
    args = parser.parse_args()

    os.makedirs('run_configs', exist_ok=True)
    shutil.copy(args.model, 'run_configs/config_{}.json'.format(_run._id))

    params = json.load(open('run_configs/config_{}.json'.format(_run._id)))
    ex.add_config({
        'tpu_name': args.tpu,
        **params
    })

    tensorboard_port = get_open_port()
    print('Tensorboard at port:', tensorboard_port)
    os.system("screen -S tensorboard_{} -d -m tensorboard --logdir {} --port {}".format(_run._id, params["model_path"], tensorboard_port))
    atexit.register(goodbye, _run._id)

    curr_step = 0

    while True:
        trainthd = threading.Thread(target=train_thread, args=(args.tpu, _run._id))
        trainthd.start()
        while trainthd.is_alive():
            time.sleep(60)
            print('Polling tensorboard for metrics...')
            data = get_run_data(tensorboard_port)
            print('polled data:', data)
            for ts, step, val in data:
                if step < curr_step:
                    continue

                _run.log_scalar('loss', val, step)
                print('Logged to sacred: step={},loss={}'.format(step, val))
                curr_step = step


def goodbye(id):
    print("You are now leaving the Python sector.")
    print("Sie verlassen den pythonischen Sektor.")

    os.system("screen -S tensorboard_{} -X quit".format(id))

        
if __name__ == '__main__':
    ex.run()
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
import socket
import subprocess
import queue
import sys
import signal


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
parser.add_argument('--eval', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--no_delete_tpu', action='store_true')
parser.add_argument('--initial_heartbeat_timeout', type=int, default=7200)
parser.add_argument('--heartbeat_timeout', type=int, default=1800) # kill and restart if nothing logged to tensorboard in this many seconds
args = parser.parse_args()

params = fetch_model_params(args.model)

ex = sacred.Experiment(args.experiment_name)
ex.observers.append(sacred.observers.QueuedMongoObserver(url='127.0.0.1:27017', db_name='db', username='user', password='password'))


def get_open_port(lo=8000, hi=8100):
    for i in range(lo, hi):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', i)) != 0:
                return i


def train_thread(args, tpu, id, q):
    print('starting training on', tpu)

    # pass binary flags through
    opts = ''
    for flag in ['auto_layout', 'auto_layout_and_mesh_shape', 'new', 'test', 'predict', 'eval', ]:
        if args.__getattribute__(flag):
            opts += ' --' + flag

    for flag in ['autostack', ]:
        if not args.__getattribute__(flag):
            opts += ' --' + flag

    cmd = "python3 main.py --tpu {tpu} --model run_configs/config_{id}.json --steps_per_checkpoint {steps_per_checkpoint} {opts} --sacred_id {run_id}".format(tpu=tpu, id=id, steps_per_checkpoint=args.steps_per_checkpoint, opts=opts, run_id=id)
    print('Running:', cmd)
    proc = subprocess.Popen(cmd, shell=True)

    # poll until it's exited
    while proc.poll() is None:
        time.sleep(60)
        try:
            nq, *nargs = q.get_nowait()
            if nq == 'kill':
                print('train thread recieved kill signal from logging thread')
                # first send SIGTERM
                proc.terminate()

                time.sleep(60)
                
                # if it still hasn't exited, we send SIGKILL
                if proc.poll() is None: 
                    print('SIGTERM not successful, sending SIGKILL')
                    proc.kill()

        except queue.Empty:
            pass

    print('exited training!')
    if proc.returncode == 0:
        print('exited gracefully')
        os.kill(os.getpid(), signal.SIGINT)
        return
    
    if args.no_delete_tpu:
        print('recreate done, exiting train_thread - not killing tpu!')
        return
    print("Recreating {} in 60sec...".format(tpu))
    time.sleep(60)
    os.system("pu recreate {} --yes --retry 3600 --retry-randomness 1.5".format(tpu))
    print('recreate done, exiting train_thread')
    
    # clear out queue
    while True:
        try:
            q.get_nowait()
            print('dropped request in queue after pu recreate')
        except queue.Empty:
            break


def get_json(uri, params=None, timeout=15):
    resp = requests.get(uri, params=params, timeout=timeout)
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
        if 'eval' in runs:
            if 'loss' in tag_sets['eval']:
                r['val_loss'] = get_scalar_data(base_uri, 'eval', 'loss')
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
    print('experiment main invoked with argv:', " ".join(sys.argv))
    print('WARNING: please remember to remove old metric log files from the model directory.')

    os.makedirs('run_configs', exist_ok=True)
    shutil.copy(args.model if args.model.endswith('.json') else 'configs/{}.json'.format(args.model), 'run_configs/config_{}.json'.format(_run._id))

    tensorboard_port = get_open_port()
    print('Tensorboard at port:', tensorboard_port)
    print('Tensorboard url: ', 'http://eleutherai.bmk.sh:'+ str(tensorboard_port))
    os.system("screen -S tensorboard_{} -d -m bash -c 'tensorboard --logdir {} --port {} --bind_all --reload_multifile=true || tensorboard --logdir {} --port {} --reload_multifile=true'".format(_run._id, params["model_path"], tensorboard_port,params["model_path"], tensorboard_port,))
    atexit.register(goodbye, _run._id)

    curr_step = {}
    seen_predictions = set()

    heartbeat_timeout = args.initial_heartbeat_timeout * 2
    while True:
        last_tb_log_time = time.time()
        start_time = time.time()
        q = queue.Queue()
        trainthd = threading.Thread(target=train_thread, args=(args, args.tpu, _run._id, q))
        trainthd.start()

        while trainthd.is_alive():
            time.sleep(60)

            if start_time + args.initial_heartbeat_timeout < time.time():
                # after initial args.initial_heartbeat_timeout grace period, now we want to set the timeout threshold much lower
                heartbeat_timeout = args.heartbeat_timeout

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
                    
                    # found something new, so logging!
                    last_tb_log_time = time.time()

                    curr_step[k] = step

            for f in glob.glob('predictions_{}_*'.format(_run._id)):
                if f in seen_predictions:
                    continue
                print('collecting prediction file', f)
                ex.add_artifact(f)
                
                seen_predictions.add(f)
            
            # collect eval metrics from jsonl
            if os.path.exists(f'eval_{_run._id}.jsonl'):
                with open(f'eval_{_run._id}.jsonl') as fh:
                    for line in fh:
                        ob = json.loads(line)
                        val_step = ob['global_step']
                        val_task = ob['task']
                        for metr in ob.keys():
                            k = 'fs.' + val_task + '.' + metr
                            if metr in ['task', 'global_step']: continue
                            if val_step <= curr_step.get(k, -1): continue
                            _run.log_scalar(k, ob[metr], val_step)
                            curr_step[k] = val_step

            if time.time() - last_tb_log_time > heartbeat_timeout:
                # the run hasn't logged in a while, so we restart it
                q.put(('kill',))

                # give training thread some time to do its thing and recreate tpu
                while trainthd.is_alive():
                    print('logging thread waiting for killing stalled run and for tpu recreate to finish')
                    time.sleep(60)
                
                # reset heartbeat timeout to initial
                heartbeat_timeout = args.initial_heartbeat_timeout
                last_tb_log_time = time.time()


        if args.no_delete_tpu:
            break


def goodbye(id):
    print("You are now leaving the Python sector.")
    print("Sie verlassen den pythonischen Sektor.")

    os.system("screen -S tensorboard_{} -X quit".format(id))

        
if __name__ == '__main__':
    for file in glob.glob("**/*", recursive=True):
        if file.split('.')[-1] in ['py']:
            print('Adding', file, 'to sacred')
            ex.add_source_file(file)

    ex.add_config({
        'tpu_name': args.tpu,
        **params
    })

    ex.run()

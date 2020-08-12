import sacred
import argparse
import main
import time
import subprocess
import shutil
import os
import json 

ex = sacred.Experiment('eleutherai-gpt3')
#ex.observers.append(sacred.observers.QueuedMongoObserver(url='127.0.0.1:27017', db_name='db', username='user', password='password'))


@ex.main
def main(_run):
    print('Starting run', _run._id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str) # JSON file that contains model parameters
    args = parser.parse_args()

    ex.add_config({
        'tpu_name': args.tpu,
        **json.load(open(args.model))
    })

    os.makedirs('run_configs', exist_ok=True)
    shutil.copy(args.model, 'run_configs/config_{}.json'.format(_run._id))

    while True:
        os.system("python3 main.py --tpu {tpu} --model run_configs/config_{id}.json".format(tpu=args.tpu, id=_run._id))
        print("Recreating {} in 60sec...".format(args.tpu))
        time.sleep(60)



if __name__ == '__main__':
    ex.run()
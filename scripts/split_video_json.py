import argparse
import json
import os

from scripts.video2tfrecord import split_equal

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('load_path', type=str,
                        help='The path to a json file containing video information, or a path to a folder containing '
                             'json files with video information.')
    parser.add_argument('split', type=int, help='The number of equal splits.')
    parser.add_argument('-prefix', type=str, default='', help='A save file prerfix.')

    args = parser.parse_args()

    load_path = args.load_path
    split = args.split
    prefix = args.prefix

    if os.listdir(load_path):
        load_path = [os.path.join(load_path, p) for p in os.listdir(load_path)]
    else:
        load_path = [load_path]

    ids = []
    duration = []

    for l in load_path:
        json_load = json.load(open(l))

        ids = ids + json_load['id']
        duration = duration + json_load['duration']

    ids, duration = split_equal(ids, duration, split, -1)

    for idx, (i, d) in enumerate(zip(ids, duration)):

        path = os.path.join(prefix, "work_split_{}.json".format(idx))
        dump = {'id': i, 'duration': d}

        json.dump(dump, open(path, 'w'))
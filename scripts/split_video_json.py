import numpy as np
import argparse
import json
import os


# Function copy from video2tfrecords.py
def split_equal(ids: list, duration: list, num: int, min_duration: int = 256):

    sort = sorted(zip(duration, ids))[::-1]

    ids_split = [[] for i in range(num)]
    duration_spit = [[] for i in range(num)]
    duration_sum = [0] * num

    for d, i in sort:
        if d > min_duration or min_duration <= 0:
            pos = np.argmin(duration_sum)

            ids_split[pos].append(i)
            duration_spit[pos].append(d)
            duration_sum[pos] = duration_sum[pos] + d

    return ids_split, duration_spit


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

    split_video_count = 0
    split_video_duration = 0

    for i in range(len(ids)):
        buffer_video_count = len(ids[i])
        buffer_video_duration = sum(duration[i])

        print('split:', i, 'videos:', buffer_video_count, 'duration:', buffer_video_duration)

        split_video_count += buffer_video_count
        split_video_duration += buffer_video_duration

    print('')
    print('total num of videos:', split_video_count, 'total video duration:', split_video_duration)

    for idx, (i, d) in enumerate(zip(ids, duration)):

        path = "{}work_split_{}.json".format(prefix, idx)
        dump = {'id': i, 'duration': d}

        json.dump(dump, open(path, 'w'))
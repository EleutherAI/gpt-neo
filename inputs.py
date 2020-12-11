import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from data.encoders import encode
import os
import random
import requests
from data.video2tfrecord import frame_decoder
from google.cloud import storage

def tf_record_dataset(name, sequence_length, time_delay, deterministic):
    data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor([name]), buffer_size=2**20, num_parallel_reads=1)
    data = data.map(frame_decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
    data = data.window(size=sequence_length + time_delay, stride=1, shift=sequence_length, drop_remainder=True)
    data = data.interleave(lambda x: x.batch(sequence_length + time_delay, drop_remainder=True),
                           cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           block_length=1)
    data = data.repeat()

    return data

def generic_data(params, eval=False):
    sequence_length = params['n_ctx']
    buffer_size = params.get('buffer_size', 1)
    frame_height = params.get('frame_height', 176)
    frame_width = params.get('frame_width', 320)
    bucket_name = params.get('bucket_name', 'text-datasets')
    time_patch = params.get('time_patch', 1)
    color_channels = params.get('color_channels', 3)
    deterministic = params.get('deterministic', False)
    batch_size = params['eval_batch_size' if eval else 'train_batch_size']

    data = tf.data.Dataset.from_tensor_slices([f'gs://{bucket_name}/{itm.name}' for itm in storage.client.Client().list_blobs(bucket_name, prefix='datasets/video')])
    data = data.interleave(lambda x: tf_record_dataset(x, sequence_length, time_patch, deterministic), 
                           cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           block_length=1)
    data = data.shuffle(buffer_size)
    data = data.batch(batch_size)

    def prepare(x):
        # Target Shape: [batch_size, sequence_length, frame_height, frame_width, color_channels]
        x = tf.reshape(x, (batch_size, sequence_length + time_patch, frame_height, frame_width, color_channels))
        x = tf.cast(x, tf.float32)
        x = x / 255.

        vals1 = x[:, :sequence_length]
        vals2 = x[:, time_patch:sequence_length + time_patch]

        vals1 = tf.reshape(vals1, (batch_size, sequence_length, frame_height, frame_width, color_channels))
        vals2 = tf.reshape(vals2, (batch_size, sequence_length, frame_height, frame_width, color_channels))

        vals1 = tf.cast(vals1, dtype=tf.float32)
        vals2 = tf.cast(vals2, dtype=tf.float32)
        return vals1, vals2

    data = data.map(prepare, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()

    return data



if __name__ == '__main__':
    test_data = ['/tfrecord/bDCLRy2qbgo.tfrecord', '/tfrecord/Q0eeWvE8mfs.tfrecord', '/tfrecord/pXYHHcwhWrQ.tfrecord',
                 '/tfrecord/pUH42iBHksQ.tfrecord', '/tfrecord/pCUPKkaWlVg.tfrecord', '/tfrecord/xXNsR9X1XQk.tfrecord',
                 '/tfrecord/J9pMGE0z-fc.tfrecord', '/tfrecord/On60TZ98eb4.tfrecord', '/tfrecord/KtvAVH4AhY8.tfrecord']

    dataset = generic_data({'n_ctx': 256, 'train_batch_size': 32})

    iterator = dataset.make_one_shot_iterator()
    #iterator = dataset.make_initializable_iterator()
    next_frame_data = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(iterator.initializer)

        while True:
            frame_data_1, frame_data_2 = sess.run(next_frame_data)
            print(frame_data_1.shape, frame_data_2.shape)

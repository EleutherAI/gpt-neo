import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from data.encoders import encode
import os
import requests
from data.video2tfrecord import frame_decoder
from google.cloud import storage

def tf_record_dataset(name, sequence_length):
    return (tf.data.TFRecordDataset(filenames=tf.convert_to_tensor([f'gs://{name}']),
                                    buffer_size=1,
                                    num_parallel_reads=1).map(frame_decoder)
                                                         .window(size=sequence_length + 1, stride=1, shift=sequence_length, drop_remainder=True)
                                                         .flat_map(lambda x: x.batch(sequence_length + 1)))

def generic_data(params, eval=False):
    sequence_length = params['n_ctx']
    shards = params.get('shards', 1)
    buffer_size = params.get('buffer', 1)
    frame_height = params.get('frame_height', 176)
    frame_width = params.get('frame_width', 320)
    color_channels = params.get('color_channels', 3)
    batch_size = params['eval_batch_size' if eval else 'train_batch_size']

    @tf.function
    def prepare(x):
        # input tensor (batch_size, sequence_length, frame_height, frame_width, color_channels)
#        x = tf.reshape(x, (batch_size, sequence_length + 1, frame_height, frame_width, color_channels))
#        x = tf.cast(x, tf.float32)
#        x = x / 255.
#        print(x.shape)

        vals1 = x[:, :sequence_length]
        vals2 = x[:, 1:sequence_length + 1]

        vals1 = tf.reshape(vals1, (batch_size, sequence_length, frame_height, frame_width, color_channels))
        vals2 = tf.reshape(vals2, (batch_size, sequence_length, frame_height, frame_width, color_channels))

        vals1 = tf.cast(vals1, dtype=tf.float32)
        vals2 = tf.cast(vals2, dtype=tf.float32)
        return vals1, vals2


    data = [tf_record_dataset(itm.name, sequence_length) for itm in storage.client.Client().list_blobs('text-datasets', prefix='datasets/video')]
    dataset = tf.data.experimental.sample_from_datasets(data).shuffle(buffer_size).batch(batch_size, drop_remainder=True).map(prepare)
    return dataset



if __name__ == '__main__':
    dataset = generic_data({'n_ctx': 256, 'train_batch_size': 32})

    iterator = dataset.make_one_shot_iterator()
    next_frame_data = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        frame_data_1, frame_data_2 = sess.run(next_frame_data)
        print(frame_data_1.shape, frame_data_2.shape)

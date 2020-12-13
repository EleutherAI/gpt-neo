import tensorflow.compat.v1 as tf
from google.cloud import storage

from .dataclass import ModelParameter
from .video2tfrecord import frame_decoder


def tf_record_dataset(name: tf.Tensor, sequence_length: int, time_delay: int):
    data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor([name]), buffer_size=2 ** 26, num_parallel_reads=1)
    data = data.map(frame_decoder, num_parallel_calls=1)
    data = data.repeat()

    data = data.window(size=sequence_length + time_delay, stride=1, shift=sequence_length, drop_remainder=True)

    data = data.interleave(lambda x: x.batch(sequence_length + time_delay, drop_remainder=True),
                           cycle_length=1,
                           num_parallel_calls=1,
                           block_length=1)

    data = data.repeat()

    return data


def generic_data(params: ModelParameter, eval: bool = False):
    sequence_length = params.n_ctx
    buffer_size = params.buffer_size
    frame_height = params.frame_height
    frame_width = params.frame_width
    bucket_name = params.bucket_name
    time_patch = params.time_patch
    color_channels = params.color_channels
    patch_size = params.patch_size
    batch_size = params.eval_batch_size if eval else params.train_batch_size
    prefix = params.prefix

    time_patch_size = sequence_length // time_patch
    frame_height_patch = frame_height // patch_size
    frame_width_patch = frame_width // patch_size
    channel_color_size = color_channels * time_patch * patch_size ** 2

    path = [f'gs://{bucket_name}/{itm.name}' for itm in storage.client.Client().list_blobs(bucket_name, prefix=prefix)]

    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.interleave(lambda x: tf_record_dataset(x, sequence_length, time_patch),
                           cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           block_length=1)

    data = data.batch(batch_size)

    def prepare(x: tf.Tensor):
        # Target Shape: [batch_size, sequence_length, frame_height, frame_width, color_channels]
        x = tf.reshape(x, (batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch, channel_color_size))
        x = tf.cast(x, tf.float32)
        x = x / 255.

        src = x[:, :time_patch_size]
        tgt = x[:, 1:time_patch_size + 1]

        src += tf.reshape(tf.range(time_patch_size, dtype=tf.float32) / (time_patch_size * 2.),
                          (1, time_patch_size, 1, 1, 1))
        src += tf.reshape(tf.range(frame_height_patch, dtype=tf.float32) / (frame_height_patch * 2.),
                          (1, 1, frame_height_patch, 1, 1))
        src += tf.reshape(tf.range(frame_width_patch, dtype=tf.float32) / (frame_width_patch * 2.),
                          (1, 1, 1, frame_width_patch, 1))
        src -= 1.5

        return src, tgt

    data = data.map(prepare, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.repeat()

    if buffer_size > 0:
        print(f"Buffering {buffer_size} elements")
        data = data.prefetch(buffer_size)

    return data


if __name__ == '__main__':
    test_data = ['/tfrecord/bDCLRy2qbgo.tfrecord', '/tfrecord/Q0eeWvE8mfs.tfrecord', '/tfrecord/pXYHHcwhWrQ.tfrecord',
                 '/tfrecord/pUH42iBHksQ.tfrecord', '/tfrecord/pCUPKkaWlVg.tfrecord', '/tfrecord/xXNsR9X1XQk.tfrecord',
                 '/tfrecord/J9pMGE0z-fc.tfrecord', '/tfrecord/On60TZ98eb4.tfrecord', '/tfrecord/KtvAVH4AhY8.tfrecord']

    dataset = generic_data({'n_ctx': 256, 'train_batch_size': 32})

    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    next_frame_data = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(iterator.initializer)

        while True:
            frame_data_1, frame_data_2 = sess.run(next_frame_data)
            print(frame_data_1.shape, frame_data_2.shape)

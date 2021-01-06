import random

import tensorflow.compat.v1 as tf
from google.cloud import storage

from .dataclass import ModelParameter
from scripts.video2tfrecord import get_decoder


def tf_record_dataset(name: tf.Tensor, sequence_length: int, time_delay: int,
                      frame_decoder: object, interleave_func: object):
    data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor([name]), buffer_size=2 ** 26, num_parallel_reads=1)
    data = data.map(frame_decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data = data.window(size=sequence_length + time_delay, stride=1, shift=sequence_length, drop_remainder=True)
    data = data.interleave(interleave_func, cycle_length=1, num_parallel_calls=1, block_length=1)

    return data


def generic_data(params: ModelParameter):
    params = ModelParameter(params)
    sequence_length = params.n_ctx
    buffer_size = params.buffer_size
    three_axes = params.three_axes
    frame_height = params.frame_height
    frame_width = params.frame_width
    bucket_name = params.bucket_name
    time_patch = params.time_patch
    color_channels = params.color_channels
    patch_size = params.patch_size
    batch_size = params.train_batch_size
    language_token_per_frame = params.language_token_per_frame
    prefix = params.prefix

    assert not (language_token_per_frame > 0 and time_patch > 1), \
        ("Time patch and language token are currently not supported together")

    if params.use_video:
        frame_decoder = get_decoder(language_token_num_per_frame=language_token_per_frame,
                                    frame_height=frame_height, frame_width=frame_width, color_channels=color_channels)
    else:
        frame_decoder = lambda x: tf.sparse.to_dense(
                tf.parse_single_example(x, {"text": tf.VarLenFeature(tf.int64)})['text'])

    time_patch_size = params.time_patch_size
    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch
    channel_color_size = params.channel_color_size

    if params.use_video and params.language_token_per_frame > 0:
        interleave_func = lambda x, y, z: tf.data.Dataset.zip((x, y, z)) \
            .batch(sequence_length + time_patch, drop_remainder=True)
    elif params.use_video:
        interleave_func = lambda x: x.batch(sequence_length + time_patch, drop_remainder=True)
    else:
        interleave_func = lambda x: x.batch(sequence_length // 1024, drop_remainder=True)

    path = [f'gs://{bucket_name}/{itm.name}'
            for folder in prefix
            for itm in storage.client.Client().list_blobs(bucket_name, prefix=folder)]
    random.shuffle(path)


    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.apply(tf.data.experimental.parallel_interleave(lambda x: tf_record_dataset(x, sequence_length, time_patch, frame_decoder, interleave_func),
                           cycle_length=params.interleaved_datasets,
                           block_length=1, sloppy=True))

    data = data.repeat()
    data = data.shuffle(params.shuffle_buffer)
    data = data.batch(batch_size)

    def with_token(*args):
        print(args)
        if params.use_video:
            frame, *args = args
        if params.use_language:
            token, *args = args
        token_x, token_y, out_frame = (None, None, None)

        if params.use_language:
            token = tf.reshape(token,
                               (batch_size,
                                sequence_length + time_patch * params.use_video + (sequence_length // 1024) * (1 - params.use_video)) +
                               ((language_token_per_frame,) if params.use_video else tuple()))
            print(token)
            token = tf.cast(token, tf.int64)

            token_x = token[:, :sequence_length]
            token_y = token[:, 1:sequence_length + 1]

        if params.use_video:
            # Target Shape: [batch_size, sequence_length, frame_height, frame_width, color_channels]
            frame = tf.reshape(frame, (
                    batch_size, time_patch_size + 1, time_patch, frame_height_patch, patch_size, frame_width_patch,
                    patch_size,
                    color_channels))
            frame = tf.transpose(frame, [0, 1, 3, 5, 2, 4, 6, 7])

            if three_axes:
                out_frame = tf.reshape(frame, (batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                                               channel_color_size))
            else:
                out_frame = tf.reshape(frame, (batch_size, time_patch_size + 1, frame_height_patch * frame_width_patch,
                                               channel_color_size))

        return {k: v for k, v in {'frame': out_frame, 'token_x': token_x, 'token_y': token_y}.items() if v is not None}

    data = data.map(with_token, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if buffer_size > 0:
        print(f"Buffering {buffer_size} elements")
        data = data.prefetch(buffer_size)

    if params.use_video:
        def memory_op(x):
            x['frame'] = tf.cast(x['frame'], tf.float32)
            return x

        data = data.map(memory_op)

    return data

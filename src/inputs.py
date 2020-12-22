import tensorflow.compat.v1 as tf
from google.cloud import storage

from .dataclass import ModelParameter
from .video2tfrecord import get_decoder


def tf_record_dataset(name: tf.Tensor, sequence_length: int, time_delay: int, frame_decoder: object):
    data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor([name]), buffer_size=2 ** 26, num_parallel_reads=1)
    data = data.repeat()

    data = data.window(size=sequence_length + time_delay, stride=1, shift=sequence_length, drop_remainder=True)
    data = data.interleave(lambda x: x.batch(sequence_length + time_delay, drop_remainder=True),
                           cycle_length=1,
                           num_parallel_calls=1,
                           block_length=1)

    data = data.repeat()

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
    batch_size = params.batch_size
    language_token_per_frame = params.language_token_per_frame
    prefix = params.prefix

    assert not (language_token_per_frame > 0 and time_patch > 1),\
        ("Time patch and language token are currently not supported together")

    frame_decoder = get_decoder(language_token_num_per_frame=language_token_per_frame,
                                frame_height=frame_height, frame_width=frame_width, color_channels=color_channels)

    time_patch_size = sequence_length // time_patch
    frame_height_patch = frame_height // patch_size
    frame_width_patch = frame_width // patch_size
    channel_color_size = color_channels * time_patch * patch_size ** 2
    batch_size = params.eval_batch_size if params.eval else params.train_batch_size

    params.time_patch_size = time_patch_size
    params.frame_height_patch = frame_height_patch
    params.frame_width_patch = frame_width_patch
    params.channel_color_size = channel_color_size

    if not three_axes:
        frame_height_patch = frame_height_patch * frame_width_patch

    path = [f'gs://{bucket_name}/{itm.name}' for itm in storage.client.Client().list_blobs(bucket_name, prefix=prefix)]

    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.interleave(lambda x: tf_record_dataset(x, sequence_length, time_patch, frame_decoder),
                           cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           block_length=1)

    data = data.map(frame_decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size)

    def frame_cpu(frame: tf.Tensor):
        # Target Shape: [batch_size, sequence_length, frame_height, frame_width, color_channels]
        # TODO: use tf.gather
        frame = tf.reshape(frame, (batch_size, sequence_length + time_patch, frame_height, frame_width, color_channels))

        if time_patch > 1:
            frame = [tf.concat([frame[:, j] for j in range(i, i + time_patch)], axis=-1)
                 for i in range(0, sequence_length + time_patch, time_patch)]
            frame = tf.stack(frame, axis=1)

        frame = [tf.reshape(frame[:, :, i:i + patch_size, j:j + patch_size],
                        [batch_size, time_patch_size + 1, channel_color_size])
             for i in range(0, frame_height, patch_size) for j in range(0, frame_width, patch_size)]

        frame = tf.stack(frame, axis=2)
        if three_axes:
            frame = tf.reshape(frame, (batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                               channel_color_size))

        return {'frame': frame}

    def with_token(frame: tf.Tensor, token: tf.Tensor, skip: tf.Tensor):

        token = tf.reshape(token, (batch_size, sequence_length + time_patch, language_token_per_frame))
        token = tf.cast(token, tf.int64)

        token_x = token[:, :sequence_length]
        token_y = token[:, 1:sequence_length + 1]

        frame = frame_cpu(frame)

        return frame.update({'token_x': token_x, 'token_y': token_y})

    def memory_op(x):
        x['frame'] = tf.cast(['frame'], tf.float32)
        return x



    if language_token_per_frame > 0:
        data = data.map(with_token, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        data = data.map(frame_cpu, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if buffer_size > 0:
        print(f"Buffering {buffer_size} elements")
        data = data.prefetch(buffer_size)

    data = data.map(memory_op)

    return data

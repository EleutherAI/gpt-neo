import tensorflow.compat.v1 as tf
from google.cloud import storage
import numpy as np

from .dataclass import ModelParameter
from .video2tfrecord import get_decoder


def linear_pos(n):
    a = np.arange(n)
    a = a / (n - 1)

    return a


def tf_record_dataset(name: tf.Tensor, sequence_length: int, time_delay: int, frame_decoder: object):
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


def embedding_pos(_1d_size: int, _2d_size: int, _3d_size: int, _4d_size: int, _3d_attention: bool = False):

    time_embedding = np.tile(linear_pos(_2d_size), (_1d_size, 1))
    time_embedding = np.stack([time_embedding for o in range(_3d_size)], axis=-1)

    height_embedding = np.tile(linear_pos(_3d_size), (_1d_size, _2d_size, 1))

    embedding = []

    if _3d_attention:
        width_embedding = np.tile(linear_pos(_4d_size), (_1d_size, _2d_size, _3d_size, 1))
        width_embedding = np.expand_dims(width_embedding, axis=-1)
        embedding.append(width_embedding)

        time_embedding = np.stack([time_embedding for o in range(_4d_size)], axis=-1)
        height_embedding = np.stack([height_embedding for o in range(_4d_size)], axis=-1)

    time_embedding = np.expand_dims(time_embedding, axis=-1)
    height_embedding = np.expand_dims(height_embedding, axis=-1)

    embedding.append(time_embedding)
    embedding.append(height_embedding)

    embedding = np.concatenate(embedding, axis=-1)

    return embedding


def generic_data(params: ModelParameter, eval: bool = False):
    params = ModelParameter(params)
    sequence_length = params.n_ctx
    buffer_size = params.buffer_size
    _3d_attention = params._3d_attention
    frame_height = params.frame_height
    frame_width = params.frame_width
    bucket_name = params.bucket_name
    time_patch = params.time_patch
    color_channels = params.color_channels
    patch_size = params.patch_size
    batch_size = params.eval_batch_size if eval else params.train_batch_size
    prefix = params.prefix
    concat_linea_encode = params.concat_linea_encode

    frame_decoder = get_decoder(language_token_num_per_frame=params.language_token_per_frame)

    time_patch_size = sequence_length // time_patch
    frame_height_patch = frame_height // patch_size
    frame_width_patch = frame_width // patch_size
    channel_color_size = color_channels * time_patch * patch_size ** 2

    if not _3d_attention:
        frame_height_patch = frame_height_patch * frame_width_patch

    if concat_linea_encode:
        embedding = embedding_pos(batch_size, time_patch_size, frame_height_patch, frame_width_patch, _3d_attention)
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)

    path = [f'gs://{bucket_name}/{itm.name}' for itm in storage.client.Client().list_blobs(bucket_name, prefix=prefix)]

    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.interleave(lambda x: tf_record_dataset(x, sequence_length, time_patch, frame_decoder),
                           cycle_length=tf.data.experimental.AUTOTUNE,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           block_length=1)

    data = data.batch(batch_size)

    def prepare_cpu(x: tf.Tensor):

        # Target Shape: [batch_size, sequence_length, frame_height, frame_width, color_channels]
        x = tf.reshape(x, (batch_size, sequence_length + time_patch, frame_height, frame_width, color_channels))

        if time_patch > 1:
            x = [tf.concat([x[:, j] for j in range(i, i+time_patch)], axis=-1)
                 for i in range(0, sequence_length + time_patch, time_patch)]
            x = tf.stack(x, axis=1)

        x = [tf.reshape(x[:, :, i:i + patch_size, j:j + patch_size], [batch_size, time_patch_size + 1,
                                                                      channel_color_size])
             for i in range(0, frame_height, patch_size) for j in range(0, frame_width, patch_size)]

        x = tf.stack(x, axis=2)
        if _3d_attention:
            x = tf.reshape(x, (batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                               channel_color_size))

        src = x[:, :time_patch_size]
        tgt = x[:, 1:time_patch_size + 1]

        return src, tgt

    def prepare_mem(src: tf.Tensor, tgt: tf.Tensor):

        src = tf.cast(src, tf.float32) / 255.
        tgt = tf.cast(tgt, tf.float32) / 255.

        if concat_linea_encode:
            src = tf.concat([src, embedding], axis=-1)
        else:
            src += tf.reshape(tf.range(time_patch_size, dtype=tf.float32) / (time_patch_size - 1),
                              (1, time_patch_size, 1, 1, 1))
            src += tf.reshape(tf.range(frame_height_patch, dtype=tf.float32) / (frame_height_patch - 1),
                              (1, 1, frame_height_patch, 1, 1))
            src += tf.reshape(tf.range(frame_width_patch, dtype=tf.float32) / (frame_width_patch - 1),
                              (1, 1, 1, frame_width_patch, 1))

        return src, tgt

    data = data.map(prepare_cpu, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if buffer_size > 0:
        print(f"Buffering {buffer_size} elements")
        data = data.prefetch(buffer_size)

    data = data.map(prepare_mem)

    return data

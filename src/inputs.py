"""
Contains input pipeline code that generates tensorflow datasets if called
"""
import logging
import random
import re
from itertools import cycle

import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter


def get_video_decoder(language_token_num_per_frame=0, frame_height=None, frame_width=None, color_channels=None):
    '''
    :param language_token_num_per_frame: The number of language tokens per single frame.
    If this is 0 (default) language tokens are disabled.
    :param frame_height:
    :param frame_width:
    :param color_channels:

    This function will return a frame decoder function, that can than be used to decode tf.records.
    '''

    decode_language_token = language_token_num_per_frame > 0
    token_range = tf.range(0, language_token_num_per_frame)

    # Decoding Key.
    features = {
            'frame': tf.FixedLenFeature([], tf.string)
            }

    if decode_language_token:
        features.update({'tokens':     tf.FixedLenFeature([language_token_num_per_frame], tf.int64),
                         'skip_frame': tf.FixedLenFeature([], tf.int64),
                         'mask':       tf.FixedLenFeature([], tf.int64)
                         })

    def frame_decoder(proto):
        '''
        :param proto: Proto buffer to be decoded.
        :return: tensor with decode frame.

        This Function will decode frame from proto buffer.
        '''

        sample = tf.parse_single_example(proto, features)
        frame = tf.image.decode_image(sample['frame'])

        if decode_language_token:
            tokens = sample['tokens']
            skip_frame = sample['skip_frame']
            mask = sample['skip_frame']

            if skip_frame > 0:
                frame = tf.zeros(shape=(frame_height, frame_width, color_channels), dtype=tf.uint8)

            b_mask = tf.less_equal(token_range, tf.cast(mask, tf.int32))

            return frame, tokens, skip_frame, b_mask

        return frame

    return tf.function(frame_decoder, experimental_compile=False)


def _text_decoder(name: tf.Tensor, ctx: int, chunk_size: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param name: protobuf object to decode
    :param ctx: context size of generated dataset
    :param chunk_size: batch size directly after creating the dataset
    :return: tensorflow dataset of token
    """

    @tf.function
    def _decode_protobuf(proto):
        text_slice = tf.parse_single_example(proto, {'text': tf.FixedLenFeature([], tf.string)})['text']
        data = tf.data.Dataset.from_tensor_slices(
                tf.reshape(tf.strings.unicode_decode(text_slice, 'UTF-8'), (-1, 1)))
        if chunk_size > 0:
            data = data.batch(chunk_size)
        data = data.window(size=ctx + 1, shift=ctx, stride=1, drop_remainder=True)
        data = data.interleave(lambda x: x.batch(ctx + 1, drop_remainder=True))
        return data

    return tf.data.TFRecordDataset(filenames=name).interleave(_decode_protobuf)


def text_decode(proto):
    x = tf.parse_single_example(proto, {'text': tf.VarLenFeature(tf.int64)})
    x = x['text']
    x = tf.sparse.to_dense(x)
    x = tf.cast(x, tf.int32)
    x = tf.data.Dataset.from_tensor_slices(x)

    return x


def dataset_text(path: str, batch_size: int, params: ModelParameter) -> tf.data.Dataset:
    """
    Creates a text dataset containing shuffled and prefetched windows.
    :param path: Path to dataset (in google cloud bucket)
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    three_axes = params.three_axes

    time_patch = params.time_patch
    token_patch_size = params.token_patch_size
    language_token_patch = params.language_token_patch
    language_token_per_frame = params.language_token_per_frame

    time_patch_size = params.time_patch_size
    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch
    channel_color_size = params.channel_color_size

    assert not (language_token_per_frame > 0 and time_patch > 1), \
        ("Time patch and language token are currently not supported together")

    padding_token = tf.constant([[[params.padding_token]] * (time_patch_size + 1)] * batch_size, dtype=tf.int32)
    padding_token = tf.data.Dataset.from_tensors(padding_token).repeat()

    if three_axes:
        padding_frame = tf.zeros((batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                                  channel_color_size), dtype=tf.uint8)
    else:
        padding_frame = tf.zeros((batch_size, time_patch_size + 1, frame_height_patch * frame_width_patch,
                                  channel_color_size), dtype=tf.uint8)

    padding_frame = tf.data.Dataset.from_tensors(padding_frame).repeat()

    padding_frame_mask = tf.zeros((batch_size, time_patch_size), dtype=tf.bool)
    padding_frame_mask = tf.data.Dataset.from_tensors(padding_frame_mask).repeat()

    padding_token_mask = tf.ones((batch_size, time_patch_size, language_token_patch, token_patch_size), dtype=tf.bool)
    padding_token_mask = tf.data.Dataset.from_tensors(padding_token_mask).repeat()

    def _memory_func(x, _padding_token, _padding_frame, _padding_frame_mask, _padding_token_mask):

        x = tf.reshape(x, (batch_size, time_patch_size + 1, language_token_per_frame - 1))
        x = tf.cast(x, tf.int32)
        x = tf.concat([x, _padding_token], axis=2)

        x = tf.reshape(x, (batch_size, time_patch_size + 1, language_token_patch, token_patch_size))

        token_x = x[:, :time_patch_size]
        token_y = x[:, 1:time_patch_size + 1]

        if three_axes:
            _padding_frame = tf.reshape(_padding_frame, (batch_size,
                                                         time_patch_size + 1,
                                                         frame_height_patch,
                                                         frame_width_patch,
                                                         channel_color_size))
        else:
            _padding_frame = tf.reshape(_padding_frame, (batch_size,
                                                         time_patch_size + 1,
                                                         frame_height_patch * frame_width_patch,
                                                         channel_color_size))

        _padding_frame_mask = tf.reshape(_padding_frame_mask, (batch_size, time_patch_size))
        _padding_token_mask = tf.reshape(_padding_token_mask,
                                         (batch_size, time_patch_size, language_token_patch, token_patch_size))

        return {'token_x': token_x, 'token_y': token_y, 'frame': _padding_frame,
                'vid_msk': _padding_frame_mask, 'tkn_msk': _padding_token_mask}

    path = tf.io.gfile.glob(path)
    random.seed(params.data_seed)
    random.shuffle(path)

    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.repeat()

    data = data.map(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data = data.flat_map(lambda x: x.flat_map(text_decode))
    data = data.window(size=(time_patch_size + 1) * (language_token_per_frame - 1),
                       shift=time_patch_size * (language_token_per_frame - 1), stride=1, drop_remainder=True)
    data = data.flat_map(lambda x: x.batch((time_patch_size + 1) * (language_token_per_frame - 1)))

    data = data.shuffle(16384, seed=params.data_seed)

    data = data.batch(batch_size)

    data = tf.data.Dataset.zip((data, padding_token, padding_frame, padding_frame_mask, padding_token_mask))
    data = data.map(_memory_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return data


def dataset_video(path: str, batch_size: int, params: ModelParameter):
    """
    Creates a video dataset containing shuffled and prefetched windows.
    :param path: Path to dataset (in google cloud bucket)
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    three_axes = params.three_axes
    frame_height = params.frame_height
    frame_width = params.frame_width

    time_patch = params.time_patch
    color_channels = params.color_channels
    patch_size = params.patch_size
    n_ctx = params.n_ctx
    token_patch_size = params.token_patch_size
    language_token_patch = params.language_token_patch
    language_token_per_frame = params.language_token_per_frame

    time_patch_size = params.time_patch_size
    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch
    channel_color_size = params.channel_color_size

    assert not (language_token_per_frame > 0 and time_patch > 1), \
        ("Time patch and language token are currently not supported together")

    def _decode_func(name: tf.Tensor):

        data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor(name), buffer_size=2 ** 26, num_parallel_reads=1)
        data = data.map(frame_decoder, num_parallel_calls=1)

        data = data.window(size=n_ctx + time_patch, stride=1, shift=n_ctx, drop_remainder=True)
        data = data.interleave(interleave_func, cycle_length=1, num_parallel_calls=1, block_length=1)

        return data

    def _pre_func(*args):

        token_x, token_y, out_frame, frame_mask, token_mask = (None, None, None, None, None)

        frame, *args = args

        if params.use_language:
            token, frame_mask, token_mask, *args = args

        frame = tf.reshape(frame, (batch_size, time_patch_size + 1, time_patch, frame_height_patch, patch_size,
                                   frame_width_patch, patch_size, color_channels))

        frame = tf.transpose(frame, [0, 1, 3, 5, 2, 4, 6, 7])

        if three_axes:
            out_frame = tf.reshape(frame, (batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                                           channel_color_size))
        else:
            out_frame = tf.reshape(frame, (batch_size, time_patch_size + 1, frame_height_patch * frame_width_patch,
                                           channel_color_size))

        if params.use_language:
            token = tf.reshape(token, (batch_size, time_patch_size + 1, language_token_patch, token_patch_size))
            token = tf.cast(token, tf.int32)

            token_x = token[:, :time_patch_size]
            token_y = token[:, 1:time_patch_size + 1]

            frame_mask = frame_mask[:, 1:time_patch_size + 1]
            frame_mask = tf.reshape(frame_mask, (batch_size, time_patch_size))
            frame_mask = 1 - frame_mask
            frame_mask = tf.cast(frame_mask, tf.bool)

            token_mask = token_mask[:, 1:time_patch_size + 1]
            token_mask = tf.reshape(token_mask, (batch_size, time_patch_size, language_token_patch, token_patch_size))
            token_mask = tf.cast(token_mask, tf.bool)

        return {k: v for k, v in {'frame':   out_frame / 255, 'token_x': token_x, 'token_y': token_y,
                                  'vid_msk': frame_mask, 'tkn_msk': token_mask
                                  }.items() if v is not None}

    if language_token_per_frame > 0:
        interleave_func = lambda x, y, z, a: tf.data.Dataset.zip((x, y, z, a)) \
            .batch(n_ctx + time_patch, drop_remainder=True)
    else:
        interleave_func = lambda x: x.batch(n_ctx + time_patch, drop_remainder=True)

    frame_decoder = get_video_decoder(language_token_num_per_frame=language_token_per_frame,
                                      frame_height=frame_height, frame_width=frame_width, color_channels=color_channels)

    path = tf.io.gfile.glob(path)
    random.seed(params.data_seed)
    random.shuffle(path)

    data = tf.data.Dataset.from_tensor_slices(path)
    data = data.repeat()
    data = data.apply(tf.data.experimental.parallel_interleave(lambda x: _decode_func(x),
                                                               cycle_length=params.interleaved_datasets, block_length=1,
                                                               sloppy=False))

    data = data.batch(batch_size)
    data = data.map(_pre_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data

    return data


def dataset(params: ModelParameter, step: int = 0, train: bool = True):
    """
    Creates any dataset containing shuffled and prefetched windows.
    :param params: ModelParameter
    :param step: number of items to skip of dataset
    :param train: is the model in train or sample mode
    :return: tensorflow dataset
    """

    #params = ModelParameter(params)

    def memory_op(x):
        print(x['frame'].shape)
        x['frame'] = tf.cast(tf.reshape(x['frame'], (257,220,768)), tf.float32)
        return [x[key] for key in x.keys()]

    def concat_op(*args):
        x, *args = args

        for key in x.keys():
            x[key] = tf.concat([x[key]] + [arg[key] for arg in args], axis=0)

        return x

    weights = [set['weight'] for set in params.dataset_configs]
    datasets = []
    batch_size = params.train_batch_size if train else 1

    assert batch_size % sum(weights) == 0, f"The batch size needs to be divisible by the sum of all weights. " \
                                           f"The batch size {batch_size} is not divisible by the sum of " \
                                           f"{weights} is not."

    weight_multi = batch_size // sum(weights)

    for set in params.dataset_configs:
        dtype = set['type']
        path = set['path']
        weight = set['weight']

        assert dtype == 'video' or dtype == 'text', \
            f"{dtype} is not a supported option for type for a dataset."

        if dtype == 'video':
            datasets.append(dataset_video(path, weight * weight_multi, params))
        elif dtype == 'text':
            datasets.append(dataset_text(path, weight * weight_multi, params))

    if len(datasets) > 1:
        datasets = tf.data.Dataset.zip(tuple(datasets))
        datasets = datasets.map(concat_op)
    else:
        datasets = datasets[0]

    datasets = datasets.prefetch(params.buffer_size)
    datasets = datasets.map(memory_op)

    datasets = datasets.skip(step)

    options = tf.data.Options()
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.hoist_random_uniform = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = False
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_vectorization.use_choose_fastest = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.apply_default_optimizations = False

    datasets = datasets.with_options(options)

    return datasets


def _get_number_of_documents(filename):
    # extracts number of files from a filename formatted "<name>_<num_documents>.tfrecords."
    # if no pattern is matched, returns None
    match = re.search("_(\d{1,}).tfrecords$", filename)
    return int(match.group(1)) if match is not None else match


def _get_number_of_documents_by_iteration(filename):
    # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
    # this could be very slow.
    logging.warning(
            "inputs/sequential_input() found no metadata found in filename - iterating through first tfrecord to find global length")
    count = 0
    for item in tf.io.tf_record_iterator(filename):
        count += 1
    return count


def _get_skip_index(all_files, n_batches):
    prev_cumsum = 0
    cumsum = 0
    global_n_documents = None
    for count, f in cycle(enumerate(all_files)):
        prev_cumsum = cumsum
        if _get_number_of_documents(f) is not None:
            cumsum += _get_number_of_documents(f)
        elif global_n_documents is None:
            global_n_documents = _get_number_of_documents_by_iteration(f)
            cumsum += global_n_documents
        else:
            cumsum += global_n_documents
        if cumsum == n_batches:
            remainder = 0
            skip_idx = count + 1
        elif cumsum > n_batches:
            remainder = n_batches - prev_cumsum
            skip_idx = count
            break
    return skip_idx, remainder


def gpt_neo_input(params, step=None, eval=False):
    """
    Input fn that reads tfrecords encoded with a fixed chunk size (== n_ctx + 1), and that either:

        - has the number of documents for each tfrecord file encoded in the title in the format
          <name>_<n_documents>.tfrecords.

          OR

        - has a fixed number of documents per tfrecord file.

    If the glob pattern above isn't matched, we assume that each document has the same number of samples as the first
    tfrecord read.
    If this isn't the case, it may result in errors, or some samples being missed.

    This means we can calculate the number of samples we've seen so far using the global step,
    and can use dataset.skip() to iterate through the list of filenames, as opposed to the whole dataset, which is
    incredibly inefficient.

    If training is starting and stopping often, as with TPU pre-emption, reading the whole dataset sequentially appears
    to improve model
    performance, as it results in less repeated data.
    :param params: serialized dict of ModelParameter instance
    :param step: Number of steps to skip
    :param eval: Whether this dataset is in evaluation mode
    :return: tensorflow dataset
    """

    params = ModelParameter(params)

    if not eval:
        assert step is not None

    logging.warning("Changing batch size with sequential_input() will result in some data being skipped or repeated."
                    "Please ensure your batch size stays constant throughout training.")

    batch_size = params.train_batch_size

    filenames = []

    # iterate through each dataset and read params
    for set in params.dataset_configs:
        path = set['path']

        # then glob all files that fit the pattern specified in dataset_configs
        filenames.extend(tf.io.gfile.glob(path))

    filenames = sorted(filenames)
    shuffle_filenames = params.get("shuffle_input_filenames", True)
    if shuffle_filenames:
        # shuffle deterministically
        seed = params.data_seed
        random.seed(seed)
        random.shuffle(filenames)

    # repeat filenames to infinity
    dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()

    def _memory_func(x):

        x = tf.reshape(x, (batch_size, params.n_ctx + 1, 1))
        x = tf.cast(x, tf.int32)

        vals1 = x[:, :params.n_ctx]
        vals2 = x[:, 1:params.n_ctx + 1]

        return {'token_x': vals1, 'token_y': vals2}

    dataset = dataset.interleave(lambda x: _text_decoder(x, params.n_ctx, -1))

    dataset = dataset.shuffle(512, seed=seed)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(_memory_func)
    dataset = dataset.prefetch(params.buffer_size * 2)

    options = tf.data.Options()
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.hoist_random_uniform = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = False
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_vectorization.use_choose_fastest = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.apply_default_optimizations = False

    # dataset = dataset.with_options(options)

    return dataset

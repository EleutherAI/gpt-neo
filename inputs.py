import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from data.encoders import encode
import random
import re
import logging
from itertools import cycle
from utils import natural_sort


### IN USE ###

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


def _parse_function(example_proto):
    features = {
        "text": tf.VarLenFeature(tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.sparse.to_dense(parsed_features["text"], parsed_features["text"].dense_shape[0])


def autoregressive_sample_text(params, x):
    vals1 = x[:params["n_ctx"]]
    vals2 = x[1:params["n_ctx"] + 1]

    vals1 = tf.reshape(vals1, [params["n_ctx"]])
    vals2 = tf.reshape(vals2, [params["n_ctx"]])
    vals1 = tf.cast(vals1, dtype=tf.int32)
    vals2 = tf.cast(vals2, dtype=tf.int32)
    return vals1, vals2


def sequential_input(params, global_step=None, eval=False):
    """
    Input fn that reads tfrecords encoded with a fixed chunk size (== n_ctx + 1), and that either:

        - has the number of documents for each tfrecord file encoded in the title in the format
          <name>_<n_documents>.tfrecords.

          OR

        - has a fixed number of documents per tfrecord file.

    If the glob pattern above isn't matched, we assume that each document has the same number of samples as the first tfrecord read.
    If this isn't the case, it may result in errors, or some samples being missed.

    This means we can calculate the number of samples we've seen so far using the global step,
    and can use dataset.skip() to iterate through the list of filenames, as opposed to the whole dataset, which is incredibly inefficient.

    If training is starting and stopping often, as with TPU pre-emption, reading the whole dataset sequentially appears to improve model
    performance, as it results in less repeated data.
    """
    if not eval:
        assert global_step is not None
    logging.warning(
        "Changing batch size with sequential_input() will result in some data being skipped or repeated. Please ensure your batch size stays constant throughout training.")
    batch_size = params['eval_batch_size' if eval else 'train_batch_size']

    filenames = []
    for dataset_config in params['dataset_configs'].values():  # iterate through each dataset and read params
        path_key = 'path' if not eval else 'eval_path'
        path = dataset_config[path_key]
        filenames.extend(
            tf.io.gfile.glob(path))  # then glob all files that fit the pattern specified in dataset_configs

    filenames = natural_sort(filenames)
    shuffle_filenames = params.get("shuffle_input_filenames", True)
    if shuffle_filenames:
        seed = params.get('seed', 1)  # shuffle deterministically
        random.seed(seed)
        random.shuffle(filenames)

    dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()  # repeat filenames to infinity

    if not eval:
        # skip forward first in the filenames list, then skip the remaining amount in the parsed tfrecords files
        skip_idx, remainder = _get_skip_index(filenames, n_batches=global_step * params[
            "train_batch_size"])  # TODO: fix for > 1 epoch
        dataset = dataset.skip(skip_idx)  # skip to skip idx

        # read tfrecord examples and skip remainder
        dataset = dataset.apply(tf.data.TFRecordDataset)
        dataset = dataset.skip(remainder)
    else:
        # shuffle filenames if in eval mode
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.apply(tf.data.TFRecordDataset)

    # parse the tokenized data from the tfrecord files and shuffle
    dataset = dataset.map(_parse_function, num_parallel_calls=1)
    dataset = dataset.map(partial(autoregressive_sample_text, params), num_parallel_calls=1)

    # batch data and repeat to infinity
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(params["iterations"] * 2)
    return dataset.repeat()


def pred_input(params, logger, enc=None,
               path_to_prompt=""):
    unicorns = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
               "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
               "researchers was the fact that the unicorns spoke perfect English."

    text = unicorns if path_to_prompt == "" else open(path_to_prompt, "r").read()
    tokens = encode(enc, text)

    if len(tokens) > params["n_ctx"]:
        logger.info("The length of your input prompt is longer than the model's context length - truncating input.")
        tokens = tokens[len(tokens) - params["n_ctx"]:]
    if len(tokens) < params["n_ctx"]:
        tokens = tf.pad(tokens, [[0, params["n_ctx"] - len(tokens)]], constant_values=params["padding_id"])
    t = tf.broadcast_to(tokens, [params["batch_size"], params["n_ctx"]])
    dataset = tf.data.Dataset.from_tensors(t)

    def _dummy_labels(x):
        return x, x

    dataset = dataset.map(_dummy_labels)
    return dataset


def handle_pred_output(predictions, logger, enc, params, out_name="test"):
    with tf.gfile.Open(f"{out_name}.txt", "w") as f:
        for i, p in enumerate(predictions):
            p = p["outputs"]

            # remove eos + padding ids from output
            idx = np.argmax(p == params['eos_id'])
            if idx > 0:
                p = p[:idx]
            idx = np.argmax(p == params['padding_id'])
            if idx > 0:
                p = p[:idx]

            text = enc.decode(p)
            f.write("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            f.write(text)
            f.write("\n" + "=" * 80 + "\n")

            logger.info("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            logger.info(text)
            logger.info("\n" + "=" * 80 + "\n")


### DEPRECATED ###

def generic_text(params, eval=False, sample_text_fn=None, **kwargs):
    logging.warning("DEPRECATION WARNING: generic_text will be phased out in future versions.")
    i = 0 if not eval else 1

    weights = []
    datasets = []

    for dataset in params["datasets"]:
        dataset_id, stitch, datatype, weight = dataset

        assert dataset_id in params[
            'dataset_configs'], f'Unknown dataset id {dataset_id} given. Please make sure your dataset ids contain that configuration'
        dataset_config = params['dataset_configs'][dataset_id]

        path_key = 'path' if not eval else 'eval_path'
        path = dataset_config[path_key]

        datasets.append(text_dataset(
            tf.io.gfile.glob(path),
            params,
            stitch=stitch,
            datatype=datatype,
            batch=False,
            sample_text_fn=sample_text_fn
        ))

        weights.append(weight)

    batch_size = params['eval_batch_size' if eval else 'train_batch_size']

    seed = params.get('seed', None)
    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(params["iterations"] * 2)
    return dataset


def text_dataset(files, params, stitch, datatype, batch=True, sample_text_fn=None):
    seed = params.get('seed', None)
    deterministic = seed is not None
    num_parallel_calls = 1 if deterministic else tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices(files)

    if deterministic:
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
    else:
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))

    if "documents" in datatype:
        def _parse_function(example_proto):
            features = {
                # "hash": tf.VarLenFeature(tf.string),
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"], parsed_features["text"].dense_shape[0]
    else:
        def _parse_function(example_proto):
            features = {
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"]  # Assuming the text is not sparse

    dataset = dataset.map(_parse_function, num_parallel_calls=1)

    # Subsample method
    if "documents" in datatype:
        # Since samples can be less than the correct length, and TPUs don't like variable lengths, this function stitches together enough samples
        # to have a text at least 1024 tokens long. For this to work the stitch parameter must be correctly tuned so that
        # stitch * min(characters_in_text) >= amount
        def _stitch_text(x, y):
            x = tf.sparse.to_dense(x)

            def _get_x(i):
                return tf.gather(x[i], tf.range(y[i]))

            out = _get_x(0)
            eos_id = params['eos_id']

            for i in range(1, stitch):
                out = tf.concat([out, [eos_id], _get_x(i)], axis=0)  # text1<|endoftext|>text2

            return out

        # Hack-y way to stitch together multiple texts

        dataset = dataset.shuffle(1000 * stitch, seed=seed).batch(stitch, drop_remainder=True).map(_stitch_text,
                                                                                                   num_parallel_calls=num_parallel_calls)

        # Sample 1024(+1) tokens from the stitched together text
        is_random_documents = datatype == "documents_random"
        if sample_text_fn is not None:
            _sample_text = partial(sample_text_fn, random_documents=is_random_documents)
        else:
            _sample_text = autoregressive_sample_text_random_documents if is_random_documents else autoregressive_sample_text
            _sample_text = partial(_sample_text, params)

        dataset = dataset.map(_sample_text, num_parallel_calls=num_parallel_calls)

    if batch:
        dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    dataset = dataset.repeat()

    return dataset


def autoregressive_sample_text_random_documents(params, x):
    seed = params.get('seed', None)
    s = tf.size(x)
    r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32, seed=seed)
    r1 = tf.range(r, r + params["n_ctx"])
    r2 = tf.range(r + 1, (r + 1) + params["n_ctx"])
    r1 = tf.reshape(r1, [params["n_ctx"]])  # Somehow, this makes the compiler happy
    r2 = tf.reshape(r2, [params[
                             "n_ctx"]])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
    vals1 = tf.gather(x, r1)
    vals2 = tf.gather(x, r2)

    vals1 = tf.reshape(vals1, [params["n_ctx"]])
    vals2 = tf.reshape(vals2, [params["n_ctx"]])
    vals1 = tf.cast(vals1, dtype=tf.int32)
    vals2 = tf.cast(vals2, dtype=tf.int32)
    return vals1, vals2


def mlm_sample_text(params, x, random_documents=False):
    seed = params.get('seed', None)
    ctx_len = params["n_ctx"]
    assert 'mlm_mask_id' in params, 'the key `mlm_mask_id` must be set on your config to do masked language model training, specifying the id of the reserved mask token'

    mask_id = params['mlm_mask_id']
    cls_token_id = params.get('mlm_cls_token_id', None)
    num_tokens = params.get('n_vocab', None)

    mask_ignore_ids = set(params.get('mlm_mask_ignore_ids', []))
    mask_ignore_ids.add(cls_token_id)

    mask_prob = params.get('mlm_mask_prob', 0.15)
    same_token_prob = params.get('mlm_same_token_prob', 0.10)
    random_token_prob = params.get('mlm_random_token_prob', 0.)

    seq_len = ctx_len if cls_token_id is None else (ctx_len - 1)

    if random_documents:
        s = tf.size(x)
        r = tf.random.uniform([], maxval=(s - seq_len), dtype=tf.dtypes.int32, seed=seed)
        r1 = tf.range(r, r + seq_len)
        r1 = tf.reshape(r1, [seq_len])
        features = tf.gather(x, r1)
    else:
        features = x[:seq_len]

    # add cls token id if specified by `mlm_cls_token_id`
    if cls_token_id is not None:
        features = tf.pad(features, [[1, 0]], constant_values=cls_token_id)

    features = tf.cast(features, dtype=tf.int32)
    shape = features.shape

    # determine which tokens are mask-able
    can_mask = tf.not_equal(features, 0)
    for ignore_id in mask_ignore_ids:
        can_mask &= tf.not_equal(features, ignore_id)

    # generate boolean mask for masking ids
    mask_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed), mask_prob)
    mask_mask &= can_mask

    # generate mask for actually replacing the tokens, for allowing a small number of tokens to stay the same
    replace_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed),
                           1 - same_token_prob)

    # randomly replace some tokens with random tokens before masking
    if random_token_prob > 0:
        random_token_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed),
                                    random_token_prob)
        random_tokens = tf.random.uniform(shape, minval=1, maxval=num_tokens, dtype=tf.dtypes.int32, seed=seed)

        # make sure random tokens do not include illegal token ids specified by `mlm_mask_ignore_ids`
        random_can_mask = tf.not_equal(random_tokens, 0)
        for ignore_id in mask_ignore_ids:
            random_can_mask &= tf.not_equal(random_tokens, ignore_id)

        features = tf.where(random_token_mask & random_can_mask, random_tokens, features)

    # mask the tokens
    mask_tokens = tf.ones(shape, dtype=tf.int32) * mask_id
    masked_features = tf.where(mask_mask & replace_mask, mask_tokens, features)

    # labels will be set to 0 for all non-masked tokens
    labels = tf.where(mask_mask, tf.zeros(shape, dtype=tf.int32), features)

    masked_features, labels = map(lambda t: tf.reshape(t, [ctx_len]), (masked_features, labels))
    return masked_features, labels

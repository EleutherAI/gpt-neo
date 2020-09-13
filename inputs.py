import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from tokenizers import Tokenizer
from encoders import encode

def test_generic_text(params, eval=False, **kwargs):
    batch_size = params['train_batch_size']

    def _generate():
        while True:
            length = params['n_ctx'] // 2 - 1
            bos = np.full((batch_size, 1), 1)
            eos = np.full((batch_size, 1), 2)
            pad = np.full((batch_size, 1), 3)
            src_seq = np.random.randint(4,  (params['n_vocab'] - 1), (batch_size, length))
            tgt_seq = src_seq + 1
            seq = np.concatenate([bos, src_seq, pad, tgt_seq, eos], axis=1)

            for ind in range(batch_size):
                yield seq[ind]

    def _sample_text(x):
        vals1 = x[:params["n_ctx"]]
        vals2 = x[1:params["n_ctx"] + 1]

        vals1 = tf.reshape(vals1, [params["n_ctx"]])
        vals2 = tf.reshape(vals2, [params["n_ctx"]])
        vals1 = tf.cast(vals1, dtype=tf.int32)
        vals2 = tf.cast(vals2, dtype=tf.int32)
        return vals1, vals2

    dataset = tf.data.Dataset.from_generator(_generate, output_types=tf.int64)
    dataset = dataset.map(_sample_text)
    dataset = dataset.batch(batch_size)
    return dataset

def generic_text(params, eval=False, sample_text_fn=None):
    i = 0 if not eval else 1
    print('##############################')
    print(params["datasets"])
    print('##############################')

    weights = []
    datasets = []

    for dataset in params["datasets"]:
        dataset_id, stitch, datatype, weight = dataset

        assert dataset_id in params['dataset_configs'], f'Unknown dataset id {dataset_id} given. Please make sure your dataset ids contain that configuration'
        dataset_config = params['dataset_configs'][dataset_id]

        path_key = 'path' if not eval else 'eval_path'
        path = dataset_config[path_key]

        datasets.append(text_dataset(
            tf.io.gfile.glob(path),
            params,
            stitch = stitch,
            datatype = datatype,
            batch = False,
            sample_text_fn = sample_text_fn
        ))

        weights.append(weight)

    batch_size = params['eval_batch_size' if eval else 'train_batch_size']

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(params["iterations"] * 2)

    return dataset

def text_dataset(files, params, stitch, datatype, batch=True, sample_text_fn=None):
    dataset = tf.data.Dataset.from_tensor_slices(files)
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
            eos_id = 50256 if params["n_vocab"] == 50257 else 0

            for i in range(1, stitch):
                out = tf.concat([out, [eos_id], _get_x(i)], axis=0)  # text1<|endoftext|>text2

            return out

        # Hack-y way to stitch together multiple texts
        dataset = dataset.shuffle(1000 * stitch).batch(stitch, drop_remainder=True).map(_stitch_text,
                                                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Sample 1024(+1) tokens from the stitched together text
        is_random_documents = datatype == "documents_random"
        if sample_text_fn is not None:
            _sample_text = partial(sample_text_fn, random_documents = is_random_documents)
        else:
            _sample_text = autoregressive_sample_text_random_documents if is_random_documents else autoregressive_sample_text
            _sample_text = partial(_sample_text, params)

        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    dataset = dataset.repeat()

    return dataset

def autoregressive_sample_text(params, x):
    vals1 = x[:params["n_ctx"]]
    vals2 = x[1:params["n_ctx"] + 1]

    vals1 = tf.reshape(vals1, [params["n_ctx"]])
    vals2 = tf.reshape(vals2, [params["n_ctx"]])
    vals1 = tf.cast(vals1, dtype=tf.int32)
    vals2 = tf.cast(vals2, dtype=tf.int32)
    return vals1, vals2

def autoregressive_sample_text_random_documents(params, x):
    s = tf.size(x)
    r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32)
    r1 = tf.range(r, r + params["n_ctx"])
    r2 = tf.range(r + 1, (r + 1) + params["n_ctx"])
    r1 = tf.reshape(r1, [params["n_ctx"]])  # Somehow, this makes the compiler happy
    r2 = tf.reshape(r2, [params["n_ctx"]])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
    vals1 = tf.gather(x, r1)
    vals2 = tf.gather(x, r2)

    vals1 = tf.reshape(vals1, [params["n_ctx"]])
    vals2 = tf.reshape(vals2, [params["n_ctx"]])
    vals1 = tf.cast(vals1, dtype=tf.int32)
    vals2 = tf.cast(vals2, dtype=tf.int32)
    return vals1, vals2

def mlm_sample_text(params, x, random_documents = False):
    ctx_len = params["n_ctx"]
    assert 'mlm_mask_id' in params, 'the key `mlm_mask_id` must be set on your config to do masked language model training, specifying the id of the reserved mask token'

    mask_id = params['mlm_mask_id']
    mask_prob = params.get('mlm_mask_prob', 0.15)
    same_token_prob = params.get('mlm_same_token_prob', 0.10)

    mask_ignore_ids = params.get('mlm_mask_ignore_ids', [])

    if random_documents:
        s = tf.size(x)
        r = tf.random.uniform([], maxval=s - ctx_len, dtype=tf.dtypes.int32)
        r1 = tf.range(r, r + ctx_len)
        r1 = tf.reshape(r1, [ctx_len])
        features = tf.gather(x, r1)
    else:
        features = x[:ctx_len]

    features = tf.cast(features, dtype=tf.int32)
    shape = features.shape

    # determine which tokens are mask-able
    can_mask = tf.not_equal(features, 0)
    for ignore_id in mask_ignore_ids:
        can_mask &= tf.not_equal(features, ignore_id)

    # generate boolean mask for masking ids
    mask_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32), mask_prob)
    mask_mask &= can_mask

    # generate mask for actually replacing the tokens, for allowing a small number of tokens to stay the same
    replace_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32), 1 - same_token_prob)

    # mask the tokens
    mask_tokens = tf.ones(shape, dtype=tf.int32) * mask_id
    masked_features = tf.where(mask_mask & replace_mask, features, mask_tokens)

    # labels will be set to 0 for all non-masked tokens
    labels = tf.where(not mask_mask, features, tf.zeros(shape, dtype=tf.int32))

    masked_features, labels = map(lambda t: tf.reshape(t, [ctx_len]), (masked_features, labels))
    return masked_features, labels

def pred_input(params, enc = None, text="In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
                                        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
                                        "researchers was the fact that the unicorns spoke perfect English."):
    tokens = encode(enc, text)

    if len(tokens) > params["n_ctx"]:
        tokens = tokens[:params["n_ctx"]]
    if len(tokens) < params["n_ctx"]:
        tokens = tf.pad(tokens, [[0,params["n_ctx"]-len(tokens)]])
    t = tf.broadcast_to(tokens, [params["batch_size"], params["n_ctx"]])
    dataset = tf.data.Dataset.from_tensors(t)
    def _dummy_labels(x):
        return x, x

    dataset = dataset.map(_dummy_labels)
    return dataset


def test_pred_input(params, enc = None):
    def _dummy_labels(x):
        return x, x

    length = params["n_ctx"] // 2 - 1
    remaining = params["n_ctx"] // 2
    bos = tf.constant(1, shape=[1, 1], dtype=tf.int64)
    src_seq = tf.random.uniform(shape=[1, length], minval=4, maxval=(params['n_vocab'] - 1), dtype=tf.int64)
    seq = tf.concat([bos, src_seq], axis=1)
    seq = tf.pad(seq, [[0, 0], [0, remaining]])
    dataset = tf.data.Dataset.from_tensors(seq)

    dataset = dataset.map(_dummy_labels)
    return dataset


def handle_pred_output(predictions, logger, enc, out_name="test"):
    with tf.gfile.Open(f"{out_name}.txt", "a") as f:
        for i, p in enumerate(predictions):
            p = p["outputs"]
            text = enc.decode(p)
            f.write("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            f.write(text)
            f.write("\n" + "=" * 80 + "\n")

            logger.info("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            logger.info(text)
            logger.info("\n" + "=" * 80 + "\n")


def test_handle_pred_output(predictions, logger, enc, **kwargs):
    for i, p in enumerate(predictions):
        logger.info("=" * 40 + " INPUT " + str(i) + " " + "=" * 40 + "\n")
        logger.info(p["inputs"])
        logger.info("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
        logger.info(p["outputs"])
        logger.info("\n" + "=" * 80 + "\n")

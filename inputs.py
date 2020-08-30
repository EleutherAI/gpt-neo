from typing import List

import numpy as np
import tensorflow.compat.v1 as tf
from tokenizers import Tokenizer
from encoders import encode
from tensorflow.python.platform import tf_logging as logging

from typing import List
from mesh_tensorflow import transformer

def test_generic_text(params, eval=False):
    batch_size = params.batch_size

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

def generic_text(params, eval=False):
    # params["datasets"] = [(train glob, eval_glob, stitch, ["random_sample", "sample", "chunk"] weight)]
    # , dsets=[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]
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
            batch = False)
        )

        weights.append(weight)

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    return dataset


def text_dataset(files, params, stitch, datatype, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
        if datatype == "documents_random":
            def _sample_text(x):
                s = tf.size(x)
                r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32)
                r1 = tf.range(r, r + params["n_ctx"])
                r2 = tf.range(r + 1, (r + 1) + params["n_ctx"])
                r1 = tf.reshape(r1, [params["n_ctx"]])  # Somehow, this makes the compiler happy
                r2 = tf.reshape(r2, [
                    params["n_ctx"]])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
                vals1 = tf.gather(x, r1)
                vals2 = tf.gather(x, r2)

                vals1 = tf.reshape(vals1, [params["n_ctx"]])
                vals2 = tf.reshape(vals2, [params["n_ctx"]])
                vals1 = tf.cast(vals1, dtype=tf.int32)
                vals2 = tf.cast(vals2, dtype=tf.int32)
                return vals1, vals2

        else:
            def _sample_text(x):
                vals1 = x[:params["n_ctx"]]
                vals2 = x[1:params["n_ctx"] + 1]

                vals1 = tf.reshape(vals1, [params["n_ctx"]])
                vals2 = tf.reshape(vals2, [params["n_ctx"]])
                vals1 = tf.cast(vals1, dtype=tf.int32)
                vals2 = tf.cast(vals2, dtype=tf.int32)
                return vals1, vals2

        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    dataset = dataset.repeat()

    return dataset

def read_example(example_proto, max_seq_len=1024) -> dict:
    features = {
        "id": tf.io.VarLenFeature(tf.int64),
        "content": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.VarLenFeature(tf.int64),
        "offset_start": tf.io.VarLenFeature(tf.int64),
        "offset_end": tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {
        "id": tf.cast(parsed_features['id'], tf.uint64),
        "content": parsed_features['content'],
        # WARNING: remapping from target to targets
        "targets": tf.sparse.to_dense(tf.cast(parsed_features['target'], tf.int64)),
        "offset_start": tf.sparse.to_dense(tf.cast(parsed_features['offset_start'], tf.uint64)),
        "offset_end": tf.sparse.to_dense(tf.cast(parsed_features['offset_end'], tf.uint64)),
    } 


class LanguageModelInputConfig:
    batch_size:int
    prefetch: int
    pack: bool = True 
    split: str = 'TRAIN'


class LanguageModelInput:
  """Wrapper class that acts as the input_fn to TPUEstimator."""

  def __init__(self, file_pattern:List[str]):
    logging.info('init ToyModelInput()')
    self._file_pattern = file_pattern

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.estimator.tpu.RunConfig` for details.
    batch_size = params['batch_size']
    max_seq_length = params['max_sequence_length'] 

    logging.info('call LanguageModelInput() with batch size {} and sequence length', 
                                                            batch_size, max_seq_length)
    
    filenames = tf.io.gfile.glob(self._file_pattern)
    logging.info("Found %s files matching %s" % (len(filenames), self._file_pattern))
    if not filenames:
        raise ValueError("No matching files found")
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=64 * 1024 * 1024)
    keys = ["target"] 
    EOS = 1
    PAD = 0
    def decode_example(serialized_example):
        """Return a dict of Tensors from a serialized tensorflow.Example."""
        decoded = tf.io.parse_example(
            serialized=[serialized_example],
            features={k: tf.VarLenFeature(tf.int64) for k in keys})
        decoded = {k: v.values for k, v in decoded.items()}
        # append EOS
        decoded = {k: tf.concat([v, [EOS]], 0) for k, v in decoded.items()}
        return decoded

    ds = ds.map(decode_example,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # pack the dataset 
    ds = transformer.datasets.pack_or_pad(
            ds,
            sequence_length=max_seq_length,
            pack=params['pack'],
            feature_keys=None, 
            ensure_eos=False)
    # ds is  
    if params['split'] == 'TRAIN':
        if params['repeat']:
            ds = ds.repeat()
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.batch(batch_size)
    if params['prefetch']:
        ds = ds.prefetch(params['prefetch'])
    return ds


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

    length = params["n_ctx"] // 2 - 2
    remaining = params["n_ctx"] // 2
    bos = tf.constant(1, shape=[1, 1], dtype=tf.int64)
    pad = tf.constant(3, shape=[1, 1], dtype=tf.int64)
    src_seq = tf.random.uniform(shape=[1, length], minval=4, maxval=(params['n_vocab'] - 1), dtype=tf.int64)
    seq = tf.concat([bos, src_seq, pad], axis=1)
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

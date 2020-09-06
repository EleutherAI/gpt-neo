import numpy as np
import tensorflow.compat.v1 as tf
from tokenizers import Tokenizer
from encoders import encode

def test_generic_text(params, eval=False):
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

def generic_text(params, eval=False):
    # params["datasets"] = [(train glob, eval_glob, ["random_sample", "sample", "chunk"] weight)]
    # , dsets=[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]
    i = 0 if not eval else 1
    current_host = get_current_host(params)
    num_hosts = get_num_hosts(params)
    print('##############################')
    print(params["datasets"])
    print('##############################')
    print(f"Host {current_host} of {num_hosts}")

    weights = []
    datasets = []

    for dataset in params["datasets"]:

        path_key = 'path' if not eval else 'eval_path'
        path = dataset[path_key]

        # fetch the filenames.
        # should this be tf.data.Dataset.list_files(pattern, shuffle=False, seed=seed)?
        filenames = tf.io.gfile.glob(path)
        # sort the filename list, for sharding across TPU hosts.
        filenames = list(sorted(filenames))
        # convert filename list to tensor.
        filenames = tf.data.Dataset.from_tensor_slices(filenames)
        # shard across each TPU host.
        filenames = filenames.shard(num_hosts, current_host)
        # if using Dataset.list_files, be sure to cache it to avoid
        # unnecessary class A operations against GCE buckets:
        #filenames = filenames.cache()
        # shuffle the filename list for each TPU host.
        filenames = filenames.shuffle(10000)

        datasets.append(text_dataset(
            filenames,
            params,
            datatype=dataset["mode"])
        )

        weights.append(dataset["weight"])

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["train_batch_size"])
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_current_host(params):
  # TODO(dehao): Replace the following with params['context'].current_host
  if 'context' in params:
    return params['context'].current_input_fn_deployment()[1]
  elif 'dataset_index' in params:
    return params['dataset_index']
  else:
    return 0


def get_num_hosts(params):
  if 'context' in params:
   return params['context'].num_hosts
  elif 'dataset_index' in params:
    return params['dataset_num_shards']
  else:
    return 1


def text_dataset(files, params, datatype):
    dataset = files
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=8, sloppy=True))

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

    eos_id = 50256 if params["n_vocab"] == 50257 else 0
    # parse each document, appending <|endoftext|>
    def parse_documents(x):
      doc, size = _parse_function(x)
      return tf.concat([doc.values, [eos_id]], axis=0)
    dataset = dataset.map(parse_documents)
    # cache the documents so that we don't read the tfrecord files
    # more than once.
    dataset = dataset.cache()
    # make an endless stream of documents.
    dataset = dataset.repeat()
    # shuffle 1,000 documents.
    dataset = dataset.shuffle(1000)
    # flatten into tokens.
    dataset = dataset.unbatch()
    # take a chunk of 32k tokens for 1,024 context, or 64k tokens for 2,048 context.
    num_tokens = 32 * params["n_ctx"]
    dataset = dataset.batch(num_tokens)
    # shuffle 1,000 chunks, which is about 30M tokens for 1,024 context,
    # or 60M tokens for 2,048 context.
    dataset = dataset.shuffle(1000)
    # given a bin that holds `total` elements, return a random
    # position such that you can take the next `subset` elements
    # without going out of bounds. E.g. randpos(1,10) will return
    # [0..9], randpos(2,10) will return [0..8], etc.
    def randpos(subset, total, dtype=tf.int64):
      assert subset <= total
      return tf.random.uniform([], maxval=(total - subset) + 1, dtype=dtype)
    # take a sample.
    def sample(tokens):
      pos = randpos(params["n_ctx"] + 1, num_tokens)
      pos += tf.range(params["n_ctx"], dtype=tf.int64)
      feature = tf.gather(tokens, pos)
      label = tf.gather(tokens, pos + 1)
      return feature, label
    dataset = dataset.map(sample)
    return dataset


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

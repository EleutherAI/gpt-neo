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

def generic_text(params, eval=False, sample_text_fn=None):
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


def autoregressive_sample_text(params, x):
    vals1 = x[:, :sequence_length]
    vals2 = x[:, 1:sequence_length + 1]

    vals1 = tf.reshape(vals1, [batch_size, sequence_length])
    vals2 = tf.reshape(vals2, [batch_size, sequence_length])
    vals1 = tf.cast(vals1, dtype=tf.int32)
    vals2 = tf.cast(vals2, dtype=tf.int32)
    return vals1, vals2

def autoregressive_sample_text_random_documents(params, x):
    seed = params.get('seed', None)
    s = tf.size(x)
    r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32, seed=seed)
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
    seed = params.get('seed', None)
    ctx_len = params["n_ctx"]
    assert 'mlm_mask_id' in params, 'the key `mlm_mask_id` must be set on your config to do masked language model training, specifying the id of the reserved mask token'

    mask_id = params['mlm_mask_id']
    mask_prob = params.get('mlm_mask_prob', 0.15)
    same_token_prob = params.get('mlm_same_token_prob', 0.10)

    mask_ignore_ids = params.get('mlm_mask_ignore_ids', [])

    if random_documents:
        s = tf.size(x)
        r = tf.random.uniform([], maxval=s - ctx_len, dtype=tf.dtypes.int32, seed=seed)
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
    mask_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed), mask_prob)
    mask_mask &= can_mask

    # generate mask for actually replacing the tokens, for allowing a small number of tokens to stay the same
    replace_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed), 1 - same_token_prob)

    # mask the tokens
    mask_tokens = tf.ones(shape, dtype=tf.int32) * mask_id
    masked_features = tf.where(mask_mask & replace_mask, features, mask_tokens)

    # labels will be set to 0 for all non-masked tokens
    labels = tf.where(not mask_mask, features, tf.zeros(shape, dtype=tf.int32))

    masked_features, labels = map(lambda t: tf.reshape(t, [ctx_len]), (masked_features, labels))
    return masked_features, labels

def pred_input(params, logger, enc=None, path_to_prompt=""):

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
    with tf.gfile.Open(f"{out_name}.txt", "a") as f:
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


if __name__ == '__main__':
    dataset = generic_text({'n_ctx': 256, 'train_batch_size': 32})

    iterator = dataset.make_one_shot_iterator()
    next_frame_data = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        frame_data_1, frame_data_2 = sess.run(next_frame_data)
        print(frame_data_1.shape, frame_data_2.shape)

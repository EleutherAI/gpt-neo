import tensorflow as tf

from models.gpt2 import gpt2

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def sample_sequence(*, params, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    length = length - params["text_len"]

    def step(params, tokens, past=None):
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                lm_output = gpt2.model(params=params, X=tokens, past=past, reuse=tf.AUTO_REUSE)

            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        else:
            lm_output = lm_output = gpt2.model(params=params, X=tokens, past=past, reuse=tf.AUTO_REUSE)


        logits = lm_output['logits'][:, :, :params["n_vocab"]]
        presents = lm_output['present']
        presents.set_shape(gpt2.past_shape(params=params, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):

        context_output = step(params, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(params, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(gpt2.past_shape(params=params, batch_size=batch_size)),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
            ],
            back_prop=False,
        )

        return tokens

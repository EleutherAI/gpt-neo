import tensorflow as tf

def perplexity_metric(loss):
    loss = tf.reduce_mean(loss)
    perplexity = tf.exp(loss)
    return {"perplexity": tf.metrics.mean(perplexity)}

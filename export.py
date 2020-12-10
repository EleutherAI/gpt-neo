import tensorflow.compat.v1 as tf

def export_model(estimator, export_dir, params,
                 checkpoint_path=None):


    def serving_input_receiver_fn():
        t = tf.placeholder(dtype=tf.int64,
                            shape=[1, params["n_ctx"]],
                            name='input_example_tensor')
        return tf.estimator.export.ServingInputReceiver(t, t)

    return estimator.export_saved_model(
        export_dir, serving_input_receiver_fn, checkpoint_path=checkpoint_path)
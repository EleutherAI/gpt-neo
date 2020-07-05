import logging
from functools import partial

import tensorflow as tf

from inputs import gpt2_pred_input
from models.gpt2 import encoder


# Takes in the user supplied text and generates output texts. Outputs to log/console and a file
def gpt2_predict(network, text, params):
    logger = logging.getLogger('tensorflow')

    enc = encoder.get_encoder(params["encoder_path"])
    predictions = network.predict(input_fn=partial(gpt2_pred_input, text=text))

    with tf.gfile.Open(params["predict_path"], "a") as f:
        for i, p in enumerate(predictions):
            p = p["tokens"]
            text = enc.decode(p)
            f.write("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            f.write(text)
            f.write("\n" + "=" * 80 + "\n")

            logger.info("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
            logger.info(text)
            logger.info("\n" + "=" * 80 + "\n")

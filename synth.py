import os
import json

from absl import logging 
from absl.flags import argparse_flags
from pydantic.dataclasses import dataclass

import tensorflow as tf
from tensorflow.compat import v1
import inputs
import config

from tqdm import auto as tqdm

def parse_args(args, parser=None):
    # Parse command line arguments
    parser.add_argument(
        "taskspec",
        type=str,
        help="the json file specifiing the configuration for this run",
    )  # Name of TPU to train on, if any
    parser.add_argument("output", type=str, help="processes the dataset and saves is to this location")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--ctx_len", 
                        type=int, 
                        help="Also called context size. The max input sequence of the final neural network.")


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])

def example_producer(infeed):
    with v1.Session(graph=tf.Graph()) as sess:
        ds = infeed({"batch_size": 8})

        it = ds.make_one_shot_iterator()
        example = it.get_next()
        while True:
            try:
                result = sess.run(example)
                yield result
            except tf.errors.OutOfRangeError:
                logging.error(
                    "dataset ended prematurely after only %d of the %d expected steps",
                    i,
                    steps,
                )

@dataclass
class TaskSpec:
    name:str
    description:str
    dataset: inputs.AddNSequenceGeneratorConfig

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def example_proto(example):
    # Helper function to avoid code duplication, writes the data as an example to the file and increments i
    content, target = example
    feature = {
         "content": _int64_feature(content),
         "target": _int64_feature(target)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main(args):
    logging.info("started synth process")

    task_dict = config.load(args.taskspec)
    task = TaskSpec(**task_dict)
    if args.vocab_size:
        task.dataset.vocab_size = args.vocab_size 
    if args.ctx_len:
        task.dataset.context_length = args.ctx_len
    seq = inputs.AddNSequenceGenerator(task.dataset)

    tf.io.gfile.makedirs(args.output)
    dscfg = dict(
        kind='datasets.TFRecordDataset',
        format='seq2seq',
        n_samples=args.n_samples,
        vocab_size=task.dataset.vocab_size,
        context_length=task.dataset.context_length
    )

    output_location = os.path.join(args.output, 'dataset.info.json')
    with tf.io.gfile.GFile(output_location, 'w') as w:
        json.dump(dscfg, w, indent=2)
    
    output_location = os.path.join(args.output, 'synth_%05d.tfrecord' % 1)
    
    with tf.io.TFRecordWriter(output_location) as w:
        it = iter(example_producer(seq))
        for _ in tqdm.tqdm(range(args.n_samples)):
            batch_ex = next(it)
            for c,t in zip(batch_ex[0], batch_ex[1]):
                proto = example_proto((c, t))
                w.write(proto.SerializeToString())

    # train
    logging.info("completed synth process. dataset generated %s", args.output)


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)


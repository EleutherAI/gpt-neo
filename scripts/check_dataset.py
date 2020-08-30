import random
import collections

import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from tokenizers import Tokenizer

PreProcessedTextLine = collections.namedtuple('PreProcessedTextLine', ['id', 'content', 'target', 'offset_start', 'offset_end'])

def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.")
    parser.add_argument("--tokenizer", type=str, default="byte-level-bpe.tokenizer.json", help="Name or path of a tokenizer spec")
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    parser.add_argument("--sample_size", type=int, default=10, help="the size of samples to inspect")
    args = parser.parse_args(argv[1:])
    return args

def load_tokenizer(location):
    return Tokenizer.from_file(location)

def read_example(example_proto, max_seq_len=1024) -> dict:
    features = {
        "id": tf.VarLenFeature(tf.uint64, default=-1),
        "content": tf.VarLenFeature(tf.bytes, default=0),
        "target": tf.VarLenFeature(tf.uint64, default=0),
        "offset_start": tf.VarLenFeature(tf.uint64, default=0),
        "offset_end": tf.VarLenFeature(tf.uint64, default=0),
    }
    return tf.parse_single_example(example_proto, features)

def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
    with tf.Session() as sess:
        files = tf.io.gfile.glob(args.input)
        if len(files) == 0:
            logging.error('no file found at %s', args.input)
            return

        for f in random.choices(files, k=args.sample_size):

            ds = tf.data.Dataset.from_tensor_slices(files)
            ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=4)
            ds = ds.map(read_example)
            ds = ds.shuffle(1024)

            it = ds.make_one_shot_iterator()
            example = it.get_next()
            
            tensor = example['content'] 
            max_id_tf = tf.reduce_max(tensor)
            min_id_tf = tf.reduce_min(tensor)
            
            while True: 
                try:
                    example, max_id, min_id = sess.run([example, max_id_tf, min_id_tf])
                    
                    example = PreProcessedTextLine(**example)

                    txt = tokenizer.decode(example['content'])

                    print(example['content'].shape)
                    print('-' * 50)
                    print(txt[:500], '\n...\n', txt[-500:])
                    print('-' * 50)
                    print('min token id: ', min_id)
                    print('max token id: ', max_id)
                    print('tokenization:', [pt.content[slice(start,end)] for start,end in zip(pt.offset_start, pt.offset_end)])
                except tf.errors.OutOfRangeError:
                    break

if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)

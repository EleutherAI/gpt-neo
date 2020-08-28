import tensorflow as tf
from tokenizers import Tokenizer
from datasets import pipeline

from absl import app, logging
from absl.flags import argparse_flags


def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.")
    parser.add_argument("--tokenizer", type=str, default="byte-level-bpe.tokenizer.json", help="Name or path of a tokenizer spec")
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    args = parser.parse_args(argv[1:])
    return args

def load_tokenizer(location):
    return Tokenizer.from_file(location)

def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
    with tf.Session() as sess:
        files = tf.io.gfile.glob(args.input)
        print(files)
        assert files, 'no file found at %s' % args.input

        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=4)
        ds = ds.map(pipeline.read_example)
        ds = ds.shuffle(1024)

        it = ds.make_one_shot_iterator()
        example = it.get_next()
        
        tensor = example['content'] 
        max_id_tf = tf.reduce_max(tensor)
        min_id_tf = tf.reduce_min(tensor)
        
        example, max_id, min_id = sess.run([example, max_id_tf, min_id_tf])

        txt = tokenizer.decode(example['content'])

        print(example['content'].shape)
        print('-' * 50)
        print(txt[:500], '\n...\n', txt[-500:])
        print('-' * 50)
        print('min token id: ', min_id)
        print('max token id: ', max_id)

if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)
import os
import random
import argparse
import shutil
from glob import glob

from lm_dataformat import Reader
from tqdm import auto as tqdm
from absl import app, logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Location of the dataset archives files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)")
    parser.add_argument("--output", type=str, help="Location to write the lm extracted files")
    parser.add_argument("--force", type=bool, default=False, help="removes the output directory if exists")
    parser.add_argument("--encoding", type=str, default="UTF-8", help="The encoding to use for the dataset")
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    random.seed(args.random_seed)

    archives = list(glob(args.input))
    if not len(archives):
        archives = list(glob(os.path.join(args.input, '*.*')))
    
    if not len(archives):
        logging.error('no files found at location %s', args.input)
        return 

    if os.path.exists(args.output):
        if not args.force:
            logging.error('output directory %s exists. use --force to remove everything inside it', args.output)
            return
        logging.error('output directory %s exists. deleting.')
        shutil.rmtree(args.output)

    os.makedirs(args.output)
    for src in tqdm(archives):
        name, file_ext = os.path.splitext(os.path.basename(src))

        dst = os.path.join(args.output, '.txt')
        if file_ext in ('xz', ):
            with open(dst, "w", encoding='UTF-8') as wf, Reader(arch) as rf:
                for s in rf.stream_data():
                    f.write(s)
                    f.write("\n\n")
        elif file_ext in ('txt', ):
            shutil.copyfile(src, dst)

    logging.info('tokenizer saved at %s', tokenizer_path)

if __name__ == "__main__":
    app.run(main)
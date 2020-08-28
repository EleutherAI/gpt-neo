import os
import random
import argparse
import shutil
from glob import glob

from lm_dataformat import Reader
from tqdm import auto as tqdm
from absl import app, logging
from absl.flags import argparse_flags

def flags_parser(args):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--input", type=str, help="Location of the dataset archives files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)")
    parser.add_argument("--output", type=str, help="Location to write the lm extracted files")
    parser.add_argument("--force", action="store_true", default=False, help="removes the output directory if exists")
    parser.add_argument("--encoding", type=str, default="UTF-8", help="The encoding to use for the dataset")
    args = parser.parse_args(args[1:])
    return args

def main(args):
    # default
    archives = list(p for p in glob(args.input) if not os.path.isdir(p))

    # try with general glob 
    if not len(archives):
        archives = list(glob(os.path.join(args.input, '*.*')))
    archives = list(p for p in glob(args.input) if not os.path.isdir(p))
    
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
    for src in tqdm.tqdm(archives):
        name, file_ext = os.path.splitext(os.path.basename(src))

        dst = os.path.join(args.output, name + '.txt')
        if file_ext in ('.xz', ):
            with open(dst, "w", encoding='UTF-8') as wf, Reader(arch) as rf:
                for s in rf.stream_data():
                    wf.write(s)
                    wf.write("\n\n")
        elif file_ext in ('.txt', ):
            shutil.copyfile(src, dst)
        else:
            logging.error('unsupported file %s with ext %s' % (src, file_ext))

    logging.info('tokenizer saved at %s', args.output)

if __name__ == "__main__":
    app.run(main, flags_parser=flags_parser)
import os
import random
import argparse
import shutil
from glob import glob
import ftfy
import time
from multiprocessing import Pool, cpu_count

from lm_dataformat import Reader
from tqdm import auto as tqdm
from absl import app, logging
from absl.flags import argparse_flags

import re
"""
Extract and cleans text and webarchive files
"""
NOA = re.compile(r'[^\x00-\x7F]+')

def clean_text(text):
    text = ftfy.fix_text(text, normalization='NFKC')
    return NOA.sub(' ', text)

def flags_parser(args):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--input", type=str, help="Location of the dataset archives files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)")
    parser.add_argument("--output", type=str, help="Location to write the lm extracted files")
    parser.add_argument("--force", action="store_true", default=False, help="removes the output directory if exists")
    parser.add_argument("--encoding", type=str, default="UTF-8", help="The encoding to use for the dataset")
    args = parser.parse_args(args[1:])
    return args

def process_single_file(src_dst):
    src, dst = src_dst
    name, file_ext = os.path.splitext(os.path.basename(src))

    if file_ext in ('.xz', ):
        with open(dst, "w", encoding='UTF-8') as wf, Reader(arch) as rf:
            for s in rf.stream_data():
                wf.write(clean_text(s))
                wf.write("\n\n")
    elif file_ext in ('.txt', ):
        with open(src, "r", encoding='UTF-8') as rf, open(dst, "w", encoding='UTF-8') as wf:
            for l in rf.readlines():
                wf.write(clean_text(l))
    else:
        logging.error('unsupported file %s with ext %s' % (src, file_ext))
        return 0
    return 1

def parallel(src_dst_list, total):
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    ret = 0
    for i in tqdm.tqdm(pool.imap(process_single_file, src_dst_list), total=total):
        ret += i
    return ret

def main(args):
    # default
    archives = list(p for p in glob(args.input) if not os.path.isdir(p))

    # try with general glob 
    if not archives:
        archives = list(glob(os.path.join(args.input, '*.*')))

    archives = list(p for p in archives if not os.path.isdir(p))
    
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

    src_dst_gen = ( 
        (src, os.path.join(args.output, os.path.splitext(os.path.basename(src))[0] + '.txt')) 
            for src in archives
    )
    
    start = time.time()
    count = parallel(src_dst_gen, total=len(archives))
    end = time.time()

    logging.info("processed {} files in {:.2f}s, {} / {} good files.".format(len(archives), end-start, count, len(archives)))

if __name__ == "__main__":
    app.run(main, flags_parser=flags_parser)
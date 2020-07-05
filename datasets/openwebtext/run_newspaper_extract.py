import json
import os
import sys

from multiprocessing import Pool

from tqdm import tqdm

from extract_text import extract_month, parse_archive, mkdir


# Extracts text from archives downloaded using the script from https://github.com/jcpeterson/openwebtext

with open("files.json", "r") as f: # This file should contain paths to all your RS_*-*-*_data.* files
    files = json.load(f)

mkdir("logs")
mkdir("parsed")
mkdir("parsed/newspaper")

def do_work(f):
    stdout = sys.stdout
    sys.stdout = open("logs/" + f.split("/")[-1] + ".log", 'w')

    try:
        month = extract_month(f)
        out_dir = mkdir(os.path.join("parsed/newspaper", month))
        parse_archive(f, out_dir, 1, 100)
        return f

    except Exception as e:
        print(e)
        return None

    finally:
        sys.stdout.close()
        sys.stdout = stdout

pool = Pool(processes=100)
i = 0
try:
    for f in tqdm(pool.imap_unordered(do_work, files), total=len(files)):
        if not f is None:
            files.remove(f)

        i += 1
        if i % 100:
            with open("files.json", "w") as fd:
                json.dump(files, fd)

except KeyboardInterrupt:
    with open("files.json", "w") as fd:
        json.dump(files, fd)
import os
import random
from glob import glob
from pathlib import Path

from lm_dataformat import Reader
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from tqdm import tqdm

data_path = Path("/home/GPTNeo/LLMD-CommonCrawl/openwebtext")
archives = glob(str(data_path / "*.xz"))

out_path = Path("../encoding")
if not out_path.is_dir():
    out_path.mkdir()

    for arch in tqdm(archives):
        name = os.path.basename(arch).split(".")[0] + ".txt"
        fp = out_path / name

        g = Reader(arch).stream_data()

        with open(fp, "w") as f:
            for s in g:
                f.write(s)
                f.write("\n\n")

data_files = glob(str(out_path / "*.txt"))
data_files = random.sample(data_files, int(0.2*len(data_files)))
# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.normalizer = NFKC()

# And then train
trainer = trainers.BpeTrainer(vocab_size=32768, min_frequency=2, special_tokens=["<|endoftext|>"])
tokenizer.train(trainer, data_files)

# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)

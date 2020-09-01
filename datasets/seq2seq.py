from typing import Optional

import tensorflow as tf
from absl import logging
from pydantic import BaseModel


class Seq2SeqFormat(BaseModel):
    vocab_size: int
    context_length: int
    has_eos: Optional[bool] = False
    keys = ["content", "target"]

"""
Contains a class as a datastore for model parameters
"""
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .utils_mtf import anonymize_dim


class ModelParameter(typing.Dict[str, typing.Any]):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__()

        self.use_video = True
        self.use_language = True
        self.use_checkpointing = False
        self.steps_per_checkpoint = 100_000
        self.time_patch = 1
        self.n_ctx = 32
        self.patch_size = 16
        self.frame_width = 320
        self.frame_height = 176
        self.vocab_size = 256
        self.bucket_name = "text-datasets"
        self.color_channels = 3
        self.three_axes = True
        self.prefix = "datasets/video"
        self.dataset_configs = []
        self.data_seed = 456772
        self.padding_token = 0
        self.n_head = 8
        self.n_embd = 256
        self.n_layer = 64
        self.buffer_size = 4
        self.shuffle_buffer = 16
        self.interleaved_datasets = 256
        self.token_patch_size = 4
        self.learning_reate = 5e-5
        self.dtype = tf.float32
        self.train_batch_size = 1
        self.mesh_shape = "x:1,y:1,h:32"
        self.layout = "batch:x,heads:y,height:h"
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.language_token_per_frame = 0
        self.warmup_steps = 3000
        self.memory_token = [0, 0, 0]
        self.weight_decay = 0.1
        self.train_steps = 572300
        self.warmup_steps = 3000
        self.iterations = 2500
        self.initial_autoregressive_position = 128
        self.use_autoregressive_sampling = False
        self.num_of_sample = 10
        self.label_smoothing = 0.2
        self.z_loss = 0.1
        self.gradient_clipping = 1.0
        self.intermediate_feed_forward_multiplier = 1
        self.group_linear_factor = 4
        self.feed_forward_per_attention = 2
        self.embedding_stddev = 0.02
        self.model_mode = 'jannet'
        self.aux_loss_factor = 0.01
        self.layer_cycle = ["feed-forward", "group-feed-forward", "attention"]

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.language_token_patch = self.language_token_per_frame // self.token_patch_size

        self.head_dim = mtf.Dimension("heads", self.n_head)
        self.head_dimensions = [self.head_dim]
        self.key_dim = mtf.Dimension("features_per_head", self.n_embd // self.n_head)
        self.anonymous_key_dim = anonymize_dim(self.key_dim)

        self.feature_dims = self.head_dimensions + [self.key_dim]
        self.anonymous_feature_dims = self.head_dimensions + [self.anonymous_key_dim]

        self.intermediate = [mtf.Dimension('_intermediate',
                                           int(np.prod([dim.size for dim in self.feature_dims])
                                               * self.intermediate_feed_forward_multiplier))]

        self.vocab_dim = mtf.Dimension("vocab", self.vocab_size)
        self.token_patch_count = self.language_token_per_frame // self.token_patch_size
        self.feature_dim_count = len(self.feature_dims)
        self.selected_head_dim = mtf.Dimension("top", (int(self.n_head ** 0.5) + 7) // 8 * 8)

        self._auxiliary_loss = 0
        self.layer_idx = -1

    def __getitem__(self, key: str) -> typing.Any:
        print(f"Getting {key} via deprecated interface")
        return self.key

    def __setitem__(self, key: str, value: typing.Any) -> None:
        print(f"Setting {key} via deprecated interface")
        self.key = value

    def get(self, key: str, default: typing.Any) -> typing.Any:
        """
        Default python get from list
        :param key: key to check for in dictionary
        :param default: default value if key doesn't exist
        :return: whatever value belongs to the key or the default
        """
        print(f"Getting {key} via deprecated interface with default value {default}")
        return self.__dict__.get(key, default)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

    def dict(self) -> typing.Dict[str, typing.Any]:
        """
        :return: dictionary containing parameters
        """
        return self.__dict__

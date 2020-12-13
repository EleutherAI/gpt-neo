class ModelParameter:
    def __init__(self, config=None, **config_kwargs):
        if isinstance(config, dict):
            config.update(config_kwargs)
        else:
            config = config_kwargs

        self.time_patch = 1
        self.n_ctx = 32
        self.patch_size = 16
        self.frame_width = 2048
        self.frame_height = 1152
        self.frame_height = 1152
        self.n_head = 8
        self.n_embd = 256
        self.n_layer = 64
        self.buffer_size = 8
        self.lr = 5e-5
        self.train_batch_size = 1
        self.recompute_grad = True
        self.mesh_shape = "x:1,y:1,z:1,h:32"
        self.layout = "batch:x,heads:y,embd:z,height:h"
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.dataset_ids = ["openwebtext-documents"]
        self.local_attention_radius = 256
        self.n_vocab = 256
        self.embed_dropout = 0
        self.lr_decay = "cosine"
        self.warmup_steps = 3000
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.epsilon = 1e-8
        self.ada_epsilon1 = 1e-30
        self.ada_epsilon2 = 1e-3
        self.opt_name = "adam"
        self.weight_decay = 0.1
        self.attn_dropout = 0
        self.train_steps = 572300
        self.eval_steps = 0
        self.predict_steps = 1
        self.res_dropout = 0
        self.warmup_steps = 3000
        self.eval_batch_size = 64
        self.predict_batch_size = 1
        self.iterations = 2500
        self.datasets = []
        self.shuffle = False
        self.residual = True
        self.scale_by_depth = True
        self.scale_by_in = False
        self.activation_function = "gelu"
        self.gradient_clipping = 1.0

        self.__dict__.update(config)

    def __getitem__(self, key):
        print(f"Getting {key} via deprecated interface")
        return self.__dict__[key]

    def __setitem__(self, key, value):
        print(f"Setting {key} via deprecated interface")
        self.__dict__[key] = value

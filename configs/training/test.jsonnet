local optimizers = import 'optimizers.libsonnet';

local lr = {
   lr: 0.0001,
   lr_decay: "cosine",
   warmup_steps: 0,
};

local trainer = {
   model_path: std.extVar('MODEL_PATH'), // the location to save the checkpoints
   learning_rate: lr,
};

local Dataset() = {
   file_pattern: 'gs://experiments-us-central1/childrensbooks/',
};

local GPT2() = {
   model: "GPT2",
   n_ctx: 512,
   n_embd: 512,
   n_head: 8,
   n_vocab: 32,
   
   n_layer: 1,
   scale_by_depth: false,
   scale_by_in: false,
   mesh_shape: "",
   layout: "",
   activation_function: "gelu",
   attention_types: [
      [["global"], 1]
   ],
   auto_layout: false,
   auto_layout_and_mesh_shape: false,
   
};

local Other() = {
   stop_at_token: 2,
   remove_partial_sequences: true,
   iterations: 500,
   scalenorm: true,
   no_weight_tie: false,
};

local InFeed() = {
   batch_size: 1,
   random: {
      context_length: 25,
      vocab_size: 16000,
   },
};

local Schedule() = {
   steps: 0,
   steps_per_checkpoint: 100,
};

local Cluster() = {
   use_tpu: false,
   num_cores: 256,
};

{
   cluster: Cluster(),
   infeed: InFeed(),
   model: GPT2(),
   trainer: std.mergePatch(trainer, {
      optimizer: optimizers.Adam(),
      schedule: Schedule(),
   }),
   regularization: {
      embed_dropout: 0.1,
      weight_decay: 0.1,
      attn_dropout: 0.1,
      res_dropout:0.1,
      gradient_clipping: 0.5,
   },
   other: Other()
} 
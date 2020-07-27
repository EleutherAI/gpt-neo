# GPT Neo

1T or bust my dudes.

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/)/[GPT3](https://arxiv.org/abs/2005.14165)-like models, with the facilities to train even larger models, using the Tensorflow Mesh library. Currently only TPU training is supported.

No pretrained model yet, everything has to be built from scratch!

# Requirements

`pip3 install tensorflow==1.15.2 mesh-tensorflow==0.1.16 tensorflow-datasets ortools google-api-python-client oauth2client`
(TODO: put into a `requirements.txt`)

# TPU Setup

Sign up for [Google Cloud Platform](https://cloud.google.com/), and create a [storage bucket](https://cloud.google.com/storage). 

Create your VM through a google shell (`ssh.google.com`) with `ctpu up --vm-only` so that it is connected to your Google bucket.

To train using some dummy data:

Some dummy data: `wget -P datasets/ https://storage.googleapis.com/connors-datasets/bundestag/bundestag_0.tfrecords`

Then `gsutil cp src dst`, with dst being the gs:// path to your bucket datasets folder.

## Generating your own Dataset
 TODO(@Daj) 
Encode data in tfrecords:
- ask @daj
- lookup https://github.com/ConnorJL/GPTNeo/blob/tfmesh/datasets/openwebtext/create_tfrecords.py

## Parameters

Pick a valid config for you from `/configs`. Then change these parameters:

- `n_heads`: the number of attention heads
- `n_embd`: size of the hidden layers, must be divisible by `n_heads`
- `encoder_path`: unused (TODO: double check & delete.)
- `n_vocab`: vocabulary size
- `embed_dropout`: Dropout chance on the word embedding, set to 0 to disable (default: 0.1)
- `lr`: learning rate, defaults will vary depending on model size. use [this](https://i.imgur.com/g5jKbjT.png) from the GPT3 paper as a guide.
- `warmup_steps`: number of steps before full learning rate is reached (linear ramp from `0` to `lr`).
- `lr_decay`: `cosine` (used by OA) or `linear`, According to OpenAI's scaling paper, the choice of setting here doesn't matter too much as long as it decays to above 0 over a suitable length of time. (diminish `lr` over time)
- `opt_name`: `adam` or `adafactor`. choice of optimizer. `adam` is considered better but takes 2-3x the amount of memory.
- `beta1`, `beta2` and `epsilon`: `adam` optimizer params.
- `beta1_adam`: unused (TODO: make params for different optimizers more clear.)
- `beta1`, `ada_epsilon1` and `ada_epsilon2`: `adafactor` optimizer params.
- `weight_decay`: Weight decay parameter, if not present no weight decay is used (the weight decay fix for Adam is used) (default: 0.01) (optional).
- `train_batch_size`: Batch size during training phase.
- `attn_dropout`: Dropout chance on attention layers, set to 0 to disable (default: 0.1).
- `train_steps`: number of training steps (batches), set to roughly ~1 epoch for now (total number of tokens in your dataset / number of tokens per batch (= `train_batch_size` / `n_ctx`)).
- `eval_steps`: set to `0` for no eval. each `steps_per_checkpoint`, the model is tested for `eval_steps` (`steps_per_checkpoint` is set with the CLI rn)
- `max_steps`, `res_dropout` and `predict_batch_size`: unused (TODO: double check and delete).
- `iterations`: number of steps queued to the TPU (also used for Tensorboard summaries), must be smaller than `steps_per_checkpoint`.
- `datapath`: path to the google storage folder containing the dataset.
- `datasets`: list of individual tfrecords files. dataset is like this: `[regex , ?? , ?? , sampling_type]`. example: `[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]`
    + `regex`: `regex` matching the names of the datasets
    + ??: ask @Daj
    + `sampling_type`: `document` (sequential order) or `random_sample` (chunks of length `n_ctx` in random order)
- `model`: `GPT2`, regular GPT model, WIP: 'GPT2MOE' GPT model with Mixture of Experts
- `model_path`: Google storage location to save model checkpoints.
- `n_ctx`: Size of context window. In smaller models, this is set to 1024. For larger models, this is 2048
- `predict_path`: unused (TODO: double check & delete.)
- `n_layer`: number of layers (blocks) in the model.
- `scale_by_depth` and `scale_by_in`: unused, i think? ask @daj (TODO: double check & delete.)
- `mesh_shape`: A Mesh is a n-dimensional array of processors with named dimensions. Each Tensor is assigned to a Mesh, instead of a device. The 'mesh_shape' is the shape of this array. E.G, for a TPU v3-128 "mesh_shape": “x:16,y:8”.
- `layout`: A Tensor is laid out on its mesh with one slice on each processor. A Tensor "layout", is an injective partial map specifying which dimensions of the tensor are (evenly) split across which dimensions of the mesh. No dimension of a tensor may be split across two dimensions of its mesh and no two dimensions of a tensor may be split across the same dimension of its mesh. The user defines a global set of layout rules in the form of (tensor-dimension-name, mesh-dimension-name) pairs. A dimension of a tensor is split across a dimension of its mesh if there is a matching rule. E.G (for the above example mesh_shape: "layout":"batch:x,heads:y"
- `activation_function`: `selu` (self normalizing) or `gelu` (used by OA), activation function for feed-forward passes
- `fixed_attn_block_size`: unused (computed internally)(TODO: double check and delete).
- `layer_offset`: unused (TODO: double check and delete).
- `local`: `true` (local attention) or `false` (global attention)
- `precision`: `float32` (use this for now) or `bf16` (change some variables to bf16 for better performance, not working yet)
- `microbatches_per_batch`: if > 1, will split the batch up into smaller microbatches to avoid OOMs. Gradients are accumulated locally and reduced once.

## Training

Connect to your VM, clone this repo and cd into the folder.

To run: `python3 main.py --model configs/your_config.json --steps_per_checkpoint n --tpu tpu-name`

- `tpu`: name of the tpu to use (passed by CLI for now)
- `steps_per_checkpoint`: The frequency in steps at which to save checkpoints.
- Optional: `--auto_layout` and `--auto_layout_and_mesh_shape`: CLI flags that auto generate a memory efficient `layout` (and `mesh_shape`)

To monitor: `tensorboard --logdir model_path`

Tensorboard exposes an http server on port `6006`. To access it on the remote machine, just do `localhost:6006`.
However, the remote machine will usually just have a terminal. If you want to actually look at the webpage.
To do so, you'll need to forward the port through SSH so that you can access it in your client machine.
An easy way to do port forwarding, is to add in your client machine `~/.ssh/config`:
```s
Host GptVM
 LocalForward 6006 localhost:6006
 HostName your_vm_ip
 User your_user
```
Then, you'll be able to access tensorboard on your browser by doing `localhost:6006`.

## Downloading Pretrained Models

TODO

## Generating Text

TODO

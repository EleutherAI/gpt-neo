# Various Things

No pretrained model yet, everything has to be built from scratch!

Those are python dependencies:
`pip3 install tensorflow==1.15.2 mesh-tensorflow==0.1.16 mesh-tensorflow tensorflow-datasets ortools google-api-python-client oauth2client`
(put into a `requirements.txt`)

Create your VM through a google shell (`ssh.google.com`) with `ctpu up --vm-only` so that it is connected to your Google bucket.

Download data into `gs://your_bucket/datasets`

Some dummy data: `wget -P datasets/ https://storage.googleapis.com/connors-datasets/bundestag/bundestag_0.tfrecords`

Then `gsutil cp src dst` to copy.

Pick a valid config for you from `/configs`. Then you can change those parameters:
- `n_heads`: the number of attention heads
- `n_embd`: size of the hidden layers, must be divisible by `n_heads`
- `encoder_path`: unused
- `n_vocab`: vocabulary size
- `embed_dropout`: dropout of the embed layer
- `lr`: learning rate, use [https://i.imgur.com/g5jKbjT.png](this from GPT3)
- `warmup_steps`: number of batches before full learning rate (linear ramp from `0` to `lr`)
- `lr_decay`: `cosine` (used by OA) or `linear`, either doesn't seem to matter much. essential to good learning (diminish `lr` over time)
- `opt_name`: `adam` or `adafactor`. choice of optimizer. `adam` is better but takes 2-3x the amount of memory
- `beta1`, `beta2` and `epsilon`: `adam` optimizer params
- `beta1_adam`: unused
- `beta1`, `ada_epsilon1` and `ada_epsilon2`: `adafactor` optimizer params
- `weight_decay`: not known yet, likely used regularization (leave untouched for now)
- `train_batch_size`: size of the training batches lel
- `attn_dropout`: dropout in attention layers
- `train_steps`: number of training steps (batches), set to roughly ~1epoch for now (total number of tokens in your dataset / number of tokens per batch (= `train_batch_size` / `n_ctx`))
- `eval_steps` and `steps_per_checkpoint`: set to `0` for no eval. each `steps_per_checkpoint`, the model is tested for `eval_steps` (`steps_per_checkpoint` is set with the CLI rn)
- `max_steps`, `res_dropout` and `predict_batch_size`: unused
- `iterations`: number of steps queued to the TPU (also used for Tensorboard summaries), must be smaller than `steps_per_checkpoint`
- `datapath`: path to the folder containing datasets
- `datasets`: list of datasets. dataset is like this: `[regex , ?? , ?? , sampling_type]`. example: `[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]`
    + `regex`: `regex` matching the names of the datasets
    + ??: ask @Daj
    + `sampling_type`: `document` (sequential order) or `random_sample` (chunks of length `n_ctx` in random order)
- `model`: `GPT2`, only one right now
- `model_path`: location where to save checkpoints and the model
- `n_ctx`: context window
- `predict_path`: unused
- `n_layer`: number of blocks (~layers) in the model
- `scale_by_depth` and `scale_by_in`: ask @daj
- `mesh_shape` and `layout`: mesh tensorflow stuff, ask @sid
- `--auto_layout` and `--auto_layout_and_mesh_shape`: CLI flags that make main generate a `layout` (and `mesh_shape`)
- `activation_function`: `selu` (self normalizing) or `gelu` (used by OA), activation function for feed-forward passes
- `fixed_attn_block_size`: unused (computed internally)
- `layer_offset`: unused
- `local`: `true` (local attention) or `false` (global attention)
- `precision`: `float32` (use this for now) or `bf16` (change some variables to bf16 for better performance, not working yet)
- `tpu`: name of the tpu to use (passed by CLI for now)

To run: `python3 main.py --model configs/your_config.json --steps_per_checkpoint n --tpu tpu-name`

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

Encode data in tfrecords:
- ask @daj
- lookup https://github.com/ConnorJL/GPTNeo/blob/tfmesh/datasets/openwebtext/create_tfrecords.py

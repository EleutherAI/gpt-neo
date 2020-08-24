# GPT Neo

1T or bust my dudes.

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/)/[GPT3](https://arxiv.org/abs/2005.14165)-like models, with the facilities to train even larger models, using the Tensorflow Mesh library. Currently only TPU training is supported.

No pretrained model yet, everything has to be built from scratch!

# Requirements

```bash
$ pip3 install \
    tensorflow==1.15.2 \
    mesh-tensorflow==0.1.16 \
    tensorflow-datasets \
    ortools \
    google-api-python-client \
    oauth2client \
    pytest \
    jsonlines \
    lm_dataformat \
    sacred \
    tokenizers \
    numpy
```

or

```bash
$ pip3 install -r requirements.txt
```

# Training Setup

Sign up for [Google Cloud Platform](https://cloud.google.com/), and create a [storage bucket](https://cloud.google.com/storage). 

Create your VM through a google shell (`https://ssh.cloud.google.com/`) with `ctpu up --vm-only` so that it can connect to your Google bucket and TPUs and install the correct versions of the libraries with pip (see above).

Download the dummy data: `wget https://storage.googleapis.com/connors-datasets/bundestag/bundestag_0.tfrecords`

Then copy the data to your bucket: `gsutil cp bundestag_0.tfrecords gs://<your bucket>/`

To use your own data, see "Generating Your Own Dataset" below.

## Basic Integration Testing

Testing is done on any python file that is prefixed with `test_`. Feel free to add your own file, or append to whatever file already exists.

```bash
$ python -m pytest .
```

## Quick Testing on Increment Task

This repository has, built in, a quick way of running a simple 1 layer attention network on a increment task (increment the source sequence by 1). This is for making sure any model changes can pass a basic test. The loss should converge to around 1.65. > 2 means the model is not working. 0 means there is likely a causal mask leakage.

To train, simply run

```bash
$ python main.py --test
```

To see the tensorboard

```bash
$ sh start_test_tb.sh 8081
```

To run a prediction

```bash
$ python main.py --test --predict
```

To clear the checkpoint and restart training (making changes to the model), append a `--new` flag

```bash
$ python main.py --test --new
```
It should converge within 5 minutes. If not, something is wrong.

## Peeking at a Dataset

If you are ever confused by the dataset of a particular config file, you can easily check the minimum and maximum token ids with a single command. This is useful for making sure that the vocabulary size of the model is at least as large as the maximum token id. Tensorflow will not error if you try to gather on a matrix with out of bounds indices, so you need to make sure your vocabulary size is sufficiently large.

```bash
$ python main --model {config_name} --check_dataset
```

## Training

Connect to your VM, clone this repo and cd into the folder. Find a fitting config in `/configs` and tweak parameters as needed (see reference at the end of this document). Then run:

`python3 main.py --model {your_config_name} --steps_per_checkpoint n --tpu tpu-name`

- `tpu`: Name of the TPU to use.
- `steps_per_checkpoint`: The frequency in steps at which to save checkpoints.
- `--auto_layout` and `--auto_layout_and_mesh_shape` (Optional): Disable training and instead auto generate a memory efficient `layout` (and `mesh_shape`)

## Training (with sacred)

Sacred helps track experiments and is much nicer to work with than tensorboard.

To use: 

1. Ensure model_dir doesnt have any metric logs in it (it trips up the metric stuff for tensorboard, which assumes that it's a continuation of the existing run). You can use `gsutil rm -r ...` to delete model dir

2. Run `python3 run_experiment.py --tpu sometpuhere --model someconfig.json` Options do the same thing as in the old script. 

3. You can go to http://server_ip_goes_here:8080/ to see the Omniboard overview. It's password protected, ask in the discord for info. If you want to see the tensorboard for some reason, the `run_experiment.py` script also spins up a tensorboard and automatically assigns it a port. The script should print out the tensorboard port near the top of the log. 

## Monitoring

To monitor: `tensorboard --logdir model_path`

Tensorboard exposes an http server on port `6006`. To access it on the remote machine, just do `localhost:6006`.
However, the remote machine will usually just have a terminal. To easily view the resulting webpage, you'll need to forward the port through SSH so that you can access it on your client machine.
An easy way to do port forwarding is to add the following to your client machine's `~/.ssh/config`:
```s
Host GptVM
 LocalForward 6006 localhost:6006
 HostName your_vm_ip
 User your_user
```
Then, you'll be able to access tensorboard with your browser at `localhost:6006`.

# Generating your own Dataset
You can use the `datasets/create_tfrecords.py` script to encode your text data into tfrecords suited for the training. First install the dependencies:

`pip3 install ftfy tqdm tokenizers lm_dataformat`

Your data must either be in the form of lots of normal text files (one document per file), or in any format supported by [lm_dataformat](https://github.com/leogao2/lm_dataformat). 

You can run the script without parameters to see help for all options. There are two main modes:

## Document Mode

Each example in the tfrecords is one (variably sized) document. This is to be used with the `documents_fixed` and `documents_random` sampling modes (see parameters, below).

`python3 create_tfrecords.py --mode documents --base_dir <base> --name <name> --output_dir <output> --encoder_path <encoder> --minimum_size <min> `

- `base_dir`: Defines the folder where your data is located. The script will encode all files present in this folder.
- `name`: Name of output files will be `name_i.tfrecords` where i is the number of the file.
- `output_dir`: Where to save the tfrecords to
- `encoder_path`: Path to your [tokenizers](https://github.com/huggingface/tokenizers) generated tokenizer json. You can use `datasets/byte-level-bpe.tokenizer.json`, or look at `datasets/openwebtext/train_tokenizer.py` for an example of how such a tokenizer can be trained.
- `minimum_size`: The minimum size (in tokens) a document must have, otherwise it is discareded. This is what will later determine your `stitch` parameter: `stitch * minimum_size` must always be greater or equal `n_ctx` (see parameters below).

## Chunk Mode

In chunk mode, all documents are concatenated (with seperator tokens between documents) and then sliced into equally sized chunks. So each tfrecords example is one uniformly sized chunk. For use with the `chunks` sampling mode (see parameters, below).

`python3 create_tfrecords.py --mode chunks --base_dir <base> --name <name> --output_dir <output> --encoder_path <encoder> --seperator <sep> --chunk_size <size>`

- `base_dir`: Defines the folder where your data is located. The script will encode all files present in this folder.
- `name`: Name of output files will be `name_i.tfrecords` where i is the number of the file.
- `output_dir`: Where to save the tfrecords to
- `encoder_path`: Path to your [tokenizers](https://github.com/huggingface/tokenizers) generated tokenizer json. You can use `datasets/byte-level-bpe.tokenizer.json`, or look at `datasets/openwebtext/train_tokenizer.py` for an example of how such a tokenizer can be trained.
- `seperator`: Written in list format, the seperator token(s) to insert between documents (e.g. "[0]"). Will depend on your encoder.
- `chunk_size`: How large each chunk should be. Must be equal to `n_ctx`. (Note: The tfrecords examples will be size `n_ctx+1`. This is normal and is to ensure the last input token has a target)

# Downloading Pretrained Models

TODO

# Generating Text

TODO

## Parameter Reference

Pick a valid config from `/configs` and tweak the parameters as needed:

- `n_heads`: The number of attention heads
- `n_embd`: Size of the hidden layers, must be divisible by `n_heads`
- `n_vocab`: Vocabulary size
- `embed_dropout`, `res_dropout`, `attn_dropout`: Dropout chance for word embedding/residuals/attention, set to 0 to disable (default: 0.1)
- `lr`: Learning rate, defaults will vary depending on model size. Use [this table](https://i.imgur.com/g5jKbjT.png) from the GPT3 paper as a guide.
- `warmup_steps`: Number of steps before full learning rate is reached (linear ramp from `0` to `lr`).
- `lr_decay`: `cosine` (used by OA) or `linear`. According to OpenAI's scaling paper, the choice of setting here doesn't matter too much as long as it decays to above 0 over a suitable length of time.
- `opt_name`: `adam` or `adafactor`. Choice of optimizer. `adam` is considered better but takes 2-3x the amount of memory.
- `beta1`, `beta2` and `epsilon`: `adam` optimizer params.
- `beta1`, `ada_epsilon1` and `ada_epsilon2`: `adafactor` optimizer params.
- `weight_decay`: Weight decay parameter, if not present no weight decay is used (the weight decay fix for Adam is used) (default: 0.01) (optional).
- `train_batch_size`: Batch size during training.
- `train_steps`: Number of training steps (batches), set to roughly ~1 epoch for now (total number of tokens in your dataset / number of tokens per batch (= `train_batch_size` / `n_ctx`)).
- `eval_steps`: Number of steps to run for each evaluation. Set to `0` for no eval. Each `steps_per_checkpoint`, the model is tested for `eval_steps` (`steps_per_checkpoint` is set with the CLI currently)
- `iterations`: Number of steps queued to the TPU (also used for Tensorboard summaries), must be smaller than `steps_per_checkpoint`. (default: 500)
- `datasets`: List of tfrecords datasets to use. Each dataset is a list with the following parameters: `[train glob , eval glob, stitch, sampling_mode, weight]`. So for example for a single dataset (note the double list): `[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]`
    + `train glob`: A [glob](https://en.wikipedia.org/wiki/Glob_(programming)) pattern for files used during training
    + `eval glob`: A [glob](https://en.wikipedia.org/wiki/Glob_(programming)) pattern for files used during evaluation
    + `stitch`: If `sampling_mode` `random_sample` is used, the input pipeline samples this amount of texts into one to sample from. You must select stitch so that `stitch * minimum_document_length >= n_ctx`
    + `sampling_mode`: `chunks` (tfrecords are preprocessed into the correct length and are read sequentially) or `documents_random` (`stitch` amount of documents are concatenated and then a `n_ctx` chunk is randomly subsampled)
    + `weights`: How much relative weight this dataset should have compared to others
- `model`: Which model to train. Currently only `GPT2` is supported, WIP: 'GPT2MOE' GPT model with Mixture of Experts
- `model_path`: Google storage location to save model checkpoints.
- `n_ctx`: Size of context window. In smaller models, this is set to 1024. For larger models, this is 2048.
- `n_layer`: Number of layers (blocks) in the model.
- `scale_by_depth`: If true, the weight initialization of layers are scaled by their depth as in the GPT2 paper. (default: true)
- `scale_by_in`: If true, the weight initialization of layers are scaled by their number of inputs as in the GPT2 paper. (default: true)
- `mesh_shape`: A Mesh is an n-dimensional array of processors with named dimensions. Each Tensor is assigned to a Mesh, instead of a device. The 'mesh_shape' is the shape of this array, e.g., for a v3-128 TPU "mesh_shape": “x:16,y:8”.
- `layout`: A Tensor is laid out on its mesh with one slice on each processor. A Tensor "layout", is an injective partial map specifying which dimensions of the tensor are (evenly) split across which dimensions of the mesh. No dimension of a tensor may be split across two dimensions of its mesh and no two dimensions of a tensor may be split across the same dimension of its mesh. The user defines a global set of layout rules in the form of (tensor-dimension-name, mesh-dimension-name) pairs. A dimension of a tensor is split across a dimension of its mesh if there is a matching rule, e.g. (for the above example mesh_shape: "layout":"batch:x,heads:y"
- `activation_function`: `selu` (self normalizing) or `gelu` (used by OA), activation function used in feed-forward passes. (default: gelu)
- `attention_types`: the type of attention for each layer in a list of the following format [[["attention_type"], n_layers]]. e.g. for a 12 layer net [[["global"], 12]] or [[["local"], 10], [["global"], 2]]
- `precision`: `float32` (use this for now) or `bf16` (change some variables to bf16 for better performance, not working yet)
- `tokens_per_mb_per_replica`: If not None, will split the batch up into smaller microbatches containing `tokens_per_mb_per_replica` tokens to avoid OOMs. Gradients are accumulated locally and reduced once.

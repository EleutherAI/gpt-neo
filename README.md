# GPT Neo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5297715.svg)](https://doi.org/10.5281/zenodo.5297715) [![arXiv](https://img.shields.io/badge/arXiv-2101.00027-f9f107.svg)](https://arxiv.org/abs/2101.00027)

**As of August, 2021 code is no longer maintained. It is preserved here in archival form for people who wish to continue to use it.*

🎉 1T or bust my dudes 🎉

An implementation of model & data parallel [GPT3](https://arxiv.org/abs/2005.14165)-like models using the [mesh-tensorflow](https://github.com/tensorflow/mesh) library.

**If you're just here to play with our pre-trained models, we strongly recommend you try out the [HuggingFace Transformer integration](https://huggingface.co/EleutherAI).**

Training and inference is officially supported on TPU and should work on GPU as well. This repository will be (mostly) archived as we move focus to our GPU-specific repo, [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/).

In addition to the functionality offered by GPT-3, we also offer the following:
* [Local attention](https://arxiv.org/abs/2004.05150)
* [Linear attention](https://arxiv.org/abs/1812.01243)
* [Mixture of Experts](https://arxiv.org/abs/1701.06538)
* [Axial Positional embedding](https://arxiv.org/abs/1912.12180)

NB, while neo can *technically* run a training step at 200B+ parameters, it is very inefficient at those scales. This, as well as the fact that many GPUs became available to us, among other things, prompted us to move development over to [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/).

# Pretrained Models

**Update 21/03/2021:**

We're proud to release two pretrained GPT-Neo models trained on The Pile, the weights and configs can be freely downloaded from [the-eye.eu](https://the-eye.eu/public/AI/gptneo-release/).

1.3B: https://mystic.the-eye.eu/public/AI/gptneo-release/GPT3_XL/

2.7B: https://mystic.the-eye.eu/public/AI/gptneo-release/GPT3_2-7B/

For more information on how to get these set up, see the colab notebook, or read through the rest of the readme.

## Model Evaluations

#### Linguistic Reasoning

| Model and Size   | Pile BPB   | Pile PPL  | Wikitext PPL | Lambada PPL | Lambada Acc | Winogrande | Hellaswag  |
|------------------|------------|-----------|--------------|-------------|-------------|------------|------------|
| **GPT-Neo 125M** | -----      | -----     | **32.285**   | **30.266**  | **37.36%**  | **50.43%** | **28.67%** |
| GPT-3 125M       | -----      | -----     | -----        | 18.6        | 42.7%       | 52.0%      | 33.7%      |
| **GPT-Neo 350M** | -----      | -----     | **22.5657**  | **13.876**  | **47.27%**  | **51.14%** | **32.16%** |
| GPT-3 350M       | -----      | -----     | -----        | 9.09        | 54.3%       | 52.1%      | 43.6%      |
| GPT-3 Ada        | 0.9631     | -----     | -----        | 9.954       | 51.60%      | 52.90%     | 35.93%     |
| **GPT-Neo 1.3B** | **0.7527** | **6.159** | **13.10**    | **7.498**   | **57.23%**  | **55.01%** | **38.66%** |
| GPT-3 1.3B       | -----      | -----     | -----        | 5.44        | 63.6%       | 58.7%      | 54.7%      |
| GPT-2 1.5B       | 1.0468     | -----     | 17.48        | 10.634      | 51.21%      | 59.40%     | 40.03%     |
| **GPT-Neo 2.7B** | **0.7165** | **5.646** | **11.39**    | **5.626**   | **62.22%**  | **56.50%** | **42.73%** |
| GPT-3 2.7B       | -----      | -----     | -----        | 4.60        | 67.1%       | 62.3%      | 62.8%      |


#### Physical and Scientific Reasoning

| Model and Size   | MathQA     | PubMedQA   | Piqa       |
|------------------|------------|------------|------------|
| **GPT-Neo 125M** | **22.78%** | **55.10%** | **63.06%** |
| GPT-3 125M       | -----      | -----      | 64.6%      |
| **GPT-Neo 350M** | **23.45%** | **53.80%** | **65.07%** |
| GPT-3 350M       | -----      | -----      | 70.2%      |
| GPT-3 Ada        | 24.29%     | 52.80%     | 68.88%     |
| **GPT-Neo 1.3B** | **24.05%** | **54.40%** | **71.11%** |
| GPT-3 1.3B       | -----      | -----      | 75.1%      |
| GPT-2 1.5B       | 23.64%     | 58.33%     | 70.78%     |
| **GPT-Neo 2.7B** | **24.72%** | **57.54%** | **72.14%** |
| GPT-3 2.7B       | -----      | -----      | 75.6%      |


**Note:** All evaluations were done using our [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness). Some results for GPT-2 and GPT-3 are inconsistent with the values reported in the respective papers. We are currently looking into why, and would greatly appreciate feedback and further testing of our eval harness.

# Setup

```bash
git clone https://github.com/EleutherAI/GPTNeo
cd GPTNeo
pip3 install -r requirements.txt
```
# Training Setup

## TPUs:

Sign up for [Google Cloud Platform](https://cloud.google.com/), and create a [storage bucket](https://cloud.google.com/storage). 

Create your VM through a google shell (`https://ssh.cloud.google.com/`) with `ctpu up --vm-only` so that it can connect to your Google bucket and TPUs and install the requirements with pip (see above).

Google colab provides tpu-v8s for free, which should be enough to finetune our models up to GPT3XL (1.5B parameter) sizes.
Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/GPTNeo/blob/master/GPTNeo_example_notebook.ipynb) to run through our example colab notebook.

For more detailed instructions, run through our [Training Guide](https://github.com/EleutherAI/GPTNeo#training-guide) below.

## GPUs:

You can also choose to train GPTNeo locally on your GPUs. To do so, you can omit the Google cloud setup steps above, and git clone the repo locally. Run through the [Training Guide](https://github.com/EleutherAI/GPTNeo#training-guide) below, then when running main.py, you simply have to omit the `tpu` flag, and pass in GPU ids instead.

Note: Some users have reported having difficulty getting MTF to recognize their GPUs. See [here](https://github.com/EleutherAI/gpt-neo/issues/150) for details and instructions on how to fix it.

# Generating Text

Once you have a trained model, or you've downloaded one of our pre-trained models, generating text is as simple as running the main.py script with the `--predict` flag on. You can pass a path to your prompt txt file with the `--prompt` flag, like so:

```bash
python3 main.py --predict --prompt <example_prompt.txt> --tpu <tpu_name> --model <config_name>
```

or, if using GPUs:

```bash
python3 main.py --predict --prompt <example_prompt.txt> --gpu_ids <device:GPU:0 device:GPU:1> --model <config_name>
```

# Training Guide

## 1. Create your Tokenizer (OPTIONAL)

We recommend you use [Huggingface's pretrained GPT2 tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2Tokenizer) with our repo (instructions provided below), but if you want to train a model with a different vocabulary size, we provide facilities to train your own tokenizer like so:

```bash
python data/train_tokenizer.py \
    --base_dir ./path/to/your/txt/files \
    --output_dir ./output/path \
    --file_type txt \
    --vocab_size 50257

# if it succeeded, you should see the message
# 'tokenizer saved at ./output/path/byte-level-bpe.tokenizer.json'
```

## 2. Tokenizing your Dataset

If you just want to test training, you can skip this step and download some dummy data like so:

```
wget https://storage.googleapis.com/connors-datasets/bundestag/bundestag_0.tfrecords
```

Then copy the data to your bucket, or if using GPUs, a local directory: 

```
gsutil cp bundestag_0.tfrecords gs://<your bucket>/
```

If using your own data to train, you can use the `data/create_tfrecords.py` script to encode your text data into tfrecords.

Your data must either be in the form of lots of normal .txt files (one document per file), or in any format supported by [lm_dataformat](https://github.com/leogao2/lm_dataformat). 

You can run the script without parameters to see help for all options.

In **document mode** Each example in the tfrecords is one (variably sized) document. This is to be used with the `documents_fixed` and `documents_random` sampling modes (For more details see the parameters reference section).
Document mode is the default mode.

The below command will tokenize all files in acceptable formats in *base_dir* using gpt2 tokenizer and save them to *output_dir*
```
python3 create_tfrecords.py --mode documents --input_dir <base> --name <name> --output_dir <output> --use_gpt2_tokenizer --minimum_size <min> 
```

- `input_dir`: Defines the folder where your data is located. The script will encode all files present in this folder.
- `name`: Name of output files will be `name_i.tfrecords` where i is the number of the file.
- `output_dir`: Where to save the tfrecords to
- `use_gpt2_tokenizer`: Whether to use the pretrained HuggingFace GPT2 tokenizer, in which case the separator will be set to [50256].
- `encoder_path`: if not using the pretrained gpt2 tokenizer, use this flag to provide a path to your generated tokenizer json.
- `separator`: Written in list format, the separator token(s) to insert between documents (e.g. "[0]"). Will depend on your encoder.
- `minimum_size`: The minimum size (in tokens) a document must have, otherwise it is discarded. This is what will later determine your `stitch` parameter: `stitch * minimum_size` must always be greater or equal `n_ctx` (For more details see the parameters reference section).

## 4. Using a Dataset in a Model

To use a dataset in a model, you must first register that dataset under `./configs/dataset_configs` folder. First choose a filename with a `.json` extension. That filename will serve as the dataset identification. The config should be filled out the following manner.

If you have a dataset encoded using the pretrained gpt2 tokenizer, you can specify that like so:

```json
{
    "n_vocab": 50257,
    "path": "gs://neo-datasets/openwebtext-documents/openwebtext_*.tfrecords",
    "eval_path": "gs://neo-datasets/openwebtext-documents/openwebtext_*.tfrecords",
    "tokenizer_is_pretrained": true,
    "tokenizer_path": "gpt2"
}
```

or if you've trained a custom tokenizer, like so:

```json
{
    "n_vocab": 32768,
    "path": "./path/to/your/*.tfrecords",
    "eval_path": "./path/to/your/eval/*.tfrecords",
    "tokenizer_path": "./path/to/your/byte-level-bpe.tokenizer.json"
}
```

Finally, in your model config, add the filename that you created above to the `datasets` array.

The `<dataset id>` will be the filename, excluding the `.json`, that you created above

```
"datasets": [[<dataset id>, <stitch>, <datatype>, <weight>]] # datasets key defines at run time how each dataset is processed for training
```

## 5. Choose a model configuration

Once you have your datasets set up, find a suitable config in `/configs`.

Here we use a GPT3-XL sized model as an example, but there are many more in `./configs`, all of which have short summaries in the Available Configs section.

All you need to do is edit the dataset id as described above, and edit `model_path` (where logs and checkpoints will be saved) to point to a cloud bucket you have write access to (or local path, if using GPUs).

```json
{
    "n_head": 32,
    "n_vocab": 50257,
    "embed_dropout": 0.1,
    "lr": 0.0002,
    "lr_decay": "cosine",
    "warmup_steps": 3000,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "opt_name": "adam",
    "weight_decay": 0.1,
    "train_batch_size": 512,
    "attn_dropout": 0.1,
    "train_steps": 286150,
    "eval_steps": 0,
    "predict_steps": 1,
    "res_dropout": 0.1,
    "eval_batch_size": 128,
    "predict_batch_size": 1,
    "iterations": 2500,
    "n_embd": 2048,
    "datasets": [["your_dataset_name", 25, "documents_random", 1.0]],
    "model_path": "gs://neo-models/GPT3_XL",
    "n_ctx": 2048,
    "n_layer": 24,
    "scale_by_depth": true,
    "scale_by_in": false,
    "attention_types" :  [[["global"],24]],
    "mesh_shape": "x:128,y:2",
    "layout": "batch:x,memory_length:y,embd:y",
    "activation_function": "gelu",
    "recompute_grad": true,
    "gradient_clipping": 1.0,
    "tokens_per_mb_per_replica": 2048
}
```


## 6. Run Training

```
python3 main.py --model <your_config_name> --steps_per_checkpoint <n> --tpu <tpu-name>
```

- `tpu`: Name of the TPU to use.
- `steps_per_checkpoint`: The frequency in steps at which to save checkpoints.
- `--auto_layout` and `--auto_layout_and_mesh_shape` (Optional): Disable training and instead auto generate a memory efficient `layout` (and `mesh_shape`)
- `gpu_ids`: if training using GPUs, omit the `tpu` flag and pass in the ids of your gpus. In the example below, we train on 3 GPUs, specifying their device ids delimited by spaces:

```
python3 main.py --model <your_config_name> --steps_per_checkpoint <n> --gpu_ids <device:GPU:0 device:GPU:1>
```

# Available Configs

We have several model sizes available, but some of our configs require large TPUs and will need tweaking to run on smaller machines, or GPUs. Below is a short guide to each model in the configs directory:

TODO

# Extra Features: 

## Training (with Sacred)

[Sacred](https://github.com/IDSIA/sacred) helps track experiments and is much nicer to work with than tensorboard.

To setup:

1. Install Docker and Docker-compose

2. Run `docker-compose up`

To use: 

1. Ensure model_dir doesn't have any metric logs in it (it trips up the metric stuff for tensorboard, which assumes that it's a continuation of the existing run). You can use `gsutil rm -r ...` to delete model dir

2. Run `python3 run_experiment.py --tpu sometpuhere --model someconfig.json` Options are the same as `main.py`. 

3. You can go to http://server_ip_goes_here:8081/ to see the Omniboard overview. If you prefer to see a tensorboard, the script also spins one up and automatically assigns it a port. The script should print out the tensorboard port near the top of the log. 

## Peeking at a Dataset

If you are ever confused by the dataset of a particular config file, you can easily check the minimum and maximum token ids with a single command. This is useful for making sure that the vocabulary size of the model is at least as large as the maximum token id. Tensorflow will not error if you try to gather on a matrix with out of bounds indices, so you need to make sure your vocabulary size is sufficiently large.

```bash
python main --model {config_name} --check_dataset
```

## Masked Language Modeling

In addition to being able to train large GPT's, this repository also allows you to easily do masked language modeling (BERT, RoBERTa). In order to do so, you must follow two additional steps.

1. When tokenizing your dataset, you must reserve a special id for the `[mask]` token.

2. In the configs, you will have to define two additional fields

```python
"mlm_training": true,                           # must be set to true
"mlm_mask_id": <mask id>                        # the mask id that you reserved from above
```

That's all you need to train a model with the MLM objective, good for any type of data that you have encoded properly. If you would like to tweak the other related hyperparameters, please continue reading.

```python
"mlm_cls_token_id": <cls token id>,                # auto append specified CLS token id on the left
"mlm_mask_prob": 0.15,                             # the probability of masking a token, defaults to 15%
"mlm_same_token_prob": 0.10,                       # probability of keeping the token the same, defaults to 10%
"mlm_random_token_prob": 0.10,                     # probability of tokens that are replaced with random tokens, 10% was recommended by the BERT paper
"mlm_mask_ignore_ids": [<cls token>, <sep token>]  # ignore masking other special tokens, if any
```

## Parameter Reference

Pick a valid config from `/configs` and tweak the parameters as needed:

- `n_heads`: The number of attention heads.
- `n_embd`: Size of the hidden layers, must be divisible by `n_heads`.
- `n_vocab`: Vocabulary size.
- `embed_dropout`, `res_dropout`, `attn_dropout`: Dropout probability for word embedding/residuals/attention
- `lr`: Learning rate
- `warmup_steps`: Number of steps before full learning rate is reached (linear ramp from `0` to `lr`).
- `lr_decay`: `cosine` or `linear`.
- `opt_name`: `adam` or `adafactor`.
- `beta1`, `beta2` and `epsilon`: `adam` optimizer params.
- `beta1`, `ada_epsilon1` and `ada_epsilon2`: `adafactor` optimizer params.
- `weight_decay`: Weight decay parameter, if not present no weight decay is used (the weight decay fix for Adam is used) (default: 0.01) (optional).
- `train_batch_size`: Batch size during training.
- `train_steps`: Number of training steps (batches), set to roughly ~1 epoch for now (total number of tokens in your dataset / number of tokens per batch (= `train_batch_size` / `n_ctx`)).
- `eval_steps`: Number of steps to run for each evaluation. Set to `0` for no eval. i.e After every checkpoint, the model is tested for `eval_steps`
- `iterations`: Number of steps queued to the TPU, must be smaller than `steps_per_checkpoint`. (default: 500)
- `datasets`: List of tfrecords datasets to use. Each dataset is a list with the following parameters: `[train glob , eval glob, stitch, sampling_mode, weight]`. So for example for a single dataset (note the double list): `[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]`
    + `dataset_id`: The name of a dataset configuration file in `./configs/dataset_configs`
    + `stitch`: If `sampling_mode` `random_sample` is used, the input pipeline samples this amount of texts into one to sample from. You must select stitch so that `stitch * minimum_document_length >= n_ctx`
    + `sampling_mode`: `chunks` (tfrecords are preprocessed into the correct length and are read sequentially) or `documents_random` (`stitch` amount of documents are concatenated and then a `n_ctx` chunk is randomly subsampled)
    + `weights`: How much relative weight this dataset should have compared to others
- `model`: Which model to train. Currently only `GPT` is supported, and it defaults to this if not present.
- `model_path`: Google storage bucket location (or local path, if using GPUs) to save model checkpoints and logs.
- `n_ctx`: Size of context window. Default is 2048
- `n_layer`: Number of layers (blocks) in the model.
- `scale_by_depth`: If true, the weight initialization of layers are scaled by their depth as in the GPT2 paper.
- `scale_by_in`: If true, the weight initialization of layers are scaled by their number of inputs as in the GPT2 paper.
- `mesh_shape`: A Mesh is an n-dimensional array of processors with named dimensions used for parallelism in the mesh-tensorflow library. Each Tensor is split evenly across mesh dimensions according to the layout (see below). The 'mesh_shape' is the shape of this array, and must be equal to the number of processors. e.g., for a v3-128 TPU "mesh_shape": “x:16,y:8”.
- `layout`: A Tensor is laid out on its mesh with one slice on each processor. A Tensor "layout", is an injective partial map specifying which dimensions of the tensor are (evenly) split across which dimensions of the mesh. No dimension of a tensor may be split across two dimensions of its mesh and no two dimensions of a tensor may be split across the same dimension of its mesh. The user defines a global set of layout rules in the form of (tensor-dimension-name, mesh-dimension-name) pairs. A dimension of a tensor is split across a dimension of its mesh if there is a matching rule, e.g. (for the above example mesh_shape: "layout":"batch:x,heads:y"
- `activation_function`: `selu` (self normalizing) or `gelu` (used by OA), activation function used in feed-forward passes. (default: gelu)
- `attention_types`: the type of attention for each layer in a list of the following format [[["attention_type"], n_layers]]. e.g. for a 12 layer net [[["global"], 12]] or [[["local"], 10], [["global"], 2]].
    + Choose from: `linear`, `global`, `local` or `none`. We have found a 50/50 mix of `global` and `linear` to work well. `none` allows you to create feed-forward only layers for more efficient [PAR Transformer](https://arxiv.org/abs/2009.04534) models.
- `precision`: `float32` or `bfloat16`.
- `tokens_per_mb_per_replica`: If not None, will split the batch up into smaller microbatches containing `tokens_per_mb_per_replica` tokens to avoid OOMs. Gradients are accumulated locally and reduced once. IMPORTANT: mb refers to *minibatch* not megabyte here. 

**Mixture of Experts**

- `moe_layers`: A list of layer numbers to append a [mixture of experts](https://arxiv.org/abs/1701.06538) layer onto. E.G: `[2,4,6,8,10,12]`.
We have experimentally found a moe layer for every two self-attention layers to work well.
-  `moe_params`: a dictionary of additional kwargs to pass in to the moe layer. E.G
    `{"moe_dropout_rate": 0.0 }`
    
**Experimental features** 

- `axial_pos_emb_`: If true, uses [axial positional embedding](https://arxiv.org/abs/1912.12180. 
- `mlp_glu`: If true, uses a gated linear unit variant of feed forward layers.
- `scalenorm`: If true, uses scalenorm instead of layernorm.
- `rezero`: If true, uses [rezero](https://www.groundai.com/project/rezero-is-all-you-need-fast-convergence-at-large-depth/1) instead of layernorm.
- `num_mem_kv`: adds memory / key values from the [all-attention paper](https://arxiv.org/pdf/1907.01470.pdf). Param is an int with the number of desired mem/key values.
- `macaron`: if true - uses a [macaron transformer](https://arxiv.org/pdf/1906.02762.pdf) for each layer block.

## TODO: 

- [x] finalize documentation
- [ ] update configs

## Citing GPT-Neo

If you have found GPT-Neo helpful in your work, you can cite this repository as

```
@software{gpt-neo,
  author       = {Black, Sid and
                  Gao, Leo and
                  Wang, Phil and
                  Leahy, Connor and
                  Biderman, Stella},
  title        = {{GPT-Neo: Large Scale Autoregressive Language 
                   Modeling with Mesh-Tensorflow}},
  month        = mar,
  year         = 2021,
  note         = {{If you use this software, please cite it using 
                   these metadata.}},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.5297715},
  url          = {https://doi.org/10.5281/zenodo.5297715}
}

```
The version number should be replaced with the version number you are using, and the year corresponds to the project's open-source release.

If you are specifically interested in citing the GPT-Neo models trained on [the Pile](https://arxiv.org/abs/2101.00027), we would appreciate also citing
```
@article{gao2020pile,
  title={The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and others},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```

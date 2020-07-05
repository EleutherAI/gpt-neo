# GPT Neo

1T or bust my dudes

An implementation of training for [GPT2](https://openai.com/blog/better-language-models/)/[GPT3](https://arxiv.org/abs/2005.14165)-like models. Supports both GPUs and TPUs. The dataset scripts are a bit hacky and will probably need to be adapted to your needs. 
## Requirements
For GPUs:

`pip3 install tensorflow-gpu==1.15.2 regex`

For TPUs:

`pip3 install tensorflow==1.15.2 regex google-api-python-client oauth2client`

For downloading the models:

`pip3 install requests tqdm`

For generating the dataset (in addition to Tensorflow):

`pip3 install ftfy tqdm newspaper3k`

## Downloading Pretrained Models
If you want to use my models, I currently have "117M", "PrettyBig" and "1.5B" to offer. 117M was trained on a single v2 TPU for a week (probably less than the original OpenAI model), PrettyBig is slightly bigger than 345M and was trained on a v2-256 pod for a week. ~~I was originally also planning to release my version of the 1.5B model, but have decided against it. You can read about my reasoning [here](https://medium.com/@NPCollapse/the-hacker-learns-to-trust-62f3c1490f51).~~ Since OpenAI has released their model, I have now also released my (inferior) 1.5B model, which was trained on a v3-512 pod for a week.

`python3 download_model.py PrettyBig`

This will create two directories, one named as the model and another named "encoder". Change the "model_dir" and "encoder_path" parameters in the .json corresponding to your model to point to these paths, respectively.

If you only want the encoder, use:

`python3 download_model.py encoder`

## Generating Text
To predict you can either pass the prompt directly in the command line, or have it read from a file. (This is useful for prompts that include newlines) Text is output to the console and the file specified in the "predict_path" parameter. You need a model checkpoint and a copy of the BPE encoder at an accessible location for this to work. (Change the "model_dir" and "encoder_path" parameters in the .json)

From command line:

`python3 main.py --model Your-Model.json [--top_k Top-K-Truncation] --predict_text "Hello there! My name is"`

From file:

`python3 main.py --model Your-Model.json [--top_k Top-K-Truncation] --predict_file input.txt`

The optional top_k parameter causes the model to only consider the top k most likely tokens at each step. Setting this around 40 tends to create better results, but with less variety. 

Prediction on TPUs is not supported.


## Training
To train a model, define its parameters in a .json file (see examples) and then simply call

`python3 main.py --model Your-Model.json [--tpu Your-TPU-Name]`

Using a TPU is optional, it runs fine on GPUs without modification. (Note: Evaluation doesn't work on TPU pods and must be commented out) 

This assumes you have a version of the openwebtext corpus stored in an accessible location. If you don't, see below how to generate your own version.



## Generating the Dataset
GPT2 is trained on the webtext corpus, which is basically all websites linked to from Reddit with at least 3 Karma. Since the database is huge and contains a lot of copyrighted material, I can't provide a download here. Instead, I'll describe how I got it. Be aware it cost me around ~500€ in cloud compute resources to download and process the whole thing, but I'm not claiming I was optimally efficient. 
1. Use the download script from [here](https://github.com/jcpeterson/openwebtext) to download the archives (I used the prefiltered URLs file)
2. Use *datasets/openwebtext/
run_newspaper_extract.py* to extract the text
3. Once you have the raw .txt files use *datasets/openwebtext/
create_tfrecords.py* to encode them into .tfrecords files (Requires a copy of the encoder, see Downloading Pretrained Models)
4. Place the .tfrecords files into an accessible folder or Google Storage bucket (Placing in a Google Storage bucket is mandatory if you're using TPUs)
5. Change the "data_path" parameter in your .json to point to where your .tfrecords files are located and, if necessary, adapt the functions in *inputs.py* to open the correct filenames, in case you changed them

## Using Your Own Data
You can also use your own text files as training data, but you'll need to modify some code by hand.
1. Modify the parameters in *datasets/openwebtext/create_tfrecords.py*:

```python
base_dir = "/home/connor/my_text_dir" # Path to where your .txt files are located
files_per = 175000 # How many txt files to put in one tfrecord, not too important
name = "my-custom-data" # Name of output files will be name_i.tfrecords where i is the number of the file
output_dir = "/home/connor/output" # Where to place the .tfrecords files
log_dir = "logs" # Some logs will be placed here to support restarting if the encoding is interrupted
files = glob.glob(os.path.join(base_dir, "**/*.txt")) # This needs to result in a list of paths to all of your txt files
processes = 64 # Number of encoding processes to run
encoder_path = "/home/connor/encoder" # Path to encoder files
minimum_size = 128 # The minimum length (in BPE tokens) a file is allowed to have, otherwise it is discarded.
```
2. Run the script. This will result in a bunch of name_i.tfrecords files. Put these somewhere accessible (must be in a Google Storage bucket if you're using TPUs).
3. Create a new input function in *inputs.py*. Any input function should have the signature *function_name(params, eval=False)*. The **stitch** value controls how many texts are concatenated so that you never end up with a sample that is too small. It should be: **ceil((n_ctx+1) / minimum_size)** So for example, if my minimum size is 128 and my n_ctx is 1024, stitch should be 9.

```python
def my_input(params, eval=False):
    if not eval:
        numbers = [0, 3, 4, 5, 6, 7, 8, 9] # A random subset of files for train
    else:
        numbers = [1, 2] # Random subset for eval
    files = [os.path.join(params["data_path"], "my-custom-data_{}.tfrecords".format(str(i))) for i in numbers] # Generates the list of files

    return bpe_text(params["batch_size"], files, amount=params["n_ctx"], iterations=params["iterations"], stitch=9, batch=True)
```
4. Register your new input in *main.py*.

```python
inputs = {
    "openwebtext": openwebtext, # Standard OpenWebtext input
    "openwebtext_longbiased": openwebtext_longbiased, # OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": openwebtext_long, # Openwebtext that only shows long examples
    "my_input": my_input,
}
```
5. Set your .json to use the new input.
```python
[...]
    "iterations": 500,
    "n_embd": 768,
    "input": "my_input",
    "model": "GPT2",
[...]
```
6. You're done. The input described here should be as close to GPT2 as possible and run perfectly on TPUs.

## Explanation of Parameters
Because passing two dozen parameters over the command line would be tedious, you pass all the model parameters in a .json file. Note that any paths also support Google Storage paths and *must* be gs:// paths if you're running on TPUs.

Values you'll definitely want to change:
* **model_path**: Where to save and load checkpoints from
* **data_path**: Where your .tfrecords files are located
* **encoder_path**: Path to the BPE encoder files. To get this, use the download_model.py script to download any model (or just the encoder). You will get a folder called "encoder". This is what you want this to point to (only required for prediction)

Values you'll probably want to change:
* **train_batch_size**: Batch size during training phase
* **eval_batch_size**: Batch size during evaluation
* **predict_batch_size**: Batch size during prediction
* **predict_path**: Where to save predictions (point this to a text file to append to)

Model parameters:
* **model**: A string that refers to which model to use. This should always just be "GPT2" (no other models are implemented here)
* **n_ctx**: Number of tokens the model looks at (default: 1024)
* **n_vocab**: Size of vocabulary (default: 50257)
* **n_embd**: Dimension of embedding layers
* **n_layer**: Number of layers in the model
* **n_head**: Number of attention heads (default: n_embd / 64)
* **scale_by_depth**: Whether or not to scale init by the number of layers (Default: true)
* **scale_by_in**: Whether to scale init by the number of input channels (Default: true)

Training parameters:
* **precision**: Whether to use float32 or bfloat16 variables (use "bfloat16" when training very large models) (optional, defaults to float32)
* **input**: Which input function to use (default: "openwebtext")
* **lr**: Learning rate (default: 0.00025)
* **warmup_steps**: Number of warmup steps. If this is set, a linear warmup + cosine decay schedule is used (default: 2000) (optional)
* **opt_name**: Name of optimizer, currently there are "adam" and "adafactor" (default: "adam")
* **weight_decay**: Weight decay parameter, if not present no weight decay is used (the weight decay fix for Adam is used) (default: 0.01) (optional)
* **beta1**: Adam/Adafactor beta1 parameter (adam default: 0.9, adafactor default: 0.0)
* **beta2**: Adam/Adafactor beta2 parameter (default: 0.98) (optional for adafactor with pow decay type)
* **epsilon**: Adam epsilon parameter (default: 1e-9)
* **decay_type**: Adafactor decay type, either "pow" or "adam" (default: "pow")
* **decay_exponent**: Adafactor pow decay exponent (default: 0.8)
* **train_steps**: Number of training steps to take between evaluations
* **eval_steps**: Number of steps per evaluation
* **max_steps**: The maximum number of training steps (important for declining lr)
* **iterations**: Number of iterations to perform on TPUs (Default: 100) (Only required for TPUs)
* **embed_dropout**: Dropout chance on the word embedding, set to 0 to disable (default: 0.1)
* **attn_dropout**: Dropout chance on attention layers, set to 0 to disable (default: 0.1)
* **res_dropout**: Dropout chance on residual connections, set to 0 to disable (default: 0.1)

* **local**: Whether or not to use local attention (default: false) 
* **fixed_attn_block_size**: Size of the attention blocks, should be a multiple of 128 on TPUs for performance reasons (default: 128)
* **layer_offset**: (default: 16)
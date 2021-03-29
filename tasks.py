import os.path
import requests
import ftfy
from data.encoders import fetch_encoder, encode
import tensorflow as tf
import numpy as np
from functools import partial
import json

lambada_src_uri = 'http://eaidata.bmk.sh/data/lambada_test.jsonl'
normalization = 'NFKC'


# Note: this task is called "lambada" but it really refers to OpenAI's version
# of the task, which actually differs in some ways from the task described in
# the original paper. So, strictly speaking, accuracy values from this task
# should not be compared to accuracy values from the original lambada task.
# For more information, see
#   https://github.com/openai/gpt-2/issues/131

def lambada_create_tokens_data(params, path):
    with open(path, 'w') as f:
        req = requests.get(lambada_src_uri)
        req.raise_for_status()
        jsons = [json.loads(l) for l in req.iter_lines()]
        texts = [ftfy.fix_text(j['text'], normalization=normalization) for j in jsons]
        enc = fetch_encoder(params)
        arrays = [encode(enc, t) for t in texts]
        json.dump(arrays, f)
        return arrays


def lambada_read_or_create_tokens_data(params, path):
    # if you tell me where the file should go, i will helpfully create it for you
    if not os.path.exists(path):
        return lambada_create_tokens_data(params, path)
    with open(path) as f:
        return json.load(f)


def bin_pack(params, tokens_data):
    eos_token = params['eos_id']
    n_ctx = params['n_ctx']
    dummy_token = 1
    pad_batch_size = params['eval_batch_size']
    bins = []
    for a in tokens_data:
        if len(bins) == 0 or len(bins[-1]) + len(a) + 1 > n_ctx:
            bins.append([])
        bins[-1] += a
        bins[-1].append(eos_token)
    while len(bins) % pad_batch_size != 0:
        bins.append([])
    bins_array = np.full((len(bins), n_ctx), dummy_token, dtype=np.uint16)
    for i, b in enumerate(bins):
        bins_array[i, 0:len(b)] = b
    return bins_array


def lambada_init(params):
    ds_configs = params['dataset_configs']
    l = [
        ds_configs[ds_id].get('lambada_tokens_path', "./lambada.json")
        for ds_id, _, _, _ in params['datasets']
    ]
    assert len(l) > 0, 'lambada_tokens_path not found in the dataset config'
    lt_path = l[0]
    assert lt_path.endswith('.json'), 'lambada_tokens_path must have extension json'

    tokens_data = lambada_read_or_create_tokens_data(params, lt_path)
    bins_array = bin_pack(params, tokens_data)
    params['lambada_tokens_path'] = lt_path
    params['lambada_n_steps'] = len(bins_array) // params['eval_batch_size']


def lambada_get_task_info(params):
    return {
        'n_steps': params['lambada_n_steps'],
    }


# The LAMBADA evaluation code looks at the logits of each position just before an eos_token
def lambada_input(params):
    eos_token = 50256 if params['n_vocab'] >= 50257 else 0
    n_ctx = params['n_ctx']
    lt_path = params['lambada_tokens_path']
    tokens_data = lambada_read_or_create_tokens_data(params, lt_path)
    bins_array = bin_pack(params, tokens_data)
    dataset = tf.data.Dataset.from_tensor_slices(bins_array)

    def _get_output(bin):
        bin = tf.cast(bin, dtype=tf.int32)
        indexes = tf.range(n_ctx)
        results = tf.gather(bin, (indexes + 1) % n_ctx)
        eos_next_positions = tf.math.equal(tf.gather(bin, (indexes + 2) % n_ctx), eos_token)
        output = tf.where(eos_next_positions, results, tf.constant(eos_token, shape=[n_ctx]))
        bin = tf.reshape(bin, [n_ctx])
        bin = tf.cast(bin, dtype=tf.int32)
        output = tf.reshape(output, [n_ctx])
        output = tf.cast(output, dtype=tf.int32)
        return bin, output

    dataset = dataset.map(_get_output)
    dataset = dataset.batch(params['eval_batch_size'], drop_remainder=True)
    dataset = dataset.repeat()
    return dataset


def save_eval_results(task, eval_results, current_step):

    def as_python(x):
        if isinstance(x, np.generic):
            return x.item()
        return x

    eval_results = {k: as_python(v) for k, v in eval_results.items()}
    import os
    save_path = f'eval_{current_step}.jsonl'
    if os.path.isfile(save_path):
        while True:
            num = 0
            new_save_path = f'{save_path}_{num}'
            if not os.path.isfile(new_save_path):
                save_path = new_save_path
                break

    with open(save_path, 'a') as fh:
        json.dump({'task': task, 'current_step': current_step, **eval_results}, fh)
        fh.write('\n')


def run_eval(params, estimator, logger, input_fn):
    logger.info("Running evaluation...")
    eval_results = estimator.evaluate(
        input_fn=partial(input_fn, eval=True),
        steps=params["eval_steps"])
    logger.info(f"Eval results: {eval_results}")
    save_eval_results('validation', eval_results)


def run_eval_tasks(params, eval_task_estimators, eval_tasks, logger, current_step):
    for task in eval_tasks:
        logger.info(f"Starting evaluation task '{task}'")
        task_info = task_descriptors[task]["get_task_info_fn"](params)
        task_estimator = eval_task_estimators[task]
        task_input_fn = task_descriptors[task]["input_fn"]
        eval_results = task_estimator.evaluate(
            input_fn=task_input_fn,
            steps=task_info["n_steps"],
            name=task)
        logger.info(f"Eval task '{task}' results: {eval_results}")
        save_eval_results(task, eval_results, current_step)


task_descriptors = {
    'lambada': {
        'init_fn': lambada_init,
        'get_task_info_fn': lambada_get_task_info,
        'input_fn': lambada_input,
    }
}

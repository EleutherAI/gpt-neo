import json
from pathlib import Path

DATASETS = {}

for path in Path('./dataset_configs').glob('*.json'):
    dataset_id = path.stem
    DATASETS[dataset_id] = json.loads(path.read_text())

def fetch_model_params(model):
    model_path = model if model.endswith('.json') else 'configs/{}.json'.format(model)
    with open(model_path, 'r') as f:
        params = json.loads(f.read())

    def parse_dataset(entry):
      config = {
          'n_vocab': 50257,
          'stitch': 3,
          'mode': 'documents_random',
          'weight': 1.0,
      }
      if isinstance(entry, str):
        if '::' in entry:
          config['name'], config['weight'] = entry.split('::')
        else:
          config['name'] = entry
      elif isinstance(entry, list):
        if len(entry) <= 1:
          config['name'], = entry
        elif len(entry) <= 2:
          config['name'], config['weight'] = entry
        elif len(entry) <= 3:
          config['name'], config['mode'], config['weight'] = entry
        elif len(entry) <= 4:
          config['name'], config['stitch'], config['mode'], config['weight'] = entry
        elif len(entry) <= 5:
          config['name'], config['eval_path'], config['stitch'], config['mode'], config['weight'] = entry
        else:
          assert len(entry) <= 5
      config['stitch'] = int(config['stitch'])
      config['weight'] = float(config['weight'])
      # if the name contains a star, assume it's a glob path
      if '*' in config['name']:
        config['path'] = config['name']
        config['eval_path'] = config['name']
      else:
        # otherwise the dataset must exist as a dataset config
        assert config['name'] in DATASETS, f'dataset {config["name"]} was not found under dataset_configs/ folder. please follow the example.json in that folder'
        dataset = DATASETS[config['name']]
        assert config['n_vocab'] >= dataset['n_vocab'], f"the embedding table size {config['n_vocab']} must be greater or equal to the vocab size used to encode the dataset {dataset_id} ({dataset['n_vocab']})"
        config.update(dataset)
      return config

    datasets = params.get('datasets')
    if isinstance(datasets, str):
      datasets = datasets.split(',')

    if datasets is not None:
      params["datasets"] = []
      for entry in datasets:
        dataset = parse_dataset(entry)
        params["datasets"].append(dataset)

    return params

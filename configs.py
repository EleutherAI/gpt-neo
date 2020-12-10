import json
from pathlib import Path
from collections import defaultdict

DATASETS = {}

for path in Path("configs/dataset_configs").glob("*.json"):
    dataset_id = path.stem
    DATASETS[dataset_id] = json.loads(path.read_text())


def fetch_model_params(model):
    model_path = model if model.endswith(".json") else f"configs/{model}.json"
    with open(model_path) as f:
        params = json.load(f)

    dataset_ids = []
    for d in params.get("datasets"):
        if isinstance(d, list):
            dataset_ids.append(d[0])
        else:
            dataset_ids.append(d)
    no_datasets = params.get("no_dataset", False)
    assert no_datasets or len(dataset_ids) > 0, "You must specify at least one dataset id in the model config"

    datasets = {}
    last_dataset = None
    for dataset_id in dataset_ids:
        assert dataset_id in DATASETS, f"Dataset '{dataset_id}' was not found under dataset_configs/ folder. Please follow the example.json in that folder."
        dataset = DATASETS[dataset_id]
        assert params["n_vocab"] >= dataset["n_vocab"], f"The embedding table size '{params['n_vocab']}' must be greater or equal to the vocab size used to encode the dataset '{dataset_id}' ({dataset['n_vocab']})"
        datasets[dataset_id] = dataset
        last_dataset = dataset

    if last_dataset is not None:
        params["padding_id"] = last_dataset.get("padding_id", 0)
        params["eos_id"] = last_dataset.get("eos_id", 1)

    params["dataset_configs"] = datasets

    # Set some other parameter defaults
    params["mlm_training"] = params.get("mlm_training") == True
    params["causal"] = not params["mlm_training"]

    # Set all other parameter values to default to None
    params = defaultdict(lambda: None, params)
    return params

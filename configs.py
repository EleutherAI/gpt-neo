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

    dataset_ids = [d[0] for d in params.get("datasets", [])]
    no_datasets = params.get("no_dataset", False)

    datasets = {}

    params["dataset_configs"] = datasets

    # Set some other parameter defaults
    params["mlm_training"] = params.get("mlm_training") == True
    params["causal"] = not params["mlm_training"]

    # Set all other parameter values to default to None
    params = defaultdict(lambda: None, params)
    return params

from typing import Dict
from . import gpt2

def from_config(config: Dict):
    if not config['type'] == 'GPT2':
        raise ValueError('only GPT2 is supported')
    config.pop('type')

    return gpt2.from_config(config) 


    # add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    # params["auto_layout"] = args.auto_layout
    # params["auto_layout_and_mesh_shape"] = args.auto_layout_and_mesh_shape
    
    # expand attention types param
    #params["attention_types"] = expand_attention_types_params(params["attention_types"])
    # assert len(params["attention_types"]) == params["n_layer"]  # assert that the length of expanded list = num layers
    # logging.info('params = {}', params)

    #TODO: we would like this to be as small as possible,
    # but if we're splitting by batch, a value < the dimensions batch is divided over will error.
    # can we change the mesh layout so batch will not be split at prediction time?
    #params["predict_batch_size"] = params.get("predict_batch_size", 1) # Default to 1
    #params["predict"] = args.predict
    return params

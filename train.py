import collections
from typing import Any, Dict, Optional

import mesh_tensorflow as mtf
import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import logging
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

import datasets
import models
from devices import tpu


@dataclass
class ClusterConfig:
    num_cores: int
    use_tpu: bool

@dataclass
class ModelConfig:
    activation_function: str

@dataclass
class TrainerConfig:
    cluster: ClusterConfig
    infeed: Dict
    model: Dict
    trainer: Dict
    other: Any
    regularization: Dict
    tpu: Optional[tpu.TPUConfig]

class Trainer:

    def __init__(self, model, dataset, config: TrainerConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = 'tpu' if config.tpu else 'cpu'

    @classmethod
    def from_config(cls, config:TrainerConfig):
        model = models.from_config(config.model)
        dataset = datasets.from_config(config.datasets)

        return cls(model, dataset, config)

    def save_checkpoint(self):
        state = self.model.state_dict()
        logging.info("saving model checkpoint to %s", self.config.ckpt_path)
        self.save(state, self.config.ckpt_path)
        logging.info("saved model checkpoint to %s", self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        # raw_model = model.module if hasattr(self.model, "module") else model
        # optimizer = raw_model.configure_optimizers(config)

        #def run_epoch(split):
            #is_train = split == 'train'
            #model.train(is_train)
            # data = self.dataset
            # loader = DataLoader(data, shuffle=True, pin_memory=True,
            #                     batch_size=config.batch_size,
            #                     num_workers=config.num_workers)

            # losses = []
            # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # for it, (x, y) in pbar:

            #     # place data on the correct device
            #     x = x.to(self.device)
            #     y = y.to(self.device)

            #     # forward the model
            #     with torch.set_grad_enabled(is_train):
            #         logits, loss = model(x, y)
            #         loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            #         losses.append(loss.item())

            #     if is_train:

            #         # backprop and update the parameters
            #         model.zero_grad()
            #         loss.backward()
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            #         optimizer.step()

            #         # decay the learning rate based on our progress
            #         if config.lr_decay:
            #             self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
            #             if self.tokens < config.warmup_tokens:
            #                 # linear warmup
            #                 lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
            #             else:
            #                 # cosine learning rate decay
            #                 progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
            #                 lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            #             lr = config.learning_rate * lr_mult
            #             for param_group in optimizer.param_groups:
            #                 param_group['lr'] = lr
            #         else:
            #             lr = config.learning_rate

            #         # report progress
            #         pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            # if not is_train:
            #     test_loss = float(np.mean(losses))
            #     logger.info("test loss: %f", test_loss)
            #     return test_loss

        # best_loss = float('inf')
        # self.tokens = 0 # counter used for learning rate decay
        # for epoch in range(config.max_epochs):

        #     run_epoch('train')
        #     if self.test_dataset is not None:
        #         test_loss = run_epoch('test')

        #     # supports early stopping based on the test loss, or just save always if no test set is provided
        #     good_model = self.test_dataset is None or test_loss < best_loss
        #     if self.config.ckpt_path is not None and good_model:
        #         best_loss = test_loss
        #         self.save_checkpoint()

def load_trainer_config(location):

    with tf.io.gfile.GFile(location) as fd: 
        params = json.loads(fd.read())

    n_vocab = params['n_vocab']
    params['datasets'] = []
    datasets = params.get('datasets', [])
    
    for d in datasets:
        with tf.io.gfile.GFile(d) as fd: 
            dataset_config = json.load(fd.read())
            params['datasets'].append(dataset_config)

    return params



def load_trainer(args) -> Trainer:
    with tf.io.gfile.GFile(args.config) as fd:
        params = json.load(fd)

    cfg = TrainerConfig(**params)
    
    json.dump(params, sys.stdout, indent=2)

    if args.test:
        # rewire to use testing related functions if --test is on
        return Trainer(
            name='test',
            config=cfg,
            model_fn=lambda *args: None,
            input_fn=load_input_fn(cfg.infeed),
            # pred_input_fn=test_pred_input,
            handle_prediction_output_fn=test_handle_pred_output
        )
    
    if args.model == '':
        raise ValueError('Model must be set')
    
    # params = load_trainer_config(args.model)

    # Fetch encoder per params
    encoder = fetch_encoder(params)
   
    # model.pred_input_fn = partial(pred_input_fn, enc = encoder)

    return Trainer(
        name=args.model,
        input_fn=generic_text,
        config=cfg,
        # pred_input_fn=pred_input,
        handle_prediction_output_fn=handle_pred_output,
    )

def check_dataset(trainer, args):
    sample_size = 10
    sampled_files = random.choices(trainer.config.infeed.dataset.src, k=sample_size)
    with v1.Session(graph=tf.Graph()) as sess:
        ds = trainer.input_fn(trainer.config.infeed)

        it = ds.make_one_shot_iterator()
        example = it.get_next()
        
        for _ in range(42):
            try:
                result = sess.run(example) #, max_id_tf, min_id_tf])
                # pt = PreProcessedTextLine(
                #     id = result['id'],
                #     content=result['content'],
                #     target=result['target'],
                #     offset_start=result['offset_start'],
                #     offset_end=result['offset_end'],
                # )

                # ids = tokenizer.decode(result['target'])

                # logging.info('gold text:    %r', pt.content.decode('utf-8'))
                # logging.info('decoded:       %r', ids),
                # logging.info('tokenization: %s', [pt.content.decode('utf-8')[slice(int(start), int(end))] for start,end in zip(pt.offset_start, pt.offset_end)])
                # logging.info('-' * 10)
                print(result)
            except tf.errors.OutOfRangeError:
                break
        
def parse_args(args, parser=None):
    # Parse command line arguments
    parser = parser if parser else argparse_flags.ArgumentParser()
    parser.add_argument('runspec', type=str, help="the json file specifiing the configuration for this run") # Name of TPU to train on, if any
    return parser.parse_args(args[1:])

def main(args):
    logging.info('starting train process')

    trainer = load_trainer(args)

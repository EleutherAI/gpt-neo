import optimizers

def from_config(config):
    return config

class AbstractModel:

    def __init__(self):
        super().__init__()

    def configure_optimizers(self, config: optimizers.OptimizerConfig):
        pass
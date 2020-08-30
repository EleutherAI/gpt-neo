import optimizers

def load_config():
    pass


class AbstractModel:

    def __init__(self):
        super().__init__()

    def configure_optimizers(self, config: optimizers.OptimizerConfig):
        pass
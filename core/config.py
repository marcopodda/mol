from argparse import Namespace
from core.utils.serialization import load_yaml, save_yaml


DEFAULTS = {
    "device": "cuda",
    "batch_size": 32,
    "pretrain_batch_size": 32,
    "first_pretrain_epochs": 50,
    "second_pretrain_epochs": 30,
    "num_samples": 1000,
    "max_length": 11,
    "gnn_num_layers": 3,
    "gnn_dim_hidden": 32,
    "gnn_dim_embed": 32,
    "num_epochs": 2000,
    "lr": 0.001,
    "vae_dim_hidden": 32,
    "vae_dim_latent": 32,
    "rnn_num_layers": 1
}


class Config(Namespace):
    @classmethod
    def default_config(cls):
        config = cls()
        config.save("config.yml")

    @classmethod
    def from_file(cls, path):
        config = load_yaml(path)
        return cls(**config)

    def __init__(self, **config):
        for key in DEFAULTS:
            setattr(self, key, DEFAULTS[key])

        for key in config:
            setattr(self, key, config[key])

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def save(self, path):
        save_yaml(self.__dict__, path)


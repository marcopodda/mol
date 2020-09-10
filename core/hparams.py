from argparse import Namespace
from pathlib import Path
from core.utils.serialization import load_yaml


class HParams(Namespace):
    @classmethod
    def from_file(cls, path):
        hparams_dict = load_yaml(Path(path))
        return cls(**hparams_dict)

    @classmethod
    def load(cls, hparams):
        if isinstance(hparams, dict):
            hparams = cls(**hparams)
        return hparams

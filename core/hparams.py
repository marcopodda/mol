from argparse import Namespace
from pathlib import Path
from core.utils.serialization import load_yaml


class HParams(Namespace):
    @classmethod
    def from_file(cls, path_or_obj):
        if isinstance(path_or_obj, dict) or isinstance(path_or_obj, cls):
            return cls.load(path_or_obj)

        hparams_dict = load_yaml(Path(path_or_obj))
        return cls(**hparams_dict)

    @classmethod
    def load(cls, hparams):
        if isinstance(hparams, dict):
            hparams = cls(**hparams)
        return hparams

from pathlib import Path

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.hparams import HParams
from core.datasets.datasets import TrainDataset, EvalDataset
from core.datasets.loaders import EvalDataLoader
from core.utils.serialization import load_yaml, save_yaml
from core.utils.os import get_or_create_dir
from core.utils.scoring import score
from layers.model import Model
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks import TRANSLATION, PRETRAINING
from tasks.pretraining import PretrainingWrapper
from tasks.runner import TaskRunner


class TranslationTrainDataset(TrainDataset):
    corrupt = False

    def __len__(self):
        return self.data[self.data.is_x==True].shape[0]

    def get_target_data(self, index):
        smiles = self.data.iloc[index].target
        mol_data = self.data[self.data.smiles==smiles].iloc[0]
        data = self._to_data(mol_data.frags, is_target=True, add_noise=False)
        fingerprint = self._get_fingerprint(mol_data.smiles, add_noise=False)
        return data, fingerprint


class TranslationWrapper(Wrapper):
    pretrain = False
    dataset_class = TranslationTrainDataset


class TranslationSampler(Sampler):
    def prepare_data(self):
        indices = self.dataset.data[self.dataset.data.is_val==True].index.tolist()
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.translate_batch_size, shuffle=False)


class TranslationTaskRunner(TaskRunner):
    dataset_class = TranslationTrainDataset
    wrapper_class = TranslationWrapper
    sampler_class = TranslationSampler

    @classmethod
    def load(cls, root_dir):
        root_dir = Path(root_dir)
        config = load_yaml(root_dir / "config.yml")
        return cls(
            task=config["task"],
            exp_name=config["exp_name"],
            root_dir=config["root_dir"],
            pretrain_path=config["pretrain_path"],
            dataset_name=config["dataset_name"],
            hparams=HParams.from_file(config["hparams"]),
            gpu=config["gpu"],
            debug=config["debug"])

    @classmethod
    def from_args(cls, args):
        return cls(
            task=args.command,
            exp_name=args.exp_name,
            root_dir=args.root_dir,
            pretrain_path=args.pretrain_path,
            dataset_name=args.dataset_name,
            hparams=HParams.from_file(args.hparams_file),
            gpu=args.gpu if torch.cuda.is_available() else None,
            debug=args.debug)

    def __init__(self, task, exp_name, pretrain_path, root_dir, dataset_name, hparams, gpu, debug):
        self.pretrain_path = None
        if pretrain_path is not None:
            self.pretrain_path = Path(pretrain_path)

        super().__init__(
            task=task,
            exp_name=exp_name,
            root_dir=root_dir,
            dataset_name=dataset_name,
            hparams=hparams,
            gpu=gpu,
            debug=debug
        )

    def post_init_wrapper(self, wrapper):
        if self.pretrain_path is not None:
            pretrain_ckpt_dir = self.pretrain_path / "checkpoints"
            pretrain_ckpt_path = sorted(pretrain_ckpt_dir.glob("*.ckpt"))[-1]
            pretrainer = PretrainingWrapper.load_from_checkpoint(
                pretrain_ckpt_path.as_posix(),
                root_dir=self.pretrain_path.parent.parent,
                dataset_name=self.pretrain_path.parts[-3])
            wrapper.model.embedder = pretrainer.model.embedder
            wrapper.model.autoencoder = pretrainer.model.autoencoder
            # wrapper.model.encoder.gru = pretrainer.model.encoder.gru
            # wrapper.model.decoder.gru = pretrainer.model.decoder.gru
        return wrapper

    def dump(self):
        config = {
            "task": self.task,
            "exp_name": self.exp_name,
            "dataset_name": self.dataset_name,
            "pretrain_path": self.pretrain_path,
            "hparams": self.hparams.__dict__,
            "gpu": self.gpu,
            "debug": self.debug
        }
        save_yaml(config, self.dirs.exp / "config.yml")
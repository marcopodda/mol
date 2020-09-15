import numpy as np
import torch
from pathlib import Path
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.hparams import HParams
from core.datasets.datasets import TrainDataset, EvalDataset
from core.datasets.loaders import TrainDataLoader, EvalDataLoader
from core.utils.serialization import load_yaml, save_yaml
from core.utils.os import get_or_create_dir
from core.utils.scoring import score
from layers.model import Model
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks import PRETRAINING
from tasks.runner import TaskRunner


class PretrainingWrapper(Wrapper):
    pretrain = True


class PretrainingSampler(Sampler):
    dataset_class = EvalDataset

    def prepare_data(self):
        num_samples = min(1000, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.pretrain_batch_size, shuffle=False)


class PretrainingTaskRunner(TaskRunner):
    wrapper_class = PretrainingWrapper
    sampler_class = PretrainingSampler
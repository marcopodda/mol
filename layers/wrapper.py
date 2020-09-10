import gc
import numpy as np
from argparse import Namespace

import torch
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.loaders import TrainDataLoader
from layers.model import Model


class Wrapper(pl.LightningModule):
    dataset_class = None
    pretrain = None

    def __init__(self, hparams, output_dir, name):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dataset = self.dataset_class(hparams, output_dir, name)
        self.vocab = self.dataset.vocab
        self.num_samples = len(self.dataset)

        self.model = Model(hparams, len(self.vocab))

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def prepare_data(self):
        train_loader = TrainDataLoader(self.hparams, self.dataset)
        batch_size = self.hparams.pretrain_batch_size if self.pretrain else self.hparams.translate_batch_size
        self.training_loader = train_loader(batch_size=batch_size, shuffle=True)

    def train_dataloader(self):
        return self.training_loader

    def training_step(self, batch, batch_idx):
        (_, y), (_, y_fingerprints), _, _ = batch
        logits, y_fingerprints_rec = self.model(batch)

        ce_loss = F.cross_entropy(logits, y.target.view(-1), ignore_index=0)
        bce_loss = F.binary_cross_entropy(y_fingerprints_rec, y_fingerprints)

        result = pl.TrainResult(ce_loss + bce_loss)
        result.log('CE', ce_loss, prog_bar=True)
        result.log('BCE', bce_loss, prog_bar=True)
        return result
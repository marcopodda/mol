import gc
import numpy as np
from argparse import Namespace

import torch
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import TrainDataLoader
from layers.model import Model


class Wrapper(pl.LightningModule):
    pretrain = True
    dataset_class = TrainDataset

    def __init__(self, hparams, dataset_name):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dataset = self.dataset_class(hparams, dataset_name)
        self.vocab = self.dataset.vocab
        self.num_samples = len(self.dataset)

        self.model = Model(hparams, len(self.vocab), seq_length=self.dataset.max_length)

    def forward(self, data, denoise):
        return self.model(data, denoise=denoise)

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
        (x_seqs, y_seqs), (_, y_fingerprints), _, _ = batch
        dec_logits, enc_logits, y_fingerprints_rec = self.model(batch, denoise=self.pretrain)
        targets = y_seqs.target.view(-1)

        dec_loss = F.cross_entropy(dec_logits, targets, ignore_index=0)
        ae_loss = F.binary_cross_entropy(y_fingerprints_rec, y_fingerprints)

        enc_loss = 0
        if self.pretrain:
            enc_loss = F.cross_entropy(enc_logits, targets, ignore_index=0)
            ae_loss += enc_loss

        result = pl.TrainResult(dec_loss + ae_loss)
        result.log('dec', dec_loss, prog_bar=True)
        result.log('AE', ae_loss, prog_bar=True)
        if self.pretrain:
            result.log('enc', enc_loss, prog_bar=True)
        return result
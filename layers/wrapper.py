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

    def __init__(self, hparams, root_dir, dataset_name):
        super().__init__()
        self.hparams = HParams.load(hparams)

        data_dir = root_dir.parent.parent / "data"
        self.dataset = self.dataset_class(hparams, dataset_name)
        self.vocab = self.dataset.vocab
        self.num_samples = len(self.dataset)

        self.model = Model(hparams, len(self.vocab), seq_length=self.dataset.max_length)

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
        (x_seqs, y_seqs), (_, y_fingerprints), _, _ = batch
        y_seqs_rec, enc_logits, y_fingerprints_rec = self.model(batch)

        ce_loss = F.cross_entropy(y_seqs_rec, y_seqs.target.view(-1), ignore_index=0)
        bce_loss = F.binary_cross_entropy(y_fingerprints_rec, y_fingerprints)

        if "denoise_targets" in x_seqs:
            bce_loss2 = F.binary_cross_entropy(torch.sigmoid(enc_logits), x_seqs.denoise_targets)
            bce_loss += bce_loss2

        result = pl.TrainResult(ce_loss + bce_loss)
        result.log('CE', ce_loss, prog_bar=True)
        result.log('BCE', bce_loss, prog_bar=True)
        return result
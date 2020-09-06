from argparse import Namespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pytorch_lightning as pl

from layers.model import Model


class Wrapper(pl.LightningModule):
    dataset_class = None
    
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.dataset = self.dataset_class(hparams, output_dir, name)
        self.max_length = self.dataset.max_length

        self.model = Model(hparams, self.output_dir, len(self.dataset.vocab), self.max_length)

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=2)
        return optimizer

    def train_dataloader(self):
        return self.training_loader

    def training_step(self, batch, batch_idx):
        _, y_batch, _, _ = batch
        outputs = self.model(batch)
        ce_loss = F.cross_entropy(outputs, y_batch.seq.view(-1), ignore_index=0)
        logs = {"CE": ce_loss}
        return {"loss": ce_loss, "logs": logs, "progress_bar": logs}
    
    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"tr_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}
from argparse import Namespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl


class Wrapper(pl.LightningModule):
    dataset_class = None
    model_class = None
    
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.hparams = hparams

        self.dataset = self.dataset_class(hparams, output_dir, name)
        self.vocab = self.dataset.vocab
        self.num_samples = len(self.dataset)
        
        self.model = self.model_class(hparams, len(self.vocab))

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return self.training_loader

    def training_step(self, batch, batch_idx):
        (_, y), (_,y_fingerprints), _, _ = batch
        logits, y_fingerprints_rec = self.model(batch)
        target_labels = y.target.view(-1)
        ce_loss = F.cross_entropy(logits, target_labels, ignore_index=0)
        bce_loss = F.binary_cross_entropy(y_fingerprints_rec, y_fingerprints)
        logs = {"CE": ce_loss, "BCE": bce_loss}
        return {"loss": ce_loss + bce_loss, "logs": logs, "progress_bar": logs}
    
    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"tr_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}
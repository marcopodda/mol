from pathlib import Path
from argparse import Namespace

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from core.datasets.utils import to_batch
from core.utils.serialization import load_yaml
from core.utils.vocab import Tokens
from core.utils.scores import accuracy
from tasks.generation.dataset import MolecularDataset
from tasks.generation.loader import MolecularDataLoader
from .model import Model


class TrainModel(pl.LightningModule):
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.dataset = MolecularDataset(hparams, output_dir, name)
        self.max_length = self.dataset.max_length

        self.model = Model(hparams, output_dir, self.max_length)

    def prepare_data(self):
        loader = MolecularDataLoader(self.hparams, self.dataset)
        self.training_loader = loader.get_train()
        self.validation_loader = loader.get_val()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=2)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.validation_loader

    def training_step(self, batch, batch_idx):
        outputs, vae_loss = self.model(batch)
        loss = self.loss(outputs, batch.outseq.view(-1), vae_loss)
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['train_loss'] for x in outputs]).mean()
        logs = {"train_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        outputs, vae_loss = self.model(batch)
        loss = self.loss(outputs, batch.outseq.view(-1), vae_loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {"val_loss": val_loss_mean}
        return {"log": logs, "progress_bar": logs}

    def loss(self, outputs, targets, vae_loss):
        loss = F.cross_entropy(outputs, targets, ignore_index=0)
        return loss + vae_loss


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir="", version="generation", name=output_dir.stem)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=30,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = TrainModel(hparams, output_dir, args.dataset_name)
    trainer.fit(train_model)

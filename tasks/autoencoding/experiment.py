from pathlib import Path
from argparse import Namespace

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import accuracy

import torch
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from core.utils.serialization import load_yaml
from core.datasets.vocab import Tokens
from core.utils.serialization import save_yaml
from core.utils.os import get_or_create_dir
from layers.model import Model
from layers.wrapper import Wrapper
from tasks import AUTOENCODING
from tasks.autoencoding.dataset import AutoencodingDataset
from tasks.autoencoding.loader import AutoencodingDataLoader
from tasks.autoencoding.model import Autoencoder


class AutoencodingWrapper(pl.LightningModule):
    dataset_class = AutoencodingDataset

    def __init__(self, hparams, output_dir, name):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.dataset = AutoencodingDataset(hparams, output_dir, name)
        self.model = Autoencoder(hparams, output_dir)

    def prepare_data(self):
        train_indices, val_indices = train_test_split(range(len(self.dataset)), test_size=0.1)
        
        train_loader = AutoencodingDataLoader(self.hparams, self.dataset, train_indices)
        self.training_loader = train_loader(batch_size=52)
        
        val_loader = AutoencodingDataLoader(self.hparams, self.dataset, val_indices)
        self.validation_loader = val_loader(batch_size=512, shuffle=False)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.validation_loader

    def training_step(self, batch, batch_idx):
        outputs, _ = self.model(batch)
        bce_loss = F.binary_cross_entropy(outputs, batch)
        return {"loss": bce_loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"train_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        outputs, _ = self.model(batch)
        bce_loss = F.binary_cross_entropy(outputs, batch)
        return {"val_loss": bce_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {"val_loss": val_loss_mean}
        return {"log": logs, "progress_bar": logs}


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=output_dir / AUTOENCODING, name="", version="logs")
    ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / AUTOENCODING / "checkpoints"), save_top_k=-1)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = AutoencodingWrapper(hparams, output_dir, args.dataset_name)
    trainer.fit(train_model)
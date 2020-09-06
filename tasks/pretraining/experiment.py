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

from core.datasets.utils import to_batch
from core.utils.serialization import load_yaml
from core.utils.vocab import Tokens
from core.utils.serialization import save_yaml
from core.utils.os import get_or_create_dir
from layers.maskedce import MaskedSoftmaxCELoss, sequence_mask
from tasks.pretraining.dataset import PretrainingDataset
from tasks.pretraining.loader import PretrainingDataLoader
from .model import Model
from .sampler import Sampler


class PLWrapper(pl.LightningModule):
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.dataset = PretrainingDataset(hparams, output_dir, name)
        self.max_length = self.dataset.max_length

        self.model = Model(hparams, self.output_dir, len(self.dataset.vocab), self.max_length)

    def prepare_data(self):
        loader = PretrainingDataLoader(self.hparams, self.dataset)
        indices_path = get_or_create_dir(self.output_dir / "pretraining" / "logs")
        save_yaml(self.dataset.val_indices, indices_path / "val_indices.yml")
        self.training_loader = loader.get_train()
        self.validation_loader = loader.get_val()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=2)
        return optimizer

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.validation_loader

    def training_step(self, batch, batch_idx):
        frags_batch, _, _ = batch
        outputs = self.model(batch)
        ce_loss = F.cross_entropy(outputs, frags_batch.outseq.view(-1), ignore_index=0)
        logs = {"CE": ce_loss}
        return {"loss": ce_loss, "logs": logs, "progress_bar": logs}
    
    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"tr_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        frags_batch, _, _ = batch
        outputs = self.model(batch)
        ce_loss = F.cross_entropy(outputs, frags_batch.outseq.view(-1), ignore_index=0)
        return {"val_loss": ce_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {"val_loss": val_loss_mean}
        return {"log": logs, "progress_bar": logs}


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=output_dir / args.task, name="", version="logs")
    ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / args.task / "checkpoints"), save_top_k=-1)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = PLWrapper(hparams, output_dir, args.dataset_name)
    trainer.fit(train_model)


def run_sampling(output_dir, dataset_name, epoch=None,  temp=1.0, batch_size=1000, greedy=True):
    assert epoch >= 1
    output_dir = Path(output_dir)
    task_dir = output_dir / "pretraining"
    ckpt_dir = task_dir / "checkpoints"
    samples_dir = get_or_create_dir(task_dir / "samples")
    
    all_samples = []
    
    checkpoint_name = list(ckpt_dir.glob(f"epoch={epoch-1}.ckpt"))[0]
    sample_path = samples_dir / f"samples_{epoch}.yml"
    
    if not sample_path.exists():
        print(f"processing {sample_path}...")
        plw = PLWrapper.load_from_checkpoint(checkpoint_name.as_posix(), output_dir=output_dir, name=dataset_name)
        sampler = Sampler(plw.model, plw.dataset)
        samples = sampler.run(temp=temp, batch_size=batch_size, greedy=greedy)
        save_yaml(samples, sample_path)
        
        

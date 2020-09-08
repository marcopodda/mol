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

from core.utils.serialization import load_yaml
from core.datasets.vocab import Tokens
from core.utils.serialization import save_yaml
from core.utils.os import get_or_create_dir
from layers.model import Model
from layers.wrapper import Wrapper
from tasks import PRETRAINING
from tasks.pretraining.dataset import PretrainingDataset
from tasks.pretraining.loader import PretrainingDataLoader

from .sampler import PretrainingSampler


class PretrainingWrapper(Wrapper):
    dataset_class = PretrainingDataset

    def prepare_data(self):
        loader = PretrainingDataLoader(self.hparams, self.dataset)
        indices_path = get_or_create_dir(self.output_dir / PRETRAINING / "logs")
        save_yaml(self.dataset.val_indices, indices_path / "val_indices.yml")
        self.training_loader = loader.get_train()
        self.validation_loader = loader.get_val()

    def val_dataloader(self):
        return self.validation_loader

    def validation_step(self, batch, batch_idx):
        x_batch, _, _, _ = batch
        outputs = self.model(batch)
        ce_loss = F.cross_entropy(outputs, x_batch.seq.view(-1), ignore_index=0)
        return {"val_loss": ce_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {"val_loss": val_loss_mean}
        return {"log": logs, "progress_bar": logs}


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=output_dir / PRETRAINING, name="", version="logs")
    ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / PRETRAINING / "checkpoints"), save_top_k=-1)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = PretrainingWrapper(hparams, output_dir, args.dataset_name)
    trainer.fit(train_model)


def run_sampling(output_dir, dataset_name, epoch=None,  temp=1.0, batch_size=1000, greedy=True, num_samples=None):
    assert epoch >= 1
    
    output_dir = Path(output_dir)
    task_dir = output_dir / PRETRAINING
    ckpt_dir = task_dir / "checkpoints"
    samples_dir = get_or_create_dir(task_dir / "samples")
    
    all_samples = []
    
    checkpoint_name = list(ckpt_dir.glob(f"epoch={epoch-1}.ckpt"))[0]
    sample_path = samples_dir / f"samples_{epoch}.yml"
    
    if not sample_path.exists():
        print(f"processing {sample_path}...")
        wrapper = PretrainingWrapper.load_from_checkpoint(checkpoint_name.as_posix(), output_dir=output_dir, name=dataset_name)
        sampler = PretrainingSampler(wrapper.model, dataset_name)
        samples = sampler.run(temp=temp, batch_size=batch_size, greedy=greedy, num_samples=num_samples)
        save_yaml(samples, sample_path)
        
        

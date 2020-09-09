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
from sklearn.model_selection import train_test_split as tts

from core.utils.serialization import load_yaml
from core.datasets.vocab import Tokens
from core.datasets.loaders import DataLoader
from core.utils.serialization import save_yaml
from core.utils.os import get_or_create_dir
from layers.model import Model
from layers.wrapper import Wrapper
from tasks import PRETRAINING
from tasks.pretraining.dataset import PretrainingDataset

from .sampler import PretrainingSampler


class PretrainingWrapper(Wrapper):
    dataset_class = PretrainingDataset
    model_class = Model

    def prepare_data(self):      
        train_loader = DataLoader(self.hparams, self.dataset)
        self.training_loader = train_loader(shuffle=True)

    def train_dataloader(self):
        return self.training_loader


def run(args):
    output_dir = Path(args.output_dir)
    pretrain_dir = output_dir / PRETRAINING
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=pretrain_dir, name="", version="logs")
    ckpt_dir = get_or_create_dir(pretrain_dir / "checkpoints")
    ckpt_callback = ModelCheckpoint(filepath=ckpt_dir, save_top_k=-1)
    
    train_model = PretrainingWrapper(hparams, output_dir, args.dataset_name)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
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
        
        

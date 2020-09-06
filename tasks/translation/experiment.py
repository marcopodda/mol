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
from tasks import TRANSLATION, PRETRAINING
from tasks.translation.dataset import TranslationDataset
from tasks.translation.loader import TranslationDataLoader
from tasks.translation.sampler import TranslationSampler
from tasks.pretraining.experiment import PretrainingWrapper


class TranslationWrapper(Wrapper):
    dataset_class = TranslationDataset

    def prepare_data(self):
        loader = TranslationDataLoader(self.hparams, self.dataset)
        self.training_loader = loader.get_train()


def load_embedder(hparams, output_dir, args):
    pretraining_dir = output_dir.parent / args.pretrain_from / PRETRAINING / "checkpoints"
    path = sorted(pretraining_dir.glob("*.ckpt"))[-1]
    pretrainer = PretrainingWrapper.load_from_checkpoint(
        path.as_posix(), 
        output_dir=output_dir.parent / "moses", 
        name=args.dataset_name)
    return pretrainer.model.embedder.gnn


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=output_dir / TRANSLATION, name="", version="logs")
    ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / TRANSLATION / "checkpoints"), save_top_k=-1)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = TranslationWrapper(hparams, output_dir, args.dataset_name)
    gnn = load_embedder(hparams, output_dir, args)
    train_model.model.embedder.gnn = gnn
    trainer.fit(train_model)
        

def run_sampling(output_dir, dataset_name, epoch=None, temp=1.0, batch_size=1000, greedy=True):
    assert epoch >= 1
    output_dir = Path(output_dir)
    task_dir = output_dir / TRANSLATION
    ckpt_dir = task_dir / "checkpoints"
    samples_dir = get_or_create_dir(task_dir / "samples")
    
    all_samples = []
    epoch = (epoch - 1) or "*"
    
    for i, checkpoint_name in enumerate(ckpt_dir.glob(f"epoch={epoch}.ckpt")):
        index = (i + 1) if epoch == "*" else (epoch + 1)
        sample_path = samples_dir / f"samples_{index}.yml"
        
        if not sample_path.exists():
            print(f"processing {sample_path}...")
            plw = TranslationWrapper.load_from_checkpoint(checkpoint_name.as_posix(), output_dir=output_dir, name=dataset_name)
            sampler = TranslationSampler(plw.model, plw.dataset)
            samples = sampler.run(temp=temp, batch_size=batch_size, greedy=greedy)
            save_yaml(samples, sample_path)
        
        

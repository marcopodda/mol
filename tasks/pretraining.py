import numpy as np
import torch
from pathlib import Path
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.hparams import HParams
from core.datasets.datasets import BaseDataset, EvalDataset
from core.datasets.loaders import TrainDataLoader, EvalDataLoader
from core.utils.serialization import load_yaml, save_yaml
from core.utils.os import get_or_create_dir
from layers.model import Model
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks import PRETRAINING


class PretrainingTrainDataset(BaseDataset):
    corrupt = False


class PretrainingWrapper(Wrapper):
    dataset_class = PretrainingTrainDataset
    pretrain = True


class PretrainingSampler(Sampler):
    def get_loader(self):
        loader = EvalDataLoader(self.hparams, self.dataset)
        num_samples = min(len(loader.indices), 30000)
        indices = sorted(np.random.choice(loader.indices, num_samples, replace=False))
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.pretrain_batch_size, shuffle=False)


def run(args):
    output_dir = Path(args.output_dir)
    pretrain_dir = output_dir / PRETRAINING
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = HParams.from_file(args.config_file)
    logger = TensorBoardLogger(save_dir=pretrain_dir, name="", version="logs")
    ckpt_dir = get_or_create_dir(pretrain_dir / "checkpoints")
    ckpt_callback = ModelCheckpoint(filepath=ckpt_dir, save_top_k=-1)

    train_model = PretrainingWrapper(hparams, output_dir, args.dataset_name)
    trainer = pl.Trainer(
        max_epochs=hparams.pretrain_num_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    trainer.fit(train_model)


def run_sampling(output_dir, dataset_name, epoch=None,  temp=1.0, greedy=True):
    assert epoch >= 1

    output_dir = Path(output_dir)
    task_dir = output_dir / PRETRAINING
    ckpt_dir = task_dir / "checkpoints"
    samples_dir = get_or_create_dir(task_dir / "samples")

    checkpoint_name = list(ckpt_dir.glob(f"epoch={epoch-1}.ckpt"))[0]
    sample_path = samples_dir / f"samples_{epoch}.yml"

    if not sample_path.exists():
        print(f"processing {sample_path}...")
        model = PretrainingWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_name.as_posix(),
            output_dir=output_dir,
            name=dataset_name).model
        hparams = model.hparams
        dataset = EvalDataset(hparams, output_dir, dataset_name)
        sampler = PretrainingSampler(hparams, model, dataset)
        samples = sampler.run(temp=temp, greedy=greedy)
        save_yaml(samples, sample_path)

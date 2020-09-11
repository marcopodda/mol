import numpy as np
import torch
from pathlib import Path
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.hparams import HParams
from core.datasets.datasets import TrainDataset, EvalDataset
from core.datasets.loaders import TrainDataLoader, EvalDataLoader
from core.utils.serialization import load_yaml, save_yaml
from core.utils.os import get_or_create_dir
from core.utils.scoring import score
from layers.model import Model
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks import PRETRAINING


class PretrainingWrapper(Wrapper):
    pretrain = True


class PretrainingSampler(Sampler):
    def prepare_data(self):
        indices = np.random.choice(len(self.dataset), 1000, replace=False)
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.pretrain_batch_size, shuffle=False)


def run(args):
    root_dir = Path(args.root_dir)
    pretrain_dir = root_dir / PRETRAINING
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = HParams.from_file(args.hparams_file)
    logger = TensorBoardLogger(save_dir=pretrain_dir, name="", version="logs")
    ckpt_dir = get_or_create_dir(pretrain_dir / "checkpoints")
    ckpt_callback = ModelCheckpoint(filepath=ckpt_dir, save_top_k=-1)

    train_model = PretrainingWrapper(hparams, root_dir, args.dataset_name)
    trainer = pl.Trainer(
        max_epochs=hparams.pretrain_num_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    trainer.fit(train_model)


def run_sampling(root_dir, dataset_name, epoch=0, temp=1.0, greedy=True):
    root_dir = Path(root_dir)
    task_dir = root_dir / PRETRAINING

    ckpt_dir = task_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"epoch={epoch}.ckpt"

    samples_dir = get_or_create_dir(task_dir / "samples")
    sample_path = samples_dir / f"samples_{epoch}.yml"

    if not sample_path.exists():
        print(f"processing {sample_path}...")
        model = PretrainingWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            root_dir=root_dir,
            name=dataset_name).model
        hparams = model.hparams
        dataset = EvalDataset(hparams, root_dir, dataset_name)
        sampler = PretrainingSampler(hparams, model, dataset)
        samples = sampler.run(temp=temp, greedy=greedy)
        save_yaml(samples, sample_path)


def sample_and_score(root_dir, dataset_name, epoch=0, temp=1.0, greedy=True):
    root_dir = Path(root_dir)
    run_sampling(root_dir, dataset_name, epoch=epoch, temp=temp, greedy=greedy)
    score(root_dir / PRETRAINING, dataset_name, epoch=epoch)
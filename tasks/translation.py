from pathlib import Path
from argparse import Namespace

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.datasets.datasets import BaseDataset, EvalDataset
from core.datasets.loaders import EvalDataLoader
from core.utils.serialization import load_yaml, save_yaml
from core.utils.os import get_or_create_dir
from layers.model import Model
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks import TRANSLATION, PRETRAINING
from tasks.pretraining import PretrainingWrapper


class TranslationTrainDataset(BaseDataset):
    corrupt = False


class TranslationWrapper(Wrapper):
    dataset_class = TranslationTrainDataset


class TranslationSampler(Sampler):
    def get_loader(self):
        indices = self.dataset.data[self.dataset.data.is_valid==1].index.tolist()
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.translate_batch_size, shuffle=False)


def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False
    return layer


def load_embedder(train_model, output_dir, args):
    pretrain_dir = Path(args.pretrain_from)
    pretrain_ckpt_dir = pretrain_dir / PRETRAINING / "checkpoints"
    pretrain_ckpt_path = sorted(pretrain_ckpt_dir.glob("*.ckpt"))[-1]
    pretrainer = PretrainingWrapper.load_from_checkpoint(
        pretrain_ckpt_path.as_posix(),
        output_dir=pretrain_dir,
        name=args.dataset_name)
    train_model.model.embedder = pretrainer.model.embedder
    # train_model.model.encoder.gru = freeze(pretrainer.model.encoder.gru)
    # train_model.model.decoder.gru = freeze(pretrainer.model.decoder.gru)
    return train_model


def run(args):
    output_dir = Path(args.output_dir)
    gpu = args.gpu if torch.cuda.is_available() else None
    hparams = Namespace(**load_yaml(args.config_file))
    logger = TensorBoardLogger(save_dir=output_dir / TRANSLATION, name="", version="logs")
    ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / TRANSLATION / "checkpoints"), save_top_k=-1)
    trainer = pl.Trainer(
        max_epochs=hparams.translate_num_epochs,
        checkpoint_callback=ckpt_callback,
        progress_bar_refresh_rate=10,
        gradient_clip_val=hparams.clip_norm,
        fast_dev_run=args.debug,
        logger=logger,
        gpus=gpu)
    train_model = TranslationWrapper(hparams, output_dir, args.dataset_name)
    # train_model = load_embedder(train_model, output_dir, args)
    trainer.fit(train_model)


def run_sampling(output_dir, dataset_name, epoch=None, temp=1.0, greedy=True):
    assert epoch >= 1

    output_dir = Path(output_dir)
    task_dir = output_dir / TRANSLATION
    ckpt_dir = task_dir / "checkpoints"
    samples_dir = get_or_create_dir(task_dir / "samples")

    epoch = (epoch - 1) or "*"

    for i, checkpoint_name in enumerate(ckpt_dir.glob(f"epoch={epoch}.ckpt")):
        index = (i + 1) if epoch == "*" else (epoch + 1)
        sample_path = samples_dir / f"samples_{index}.yml"

        if not sample_path.exists():
            print(f"processing {sample_path}...")
            model = TranslationWrapper.load_from_checkpoint(
                checkpoint_path=checkpoint_name.as_posix(),
                output_dir=output_dir,
                name=dataset_name).model
            hparams = model.hparams
            dataset = EvalDataset(hparams, output_dir, dataset_name)
            sampler = TranslationSampler(hparams, model, dataset)
            samples = sampler.run(temp=temp, greedy=greedy)
            save_yaml(samples, sample_path)

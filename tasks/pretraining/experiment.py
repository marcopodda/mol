from pathlib import Path
from argparse import Namespace

import numpy as np

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from core.datasets.utils import to_batch
from core.utils.os import get_or_create_dir
from core.utils.serialization import load_yaml
from core.utils.vocab import Tokens, Vocab

from tasks.pretraining.dataset import PretrainDataset, VocabDataset
from tasks.pretraining.loader import PretrainDataLoader
from tasks.pretraining.model import SkipGram


class Pretrainer(pl.LightningModule):
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.model = SkipGram(hparams)

    def prepare_data(self):
        self.dataset = PretrainDataset(self.hparams, self.output_dir, self.name)

    def forward(self, batch):
        target, context, negatives = batch
        pos_score, neg_score = self.model(target, context, negatives)
        return pos_score, neg_score

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return PretrainDataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.pretrain_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True)

    def training_step(self, batch, batch_idx):
        pos_score, neg_score = self.forward(batch)
        loss = self.model.loss(pos_score, neg_score)
        return {'loss': loss, "log": {"train_loss": loss}}


def save_embeddings(hparams, model, vocab, filename, device="cpu"):
    num_tokens = len(Tokens)
    dataset = VocabDataset(vocab)
    loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.pretrain_batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
        pin_memory=True)

    embeddings = []
    model = model.to(device)

    for batch in loader:
        batch.to(device)
        embedding = model.gnn_in(batch).detach().cpu()
        embeddings.append(embedding)

    embed_dim = hparams.gnn_dim_embed
    pad = [torch.zeros(1, embed_dim)]
    tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens - 1)]
    embeddings = torch.cat(pad + tokens + embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    torch.save(embeddings, filename)


def run_train(args):
    output_dir = Path(args.output_dir)
    embeddings_dir = get_or_create_dir(output_dir / "embeddings")
    dataset_name = args.dataset_name
    hparams = Namespace(**load_yaml(args.config_file))

    dataset = PretrainDataset(hparams, output_dir, dataset_name)
    pretrain_model = Pretrainer(hparams, output_dir, dataset_name)

    if not (embeddings_dir / "untrained.pt").exists():
        print("untrained embeddings...")
        save_embeddings(
            hparams=hparams,
            model=pretrain_model.model,
            vocab=dataset.vocab,
            filename=embeddings_dir / "untrained.pt",
            device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",)

    if not (embeddings_dir / "skipgram.pt").exists():
        print("skipgram embeddings...")
        gpu = args.gpu if torch.cuda.is_available() else None
        logger = TensorBoardLogger(save_dir="", version="pretrain", name=output_dir.stem)
        ckpt_callback = ModelCheckpoint(monitor="train_loss", save_last=True)
        trainer = pl.Trainer(
            max_epochs=hparams.pretrain_epochs,
            checkpoint_callback=ckpt_callback,
            fast_dev_run=args.debug,
            logger=logger,
            gpus=gpu)
        trainer.fit(pretrain_model)

        save_embeddings(
            hparams=hparams,
            model=pretrain_model.model,
            vocab=dataset.vocab,
            filename=embeddings_dir / "skipgram.pt",
            device=next(pretrain_model.parameters()).device)

    if not (embeddings_dir / "random.pt").exists():
        print("random embeddings...")
        num_tokens = len(Tokens)
        embed_dim = hparams.gnn_dim_embed
        pad = [torch.zeros(1, embed_dim)]
        tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens - 1)]
        embeddings = [torch.randn(1, embed_dim) for _ in range(len(dataset.vocab))]
        embeddings = torch.cat(pad + tokens + embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        torch.save(embeddings, embeddings_dir / "random.pt")

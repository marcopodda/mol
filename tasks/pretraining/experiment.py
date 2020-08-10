from pathlib import Path
from argparse import Namespace

import numpy as np

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torch_geometric.data import Batch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from core.datasets.utils import to_batch
from core.utils.os import get_or_create_dir
from core.utils.serialization import load_yaml
from core.utils.vocab import Tokens, Vocab

from tasks.pretraining.dataset import PretrainDataset, VocabDataset
from tasks.pretraining.loader import PretrainDataLoader, VocabLoader
from tasks.pretraining.model import PretrainModel


class Pretrainer(pl.LightningModule):
    def __init__(self, hparams, output_dir, name):
        super().__init__()

        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name

        self.dataset = VocabDataset(hparams, output_dir, name)
        self.vocab = self.dataset.vocab
        
        self.model = PretrainModel(hparams)
        self.loader = VocabLoader(hparams, self.dataset)

    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.loader.get()

    def training_step(self, batch, batch_idx):
        anc, pos, neg = self.forward(batch)
        loss = F.triplet_margin_loss(anc, pos, neg)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"train_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}


def save_embeddings(hparams, model, dataset, vocab, filename, device="cpu"):
    num_tokens = len(Tokens)
    loader = VocabLoader(hparams, dataset)

    embeddings = []
    model = model.to(device)

    for batch in loader.get(shuffle=False):
        embedding, _, _ = model(batch)
        embeddings.append(embedding)

    embed_dim = hparams.gnn_dim_embed
    pad = [torch.zeros(1, embed_dim)]
    tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens - 1)]
    embeddings = torch.cat(pad + tokens + embeddings, dim=0)
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    torch.save(embeddings, filename)


def run(args):
    output_dir = Path(args.output_dir)
    embeddings_dir = get_or_create_dir(output_dir / "embeddings")
    dataset_name = args.dataset_name
    hparams = Namespace(**load_yaml(args.config_file))

    dataset = VocabDataset(hparams, output_dir, dataset_name)
    pretrainer = Pretrainer(hparams, output_dir, dataset_name)

    embeddings_filename = f"emb_{hparams.gnn_dim_embed}.pt"
    if not (embeddings_dir / embeddings_filename).exists():
        print("learning embeddings...")
        gpu = args.gpu if torch.cuda.is_available() else None
        logger = TensorBoardLogger(save_dir=output_dir / args.task, name="", version="logs")
        ckpt_callback = ModelCheckpoint(filepath=get_or_create_dir(output_dir / args.task / "checkpoints"), monitor="train_loss", save_last=True)
        trainer = pl.Trainer(
            max_epochs=hparams.pretrain_epochs,
            checkpoint_callback=ckpt_callback,
            fast_dev_run=args.debug,
            logger=logger,
            gpus=gpu)
        trainer.fit(pretrainer)

        save_embeddings(
            hparams=pretrainer.hparams,
            model=pretrainer.model,
            dataset=dataset,
            vocab=dataset.vocab,
            filename=embeddings_dir / embeddings_filename,
            device=next(pretrainer.parameters()).device)

    embeddings_filename = f"random_{hparams.gnn_dim_embed}.pt"
    if not (embeddings_dir / embeddings_filename).exists():
        print("random embeddings...")
        num_tokens = len(Tokens)
        embed_dim = hparams.gnn_dim_embed
        pad = [torch.zeros(1, embed_dim)]
        tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens - 1)]
        embeddings = [torch.randn(1, embed_dim) for _ in range(len(dataset.vocab))]
        embeddings = torch.cat(pad + tokens + embeddings, dim=0)
        torch.save(embeddings, embeddings_dir / embeddings_filename)

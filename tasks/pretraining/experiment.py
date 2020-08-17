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

from tasks.pretraining.dataset import SkipgramDataset, TripletDataset, VocabDataset
from tasks.pretraining.loader import VocabLoader, TripletLoader, SkipgramLoader
from tasks.pretraining.model import TripletModel, SkipgramModel


class Pretrainer(pl.LightningModule):
    def __init__(self, hparams, model, dataset, loader):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams

        self.dataset = dataset
        self.vocab = self.dataset.vocab
        
        self.model = model  # TripletModel(hparams)
        self.loader = loader  # TripletLoader(hparams, self.dataset)

    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return optimizer  # [optimizer], [scheduler]

    def train_dataloader(self):
        return self.loader.get()

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.model.loss(outputs)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"train_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}
    
    def create_embeddings(self, filename):
        dataset = VocabDataset(self.vocab)
        loader = VocabLoader(self.hparams, dataset)
        device = next(self.model.parameters()).device
        
        embeddings = []
        batch_size = self.hparams.pretrain_batch_size
        for batch in loader.get(shuffle=False, batch_size=batch_size):
            batch = batch.to(device)
            embeddings.append(self.model.predict(batch))
        
        num_tokens = len(Tokens)
        embed_dim = self.hparams.gnn_dim_embed
        pad = [torch.zeros(1, embed_dim)]
        tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens - 1)]
        embeddings = torch.cat(pad + tokens + embeddings, dim=0)
        torch.save(embeddings, filename)


def run(args):
    output_dir = Path(args.output_dir)
    embeddings_dir = get_or_create_dir(output_dir / "embeddings")
    dataset_name = args.dataset_name
    hparams = Namespace(**load_yaml(args.config_file))

    if hparams.embedding_type == "skipgram":
        model = SkipgramModel(hparams)
        dataset = SkipgramDataset(hparams, output_dir, dataset_name)
        loader = SkipgramLoader(hparams, dataset)
    elif hparams.embedding_type == "triplet":
        model = TripletModel(hparams)
        dataset = TripletDataset(hparams, output_dir, dataset_name)
        loader = TripletLoader(hparams, dataset)
    elif hparams.embedding_type == "random":
        return
    else:
        raise ValueError("Unkwnown embedding type!")
    
    pretrainer = Pretrainer(hparams, model, dataset, loader)

    embeddings_filename = f"{hparams.embedding_type}_{hparams.gnn_dim_embed}.pt"
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
        pretrainer.create_embeddings(embeddings_dir / embeddings_filename)
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
from layers.maskedce import MaskedSoftmaxCELoss, sequence_mask
from tasks.pretraining.dataset import SkipgramDataset, TripletDataset, VocabDataset, EncoderDecoderDataset
from tasks.pretraining.loader import VocabLoader, TripletLoader, SkipgramLoader, EncoderDecoderLoader
from tasks.pretraining.model import TripletModel, SkipgramModel, EncoderDecoderModel


class Pretrainer(pl.LightningModule):
    def __init__(self, hparams, model, dataset, loader):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams

        self.dataset = dataset
        self.vocab = self.dataset.vocab
        
        self.model = model  
        self.loader = loader
        self.ce = MaskedSoftmaxCELoss()
        
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
        loss = self.ce(outputs, batch.outseq, batch.length)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {"train_loss": train_loss_mean}
        return {"log": logs, "progress_bar": logs}
    
    def create_embeddings(self, filename):
        if isinstance(self.model, EncoderDecoderModel):
            embeddings = self.model.dec_embedder.weight.data.detach()
            torch.save(embeddings, filename)
            name, dim = filename.stem.split("_")
            filename = filename.parent / f"{name}out_{dim}.pt"
            embeddings = self.model.decoder.out.weight.data.detach()
            torch.save(embeddings, filename)
        else:
            dataset = VocabDataset(self.vocab)
            loader = VocabLoader(self.hparams, dataset)
            device = next(self.model.parameters()).device
            
            embeddings = []
            for batch in loader.get(shuffle=False):
                batch = batch.to(device)
                embedding = self.model.predict(batch)
                embeddings.append(embedding.detach())
            
            num_tokens = len(Tokens)
            embed_dim = self.hparams.frag_dim_embed
            tokens = [torch.randn(1, embed_dim) for _ in range(num_tokens)]
            embeddings = torch.cat(tokens + embeddings, dim=0)
            torch.save(embeddings, filename)
        print("embeddings size:", embeddings.size())


def run(args):
    output_dir = Path(args.output_dir)
    embeddings_dir = get_or_create_dir(output_dir / "embeddings")
    dataset_name = args.dataset_name
    hparams = Namespace(**load_yaml(args.config_file))

    if hparams.embedding_type == "skipgram":
        dataset = SkipgramDataset(hparams, output_dir, dataset_name)
        loader = SkipgramLoader(hparams, dataset)
        model = SkipgramModel(hparams)
    elif hparams.embedding_type == "triplet":
        dataset = TripletDataset(hparams, output_dir, dataset_name)
        loader = TripletLoader(hparams, dataset)
        model = TripletModel(hparams)
    elif hparams.embedding_type == "encdec":
        dataset = EncoderDecoderDataset(hparams, output_dir, dataset_name)
        loader = EncoderDecoderLoader(hparams, dataset)
        model = EncoderDecoderModel(hparams, len(dataset.vocab), dataset.max_length)
    elif hparams.embedding_type == "random":
        return
    else:
        raise ValueError("Unkwnown embedding type!")
    
    pretrainer = Pretrainer(hparams, model, dataset, loader)

    embeddings_filename = f"{hparams.embedding_type}_{hparams.frag_dim_embed}.pt"
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
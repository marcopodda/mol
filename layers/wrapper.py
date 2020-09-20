import torch
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import TrainDataLoader
from core.datasets.vocab import Tokens
from layers.model import Model


class Wrapper(pl.LightningModule):
    dataset_class = TrainDataset

    def __init__(self, hparams, dataset_name):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dataset = self.dataset_class(hparams, dataset_name)
        self.vocab = self.dataset.vocab
        self.dim_output = len(self.vocab) + len(Tokens)

        self.model = Model(hparams, dim_output=self.dim_output)

    def forward(self, data):
        return self.model(data)

    def get_batch_size(self):
        return 32

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def prepare_data(self):
        train_loader = TrainDataLoader(self.hparams, self.dataset)
        batch_size = self.get_batch_size()
        self.training_loader = train_loader(batch_size=batch_size, shuffle=True)

    def train_dataloader(self):
        return self.training_loader

    def training_step(self, batch, batch_idx):
        (_, dec_batch), _ = batch

        decoder_outputs, decoder_bag, encoder_bag = self.model(batch)

        decoder_ce_loss = F.cross_entropy(decoder_outputs, dec_batch.target, ignore_index=0)
        cos_sim = F.cosine_similarity(encoder_bag, decoder_bag).mean(dim=0)

        total_loss = decoder_ce_loss

        result = pl.TrainResult(minimize=total_loss)
        result.log('ce', decoder_ce_loss, prog_bar=True)
        result.log('cs', cos_sim, prog_bar=True)

        return result

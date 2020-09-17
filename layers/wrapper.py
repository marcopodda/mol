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
        batch_data, mlp_targets, _, _ = batch
        encoder_batch, decoder_batch = batch_data

        decoder_outputs, mlp_outputs, bag_of_frags = self.model(batch)
        decoder_bag_of_frags, encoder_bag_of_frags = bag_of_frags

        decoder_ce_loss = F.cross_entropy(decoder_outputs, decoder_batch.target, ignore_index=0)
        bce_loss = F.binary_cross_entropy_with_logits(mlp_outputs, mlp_targets)
        cos_sim = F.cosine_similarity(decoder_bag_of_frags, encoder_bag_of_frags)
        cos_sim = -torch.log(cos_sim).mean(dim=0)
        total_loss = decoder_ce_loss + bce_loss + cos_sim
        result = pl.TrainResult(minimize=total_loss)
        result.log('ce', decoder_ce_loss, prog_bar=True)
        result.log('bce', bce_loss, prog_bar=True)
        result.log('cs', cos_sim, prog_bar=True)

        return result

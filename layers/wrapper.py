import torch
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import TrainDataLoader
from core.datasets.vocab import Tokens
from layers.model import Model
from layers.loss import ContrastiveLoss


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
        raise NotImplementedError

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
        batches, fingerprints, _ = batch
        x_batch, y_batch = batches
        x_fp_target, y_fp_target = fingerprints

        outputs, bags = self.model(batch)
        fp_outputs, outputs = outputs
        x_bag_of_frags, y_bag_of_frags = bags

        ce_loss = F.cross_entropy(outputs, y_batch.target, ignore_index=0)
        fp_loss = F.binarcross_entropy_with_logits(fp_outputs, y_fp_target)
        total_loss = ce_loss + fp_loss

        result = pl.TrainResult(minimize=total_loss)
        result.log('ce', ce_loss, prog_bar=True)
        result.log('bce', fp_loss, prog_bar=True)

        return result

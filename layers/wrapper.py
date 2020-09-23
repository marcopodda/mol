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
        x_batch, y1_batch, y2_batch = batches
        x_fp_target, y1_fp_target, y2_fp_target = fingerprints

        outputs, bags = self.model(batch)
        x_fp_outputs, y1_outputs, y2_outputs = outputs
        x_bag, y1_bag, y2_bag = bags

        y1_ce_loss = F.cross_entropy(y1_outputs, y1_batch.target, ignore_index=0)
        y1_fp_loss = F.binary_cross_entropy_with_logits(x_fp_outputs, y1_fp_target)
        y1_loss = y1_ce_loss + y1_fp_loss

        y2_ce_loss = F.cross_entropy(y2_outputs, y2_batch.target, ignore_index=0)
        y2_fp_loss = F.binary_cross_entropy_with_logits(x_fp_outputs, y2_fp_target)
        y2_loss = y2_ce_loss + y2_fp_loss

        kl = F.kl_div(torch.softmax(y2_outputs, dim=-1), torch.softmax(y1_outputs, dim=-1))

        total_loss = 0.25 * y1_loss + 0.75 * y2_loss

        result = pl.TrainResult(minimize=total_loss)
        result.log('y1l', y1_loss, prog_bar=True)
        result.log('y2l', y2_loss, prog_bar=True)
        # result.log('cl', closs, prog_bar=True)

        return result

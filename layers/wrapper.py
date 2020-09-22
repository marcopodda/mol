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
        self.contrastive_loss = ContrastiveLoss(self.get_batch_size())

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
        anc_batch, pos_batch, neg_batch = batches
        anc_fingerprint, pos_fingerprint, neg_fingerprint = fingerprints

        outputs, output_fingerprint, bags = self.model(batch)
        anc_outputs, pos_outputs, neg_outputs = outputs
        anc_bag, pos_bag, neg_bag = bags

        pos_loss = F.cross_entropy(pos_outputs, pos_batch.target, ignore_index=0)
        neg_loss = F.cross_entropy(neg_outputs, neg_batch.target, ignore_index=0)
        fp_loss = F.binary_cross_entropy_with_logits(output_fingerprint, pos_fingerprint)
        bof_loss = F.triplet_margin_loss(anc_bag, pos_bag, neg_bag)

        total_loss = pos_loss + neg_loss + fp_loss + bof_loss

        result = pl.TrainResult(minimize=total_loss)
        result.log('pl', pos_loss, prog_bar=True)
        result.log('nl', neg_loss, prog_bar=True)
        result.log('fl', fp_loss, prog_bar=True)
        result.log('bl', bof_loss, prog_bar=True)

        return result

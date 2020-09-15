from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import TrainDataLoader
from core.datasets.vocab import Tokens
from layers.model import Model


class Wrapper(pl.LightningModule):
    pretrain = True
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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def prepare_data(self):
        train_loader = TrainDataLoader(self.hparams, self.dataset)
        batch_size = self.hparams.pretrain_batch_size if self.pretrain else self.hparams.translate_batch_size
        self.training_loader = train_loader(batch_size=batch_size, shuffle=True)

    def train_dataloader(self):
        return self.training_loader

    def training_step(self, batch, batch_idx):
        (_, y_seqs), (_, y_fingerprints), _, _ = batch
        dec_logits, y_fingerprints_rec = self.model(batch)
        targets = y_seqs.target.view(-1)

        dec_loss = F.cross_entropy(dec_logits, targets, ignore_index=0)
        ae_loss = F.binary_cross_entropy_with_logits(y_fingerprints_rec, y_fingerprints)

        result = pl.TrainResult(dec_loss + ae_loss)
        result.log('dec', dec_loss, prog_bar=True)
        result.log('ae', ae_loss, prog_bar=True)
        return result

import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from core.hparams import HParams
from core.datasets.datasets import TrainDataset, EvalDataset

def collate_frags(data_list):
    cumsum = 0

    for i, frag_x in enumerate(data_list):
        inc = (frag_x.frags_batch.numel() + 1)
        frag_x.frags_batch += cumsum
        cumsum += inc

    return Batch.from_data_list(data_list)


def prefilled_tensor(dims, fill_with, fill_at):
    tx = torch.zeros(dims)
    tx[:, fill_at, :] = fill_with.repeat(dims[0], 1)
    return tx


class BaseDataLoader:
    def __init__(self, hparams, dataset, indices=None):
        self.hparams = HParams.load(hparams)
        self.dataset = dataset

        self._check_dataset()

        self.max_length = self.dataset.max_length
        if indices is None:
            indices = list(range(len(dataset)))
        self.indices = indices

    def __call__(self, batch_size, shuffle=False):
        dataset = Subset(self.dataset, self.indices)
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def _check_dataset(self):
        raise NotImplementedError

    def collate(self, data_list):
        raise NotImplementedError


class TrainDataLoader(BaseDataLoader):
    def _check_dataset(self):
        if not isinstance(self.dataset, TrainDataset):
            raise Exception("Works only for TrainDataset")

    def collate(self, data_list):
        frags_x, fps_x, frags_y, fps_y = zip(*data_list)

        frags_x_batch = collate_frags(frags_x)
        frags_y_batch = collate_frags(frags_y)

        B = len(frags_x)
        L = self.dataset.max_length
        D = self.hparams.frag_dim_embed

        lengths = [m.length for m in frags_x]
        enc_inputs = prefilled_tensor(dims=(B, L, D), fill_with=self.dataset.eos, fill_at=lengths)

        x_fingerprints = torch.cat(fps_x, dim=0)
        y_fingerprints = torch.cat(fps_y, dim=0)

        dec_inputs = prefilled_tensor(dims=(B, L, D), fill_with=self.dataset.sos, fill_at=0)

        return (frags_x_batch, frags_y_batch), (x_fingerprints, y_fingerprints), enc_inputs, dec_inputs


class EvalDataLoader(BaseDataLoader):
    def _check_dataset(self):
        if not isinstance(self.dataset, EvalDataset):
            raise Exception("Works only for EvalDataset")

    def collate(self, data_list):
        frags_batch, fps_x = zip(*data_list)

        frags_x_batch = collate_frags(frags_batch)

        B = len(frags_batch)
        L = self.dataset.max_length
        D = self.hparams.frag_dim_embed

        lengths = [m.length.item() for m in frags_batch]
        enc_inputs = prefilled_tensor(dims=(B, L, D), fill_with=self.dataset.eos, fill_at=lengths)

        x_fingerprints = torch.cat(fps_x, dim=0)

        return frags_x_batch, x_fingerprints, enc_inputs


class VocabDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = HParams.load(hparams)
        self.dataset = dataset

    def __call__(self, shuffle=False, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: Batch.from_data_list(b),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from core.hparams import HParams
from core.datasets.datasets import TrainDataset, EvalDataset


def collate_frags(data_list):
    cumsum = 0

    for i, data in enumerate(data_list):
        inc = (data.frags_batch.max() + 1)
        data.frags_batch += cumsum
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
        x, y1, y2, x_fingerprint, y1_fingerprint, y2_fingerprint = zip(*data_list)
        sos = self.dataset.sos
        eos = self.dataset.eos

        x_batch = collate_frags(x)
        y1_batch = collate_frags(y1)
        y2_batch = collate_frags(y2)

        B = len(x)
        L = self.dataset.max_length
        D = self.hparams.frag_dim_embed

        x_inputs = prefilled_tensor(dims=(B, L, D), fill_with=eos.clone(), fill_at=0)
        lengths = [m.length for m in y1]
        y1_inputs = prefilled_tensor(dims=(B, L, D), fill_with=sos.clone(), fill_at=lengths)
        lengths = [m.length for m in y2]
        y2_inputs = prefilled_tensor(dims=(B, L, D), fill_with=sos.clone(), fill_at=lengths)

        x_fingerprint = torch.cat(x_fingerprint, dim=0)
        y1_fingerprint = torch.cat(y1_fingerprint, dim=0)
        y2_fingerprint = torch.cat(y2_fingerprint, dim=0)

        return (x_batch, y1_batch, y2_batch), (x_fingerprint, y1_fingerprint, y2_fingerprint), (x_inputs, y1_inputs, y2_inputs)


class EvalDataLoader(BaseDataLoader):
    def _check_dataset(self):
        if not isinstance(self.dataset, EvalDataset):
            raise Exception("Works only for EvalDataset")

    def collate(self, data_list):
        frags_batch, fingerprints = zip(*data_list)

        frags_x_batch = collate_frags(frags_batch)

        B = len(frags_batch)
        L = self.dataset.max_length
        D = self.hparams.frag_dim_embed

        lengths = [m.length for m in frags_batch]
        enc_inputs = prefilled_tensor(dims=(B, L, D), fill_with=self.dataset.eos, fill_at=lengths)

        fingerprints = torch.cat(fingerprints, dim=0)

        return frags_x_batch, fingerprints, enc_inputs


class VocabDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = HParams.load(hparams)
        self.dataset = dataset

    def __call__(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: Batch.from_data_list(b),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

from itertools import chain

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class TripletLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def collate(self, data_list):
        anc, pos, neg = zip(*data_list)
        return Batch.from_data_list(anc), Batch.from_data_list(pos), Batch.from_data_list(neg),

    def get(self, shuffle=True):
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)


class VocabLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def collate(self, data_list):
        return Batch.from_data_list(data_list)

    def get(self, shuffle=True):
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


class SkipgramLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def collate(self, batch):
        target, context, negatives = zip(*batch)
        target = Batch.from_data_list(target)
        context = Batch.from_data_list(context)
        negatives = Batch.from_data_list(flatten(negatives))
        return target, context, negatives

    def get(self, shuffle=True):
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
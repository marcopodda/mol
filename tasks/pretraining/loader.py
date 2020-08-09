import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from sklearn.model_selection import train_test_split


class PretrainDataLoader:
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def collate(self, data_list):
        return Batch.from_data_list(data_list)

    def get(self):
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

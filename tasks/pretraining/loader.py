import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

from core.datasets.common import collate


class PretrainingDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset

    def get_train(self, batch_size=None):
        return self.get_loader(
            indices=self.dataset.train_indices,
            batch_size=batch_size,
            shuffle=False)
        
    def get_val(self, batch_size=None):
        return self.get_loader(
            indices=self.dataset.val_indices,
            batch_size=batch_size,
            shuffle=False)
    
    def get_loader(self, indices, batch_size=None, shuffle=False):
        dataset = Subset(self.dataset, indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from core.datasets.common import collate


class TranslationDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def get_train(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.train_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0) # self.hparams.num_workers)

    def get_val(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.val_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        
    def get_test(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.test_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

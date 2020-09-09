import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split


class AutoencodingDataLoader:
    pass
    # def __init__(self, hparams, dataset, indices):
    #     self.hparams = hparams
    #     self.dataset = dataset
    #     self.indices = indices
    
    # def __call__(self, batch_size=None, shuffle=True):
    #     dataset = Subset(self.dataset, self.indices)
    #     batch_size = batch_size or self.hparams.batch_size
        
    #     return DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         pin_memory=True,
    #         num_workers=self.hparams.num_workers)
    

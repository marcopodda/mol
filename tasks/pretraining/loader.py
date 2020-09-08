import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

from core.datasets.common import collate_single, prefilled_tensor


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
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
    
    def collate(self, data_list):
        x_frags, y_frags = zip(*data_list)
        
        x_frag_batch = collate_single(x_frags)
        y_frag_batch = collate_single(y_frags)
        
        B = len(x_frags)
        M = self.dataset.max_length
        H = self.hparams.frag_dim_embed
        L = [m.length.item() for m in x_frags]
        
        enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        dec_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.sos, fill_at=0)

        return x_frag_batch, y_frag_batch, enc_inputs, dec_inputs
    

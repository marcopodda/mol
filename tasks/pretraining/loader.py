import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch

from sklearn.model_selection import train_test_split


class PretrainingDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset

    def collate(self, frags):
        batch_size = len(frags)
        
        cumsum = 0
        for i, frag in enumerate(frags):
            inc = (frag.frags_batch.max() + 1).item()
            frag.frags_batch += cumsum
            cumsum += inc
        
        lengths = [m.length.item() for m in frags]
        enc_inputs = torch.zeros((batch_size, self.dataset.max_length, self.hparams.frag_dim_embed))
        enc_inputs[:, lengths, :] = self.dataset.eos.repeat(batch_size, 1)
        
        dec_inputs = torch.zeros((batch_size, self.dataset.max_length, self.hparams.frag_dim_embed))
        dec_inputs[:, 0, :] = self.dataset.sos.repeat(batch_size, 1)
    
        # duplicating frags batch for compatibility with Model
        return Batch.from_data_list(frags), Batch.from_data_list(frags), enc_inputs, dec_inputs

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

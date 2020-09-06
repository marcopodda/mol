import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch

from sklearn.model_selection import train_test_split


def collate(data_list, dataset, hparams):
    mols, frags = zip(*data_list)
    batch_size = len(mols)
    
    cumsum = 0
    for i, frag in enumerate(frags):
        inc = (frag.frags_batch.max() + 1).item()
        frag.frags_batch += cumsum
        cumsum += inc
    
    lengths = [m.length.item() for m in mols]
    enc_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    enc_inputs[:, lengths, :] = dataset.eos.repeat(batch_size, 1)
    
    dec_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    dec_inputs[:, 0, :] = dataset.sos.repeat(batch_size, 1)

    return Batch.from_data_list(mols), Batch.from_data_list(frags), enc_inputs, dec_inputs


class PretrainingDataLoader:
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
    
        return Batch.from_data_list(frags), enc_inputs, dec_inputs

    def get_train(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.train_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def get_val(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.val_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
    
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

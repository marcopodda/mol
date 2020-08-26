import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch

from sklearn.model_selection import train_test_split


class MolecularDataLoader:
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
        self.train_indices, self.val_indices = train_test_split(range(self.num_samples), test_size=0.1)
        self.sos = torch.randn((1, self.hparams.frag_dim_embed))

    def collate(self, data_list):
        mols, frags = zip(*data_list)
        
        cumsum = 0
        for i, frag in enumerate(frags):
            inc = (frag.frags_batch.max() + 1).item()
            frag.frags_batch += cumsum
            cumsum += inc
        
        seq_matrix = torch.zeros((len(mols), self.dataset.max_length, self.hparams.frag_dim_embed))
        seq_matrix[:, 0, :] = self.sos.repeat(len(mols), 1)
        
        return Batch.from_data_list(mols), Batch.from_data_list(frags), seq_matrix

    def get_train(self, batch_size=None):
        dataset = Subset(self.dataset, self.train_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def get_val(self, batch_size=None):
        dataset = Subset(self.dataset, self.val_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

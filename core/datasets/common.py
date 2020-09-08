import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from core.datasets.utils import frag2data


class VocabDataset:
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, index):
        mol_data = self.vocab[index]
        data = frag2data(mol_data)
        return data

    def __len__(self):
        return len(self.vocab)
    

class VocabDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset
    
    def __call__(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: Batch.from_data_list(b),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
    

def collate_single(data_list):
    cumsum = 0
    
    for i, frag_x in enumerate(data_list):
        inc = (frag_x.frags_batch.max() + 1).item()
        frag_x.frags_batch += cumsum
        cumsum += inc
        
    return Batch.from_data_list(data_list)

def prefilled_tensor(dims, fill_with, fill_at):
    mat = torch.zeros(dims)
    mat[:, fill_at, :] = fill_with.repeat(dims[0], 1)
    return mat
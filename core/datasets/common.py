import torch
from torch.utils import data
from torch_geometric.data import Batch
from core.datasets.utils import frag2data


class VocabDataset(data.Dataset):
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, index):
        mol_data = self.vocab[index]
        data = frag2data(mol_data)
        return data

    def __len__(self):
        return len(self.vocab)


def collate(data_list, dataset, hparams):
    x_frags, y_frags = zip(*data_list)
    batch_size = len(x_frags)
    
    cumsum = 0
    for i, frag_x in enumerate(x_frags):
        inc = (frag_x.frags_batch.max() + 1).item()
        frag_x.frags_batch += cumsum
        cumsum += inc
    
    cumsum = 0
    for i, frag_y in enumerate(y_frags):
        inc = (frag_y.frags_batch.max() + 1).item()
        frag_y.frags_batch += cumsum
        cumsum += inc
    
    lengths = [m.length.item() for m in x_frags]
    enc_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    enc_inputs[:, lengths, :] = dataset.eos.repeat(batch_size, 1)
    
    dec_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    dec_inputs[:, 0, :] = dataset.sos.repeat(batch_size, 1)

    return Batch.from_data_list(x_frags), Batch.from_data_list(y_frags), enc_inputs, dec_inputs
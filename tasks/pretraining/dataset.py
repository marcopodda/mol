import numpy as np
from argparse import Namespace

import torch
from torch.utils import data

from core.datasets.preprocess import get_data
from core.datasets.utils import to_data
from core.mols.props import bulk_tanimoto
from core.utils.vocab import Tokens

class VocabDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        _, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.num_frags = len(self.vocab)
        self.num_samples = int(0.9 * self.num_frags)
     
    def __getitem__(self, index):
        smiles = self.vocab[index]
        samples = np.random.choice(self.num_frags, self.num_samples).tolist()
        if index in samples:
            samples.remove(index)
        smiles_samples = [self.vocab[i] for i in samples]
        td = bulk_tanimoto(smiles, smiles_samples)
        
        positive_idx, negative_idx = (-np.array(td)).argsort()[:2]
        return to_data(smiles), to_data(smiles_samples[positive_idx]), to_data(smiles_samples[negative_idx])

    def __len__(self):
        return self.num_frags


class PretrainDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.max_length = self.data.length.max() + 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data.iloc[index]
        # return to_data(data, self.vocab, self.max_length)
        seq_len = data.length
        data = to_data(data, self.vocab, self.max_length)
        probs = torch.zeros_like(data.inseq)
        probs[:, 1:seq_len] = torch.rand(seq_len - 1)
        data.inseq[probs > 0.5] = Tokens.MASK.value
        return data
        
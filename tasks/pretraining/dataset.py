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
     
    def __getitem__(self, index):
        anchor_smiles = self.vocab[index]
        positive_smiles = np.random.choice(self.vocab.most_similar_1[index])
        negative_smiles = np.random.choice(self.vocab.most_similar_2[index])
        return (
            to_data(anchor_smiles), 
            to_data(positive_smiles), 
            to_data(negative_smiles))

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
        
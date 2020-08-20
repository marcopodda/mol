import numpy as np
from argparse import Namespace

import torch
from torch.utils import data

from core.datasets.preprocess import get_data
from core.datasets.utils import to_data
from core.utils.vocab import Tokens


class MolecularDataset(data.Dataset):
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
        seq_len = data.length
        return to_data(data, self.vocab, self.max_length)
        # data = to_data(data, self.vocab, self.max_length)
        # probs = torch.zeros_like(data.inseq)
        # probs[:, 1:seq_len] = torch.rand(seq_len - 1)
        # data.inseq[probs > 0.5] = Tokens.MASK.value
        # return data
        
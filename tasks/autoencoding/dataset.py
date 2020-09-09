import numpy as np
import networkx as nx
from argparse import Namespace

import torch
from torch.utils import data

from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, from_networkx

from sklearn.model_selection import train_test_split

from core.datasets.features import mol2nx
from core.datasets.utils import pad, frag2data, fragslist2data, build_frag_sequence, get_data
from core.datasets.vocab import Tokens
from core.mols.props import get_fingerprint
from core.utils.os import get_or_create_dir, dir_is_empty


class AutoencodingDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name
        self.noise_prob = 0.05
        
        data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data.iloc[index]
        fingerprint = get_fingerprint(data.smiles)
        data = torch.LongTensor(fingerprint)
        return self.add_noise(data)
    
    def add_noise(self, data):
        noise = torch.rand(*data.size())
        noise = (noise >= self.noise_prob).int()
        return data.logical_and(noise).float()
        
        
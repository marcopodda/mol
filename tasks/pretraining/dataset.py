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
from core.utils.os import get_or_create_dir, dir_is_empty
from core.utils.serialization import load_numpy, save_numpy, load_yaml, save_yaml
from tasks import PRETRAINING

class PretrainingDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name
        
        self.data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.load_indices()
        
        self.max_length = self.data.length.max() + 1
        
        self.sos = self._initialize_token("sos")
        self.eos = self._initialize_token("eos")
    
    def load_indices(self):
        logs_dir = get_or_create_dir(self.output_dir / PRETRAINING / "logs")
        train_indices_path = logs_dir / "train_indices.yml"
        val_indices_path = logs_dir / "val_indices.yml"
        if dir_is_empty(logs_dir):
            self.train_indices, self.val_indices = train_test_split(range(self.data.shape[0]), test_size=0.1)
            save_yaml(self.train_indices, train_indices_path)
            save_yaml(self.val_indices, val_indices_path)
        else:
            self.train_indices = load_yaml(train_indices_path)
            self.val_indices = load_yaml(val_indices_path)
    
    def _initialize_token(self, name):
        path = self.output_dir / "DATA" / f"{name}_{self.hparams.frag_dim_embed}.dat"    
        if path.exists():
            token = torch.FloatTensor(load_numpy(path))
        else:
            token = torch.randn((1, self.hparams.frag_dim_embed))
            save_numpy(token.numpy(), path)
        return token

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        frags_list = self.data.iloc[index].frags
        data = fragslist2data(frags_list)
        data["seq"] = build_frag_sequence(frags_list, self.vocab, self.max_length)
        return data, data.clone()
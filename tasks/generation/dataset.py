import numpy as np
import networkx as nx
from argparse import Namespace

import torch
from torch.utils import data

from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, from_networkx

from core.datasets.preprocess import get_data
from core.datasets.features import mol2nx
from core.datasets.utils import to_data, pad
from core.utils.vocab import Tokens
from core.utils.serialization import load_numpy, save_numpy


class MolecularDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name
        
        self.data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.max_length = self.data.length.max() + 1
        
        self.sos = self._initialize_token("sos")
        self.eos = self._initialize_token("eos")
    
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
        data = self.data.iloc[index]
        seq_len = data.length
        
        frags = [mol2nx(f) for f in data.frags]       
        num_nodes = [f.number_of_nodes() for f in frags]
        frags_batch = torch.cat([torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)])
        frags_graph = from_networkx(nx.disjoint_union_all(frags))
        frags_graph["frags_batch"] = frags_batch
        frags_graph["length"] = torch.LongTensor([[len(frags)]])
        
        mol_data = to_data(data, self.vocab, self.max_length)
        return mol_data, frags_graph
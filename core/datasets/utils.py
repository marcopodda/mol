import ast

import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

from rdkit import Chem

from core.datasets.features import get_features, mol2nx
from core.utils.vocab import Tokens



def pad(vec, length, pad_symbol=Tokens.PAD.value):
    padded = np.ones(length) * pad_symbol
    padded[:len(vec)] = vec
    return torch.LongTensor([padded])


def get_graph_data(row):
    if isinstance(row, pd.Series):
        mol = Chem.MolFromSmiles(row.smiles)
    else:
        mol = Chem.MolFromSmiles(row)
    
    G = mol2nx(mol)
    return from_networkx(G)


def to_data(row, vocab=None, max_length=None):
    data = get_graph_data(row)
    
    if vocab is not None and max_length is not None:
        seq = [vocab[f] + len(Tokens) for f in row.frags]
        data["outseq"] = pad(seq + [Tokens.EOS.value], max_length)
        data["length"] = torch.LongTensor([[len(seq)]])

    return data


def to_batch(smi):
    data = to_data(smi)
    return Batch.from_data_list([data])


def load_csv_data(path, convert=None, cast=None):
    converters = {}
    if convert:
        for col in convert:
            converters[col] = ast.literal_eval

    return pd.read_csv(
        filepath_or_buffer=path,
        index_col=0,
        converters=converters,
        dtype=cast
    )

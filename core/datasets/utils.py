import ast

import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data, Batch

from rdkit import Chem

from core.datasets.features import get_features
from core.utils.vocab import Tokens



def pad(vec, length, pad_symbol=Tokens.PAD.value):
    padded = np.ones(length) * pad_symbol
    padded[:len(vec)] = vec
    return torch.LongTensor([padded])


def to_data(row, vocab=None, max_length=None):
    if isinstance(row, pd.Series):
        mol = Chem.MolFromSmiles(row.smiles)
    else:
        mol = Chem.MolFromSmiles(row)

    edge_index, x, edge_attr = get_features(mol)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    try:
        assert vocab is not None
        assert max_length is not None
        seq = [vocab[f] + len(Tokens) for f in row.frags]
        data["inseq"] = pad([Tokens.SOS.value] + seq, max_length)
        data["outseq"] = pad(seq + [Tokens.EOS.value], max_length)
        data["length"] = torch.LongTensor([[len(seq) + 1]])
    except Exception as e:
        pass
    
    if isinstance(row, pd.Series):
        props = torch.Tensor([row.qed, row.SAS, row.logP, row.mr, row.mw])
        data['props'] = props

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

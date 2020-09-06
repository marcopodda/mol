import ast

import pandas as pd
import numpy as np
import networkx as nx
import warnings
from pathlib import Path

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

from core.datasets.features import mol2nx
from core.datasets.settings import DATA_DIR, CONFIG_DIR
from core.datasets.vocab import Vocab, Tokens, get_vocab
from core.mols.utils import mol_from_smiles, mol_to_smiles
from core.utils.os import get_or_create_dir, dir_is_empty
from core.utils.misc import get_n_jobs


def pad(vec, length, pad_symbol=Tokens.PAD.value):
    padded = np.ones(length) * pad_symbol
    padded[:len(vec)] = vec
    return torch.LongTensor([padded])


def fragslist2data(frags_list):
    assert isinstance(frags_list, list)
    assert len(frags_list) > 0
    
    if isinstance(frags_list[0], str):
        frags_list = [mol_from_smiles(f) for f in frags_list]
    
    seq_length = len(frags_list)
    frags = [mol2nx(f) for f in frags_list]       
    num_nodes = [f.number_of_nodes() for f in frags]
    frags_batch = torch.cat([torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)])
    data = from_networkx(nx.disjoint_union_all(frags))
    data["frags_batch"] = frags_batch
    data["length"] = torch.LongTensor([[seq_length]])
    return data


def frag2data(frag):
    if isinstance(frag, str):
        frag = mol_from_smiles(frag)
        
    graph = mol2nx(frag)
    data = from_networkx(graph)
    return data


def build_frag_sequence(frags_list, vocab, max_length):
    assert isinstance(frags_list, list)
    assert len(frags_list) > 0
    assert isinstance(frags_list[0], str)
    
    seq = [vocab[f] + len(Tokens) for f in frags_list] + [Tokens.EOS.value]
    return pad(seq, max_length)


def get_data(dest_dir, dataset_name, num_samples=None):
    dest_dir = Path(dest_dir)

    processed_dir = DATA_DIR / dataset_name / "PROCESSED"
    processed_data_path = processed_dir / "data.csv"
    processed_vocab_path = processed_dir / "vocab.csv"

    if not processed_data_path.exists():
        print("Preprocess your dataset first!")
        exit(1)

    if not processed_vocab_path.exists():
        print("Create the vocabulary first!")
        exit(1)

    dest_dir = get_or_create_dir(dest_dir / "DATA")
    dest_data_path = dest_dir / "data.csv"
    dest_vocab_path = dest_dir / "vocab.csv"

    n_jobs = get_n_jobs()

    if dest_data_path.exists():
        data = load_csv_data(dest_data_path, convert=["frags"], cast={"length": int})
        vocab = Vocab.from_file(dest_vocab_path)
        if num_samples is not None and num_samples != data.shape[0]:
            warnings.warn(f"Got num samples: {num_samples} != data size: {data.shape[0]}. Overriding.")
            if num_samples is not None:
                data = data.sample(n=num_samples)
                data = data.reset_index(drop=True)
            data.to_csv(dest_data_path)
            
            vocab = get_vocab(data, processed_vocab_path)
            vocab.save(dest_vocab_path)
    else:
        data = load_csv_data(processed_data_path, convert=["frags"], cast={"length": int})
        if num_samples is not None:
            data = data.sample(n=num_samples)
            data = data.reset_index(drop=True)
        data.to_csv(dest_data_path)
        
        vocab = get_vocab(data, processed_vocab_path)
        vocab.save(dest_vocab_path)

    data = load_csv_data(dest_data_path, convert=["frags"], cast={"length": int})
    vocab = Vocab.from_file(dest_vocab_path)
    return data, vocab



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

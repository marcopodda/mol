import ast

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

import torch

from core.datasets.settings import DATA_DIR, CONFIG_DIR
from core.datasets.vocab import Vocab, Tokens, get_vocab
from core.mols.utils import mol_from_smiles, mol_to_smiles
from core.utils.os import get_or_create_dir
from core.utils.misc import get_n_jobs
from core.utils.serialization import load_yaml


def pad(vec, length, pad_symbol=Tokens.PAD.value):
    padded = np.ones(length) * pad_symbol
    padded[:len(vec)] = vec
    return torch.LongTensor([padded])


def load_dataset_info(name):
    path = CONFIG_DIR / f'{name}.yml'
    return load_yaml(path)


def load_csv(path, convert=None, cast=None):
    converters = {}
    if convert:
        for col in convert:
            converters[col] = ast.literal_eval

    return pd.read_csv(
        filepath_or_buffer=path,
        index_col=0,
        converters=converters,
        dtype=cast)
    

def load_data(dest_dir, dataset_name, num_samples=None):
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
        data = load_csv(dest_data_path, convert=["frags"], cast={"length": int})
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
        data = load_csv(processed_data_path, convert=["frags"], cast={"length": int})
        if num_samples is not None:
            data = data.sample(n=num_samples)
            data = data.reset_index(drop=True)
        data.to_csv(dest_data_path)
        
        vocab = get_vocab(data, processed_vocab_path)
        vocab.save(dest_vocab_path)

    data = load_csv(dest_data_path, convert=["frags"], cast={"length": int})
    vocab = Vocab.from_file(dest_vocab_path)
    return data, vocab, data.length.max() + 1

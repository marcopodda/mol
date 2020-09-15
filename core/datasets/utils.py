import ast

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

import torch

from core.datasets.settings import DATA_DIR, CONFIG_DIR
from core.datasets.vocab import Vocab, Tokens
from core.mols.props import get_fingerprint
from core.utils.misc import get_n_jobs
from core.utils.serialization import load_yaml


def pad(vec, length, pad_symbol=Tokens.PAD.value):
    padded = np.ones(length) * pad_symbol
    padded[:len(vec)] = vec
    return torch.LongTensor(padded)


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


def load_data(dataset_name):
    processed_dir = DATA_DIR / dataset_name
    processed_data_path = processed_dir / "data.csv"
    processed_vocab_path = processed_dir / "vocab.csv"

    if not processed_data_path.exists():
        print("Preprocess your dataset first!")
        exit(1)

    data = load_csv(processed_data_path, convert=["frags"], cast={"length": int})

    if not processed_vocab_path.exists():
        print("Create the vocabulary first!")
        exit(1)

    vocab = Vocab.from_file(processed_vocab_path)
    return data, vocab, data.length.max() + 1


def hamming_distance(x, y):
    x_fp = get_fingerprint(x)
    y_fp = get_fingerprint(y)
    return np.logical_xor(x_fp, y_fp).sum()


def get_counts(data):
    n_jobs = get_n_jobs()
    xs = data[data.is_x == True].smiles.tolist()
    ys = data[data.is_y == True].smiles.tolist()

    P = Parallel(n_jobs=n_jobs, verbose=1)
    return P(delayed(hamming_distance)(x, y) for (x, y) in zip(xs, ys))

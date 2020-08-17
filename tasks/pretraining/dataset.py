import itertools
from argparse import Namespace

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from joblib import Parallel, delayed

from core.datasets.preprocess import get_data
from core.datasets.utils import to_data, load_csv_data
from core.mols.props import bulk_tanimoto
from core.utils.misc import get_n_jobs
from core.utils.vocab import Tokens


class VocabDataset(data.Dataset):
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, index):
        return to_data(self.vocab[index])

    def __len__(self):
        return len(self.vocab)



class TripletDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
            
        _, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.vocab_size = len(self.vocab)
     
    def __getitem__(self, index):
        anchor_smiles = self.vocab[index]
        positive_smiles = np.random.choice(self.vocab.most_similar_1[index])
        negative_smiles = np.random.choice(self.vocab.most_similar_2[index])
        return (
            to_data(anchor_smiles), 
            to_data(positive_smiles), 
            to_data(negative_smiles))

    def __len__(self):
        return self.vocab_size


def process(context):
    rows = []
    for (i, j) in itertools.permutations(range(len(context)), 2):
        target_idx, context_idx = context[i], context[j]
        rows.append({"target_idx": target_idx, "context_idx": context_idx, "context": context})
    return rows


def get_context_pairs(root_dir, data, vocab, num_negatives):
    data_dir = root_dir / "DATA"
    dataset_path = data_dir / "skipgram.csv"

    if not dataset_path.exists():
        contexts = [[vocab[f] for f in frag] for frag in data.frags]
        P = Parallel(n_jobs=get_n_jobs(), verbose=1)
        rows = P(delayed(process)(ctx) for ctx in contexts)
        dataset = pd.DataFrame(row for subitem in rows for row in subitem)
        dataset = dataset.drop_duplicates(subset=['target_idx', 'context_idx']).reset_index(drop=True)
        dataset.to_csv(dataset_path)

    dataset = load_csv_data(dataset_path, convert=["context"])
    return dataset


class SkipgramDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
            
        data, vocab = get_data(output_dir, name, hparams.num_samples)
        self.context_pairs = get_context_pairs(output_dir, data, vocab, hparams.num_negatives)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.p = vocab.unigram_prob()
        self.num_negatives = hparams.num_negatives
        self.sample_size = self.num_negatives + data.length.max()

    def sample_negatives(self, context):
        candidates = np.random.choice(self.vocab_size, size=self.sample_size, p=self.p, replace=False)
        negative_idxs = np.setdiff1d(candidates, context)[:self.num_negatives]
        return negative_idxs.tolist()

    def __getitem__(self, index):
        data = self.context_pairs.iloc[index]
        target_idx = int(data.target_idx)
        context_idx = int(data.context_idx)
        negative_idxs = self.sample_negatives(data.context)

        target = to_data(self.vocab[target_idx])
        context = to_data(self.vocab[context_idx])

        negatives = []
        for negative_idx in negative_idxs:
            negative = to_data(self.vocab[negative_idx])
            negatives.append(negative)
        
        return target, context, negatives

    def __len__(self):
        return len(self.context_pairs)
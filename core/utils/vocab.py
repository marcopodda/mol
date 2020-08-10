from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd

from core.mols.props import bulk_tanimoto
from core.utils.serialization import load_pickle, save_pickle


class Tokens(Enum):
    PAD = 0
    SOS = 1
    EOS = 2
    MASK = 3


def compute_most_similar(frag, other_frags):
    other_frags.remove(frag)
    sim = np.array(bulk_tanimoto(frag, other_frags))
    second_best, best = np.unique(sorted(sim)).tolist()[-2:]
    best_idxs = np.where(sim == best)[0]
    second_best_idxs = np.where(sim == second_best)[0]
    return frag, [other_frags[i] for i in best_idxs], [other_frags[i] for i in second_best_idxs]


class Vocab:
    @classmethod
    def from_file(cls, filename):
        vocab = cls()
        
        data = pd.read_csv(filename, index_col=0)
        
        vocab.most_similar_1 = [None] * data.shape[0]
        vocab.most_similar_2 = [None] * data.shape[0]

        for idx, smi in data.smiles.iteritems():
            vocab._frag2idx[smi] = idx
            vocab._idx2frag[idx] = smi
            vocab.most_similar_1[idx] = eval(data.most_similar_1.iloc[idx])
            vocab.most_similar_2[idx] = eval(data.most_similar_2.iloc[idx])
            
        vocab._freq = data.freqs.to_dict()
        return vocab

    def __init__(self):
        self._freq = {}
        self._frag2idx = {}
        self._idx2frag = {}
        self.most_similar_1 = []
        self.most_similar_2 = []

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._idx2frag[key]
        return self._frag2idx[key]

    def update(self, frag):
        if frag not in self._frag2idx:
            idx = len(self._frag2idx)
            self._frag2idx[frag] = idx
            self._idx2frag[idx] = frag

        if frag not in self._freq:
            self._freq[frag] = 1
        else:
            self._freq[frag] += 1

    def __len__(self):
        return len(self._idx2frag)

    def __iter__(self):
        return iter(self._frag2idx)

    def freq(self, key):
        if isinstance(key, int):
            key = self._idx2frag[key]
        return self._freq[key]

    def to_dataframe(self):
        idx, smiles = zip(*self._idx2frag.items())
        freqs = list(self._freq.values())
        df = pd.DataFrame.from_dict({
            "smiles": smiles, 
            "freqs": freqs, 
            "most_similar_1": self.most_similar_1, 
            "most_similar_2": self.most_similar_2})
        df.index = idx
        return df

    def save(self, path):
        df = self.to_dataframe()
        df.to_csv(path)

    def unigram_prob(self, use_tokens=False):
        freqs = list(self._freq.values())
        if use_tokens:
            freqs = ([0] * len(Tokens)) + freqs
        freqs = np.array(freqs, dtype=np.float) ** 0.75
        return freqs / freqs.sum()


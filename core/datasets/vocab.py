from enum import Enum
import numpy as np
import pandas as pd


class Tokens(Enum):
    PAD = 0
    SOS = 1
    EOS = 2
    MASK = 3


class Vocab:
    @classmethod
    def from_file(cls, filename):
        vocab = cls()
        
        data = pd.read_csv(filename, index_col=0)
        
        for idx, smi in data.smiles.iteritems():
            vocab._frag2idx[smi] = idx
            vocab._idx2frag[idx] = smi
            
        vocab._freq = data.freqs.to_dict()
        return vocab

    def __init__(self):
        self._freq = {}
        self._frag2idx = {}
        self._idx2frag = {}

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
            "freqs": freqs})
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
        

def populate_vocab(df, n_jobs):
    print("Create vocab...", end=" ")
    vocab = Vocab()
    
    for _, frags in df.frags.iteritems():
        [vocab.update(f) for f in frags]
    print("Done.")
    
    return vocab


def get_vocab(df, path):
    vocab = Vocab.from_file(path)
    new_vocab = Vocab()
    
    for _, frags in df.frags.iteritems():
        [new_vocab.update(f) for f in frags]
    
    return new_vocab
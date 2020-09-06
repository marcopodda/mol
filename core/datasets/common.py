from torch.utils import data
from core.datasets.utils import frag2data


class VocabDataset(data.Dataset):
    def __init__(self, vocab):
        self.vocab = vocab

    def __getitem__(self, index):
        mol_data = self.vocab[index]
        data = frag2data(mol_data)
        return data

    def __len__(self):
        return len(self.vocab)

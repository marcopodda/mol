import numpy as np
from argparse import Namespace

from torch.utils import data

from core.datasets.preprocess import get_data
from core.datasets.utils import to_data
from core.utils.vocab import Tokens


class MolecularDataset(data.Dataset):
    def __init__(self, hparams, output_dir, name):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            
        self.data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.max_length = self.data.length.max() + 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data.iloc[index]
        seq_len = data.length
        data = to_data(data, self.vocab, self.max_length)
        to_mask = np.random.choice(range(1, seq_len))
        data.inseq[:, to_mask] = Tokens.MASK.value
        return data
        
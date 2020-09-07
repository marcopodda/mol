from argparse import Namespace

import torch

from core.datasets.utils import get_data, frag2data, fragslist2data, build_frag_sequence
from core.utils.serialization import load_numpy, save_numpy


class TranslationDataset:
    def __init__(self, hparams, output_dir, name):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.name = name
        
        self.data, self.vocab = get_data(output_dir, name, hparams.num_samples)
        self.max_length = self.data.length.max() + 1
        
        self.sos = self._initialize_token("sos")
        self.eos = self._initialize_token("eos")
    
    def _initialize_token(self, name):
        path = self.output_dir / "DATA" / f"{name}_{self.hparams.frag_dim_embed}.dat"    
        if path.exists():
            token = torch.FloatTensor(load_numpy(path))
        else:
            token = torch.randn((1, self.hparams.frag_dim_embed))
            save_numpy(token.numpy(), path)
        return token

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        mol_data_x = self.data.iloc[index]
        frags_list_x = mol_data_x.frags
        data_x = fragslist2data(frags_list_x)
        data_x["seq"] = build_frag_sequence(frags_list_x, self.vocab, self.max_length)
        
        if mol_data_x.is_x == 1:
            target = self.data[self.data.smiles==mol_data_x.target]
            mol_data_y = self.data[self.data.smiles==mol_data_x.target].iloc[0]
            frags_list_y = mol_data_y.frags
            data_y = fragslist2data(frags_list_y)
            data_y["seq"] = build_frag_sequence(frags_list_y, self.vocab, self.max_length)
            return data_x, data_y
            
        return data_x, data_x

    @property
    def train_indices(self):
        return self.data[self.data.is_x==1].index.tolist()
    
    @property
    def val_indices(self):
        return self.data[self.data.is_valid==1].index.tolist()
    
    @property
    def test_indices(self):
        return self.data[self.data.is_test==1].index.tolist()
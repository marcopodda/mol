import numpy as np
from argparse import Namespace

import torch

from core.mols.props import get_fingerprint
from core.utils.serialization import load_numpy, save_numpy


class TranslationBaseDataset:
    pass
    # def __init__(self, hparams, output_dir, name):
    #     if isinstance(hparams, dict):
    #         hparams = Namespace(**hparams)
        
    #     self.hparams = hparams
    #     self.output_dir = output_dir
    #     self.name = name
        
    #     data, self.vocab = get_data(output_dir, name, hparams.num_samples)
    #     self.data = self.filter_data(data).reset_index(drop=True)
    #     self.max_length = data.length.max() + 1
        
    #     self.sos = self._initialize_token("sos")
    #     self.eos = self._initialize_token("eos")
        
    # def _initialize_token(self, name):
    #     path = self.output_dir / "DATA" / f"{name}_{self.hparams.frag_dim_embed}.dat"    
    #     if path.exists():
    #         token = torch.FloatTensor(load_numpy(path))
    #     else:
    #         token = torch.randn((1, self.hparams.frag_dim_embed))
    #         save_numpy(token.numpy(), path)
    #     return token

    # def filter_data(self, data):
    #     raise NotImplementedError
    
    # def __len__(self):
    #     return self.data.shape[0]
    
    # def __getitem__(self, index):
    #     mol_data_x = self.data.iloc[index]
    #     data_x = fragslist2data(mol_data_x.frags)
    #     data_x["seq"] = build_frag_sequence(mol_data_x.frags, self.vocab, self.max_length)
    #     fingerprint_x =  torch.FloatTensor(get_fingerprint(mol_data_x.smiles)).view(1, -1)
    #     return data_x, fingerprint_x


class TranslationTrainDataset(TranslationBaseDataset):
    pass
    # def filter_data(self, data):
    #     return data[(data.is_x==1) | (data.is_y==1)]
    
    # def __len__(self):
    #     return self.data[self.data.is_x==1].shape[0]

    # def __getitem__(self, index):
    #     mol_data_x = self.data.iloc[index]
    #     data_x = fragslist2data(mol_data_x.frags)
    #     data_x["seq"] = build_frag_sequence(mol_data_x.frags, self.vocab, self.max_length)
    #     fingerprint_x = torch.FloatTensor(get_fingerprint(mol_data_x.smiles)).view(1, -1)
        
    #     target = self.data.smiles==mol_data_x.target
    #     mol_data_y = self.data[target].iloc[0]
    #     data_y = fragslist2data(mol_data_y.frags)
    #     data_y["seq"] = build_frag_sequence(mol_data_y.frags, self.vocab, self.max_length)
    #     fingerprint_y = torch.FloatTensor(get_fingerprint(mol_data_y.smiles)).view(1, -1)
        
    #     return data_x, fingerprint_x, data_y, fingerprint_y
    

class TranslationValDataset(TranslationBaseDataset):
    pass
    # def filter_data(self, data):
    #     return data[data.is_valid==1]
    

class TranslationTestDataset(TranslationBaseDataset):
    pass
    # def filter_data(self, data):
    #     return data[data.is_test==1]
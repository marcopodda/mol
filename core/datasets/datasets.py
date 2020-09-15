import numpy as np
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from core.hparams import HParams
from core.datasets.features import mol2nx, FINGERPRINT_DIM
from core.datasets.settings import DATA_DIR
from core.datasets.utils import pad, load_data
from core.datasets.vocab import Tokens
from core.mols.utils import mol_from_smiles
from core.mols.props import get_fingerprint
from core.utils.serialization import load_numpy, save_numpy


class BaseDataset:
    corrupt_input = None

    def __init__(self, hparams, dataset_name):
        self.hparams = HParams.load(hparams)
        self.dataset_name = dataset_name

        self.data, self.vocab, self.max_length = self.get_dataset()
        self.sos = self._initialize_token("sos")
        self.eos = self._initialize_token("eos")
        self.mask = self._initialize_token("mask")

    def _initialize_token(self, name):
        path = DATA_DIR / self.dataset_name / f"{name}_{self.hparams.frag_dim_embed}.dat"
        if path.exists():
            token = torch.FloatTensor(load_numpy(path))
        else:
            token = torch.randn((1, self.hparams.frag_dim_embed))
            save_numpy(token.numpy(), path)
        return token

    def _to_data(self, frags_smiles, is_target):
        targets = self._get_target_sequence(frags_smiles)

        if self.corrupt_input is True and is_target is False:
            frags_smiles = self._corrupt_input_seq(frags_smiles)

        frags_list = [mol_from_smiles(f) for f in frags_smiles]
        frag_graphs = [mol2nx(f) for f in frags_list]
        num_nodes = [f.number_of_nodes() for f in frag_graphs]

        data = from_networkx(nx.disjoint_union_all(frag_graphs))
        frags_batch = [torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)]
        data["frags_batch"] = torch.cat(frags_batch)
        data["length"] = torch.LongTensor([[len(frags_list)]])
        data["target"] = targets
        return data

    def _get_fingerprint(self, smiles, is_target):
        fingerprint = np.array(get_fingerprint(smiles), dtype=np.int)
        if self.corrupt_input is True and is_target is False:
            fingerprint = self._corrupt_input_fingerprint(fingerprint)
        fingerprint_tx = torch.FloatTensor(fingerprint).view(1, -1)
        return fingerprint_tx

    def _get_target_sequence(self, frags_list):
        seq = [self.vocab[f] + len(Tokens) for f in frags_list] + [Tokens.EOS.value]
        padded_seq = pad(seq, self.max_length)
        return padded_seq

    def _corrupt_input_seq(self, seq):
        if np.random.rand() > 0.25 and len(seq) > 2:
            delete_index = np.random.choice(len(seq)-1)
            seq.pop(delete_index)

        mask_index = None
        if np.random.rand() > 0.25:
            mask_index = np.random.choice(len(seq)-1)
            seq[mask_index] = self.vocab.sample()

        if np.random.rand() > 0.25 and len(seq) + 2 <= self.max_length:
            add_index = np.random.choice(len(seq)-1)
            seq.insert(add_index, self.vocab.sample())

        return seq

    def _corrupt_input_fingerprint(self, fingerprint):
        num_to_flip = np.clip(int(np.random.randn() * 20 + 68), a_min=1, a_max=None)
        flip_indices = np.random.choice(FINGERPRINT_DIM-1, num_to_flip)
        fingerprint[flip_indices] = np.logical_not(fingerprint[flip_indices])
        return fingerprint

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x_molecule, x_fingerprint = self.get_input_data(index)
        y_molecule, y_fingerprint = self.get_target_data(index)
        return x_molecule, x_fingerprint, y_molecule, y_fingerprint

    def get_dataset(self):
        data, vocab, max_length = load_data(self.dataset_name)
        return data, vocab, max_length

    def get_input_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._to_data(mol_data.frags, is_target=False)
        fingerprint = self._get_fingerprint(mol_data.smiles, is_target=False)
        return data, fingerprint

    def get_target_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._to_data(mol_data.frags, is_target=True)
        fingerprint = self._get_fingerprint(mol_data.smiles, is_target=True)
        return data, fingerprint


class TrainDataset(BaseDataset):
    corrupt_input = True

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_train == True].reset_index(drop=True)
        return data, vocab, max_length


class EvalDataset(BaseDataset):
    corrupt_input = False

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_val == True].reset_index(drop=True)
        return data, vocab, max_length

    def __getitem__(self, index):
        x_molecule, x_fingerprint = self.get_input_data(index)
        return x_molecule, x_fingerprint


class VocabDataset:
    def __init__(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, index):
        frag_smiles = self.vocab[index]
        graph = mol2nx(frag_smiles)
        data = from_networkx(graph)
        return data

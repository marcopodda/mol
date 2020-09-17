import numpy as np
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from core.hparams import HParams
from core.datasets.features import mol2nx, FINGERPRINT_DIM
from core.datasets.settings import DATA_DIR
from core.datasets.utils import pad, load_data
from core.datasets.vocab import Tokens
from core.mols.utils import mol_from_smiles, mol_to_smiles, mols_from_smiles
from core.mols.split import join_fragments
from core.mols.props import get_fingerprint, similarity, drd2, logp
from core.utils.serialization import load_numpy, save_numpy


class BaseDataset:
    def __init__(self, hparams, dataset_name):
        self.hparams = HParams.load(hparams)
        self.dataset_name = dataset_name

        self.data, self.vocab, self.max_length = self.get_dataset()

        self.sos = self._initialize_token("sos")
        self.eos = self._initialize_token("eos")
        self.mask = self._initialize_token("mask")

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.data.shape[0]

    def _initialize_token(self, name):
        path = DATA_DIR / self.dataset_name / f"{name}_{self.hparams.frag_dim_embed}.dat"
        if path.exists():
            token = torch.FloatTensor(load_numpy(path))
        else:
            token = torch.randn((1, self.hparams.frag_dim_embed))
            save_numpy(token.numpy(), path)
        return token

    def _corrupt_seq(self, seq, reps=1):
        seq = seq[:]

        # deletion
        for _ in range(reps):
            if np.random.rand() > 0.25 and len(seq) > 2:
                delete_index = np.random.choice(len(seq)-1)
                seq.pop(delete_index)

        # replacement
        for _ in range(reps):
            if  np.random.rand() > 0.25:
                mask_index = np.random.choice(len(seq)-1)
                probs = self.vocab.condition(seq[mask_index])
                seq[mask_index] = self.vocab.sample(probs=probs)

        # insertion
        for _ in range(reps):
            if np.random.rand() > 0.25 and len(seq) + 2 <= self.max_length:
                add_index = np.random.choice(len(seq)-1)
                probs = self.vocab.condition(seq[add_index])
                seq.insert(add_index, self.vocab.sample(probs=probs))

        return seq

    def _get_data(self, frags_smiles, corrupt=False):
        if corrupt is True:
            frags_smiles = self._corrupt_seq(frags_smiles)

        frags_list = [mol_from_smiles(f) for f in frags_smiles]
        frag_graphs = [mol2nx(f) for f in frags_list]
        num_nodes = [f.number_of_nodes() for f in frag_graphs]

        data = from_networkx(nx.disjoint_union_all(frag_graphs))
        frags_batch = [torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)]
        data["frags_batch"] = torch.cat(frags_batch)
        data["length"] = torch.LongTensor([len(frags_list)])
        data["target"] = self._get_target_sequence(frags_smiles)
        return data

    def _get_target_sequence(self, frags_list):
        seq = [self.vocab[f] + len(Tokens) for f in frags_list] + [Tokens.EOS.value]
        padded_seq = pad(seq, self.max_length)
        return padded_seq

    def get_dataset(self):
        data, vocab, max_length = load_data(self.dataset_name)
        return data, vocab, max_length


class TrainDataset(BaseDataset):
    def __getitem__(self, index):
        x_molecule, x_smiles = self.get_input_data(index)
        y_molecule, y_smiles = self.get_target_data(index)
        sim = similarity(x_smiles, y_smiles)
        target = torch.FloatTensor([[sim]])
        return x_molecule, y_molecule, target

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_train == True].reset_index(drop=True)
        return data, vocab, max_length

    def get_input_data(self, index):
        mol_data = self.data.iloc[index]
        corrupt = bool(round(np.random.rand()))
        data = self._get_data(mol_data.frags, corrupt=True)
        return data, mol_data.smiles

    def get_target_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._get_data(mol_data.frags, corrupt=False)
        return data, mol_data.smiles


class EvalDataset(BaseDataset):
    def __getitem__(self, index):
        x_molecule = self.get_input_data(index)
        return x_molecule

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_train == True][:1000].reset_index(drop=True)
        return data, vocab, max_length

    def get_input_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._get_data(mol_data.frags, corrupt=False)
        return data


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

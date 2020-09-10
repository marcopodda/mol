import numpy as np
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from core.hparams import HParams
from core.datasets.features import mol2nx, FINGERPRINT_DIM
from core.datasets.utils import pad, load_data
from core.datasets.vocab import Tokens
from core.mols.utils import mol_from_smiles
from core.mols.props import get_fingerprint
from core.utils.serialization import load_numpy, save_numpy


class BaseDataset:
    corrupt = False

    def __init__(self, hparams, output_dir, dataset_name):
        self.hparams = HParams.load(hparams)
        self.output_dir = output_dir
        self.dataset_name = dataset_name

        self.data, self.vocab, self.max_length = self.get_data()
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

    def _to_data(self, frags_smiles, is_target, add_noise=False):
        if add_noise and self.corrupt:
            frags_smiles = self._corrupt_seq(frags_smiles)
        frags_list = [mol_from_smiles(f) for f in frags_smiles]
        frag_graphs = [mol2nx(f) for f in frags_list]
        num_nodes = [f.number_of_nodes() for f in frag_graphs]

        data = from_networkx(nx.disjoint_union_all(frag_graphs))
        frags_batch = [torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)]
        data["frags_batch"] = torch.cat(frags_batch)
        data["length"] = torch.LongTensor([[len(frags_list)]])
        if is_target:
            data["target"] = self._get_target_sequence(frags_smiles)
        return data

    def _get_fingerprint(self, smiles, add_noise):
        fingerprint = np.array(get_fingerprint(smiles), dtype=np.int)
        if add_noise and self.corrupt:
            fingerprint = self._corrupt_fingerprint(fingerprint)
        fingerprint_tx = torch.FloatTensor(fingerprint).view(1, -1)
        return fingerprint_tx

    def _get_target_sequence(self, frags_list):
        seq = [self.vocab[f] + len(Tokens) for f in frags_list] + [Tokens.EOS.value]
        padded_seq = pad(seq, self.max_length)
        return padded_seq

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x_molecule, x_fingerprint = self.get_input_data(index)
        y_molecule, y_fingerprint = self.get_target_data(index)
        return x_molecule, x_fingerprint, y_molecule, y_fingerprint

    def get_data(self):
        path = self.output_dir
        name = self.dataset_name
        num_samples = self.hparams.num_samples
        data, vocab, max_length = load_data(path, name, num_samples)
        return data, vocab, max_length

    def get_input_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._to_data(mol_data.frags, is_target=False, add_noise=True)
        fingerprint = self._get_fingerprint(mol_data.smiles, add_noise=True)
        return data, fingerprint

    def get_target_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._to_data(mol_data.frags, is_target=True, add_noise=False)
        fingerprint = self._get_fingerprint(mol_data.smiles, add_noise=False)
        return data, fingerprint

    def _sample_poisson(self, length, lam):
        sample = np.random.poisson(lam)
        while sample >= length - 1:
            sample = np.random.poisson(lam)
        return sample

    def _corrupt_seq(self, seq):
        seq = seq[:]
        num_to_delete = min(1, self._sample_poisson(len(seq), lam=0.25))
        delete_indices = np.random.choice(len(seq)-1, num_to_delete).tolist()
        for delete_index in delete_indices:
            seq.remove(seq[delete_index])

        num_to_replace = self._sample_poisson(len(seq), lam=0.5)
        replacement_indices = np.random.choice(len(seq)-1, num_to_replace).tolist()
        for replacement_index in replacement_indices:
            seq[replacement_index] = self.vocab.sample()

        num_to_add = self._sample_poisson(len(seq), lam=0.5)
        add_indices = np.random.choice(len(seq)-1, num_to_add).tolist()
        for add_index in add_indices:
            if len(seq) + 2 < self.max_length:
                seq.insert(add_index, self.vocab.sample())

        return seq

    def _corrupt_fingerprint(self, fingerprint):
        num_to_flip = np.random.poisson(8)
        flip_indices = np.random.choice(FINGERPRINT_DIM-1, num_to_flip)
        fingerprint[flip_indices] = np.logical_not(fingerprint[flip_indices])
        return fingerprint


class EvalDataset(BaseDataset):
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
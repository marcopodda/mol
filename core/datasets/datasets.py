import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from core.datasets.features import mol2nx
from core.datasets.utils import pad, load_data
from core.datasets.vocab import Tokens
from core.mols.utils import mol_from_smiles
from core.mols.props import get_fingerprint
from core.utils.serialization import load_numpy, save_numpy


class BaseDataset:
    def __init__(self, hparams, output_dir, dataset_name):
        self.hparams = hparams
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

    def _to_data(self, frags_smiles, is_target):
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

    def _get_fingerprint(self, smiles):
        fingerprint = get_fingerprint(smiles)
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
        data = self._to_data(mol_data.frags, is_target=False)
        fingerprint = self._get_fingerprint(mol_data.smiles)
        return data, fingerprint

    def get_target_data(self, index):
        mol_data = self.data.iloc[index]
        data = self._to_data(mol_data.frags, is_target=True)
        fingerprint = self._get_fingerprint(mol_data.smiles)
        return data, fingerprint


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

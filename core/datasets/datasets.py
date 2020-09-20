import math
import numpy as np
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from core.hparams import HParams
from core.datasets.features import mol2nx, FINGERPRINT_DIM
from core.datasets.settings import DATA_DIR
from core.datasets.utils import pad, load_data
from core.datasets.vocab import Tokens
from core.mols.utils import mol_from_smiles, mol_to_smiles, mols_from_smiles, mols_to_smiles
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
            stdv = 1. / math.sqrt(token.size(1))
            token.data.uniform_(-stdv, stdv)
            save_numpy(token.numpy(), path)
        return token

    def _corrupt_seq(self, seq, reps=1):
        seq = seq[:]

        # deletion
        for _ in range(reps):
            if np.random.rand() > 0.1 and len(seq) > 2:
                delete_index = np.random.choice(len(seq)-1)
                seq.pop(delete_index)

        # replacement
        for _ in range(reps):
            if  np.random.rand() > 0.1:
                mask_index = np.random.choice(len(seq)-1)
                probs = self.vocab.condition(seq[mask_index])
                seq[mask_index] = self.vocab.sample(probs=probs)

        # insertion
        for _ in range(reps):
            if np.random.rand() > 0.1 and len(seq) + 2 <= self.max_length:
                add_index = np.random.choice(len(seq)-1)
                probs = self.vocab.condition(seq[add_index])
                seq.insert(add_index, self.vocab.sample(probs=probs))

        return seq

    def _get_data(self, frags_smiles, corrupt=False, reps=1):
        if corrupt is True:
            frags_smiles = self._corrupt_seq(frags_smiles, reps=reps)

        frags_list = [mol_from_smiles(f) for f in frags_smiles]
        frag_graphs = [mol2nx(f) for f in frags_list]
        num_nodes = [f.number_of_nodes() for f in frag_graphs]

        data = from_networkx(nx.disjoint_union_all(frag_graphs))
        frags_batch = [torch.LongTensor([i]).repeat(n) for (i, n) in enumerate(num_nodes)]
        data["frags_batch"] = torch.cat(frags_batch)
        data["length"] = torch.LongTensor([len(frags_list)])
        data["target"] = self._get_target_sequence(frags_smiles)
        return data, frags_list

    def _get_target_sequence(self, frags_smiles):
        seq = [self.vocab[f] + len(Tokens) for f in frags_smiles] + [Tokens.EOS.value]
        padded_seq = pad(seq, self.max_length)
        return padded_seq

    def compute_similarity(self, frags1, frags2):
        joined1 = mol_to_smiles(join_fragments(frags1))
        joined2 = mol_to_smiles(join_fragments(frags2))
        return similarity(joined1, joined2)

    def get_dataset(self):
        data, vocab, max_length = load_data(self.dataset_name)
        return data, vocab, max_length

    def get_input_data(self, index, corrupt, reps=1):
        mol_data = self.data.iloc[index]
        data, frags_list = self._get_data(mol_data.frags, corrupt=corrupt, reps=reps)
        return data, mol_data.smiles, frags_list

    def get_target_data(self, index, corrupt, reps=1):
        return self.get_input_data(index, corrupt=corrupt, reps=reps)


class TrainDataset(BaseDataset):
    def __getitem__(self, index):
        anc, anc_smiles, anc_frags = self.get_input_data(index, corrupt=True, reps=1)
        pos, pos_smiles, pos_frags = self.get_target_data(index, corrupt=False)
        neg, neg_smiles, neg_frags = self.get_target_data(index, corrupt=True, reps=2)

        sim1 = self.compute_similarity(anc_frags, pos_frags)
        sim2 = self.compute_similarity(anc_frags, neg_frags)

        while sim1 == sim2:
            anc, anc_smiles, anc_frags = self.get_input_data(index, corrupt=True, reps=1)
            neg, neg_smiles, neg_frags = self.get_target_data(index, corrupt=True, reps=2)

            sim1 = self.compute_similarity(anc_frags, pos_frags)
            sim2 = self.compute_similarity(neg_frags, pos_frags)

        if sim2 > sim1:
            temp = anc.clone()
            anc = neg.clone()
            neg = temp.clone()
            del temp

        prop_anc = torch.FloatTensor([[0.0]])
        prop_pos = torch.FloatTensor([[0.0]])
        prop_neg = torch.FloatTensor([[0.0]])

        return anc, pos, neg, prop_anc, prop_pos, prop_neg

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_train == True].reset_index(drop=True)
        return data, vocab, max_length


class EvalDataset(BaseDataset):
    def __getitem__(self, index):
        x_molecule, _, _ = self.get_input_data(index, corrupt=False)
        return x_molecule

    def get_dataset(self):
        data, vocab, max_length = super().get_dataset()
        data = data[data.is_val == True].reset_index(drop=True)
        return data, vocab, max_length


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

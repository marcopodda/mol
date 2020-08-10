from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN
from layers.vae import VAE, MMDVAE
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder


class PretrainModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams

        self.gnn = GNN(hparams)

    def forward(self, batch):
        anc, pos, neg = batch
        anc_h = self.gnn(anc)
        pos_h = self.gnn(pos)
        neg_h = self.gnn(neg)
        return anc_h, pos_h, neg_h

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


class TripletModel(nn.Module):
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
    
    def loss(self, outputs):
        anc, pos, neg = outputs
        return F.triplet_margin_loss(anc, pos, neg)
    
    def predict(self, batch):
        return self.gnn(batch)


class SkipgramModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gnn_in = GNN(hparams)
        self.gnn_out = GNN(hparams)

    def forward(self, batch):
        target, context, negatives = batch
        emb_target = self.gnn_in(target)
        emb_context = self.gnn_out(context)
        emb_negatives = self.gnn_out(negatives)

        pos_score = torch.mul(emb_target, emb_context).squeeze()
        pos_score = torch.sum(pos_score, dim=1)

        num_negatives, dim_embed = self.hparams.num_negatives, self.hparams.gnn_dim_embed
        emb_negatives = emb_negatives.view(-1, num_negatives, dim_embed)
        neg_score = torch.bmm(emb_negatives, emb_target.unsqueeze(2)).squeeze()
        return pos_score, neg_score

    def loss(self, outputs): 
        pos_score, neg_score = outputs
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(-neg_score)
        return -(torch.sum(pos_score) + torch.sum(neg_score))

    def predict(self, batch):
        return self.gnn_in(batch)
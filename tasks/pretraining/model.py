import torch
from torch import nn
from torch.nn import functional as F

from layers.graphconv import GNN


class SkipGram(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gnn_in = GNN(hparams)
        self.gnn_out = GNN(hparams)

    def forward(self, target, context, negatives):
        emb_target = self.gnn_in(target)
        emb_context = self.gnn_out(context)
        emb_negatives = self.gnn_out(negatives)

        pos_score = torch.mul(emb_target, emb_context).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        
        num_negatives, dim_embed = self.hparams.num_negatives, self.hparams.gnn_dim_embed
        emb_negatives = emb_negatives.view(-1, num_negatives, dim_embed)
        neg_score = torch.bmm(emb_negatives, emb_target.unsqueeze(2)).squeeze()
        return pos_score, neg_score

    def loss(self, pos_score, neg_score):
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(-neg_score)
        return -(torch.sum(pos_score) + torch.sum(neg_score))

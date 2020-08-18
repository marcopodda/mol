from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder


class TripletModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams

        self.gnn = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed, 
            dim_hidden=hparams.gnn_dim_hidden, 
            dim_output=hparams.frag_dim_embed)

    def forward(self, batch):
        anc, pos, neg = batch
        anc_h = self.gnn(anc)
        pos_h = self.gnn(pos)
        neg_h = self.gnn(neg)
        return anc_h, pos_h, neg_h
    
    def loss(self, outputs, batch):
        anc, pos, neg = outputs
        return F.triplet_margin_loss(anc, pos, neg)
    
    def predict(self, batch):
        with torch.no_grad():
            return self.gnn(batch)


class SkipgramModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        self.gnn_in = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed, 
            dim_hidden=hparams.gnn_dim_hidden, 
            dim_output=hparams.frag_dim_embed)
        
        self.gnn_out = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed, 
            dim_hidden=hparams.gnn_dim_hidden, 
            dim_output=hparams.frag_dim_embed)

    def forward(self, batch):
        target, context, negatives = batch
        emb_target = self.gnn_in(target)
        emb_context = self.gnn_out(context)
        emb_negatives = self.gnn_out(negatives)

        pos_score = torch.mul(emb_target, emb_context).squeeze()
        pos_score = torch.sum(pos_score, dim=1)

        num_negatives, dim_embed = self.hparams.num_negatives, self.hparams.frag_dim_embed
        emb_negatives = emb_negatives.view(-1, num_negatives, dim_embed)
        neg_score = torch.bmm(emb_negatives, emb_target.unsqueeze(2)).squeeze()
        return pos_score, neg_score

    def loss(self, outputs, batch): 
        pos_score, neg_score = outputs
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(-neg_score)
        return -(torch.sum(pos_score) + torch.sum(neg_score))

    def predict(self, batch):
        with torch.no_grad():
            return self.gnn(batch)
        

class EncoderDecoderModel(nn.Module):
    def __init__(self, hparams, vocab_size, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.num_embeddings = vocab_size + len(Tokens)
        self.embedding_dropout = hparams.embedding_dropout
        self.dec_embedder = nn.Embedding(self.num_embeddings, hparams.frag_dim_embed, padding_idx=Tokens.PAD.value)

        self.encoder = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed,
            dim_hidden=hparams.gnn_dim_hidden,
            dim_output=hparams.frag_dim_embed)

        self.decoder = Decoder(
            hparams=hparams,
            max_length=max_length,
            rnn_dropout=hparams.rnn_dropout,
            num_layers = hparams.rnn_num_layers,
            dim_input=hparams.frag_dim_embed,
            dim_hidden=hparams.frag_dim_embed,
            dim_output=self.num_embeddings)

    def forward(self, batch):
        h = self.encoder(batch)
        x = self.dec_embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        
        h = h[None, :, :].repeat(self.hparams.rnn_num_layers, 1, 1)
        outputs, _ = self.decoder(x, h)
        return outputs

    def loss(self, outputs, batch):
        # outputs = outputs.view(-1)
        targets = batch.outseq.view(-1)
        return F.cross_entropy(outputs, targets, ignore_index=Tokens.PAD.value)
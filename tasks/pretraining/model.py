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
    def __init__(self, hparams, vocab_size, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams

        self.max_length = max_length
        self.gnn_num_layers = hparams.gnn_num_layers
        self.gnn_dim_input = ATOM_FDIM
        self.gnn_dim_hidden = hparams.gnn_dim_hidden
        self.gnn_dim_embed = hparams.gnn_dim_embed

        self.embedding_dropout = hparams.embedding_dropout
        self.rnn_dropout = hparams.rnn_dropout
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_dim_input = self.gnn_dim_embed
        self.rnn_dim_hidden = self.gnn_dim_embed
        self.rnn_dim_output = vocab_size + len(Tokens)

        self.gnn = GNN(hparams)

        self.embedder = nn.Embedding(self.rnn_dim_output, self.gnn_dim_embed, padding_idx=0)

        self.decoder = Decoder(
            hparams=hparams,
            max_length=self.max_length,
            rnn_dropout=self.rnn_dropout,
            num_layers = self.rnn_num_layers,
            dim_input=self.rnn_dim_input,
            dim_hidden=self.rnn_dim_hidden,
            dim_output=self.rnn_dim_output
        )

        if self.hparams.tie_weights:
            self.decoder.tie_weights(self.embedder)

    def forward(self, batch):
        h = self.gnn(batch)
        h = h[None, :, :].repeat(self.rnn_num_layers, 1, 1)
        
        x = self.embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        output, hidden = self.decoder(x, h)
        return output

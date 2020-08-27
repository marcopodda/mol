from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN, GNNEmbedder
from layers.vae import VAE
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder

from torch_geometric.nn import global_add_pool


class Model(nn.Module):        
    def __init__(self, hparams, output_dir, vocab_size, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.num_embeddings = vocab_size + len(Tokens)
        self.embedding_dropout = hparams.embedding_dropout
        self.max_length = max_length

        self.encoder = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed,
            dim_hidden=hparams.gnn_dim_hidden,
            dim_output=hparams.rnn_dim_state)
            # dim_output=hparams.rnn_dim_state // 2)
            
        self.gnn_logv = GNN(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed,
            dim_hidden=hparams.gnn_dim_hidden,
            dim_output=hparams.rnn_dim_state // 2)

        self.vae = VAE(
            hparams=hparams,
            vocab_size=vocab_size,
            dim_input=hparams.rnn_dim_state,
            dim_latent=hparams.rnn_dim_state // 2,
            dim_output=hparams.rnn_dim_state)

        self.embedder = GNNEmbedder(
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
            dim_hidden=hparams.rnn_dim_state,
            dim_output=vocab_size + len(Tokens))

    def forward(self, batch):
        graphs_batch, frags_batch, seq_matrix = batch
        
        h = self.encoder(graphs_batch)
        # logv = self.gnn_logv(graphs_batch)
        # h, vae_loss = self.vae(mean, logv)
        h = h.unsqueeze(0).repeat(self.hparams.rnn_num_layers, 1, 1)
        vae_loss = 0
        
        x = self.embedder(frags_batch, seq_matrix)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        output, h_dec = self.decoder(x, h)
        
        return output, vae_loss
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

        self.embedder = GNNEmbedder(
            hparams=hparams,
            num_layers=hparams.gnn_num_layers,
            dim_edge_embed=hparams.gnn_dim_edge_embed,
            dim_hidden=hparams.gnn_dim_hidden,
            dim_output=hparams.frag_dim_embed)

        self.encoder = Encoder(
            hparams=hparams, 
            rnn_dropout=hparams.rnn_dropout, 
            num_layers=hparams.rnn_num_layers, 
            dim_input=hparams.frag_dim_embed, 
            dim_hidden=hparams.rnn_dim_state)

        self.decoder = Decoder(
            hparams=hparams,
            max_length=max_length,
            rnn_dropout=hparams.rnn_dropout,
            num_layers = hparams.rnn_num_layers,
            dim_input=hparams.frag_dim_embed * 2,
            dim_hidden=hparams.rnn_dim_state,
            dim_output=vocab_size + len(Tokens))

    def forward(self, batch):
        frags_x, frags_y, enc_inputs, dec_inputs = batch
        bof, enc_hidden, enc_outputs = self.encode(frags_x, enc_inputs)
        return self.decode(frags_y, bof, enc_hidden, enc_outputs, dec_inputs)
    
    def encode(self, frags_batch, enc_inputs):
        enc_inputs = self.embedder(frags_batch, enc_inputs, input=False)
        bof = self.compute_bof(frags_batch, enc_inputs)
        enc_inputs = F.dropout(enc_inputs, p=self.embedding_dropout, training=self.training)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        return bof, enc_hidden, enc_outputs
        
    def decode(self, frags_batch, bof, enc_hidden, enc_outputs, dec_inputs):
        dec_inputs = self.embedder(frags_batch, dec_inputs, input=True)
        dec_inputs = torch.cat([dec_inputs, bof.repeat(1, dec_inputs.size(1), 1)], dim=-1)
        dec_inputs = F.dropout(dec_inputs, p=self.embedding_dropout, training=self.training)
        return self.decoder.decode_with_attention(dec_inputs, enc_hidden, enc_outputs)
    
    def compute_bof(self, batch, enc_inputs):
        B, S, D = enc_inputs.size()
        
        bofs = []
        for i, l in enumerate(batch.length):
            bof = enc_inputs[i, :l, :].sum(dim=0, keepdim=True)
            bofs.append(bof)
        
        return torch.cat(bofs, dim=0).view(B, 1, -1)
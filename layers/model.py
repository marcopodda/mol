from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.vocab import Tokens
from layers.embedder import GNNEmbedder
from layers.encoder import Encoder
from layers.decoder import Decoder


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
            dim_input=hparams.frag_dim_embed,
            dim_hidden=hparams.rnn_dim_state,
            dim_output=vocab_size + len(Tokens))

    def forward(self, batch):
        x_batch, y_batch, enc_inputs, dec_inputs = batch
        enc_hidden, enc_outputs = self.encode(x_batch, enc_inputs)
        logits = self.decode(y_batch, enc_hidden, enc_outputs, dec_inputs)
        return logits

    def encode(self, batch, enc_inputs):
        enc_inputs = self.embedder(batch, enc_inputs, input=False)
        enc_inputs = F.dropout(enc_inputs, p=self.embedding_dropout, training=self.training)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        return enc_hidden, enc_outputs

    def decode(self, batch, enc_hidden, enc_outputs, dec_inputs):
        dec_inputs = self.embedder(batch, dec_inputs, input=True)
        dec_inputs = F.dropout(dec_inputs, p=self.embedding_dropout, training=self.training)
        return self.decoder.decode_with_attention(dec_inputs, enc_hidden, enc_outputs)
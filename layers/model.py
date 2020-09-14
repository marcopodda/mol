from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.hparams import HParams
from core.datasets.vocab import Tokens
from core.datasets.features import ATOM_FDIM, BOND_FDIM, FINGERPRINT_DIM
from layers.embedder import Embedder
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.autoencoder import Autoencoder


class Model(nn.Module):
    def __init__(self, hparams, vocab_size, seq_length):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dim_output = vocab_size + len(Tokens)
        self.seq_length = seq_length
        self.set_dimensions()

        self.embedder = Embedder(
            hparams=hparams,
            num_layers=self.embedder_num_layers,
            dim_input=self.embedder_dim_input,
            dim_edge_features=self.embedder_dim_edge_features,
            dim_edge_embed=self.embedder_dim_edge_embed,
            dim_hidden=self.embedder_dim_hidden,
            dim_output=self.embedder_dim_output)

        self.encoder = Encoder(
            hparams=hparams,
            num_layers=self.encoder_num_layers,
            dim_input=self.encoder_dim_input,
            dim_state=self.encoder_dim_state,
            dropout=self.encoder_dropout,
            dim_output=self.dim_output)

        self.autoencoder = Autoencoder(
            hparams=hparams,
            dim_input=self.autoencoder_dim_input,
            dim_hidden=self.autoencoder_dim_hidden)

        self.decoder = Decoder(
            hparams=hparams,
            num_layers=self.decoder_num_layers,
            dim_input=self.decoder_dim_input,
            dim_state=self.decoder_dim_state,
            dim_output=self.decoder_dim_output,
            dim_attention_input=self.decoder_dim_attention_input,
            dim_attention_output=self.decoder_dim_attention_output,
            dropout=self.decoder_dropout)

    def set_dimensions(self):
        self.embedder_num_layers = self.hparams.gnn_num_layers
        self.embedder_dim_input = ATOM_FDIM
        self.embedder_dim_hidden = self.hparams.gnn_dim_hidden
        self.embedder_dim_edge_features = BOND_FDIM
        self.embedder_dim_edge_embed = self.hparams.gnn_dim_edge_embed
        self.embedder_dim_output = self.hparams.frag_dim_embed
        self.embedder_dropout = self.hparams.embedder_dropout

        self.encoder_num_layers = self.hparams.rnn_num_layers
        self.encoder_dim_input = self.embedder_dim_output
        self.encoder_dim_state = self.hparams.rnn_dim_state
        self.encoder_dropout = self.hparams.rnn_dropout

        self.autoencoder_dim_input = FINGERPRINT_DIM
        self.autoencoder_dim_hidden = self.hparams.autoencoder_dim_hidden

        self.decoder_dim_state = self.autoencoder_dim_hidden
        if self.hparams.concat:
            self.decoder_dim_state += self.encoder_dim_state

        self.decoder_num_layers = self.hparams.rnn_num_layers
        self.decoder_dim_input = self.encoder_dim_input + self.encoder_dim_state
        self.decoder_dim_attention_input = self.decoder_dim_state + self.encoder_dim_state
        self.decoder_dim_attention_output = self.decoder_dim_state
        self.decoder_dim_output = self.dim_output
        self.decoder_dropout = self.encoder_dropout

    def encode(self, batch, enc_inputs, denoise):
        enc_inputs = self.embedder(batch, enc_inputs, input=False)
        enc_inputs = F.dropout(enc_inputs, p=self.embedder_dropout, training=self.training)
        enc_logits, enc_outputs, enc_hidden = self.encoder(enc_inputs, denoise=denoise)
        return enc_logits, enc_hidden, enc_outputs

    def decode(self, batch, enc_hidden, enc_outputs, dec_inputs):
        dec_inputs = self.embedder(batch, dec_inputs, input=True)
        dec_inputs = F.dropout(dec_inputs, p=self.embedder_dropout, training=self.training)
        return self.decoder(dec_inputs, enc_hidden, enc_outputs)

    def forward(self, batch, denoise):
        batch_data, batch_fps, enc_inputs, dec_inputs = batch
        x_batch, y_batch = batch_data
        x_fps, _ = batch_fps

        # embed fragment sequence
        enc_logits, h, enc_outputs = self.encode(x_batch, enc_inputs, denoise=denoise)

        # autoencode fingerprint
        y_hat_fps, autoenc_hidden = self.autoencoder(x_fps)
        # h = autoenc_hidden.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)

        # if self.hparams.concat:
        #     h = torch.cat([h, enc_hidden], dim=-1)

        # decode fragment sequence
        logits = self.decode(y_batch, h, enc_outputs, dec_inputs)
        return logits, enc_logits, y_hat_fps

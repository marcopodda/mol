import torch
from torch import nn
from torch.nn import functional as F

from core.hparams import HParams
from core.datasets.features import ATOM_FDIM, BOND_FDIM, FINGERPRINT_DIM
from layers.embedder import Embedder
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.mlp import MLP
from layers.autoencoder import Autoencoder


class Model(nn.Module):
    def __init__(self, hparams, dim_output):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dim_output = dim_output
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

        self.decoder = Decoder(
            hparams=hparams,
            num_layers=self.decoder_num_layers,
            dim_input=self.decoder_dim_input,
            dim_state=self.decoder_dim_state,
            dim_output=self.decoder_dim_output,
            dim_attention_input=self.decoder_dim_attention_input,
            dim_attention_output=self.decoder_dim_attention_output,
            dropout=self.decoder_dropout)

        self.autoencoder = Autoencoder(
            hparams=self.hparams,
            dim_input=self.autoencoder_dim_input,
            dim_hidden=self.autoencoder_dim_hidden)

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

        self.decoder_dim_state = self.encoder_dim_state
        self.decoder_num_layers = self.hparams.rnn_num_layers
        self.decoder_dim_input = self.encoder_dim_input + self.encoder_dim_state
        self.decoder_dim_attention_input = self.decoder_dim_state + self.encoder_dim_state
        self.decoder_dim_attention_output = self.decoder_dim_state
        self.decoder_dim_output = self.dim_output
        self.decoder_dropout = self.encoder_dropout

        self.autoencoder_dim_input = FINGERPRINT_DIM
        self.autoencoder_dim_hidden = self.encoder_dim_state

    def encode(self, input_frags, encoder_inputs):
        encoder_inputs, bag_of_frags = self.embedder(input_frags, encoder_inputs, input=False)
        encoder_outputs, encoder_hidden = self.encoder(encoder_inputs)
        return encoder_outputs, encoder_hidden, bag_of_frags

    def decode(self, output_frags, decoder_inputs, encoder_outputs, encoder_hidden):
        decoder_inputs, bag_of_frags = self.embedder(output_frags, decoder_inputs, input=True)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, encoder_hidden)
        return decoder_outputs, bag_of_frags

    def forward(self, batch):
        batches, fingerprints, inputs = batch
        x_batch, y_batch = batches
        x_fingerprint, y_fingerprint = fingerprints
        x_inputs, y_inputs = inputs

        # encode input fragment sequence
        x_outputs, x_hidden, x_bag_of_frags = self.encode(x_batch, x_inputs)

        # autoencode input fingerprint
        x_fp_outputs, x_fp_hidden = self.autoencoder(x_fingerprint)
        x_fp_hidden = x_fp_hidden.transpose(1, 0).repeat(self.decoder_num_layers, 1, 1)

        # decode input sequence
        y_outputs, y_bag_of_frags = self.decode(y_batch, y_inputs, x_outputs, x_hidden + x_fp_hidden)

        return (x_fp_outputs, y_outputs), (x_bag_of_frags, y_bag_of_frags)

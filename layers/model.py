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

    def decode(self, output_frags, decoder_inputs, encoder_hidden, encoder_outputs):
        decoder_inputs, bag_of_frags = self.embedder(output_frags, decoder_inputs, input=True)
        decoder_outputs = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
        return decoder_outputs, bag_of_frags

    def forward(self, batch):
        batches, fingerprints, inputs = batch
        anc_batch, pos_batch, neg_batch = batches
        anc_fingerprint, pos_fingerprint, neg_fingerprint = fingerprints
        anc_inputs, pos_inputs, neg_inputs = inputs

        # encode positive fragment sequence
        pos_outputs, pos_hidden, pos_bag_of_frags = self.encode(pos_batch, pos_inputs)

        # autoencode positive fingerprint
        pos_fp_outputs, pos_fp_hidden = self.autoencoder(pos_fingerprint)
        pos_fp_hidden = pos_fp_hidden.transpose(1, 0).repeat(self.decoder_num_layers, 1, 1)

        # encode negative fragment sequence
        neg_outputs, neg_hidden, neg_bag_of_frags = self.encode(neg_batch, neg_inputs)

        # autoencode negative fingerprint
        neg_fp_outputs, neg_fp_hidden = self.autoencoder(neg_fingerprint)
        neg_fp_hidden = neg_fp_hidden.transpose(1, 0).repeat(self.decoder_num_layers, 1, 1)

        # decode anchor fragment sequence
        anc_pos_outputs, anc_bag_of_frags = self.decode(anc_batch.clone(), pos_inputs, pos_hidden + pos_fp_hidden, pos_outputs)
        anc_neg_outputs, anc_bag_of_frags = self.decode(anc_batch.clone(), neg_inputs, neg_hidden + neg_fp_hidden, neg_outputs)

        return (anc_pos_outputs, anc_neg_outputs), (pos_fp_outputs, neg_fp_outputs), (anc_bag_of_frags, pos_bag_of_frags, neg_bag_of_frags)

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

        self.encoder_mlp = MLP(
            hparams=hparams,
            dim_input=self.embedder_dim_output,
            dim_hidden=64,
            dim_output=1,
            num_layers=2
        )

        self.decoder_mlp =MLP(
            hparams=hparams,
            dim_input=self.embedder_dim_output,
            dim_hidden=64,
            dim_output=1,
            num_layers=2
        )

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

    def encode(self, input_frags, enc_inputs):
        enc_inputs, bag_of_frags = self.embedder(input_frags, enc_inputs, input=False)
        enc_inputs = F.dropout(enc_inputs, p=self.embedder_dropout, training=self.training)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        return enc_outputs, enc_hidden, bag_of_frags

    def decode(self, output_frags, dec_inputs, enc_hidden, enc_outputs):
        dec_inputs, bag_of_frags = self.embedder(output_frags, dec_inputs, input=True)
        dec_inputs = F.dropout(dec_inputs, p=self.embedder_dropout, training=self.training)
        dec_outputs = self.decoder(dec_inputs, enc_hidden, enc_outputs)
        return dec_outputs, bag_of_frags

    def forward(self, batch):
        batch_data, _, enc_inputs, dec_inputs = batch
        input_frags, output_frags = batch_data

        # embed fragment sequence
        enc_outputs, enc_hidden, enc_bag_of_frags = self.encode(input_frags, enc_inputs)

        # decode fragment sequence
        dec_outputs, dec_bag_of_frags = self.decode(output_frags, dec_inputs, enc_hidden, enc_outputs)

        enc_mlp_outputs = self.encoder_mlp(enc_bag_of_frags)
        dec_mlp_outputs = self.decoder_mlp(dec_bag_of_frags)
        cos_sim = F.cosine_similarity(enc_bag_of_frags, dec_bag_of_frags).mean()

        return dec_outputs, dec_mlp_outputs, enc_mlp_outputs, cos_sim

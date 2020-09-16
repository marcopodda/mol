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

        self.mlp = MLP(
            hparams=hparams,
            dim_input=self.embedder_dim_output,
            dim_hidden=64,
            dim_output=1,
            num_layers=2
        )

        # self.decoder_mlp =MLP(
        #     hparams=hparams,
        #     dim_input=self.embedder_dim_output,
        #     dim_hidden=64,
        #     dim_output=1,
        #     num_layers=2
        # )

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

    def encode(self, input_frags, encoder_inputs):
        encoder_inputs, bag_of_frags = self.embedder(input_frags, encoder_inputs, input=False)
        encoder_inputs = F.dropout(encoder_inputs, p=self.embedder_dropout, training=self.training)
        encoder_outputs, encoder_hidden = self.encoder(encoder_inputs)
        return encoder_outputs, encoder_hidden, bag_of_frags

    def decode(self, output_frags, decoder_inputs, encoder_hidden, encoder_outputs):
        decoder_inputs, bag_of_frags = self.embedder(output_frags, decoder_inputs, input=True)
        decoder_inputs = F.dropout(decoder_inputs, p=self.embedder_dropout, training=self.training)
        decoder_outputs = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
        return decoder_outputs, bag_of_frags

    def forward(self, batch):
        batch_data, _, encoder_inputs, decoder_inputs = batch
        encoder_batch, decoder_batch = batch_data

        # embed fragment sequence
        encoder_outputs, encoder_hidden, encoder_bag_of_frags = self.encode(encoder_batch, encoder_inputs)

        # decode fragment sequence
        decoder_outputs, decoder_bag_of_frags = self.decode(decoder_batch, decoder_inputs, encoder_hidden, encoder_outputs)

        # encoder_mlp_outputs = self.encoder_mlp(encoder_bag_of_frags)
        # decoder_mlp_outputs = self.decoder_mlp(decoder_bag_of_frags)

        diff = torch.abs(decoder_bag_of_frags - encoder_bag_of_frags)
        mlp_outputs = self.mlp(diff)

        return decoder_outputs, mlp_outputs, (decoder_bag_of_frags, encoder_bag_of_frags)

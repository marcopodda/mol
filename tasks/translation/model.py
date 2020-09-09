from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.vocab import Tokens
from core.datasets.features import ATOM_FDIM, BOND_FDIM, FINGERPRINT_DIM
from layers.embedder import Embedder
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.autoencoder import Autoencoder


class TranslationModel(nn.Module):
    def __init__(self, hparams, vocab_size):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.hparams = hparams
           
        self.num_embeddings = vocab_size + len(Tokens)
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
            dropout=self.encoder_dropout)

        self.autoencoder = Autoencoder(
            hparams=hparams,
            dim_input=self.autoencoder_dim_input,
            dim_hidden=self.autoencoder_dim_hidden,
            noise_amount=self.autoencoder_noise_amount)

        self.decoder = Decoder(
            hparams=hparams,
            num_layers=self.decoder_num_layers,
            dim_input=self.decoder_dim_input,
            dim_state=self.decoder_dim_state,
            dim_output=self.decoder_dim_output,
            dropout=self.decoder_dropout)

    def encode(self, batch, enc_inputs):
        enc_inputs = self.embedder(batch, enc_inputs, input=False)
        enc_inputs = F.dropout(enc_inputs, p=self.embedder_dropout, training=self.training)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        return enc_hidden, enc_outputs

    def decode(self, batch, enc_hidden, enc_outputs, dec_inputs):
        dec_inputs = self.embedder(batch, dec_inputs, input=True)
        dec_inputs = F.dropout(dec_inputs, p=self.embedder_dropout, training=self.training)
        return self.decoder.decode_with_attention(dec_inputs, enc_hidden, enc_outputs)

    def forward(self, batch):
        batch_data, batch_fps, x_enc_inputs, dec_inputs = batch
        x_batch, y_batch = batch_data
        x_fps, _ = batch_fps

        enc_hidden, x_enc_outputs = self.encode(x_batch, x_enc_inputs)
        y_hat_fps, hidden = self.get_decoder_hidden_state(x_fps)
        dec_hidden = torch.cat([enc_hidden, hidden], dim=-1)

        logits = self.decode(y_batch, dec_hidden, x_enc_outputs, dec_inputs)
        return logits, y_hat_fps

    def get_decoder_hidden_state(self, x_fps):
        y_hat_fps, x_enc_hidden = self.autoencoder(x_fps, with_noise=False)
        x_enc_hidden = x_enc_hidden.unsqueeze(0).repeat(self.hparams.rnn_num_layers, 1, 1)
        return y_hat_fps, x_enc_hidden
    
    def set_dimensions(self):
        self.embedder_num_layers = self.hparams.gnn_num_layers
        self.embedder_dim_input = ATOM_FDIM
        self.embedder_dim_hidden = self.hparams.gnn_dim_hidden
        self.embedder_dim_edge_features = BOND_FDIM
        self.embedder_dim_edge_embed = self.hparams.gnn_dim_edge_embedd
        self.embedder_dim_output = self.hparams.frag_dim_embed
        self.embedder_dropout = self.hparams.embedder_dropout
        
        self.encoder_dropout = self.hparams.rnn_dropout
        self.encoder_dim_input = self.embedder_dim_output
        self.encoder_dim_state = self.hparams.rnn_dim_state
        
        self.autoencoder_dim_input = FINGERPRINT_DIM
        self.autoencoder_dim_hidden = self.encoder_dim_state
        self.autoencoder_noise_amount = self.hparams.autoencoder_noise
        
        self.decoder_dim_state = self.encoder_dim_state
        if self.hparams.concat:
            self.decoder_dim_state += self.autoencoder_dim_hidden
        
        self.decoder_dropout = self.encoder_dropout
        self.decoder_num_layers = self.hparams.rnn_num_layers
        self.decoder_dim_input = self.encoder_dim_output + self.decoder_dim_state
        self.decoder_dim_output = self.num_embeddings
        
        self.embedding_dropout = hparams.embedding_dropout
        
        
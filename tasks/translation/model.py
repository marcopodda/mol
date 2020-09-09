from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.vocab import Tokens
from layers.embedder import GNNEmbedder
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.autoencoder import Autoencoder


class TranslationModel(nn.Module):  
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

        self.autoencoder = Autoencoder(
            hparams=hparams, 
            output_dir=output_dir)

        self.decoder = Decoder(
            hparams=hparams,
            max_length=max_length,
            rnn_dropout=hparams.rnn_dropout,
            num_layers = hparams.rnn_num_layers,
            dim_input=hparams.frag_dim_embed,
            dim_hidden=hparams.rnn_dim_state * 2,
            dim_output=vocab_size + len(Tokens))

    def encode(self, batch, enc_inputs):
        enc_inputs = self.embedder(batch, enc_inputs, input=False)
        enc_inputs = F.dropout(enc_inputs, p=self.embedding_dropout, training=self.training)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        return enc_hidden, enc_outputs
        
    def decode(self, batch, enc_hidden, enc_outputs, dec_inputs):
        dec_inputs = self.embedder(batch, dec_inputs, input=True)
        dec_inputs = F.dropout(dec_inputs, p=self.embedding_dropout, training=self.training)
        return self.decoder.decode_with_attention(dec_inputs, enc_hidden, enc_outputs)   
        
    def forward(self, batch):
        batch_data, batch_fps, x_enc_inputs, dec_inputs = batch
        x_batch, y_batch = batch_data
        x_fps, y_fps = batch_fps
        
        enc_hidden, x_enc_outputs = self.encode(x_batch, x_enc_inputs)
        y_hat_fps, hidden = self.get_decoder_hidden_state(x_fps)
        dec_hidden = torch.cat([enc_hidden, hidden], dim=-1)
        
        logits = self.decode(y_batch, dec_hidden, x_enc_outputs, dec_inputs)
        return logits, y_hat_fps
    
    def get_decoder_hidden_state(self, x_fps):
        y_hat_fps, x_enc_hidden = self.autoencoder(x_fps)
        x_enc_hidden = x_enc_hidden.unsqueeze(0).repeat(self.hparams.rnn_num_layers, 1, 1)
        return y_hat_fps, x_enc_hidden
    
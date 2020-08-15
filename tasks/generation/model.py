from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN
from layers.vae import VAE, MMDVAE
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder


def get_vae_class(name):
    return VAE if name == "VAE" else MMDVAE


class Model(nn.Module):
    def __init__(self, hparams, output_dir, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        
        embeddings_filename = f"{hparams.embedding_type}_{hparams.gnn_dim_embed}.pt"
        embeddings = torch.load(output_dir / "embeddings" / embeddings_filename)
        num_embeddings = embeddings.size(0)

        self.max_length = max_length
        self.dim_embed = embeddings.size(1)
        
        self.vae_dim_input = self.dim_embed
        self.vae_dim_hidden = hparams.vae_dim_hidden
        self.vae_dim_latent = hparams.vae_dim_latent

        self.embedding_dropout = hparams.embedding_dropout
        self.rnn_dropout = hparams.rnn_dropout
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_dim_input = self.dim_embed
        self.rnn_dim_hidden = self.dim_embed
        self.rnn_dim_output = num_embeddings
        
        self.enc_embedder = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.dec_embedder = nn.Embedding(*embeddings.size())

        self.encoder = Encoder(  
            hparams=hparams,
            rnn_dropout=self.rnn_dropout,
            num_layers = self.rnn_num_layers,
            dim_input=self.rnn_dim_input,
            dim_hidden=self.rnn_dim_hidden,
            dim_output=self.rnn_dim_output
        )

        self.vae = get_vae_class(hparams.vae_class)(
            hparams=hparams,
            dim_input=self.vae_dim_input,
            dim_hidden=self.vae_dim_hidden,
            dim_latent=self.vae_dim_latent
        )

        self.decoder = Decoder(
            hparams=hparams,
            max_length=self.max_length,
            rnn_dropout=self.rnn_dropout,
            num_layers = self.rnn_num_layers,
            dim_input=self.rnn_dim_input,
            dim_hidden=self.rnn_dim_hidden,
            dim_output=self.rnn_dim_output
        )

        if self.hparams.tie_weights:
            self.decoder.tie_weights(self.enc_embedder)
        
        self.mlp_dim_input = self.rnn_num_layers * self.dim_embed
        self.mlp_dim_hidden = self.mlp_dim_input // 2
        self.mlp_dim_output = 5
        
        self.mlp = MLP(
            dim_input=self.mlp_dim_input, 
            dim_hidden=self.mlp_dim_hidden, 
            dim_output=self.mlp_dim_output)

    def forward(self, batch):
        x = self.enc_embedder(batch.outseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        enc_outputs, h = self.encoder(x)

        hidden_enc, vae_loss = self.vae(h)

        x = self.dec_embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        output, hidden_dec = self.decoder(x, hidden_enc)
        h = hidden_enc.view(-1, self.rnn_dim_input * self.rnn_num_layers)
        props = None
        # props = self.mlp(h)
        
        return output, vae_loss, hidden_enc, hidden_dec, props
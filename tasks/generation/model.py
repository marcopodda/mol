from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN
from layers.vae import VAE, MMDVAE, InfoVAE
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder


def get_vae_class(name):
    if name == "VAE":
        return VAE
    if name == "InfoVAE":
        return InfoVAE
    if name == "MMDVAE":
        return MMDVAE
    raise ValueError("Unknown VAE class")


class Model(nn.Module):        
    def __init__(self, hparams, output_dir, vocab_size, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        
        self.max_length = max_length
        self.dim_embed = hparams.gnn_dim_embed
        self.num_embeddings = vocab_size + len(Tokens)
        
        self.vae_dim_input = self.dim_embed
        self.vae_dim_hidden = hparams.vae_dim_hidden
        self.vae_dim_latent = hparams.vae_dim_latent

        self.embedding_dropout = hparams.embedding_dropout
        self.rnn_dropout = hparams.rnn_dropout
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_dim_input = self.dim_embed
        self.rnn_dim_hidden = self.dim_embed
        self.rnn_dim_output = self.num_embeddings
                    
        embeddings = self.load_embeddings()
        
        self.enc_embedder = None
        if self.hparams.encoder_type == "rnn":
            self.enc_embedder = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=Tokens.PAD.value)
        
        self.dec_embedder = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=Tokens.PAD.value)

        if self.hparams.encoder_type == "gnn":
            self.encoder = GNN(hparams)
        elif self.hparams.encoder_type == "rnn":
            self.encoder = Encoder(  
                hparams=hparams,
                rnn_dropout=self.rnn_dropout,
                num_layers = self.rnn_num_layers,
                dim_input=self.rnn_dim_input,
                dim_hidden=self.rnn_dim_hidden,
                dim_output=self.rnn_dim_output
            )
        else:
            raise ValueError("Unknown encoder type!")

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
            self.decoder.tie_weights(self.dec_embedder)
        
        self.mlp_dim_input = self.rnn_num_layers * self.dim_embed
        self.mlp_dim_hidden = self.mlp_dim_input // 2
        self.mlp_dim_output = 5
        
        self.mlp = MLP(
            dim_input=self.mlp_dim_input, 
            dim_hidden=self.mlp_dim_hidden, 
            dim_output=self.mlp_dim_output)

    def load_embeddings(self):
        embeddings_filename = f"{self.hparams.embedding_type}_{self.dim_embed}.pt"
        embeddings_path = self.output_dir / "embeddings" / embeddings_filename
        
        if not embeddings_path.exists():
            print(f"Embeddings {embeddings_filename} don't exist!")
            exit(1)
            
        return torch.load(embeddings_path)

    def forward(self, batch):
        if self.hparams.encoder_type == "rnn":
            x = self.enc_embedder(batch.outseq)
            x = F.dropout(x, p=self.embedding_dropout, training=self.training)
            _, h = self.encoder(x)
        elif self.hparams.encoder_type == "gnn":
            h = self.encoder(batch)
        else:
            raise ValueError("Unknown encoder type!")

        hidden_enc, vae_loss = self.vae(h)

        x = self.dec_embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        output, hidden_dec = self.decoder(x, hidden_enc)
        h = hidden_enc.view(-1, self.rnn_dim_input * self.rnn_num_layers)
        props = None
        # props = self.mlp(h)
        
        return output, vae_loss, hidden_enc, hidden_dec, props
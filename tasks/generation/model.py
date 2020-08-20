from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from core.datasets.features import ATOM_FDIM
from core.utils.vocab import Tokens
from layers.graphconv import GNN
from layers.vae import VAE
from layers.mlp import MLP
from layers.encoder import Encoder
from layers.decoder import Decoder


class Model(nn.Module):        
    def __init__(self, hparams, output_dir, vocab_size, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        self.output_dir = output_dir
        self.embedding_dropout = hparams.embedding_dropout
                    
        embeddings = self.load_embeddings()
        self.enc_embedder = None
        if hparams.encoder_type == "rnn":
            self.enc_embedder = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=Tokens.PAD.value)
        self.dec_embedder = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=Tokens.PAD.value)

        self.enc_embedder = nn.Embedding(*embeddings.size(), padding_idx=Tokens.PAD.value)
        self.dec_embedder = nn.Embedding(*embeddings.size(), padding_idx=Tokens.PAD.value)

        if hparams.encoder_type == "gnn":
            self.encoder = GNN(
                hparams=hparams,
                num_layers=hparams.gnn_num_layers,
                dim_edge_embed=hparams.gnn_dim_edge_embed,
                dim_hidden=hparams.gnn_dim_hidden,
                dim_output=hparams.rnn_dim_state)
        elif hparams.encoder_type == "rnn":
            self.encoder = Encoder(  
                hparams=hparams,
                rnn_dropout=hparams.rnn_dropout,
                num_layers = hparams.rnn_num_layers,
                dim_input=hparams.frag_dim_embed,
                dim_hidden=hparams.rnn_dim_state)
        else:
            raise ValueError("Unknown encoder type!")

        self.vae = VAE(
            hparams=hparams,
            vocab_size=vocab_size,
            dim_input=hparams.rnn_dim_state,
            dim_latent=hparams.rnn_dim_state // 2,
            dim_output=hparams.rnn_dim_state)

        self.decoder = Decoder(
            hparams=hparams,
            max_length=max_length,
            rnn_dropout=hparams.rnn_dropout,
            num_layers = hparams.rnn_num_layers,
            dim_input=hparams.frag_dim_embed,
            dim_hidden=hparams.rnn_dim_state,
            dim_output=vocab_size + len(Tokens))
        
        self.mlp_dim_input = hparams.rnn_num_layers * hparams.rnn_dim_state
        self.mlp_dim_hidden = self.mlp_dim_input // 2
        self.mlp_dim_output = 5
        
        self.mlp = MLP(
            dim_input=self.mlp_dim_input, 
            dim_hidden=self.mlp_dim_hidden, 
            dim_output=self.mlp_dim_output)

    def load_embeddings(self):
        embeddings_filename = f"{self.hparams.embedding_type}_{self.hparams.frag_dim_embed}.pt"
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
        # h = hidden_enc.view(-1, self.hparams.rnn_dim_state * self.hparams.rnn_num_layers)
        props = None
        # props = self.mlp(h)
        
        return output, vae_loss, hidden_enc, hidden_dec, props
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
    def __init__(self, hparams, output_dir, max_length):
        super().__init__()
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.hparams = hparams
        
        embeddings_filename = f"{hparams.embedding_type}_{hparams.gnn_dim_embed}.pt"
        embeddings = torch.load(output_dir / "embeddings" / embeddings_filename)
        num_embeddings = embeddings.size(0)

        self.max_length = max_length
        self.gnn_num_layers = hparams.gnn_num_layers
        self.gnn_dim_input = ATOM_FDIM
        self.gnn_dim_hidden = hparams.gnn_dim_hidden
        self.gnn_dim_embed = hparams.gnn_dim_embed

        self.vae_dim_input = self.gnn_dim_embed
        self.vae_dim_hidden = hparams.vae_dim_hidden
        self.vae_dim_latent = hparams.vae_dim_latent

        self.embedding_dropout = hparams.embedding_dropout
        self.rnn_dropout = hparams.rnn_dropout
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_dim_input = self.gnn_dim_embed
        self.rnn_dim_hidden = self.gnn_dim_embed
        self.rnn_dim_output = num_embeddings

        self.mlp_dim_input = self.gnn_dim_embed
        self.mlp_dim_hidden = hparams.mlp_dim_hidden if "mlp_dim_hidden" in hparams else self.mlp_dim_input // 2
        self.mlp_dim_output = hparams.mlp_dim_output if "mlp_dim_output" in hparams else 1

        self.embedder = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.encoder = Encoder(
            hparams=hparams,
            rnn_dropout=self.rnn_dropout,
            num_layers = self.rnn_num_layers,
            dim_input=self.rnn_dim_input,
            dim_hidden=self.rnn_dim_hidden,
            dim_output=self.rnn_dim_output
        )

        self.vae = VAE(hparams.vae_class)(
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
            self.decoder.tie_weights(self.embedder)

        self.mlp = MLP(
            dim_input=self.mlp_dim_input,
            dim_hidden=self.mlp_dim_hidden,
            dim_output=self.mlp_dim_output
        )

    def _forward(self, batch):
        x = self.embedder(batch.outseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        enc_outputs, h = self.encoder(x)

        h, vae_loss = self.vae(h)

        x = self.embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        output, hidden = self.decoder(x, h)
        return output, vae_loss

    def _forward_att(self, batch):
        x = self.embedder(batch.outseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        enc_outputs, h = self.encoder(x)

        h, vae_loss = self.vae(h)

        x = self.embedder(batch.inseq)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)

        batch_size, seq_length, dim_embed = x.size()
        outputs = []

        for i in range(seq_length):
            inp = x[:, i, :].unsqueeze(1)
            ctx = torch.zeros_like(inp, device=inp.device) if i == 0 else ctx
            out, h, ctx, w = self.decoder.forward_attention(inp, h, ctx, enc_outputs)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs.view(-1, outputs.size(2)), vae_loss

    def forward(self, batch):
        use_attention = self.hparams.use_attention
        return self._forward_att(batch) if use_attention else self._forward(batch)

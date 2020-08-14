import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BaseVAE(nn.Module):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__()
        self.hparams = hparams

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.rnn_num_layers = hparams.rnn_num_layers

        self.fc_enc = nn.Linear(self.rnn_num_layers * dim_input, dim_hidden)
        self.fc_mean = nn.Linear(dim_hidden, dim_latent)

        self.fc_dec = nn.Linear(dim_latent, dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, self.rnn_num_layers * dim_input)

    def decode(self, z=None):
        if z is None:
            z = self.sample_prior()    
        z = F.relu(self.fc_dec(z))
        z = self.fc_out(z)
        return z.view(self.rnn_num_layers, -1, self.dim_input)
    
    def sample_prior(self, z=None):
        if z is None:
            device = next(self.parameters()).device
            return torch.randn((1, self.fc_mean.out_features), device=device)
        return torch.randn_like(z)


class MMDVAE(BaseVAE):
    def encode(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.fc_enc(x))
        mean = self.fc_mean(x)
        return mean

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        loss = self.mmd(z)
        return x_rec, loss

    def rbf_kernel(self, x, y):
        diff = x[:, None, :] - y[None, :, :]
        dist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-dist)

    def mmd(self, p):
        q = self.sample_prior(like=p)
        p_kernel = self.rbf_kernel(p, p).mean()
        q_kernel = self.rbf_kernel(q, q).mean()
        pq_kernel = self.rbf_kernel(p, q).mean()
        return p_kernel + q_kernel - 2 * pq_kernel


class VAE(BaseVAE):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__(hparams, dim_input, dim_hidden, dim_latent)
        self.fc_logv = nn.Linear(dim_hidden, dim_latent)

    def encode(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.fc_enc(x))
        mean = self.fc_mean(x)
        logv = self.fc_logv(x)
        return mean, logv

    def reparameterize(self, mean, logv):
        std = torch.exp(0.5 * logv)
        eps = self.sample_prior(like=std)
        return mean + eps * std

    def forward(self, x):
        mean, logv = self.encode(x)
        z = self.reparameterize(mean, logv)
        x_rec = self.decode(z)
        loss = self.kl_div(mean, logv)
        return x_rec, loss

    def kl_div(self, mean, logv):
        return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

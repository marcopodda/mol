import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# def gaussian_kernel(a, b):
#     dim_a, dim_b, sigma = a.size(0), b.size(0), a.size(1)
#     a = a.view(dim_a, 1, sigma)
#     b = b.view(1, dim_b, sigma)
#     a_core = a.expand(dim_a, dim_b, sigma)
#     b_core = b.expand(dim_a, dim_b, sigma)
#     numerator = (a_core - b_core).pow(2).mean(2) / sigma
#     return torch.exp(-numerator)


def gaussian_kernel(x, y, sigma_sqr=2.):
    diff = x[:, None, :] - y[None, :, :]
    pairwise_dist = torch.sum(diff ** 2, dim=-1)
    return torch.exp(-pairwise_dist / sigma_sqr)


class BaseVAE(nn.Module):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__()
        self.hparams = hparams

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.rnn_num_layers = hparams.rnn_num_layers

        self.fc_enc = nn.Linear(self.rnn_num_layers * dim_input, dim_hidden)
        self.bn_enc = nn.BatchNorm1d(dim_hidden)
        self.fc_mean = nn.Linear(dim_hidden, dim_latent)

        self.fc_dec = nn.Linear(dim_latent, dim_hidden)
        self.bn_dec = nn.BatchNorm1d(dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, self.rnn_num_layers * dim_input)

    def decode(self, z=None):
        z = z or self.sample_prior()    
        z = F.relu(self.bn_dec(self.fc_dec(z)))
        z = self.fc_out(z)
        return z.view(self.rnn_num_layers, -1, self.dim_input)
    
    def sample_prior(self):
        device = next(self.parameters()).device
        return torch.randn((1, self.fc_mean.out_features), device=device)


class MMDVAE(BaseVAE):
    def encode(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.bn_enc(self.fc_enc(x)))
        mean = self.fc_mean(x)
        return mean

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        loss = self.loss(x, x_rec, z)
        return x_rec, loss

    def loss(self, x, x_rec, mean):
        mmd_term = self.mmd(mean)
        rec_term = F.mse_loss(x_rec, x)
        return mmd_term + rec_term

    def mmd(self, z):
        q = torch.randn_like(z)
        p_kernel = gaussian_kernel(z, z).mean()
        q_kernel = gaussian_kernel(q, q).mean()
        pq_kernel = gaussian_kernel(z, q).mean()
        return p_kernel + q_kernel - 2 * pq_kernel


class VAE(BaseVAE):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__(hparams, dim_input, dim_hidden, dim_latent)
        self.fc_std = nn.Linear(dim_hidden, dim_latent)

    def encode(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.bn_enc(self.fc_enc(x)))
        mean = self.fc_mean(x)
        logvar = self.fc_std(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_rec = self.decode(z)
        loss = self.loss(x, x_rec, mean, logvar)
        return x_rec, loss

    def loss(self, x, x_rec, mean, logvar):
        kl_term = self.kl(mean, logvar)
        rec_term = F.mse_loss(x_rec, x)
        return kl_term + rec_term

    def kl(self, mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

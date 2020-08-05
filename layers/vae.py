import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.size(0), b.size(0)
    depth = a.size(1)
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)


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

    def decoder(self, z):
        z = F.relu(self.bn_dec(self.fc_dec(z)))
        z = self.fc_out(z)
        return z.view(self.rnn_num_layers, -1, self.dim_input)


class MMDVAE(BaseVAE):
    def encoder(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.bn_enc(self.fc_enc(x)))
        mean = self.fc_mean(x)
        return mean

    def forward(self, x):
        device = next(self.parameters()).device
        mean = self.encoder(x)
        x_rec = self.decoder(mean)
        loss = self.loss(x, x_rec, mean)
        return x_rec, loss

    def loss(self, x, x_rec, mean):
        mmd_term = self.mmd(mean)
        rec_term = F.mse_loss(x_rec, x)
        return mmd_term + rec_term

    def mmd(self, mu):
        z = torch.randn(100, mu.size(1), device=mu.device)
        mu2 = gaussian_kernel(mu, mu).mean()
        z2 = gaussian_kernel(z, z).mean()
        mu_z = gaussian_kernel(z, mu).mean()
        return mu2 + z2 - 2 * mu_z


class VAE(BaseVAE):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__(hparams, dim_input, dim_hidden, dim_latent)
        self.fc_std = nn.Linear(dim_hidden, dim_latent)

    def encoder(self, x):
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
        device = next(self.parameters()).device
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_rec = self.decoder(z)
        loss = self.loss(x, x_rec, mean, logvar)
        return x_rec, loss

    def loss(self, x, x_rec, mean, logvar):
        kl_term = self.kl(mean, logvar)
        rec_term = F.mse_loss(x_rec, x)
        return kl_term + rec_term

    def kl(self, mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, hparams, vocab_size, dim_input, dim_latent, dim_output):
        super().__init__()
        self.hparams = hparams

        self.dim_latent = dim_latent
        self.dim_output = dim_output
        self.rnn_num_layers = hparams.rnn_num_layers
        self.vocab_size = vocab_size
        
        if self.hparams.encoder_type == "rnn":
            dim_input *= self.rnn_num_layers
        
        self.fc_mean = nn.Linear(dim_input, dim_latent)
        self.fc_logv = nn.Linear(dim_input, dim_latent)

        self.fc_out = nn.Linear(dim_latent, self.rnn_num_layers * self.dim_output)
    
    def sample_prior(self, z=None, batch_size=1):
        if z is None:
            device = next(self.parameters()).device
            return torch.randn((batch_size, self.fc_mean.out_features), device=device)
        return torch.randn_like(z)

    def reparameterize(self, mean, logv):
        std = torch.exp(0.5 * logv)
        eps = self.sample_prior(z=std)
        return mean + eps * std

    def encode(self, x):
        if x.ndim > 2:
            x = x.view(-1, x.size(0) * x.size(2))
        mean = self.fc_mean(x)
        logv = self.fc_logv(x)
        return mean, logv
    
    def decode(self, z=None):
        if z is None:
            z = self.sample_prior()    
        z = self.fc_out(z)
        return z.view(self.rnn_num_layers, -1, self.dim_output)

    def forward(self, x):
        mean, logv = self.encode(x)
        z = self.reparameterize(mean, logv)
        x = self.decode(z)
        loss = self.loss_function(mean, logv)
        return x, loss

    def loss_function(self, mean, logv):
        kl_div = -0.5 * torch.sum(1.0 + logv - mean.pow(2) - logv.exp())
        return kl_div # / (self.hparams.batch_size * self.vocab_size)
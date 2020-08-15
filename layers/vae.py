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
        self.fc_logv = nn.Linear(dim_hidden, dim_latent)

        self.fc_dec = nn.Linear(dim_latent, dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, self.rnn_num_layers * dim_input)

    def decode(self, z=None):
        if z is None:
            z = self.sample_prior()    
        z = F.relu(self.fc_dec(z))
        z = self.fc_out(z)
        return z.view(self.rnn_num_layers, -1, self.dim_input)
    
    def sample_prior(self, z=None, batch_size=1):
        if z is None:
            device = next(self.parameters()).device
            return torch.randn((batch_size, self.fc_mean.out_features), device=device)
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

    def compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / dim * 1.0)

    def mmd(self, p):
        q = self.sample_prior(z=p)
        p_kernel = self.compute_kernel(p, p).mean()
        q_kernel = self.compute_kernel(q, q).mean()
        pq_kernel = self.compute_kernel(p, q).mean()
        return p_kernel + q_kernel - 2 * pq_kernel


class VAE(BaseVAE):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__(hparams, dim_input, dim_hidden, dim_latent)
        

    def encode(self, x):
        x = x.view(-1, x.size(0) * x.size(2))
        x = F.relu(self.fc_enc(x))
        mean = self.fc_mean(x)
        logv = self.fc_logv(x)
        return mean, logv

    def reparameterize(self, mean, logv):
        std = torch.exp(0.5 * logv)
        eps = self.sample_prior(z=std)
        return mean + eps * std

    def forward(self, x):
        mean, logv = self.encode(x)
        z = self.reparameterize(mean, logv)
        x_rec = self.decode(z)
        loss = self.kl_div(mean, logv)
        return x_rec, loss

    def kl_div(self, mean, logv):
        return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())


class InfoVAE(BaseVAE):
    def __init__(self, hparams, dim_input, dim_hidden, dim_latent):
        super().__init__(hparams, dim_input, dim_hidden, dim_latent)
        
        self.alpha = -0.5
        self.reg_weight = 100
        self.kernel_type = "imq"
        self.z_var = 2.0
        
    def forward(self, x):
        mean, logv = self.encode(x)
        z = self.reparameterize(mean, logv)
        x_rec = self.decode(z)
        loss = self.loss_function(z, mean, logv)
        return x_rec, loss
    
    def compute_kernel(self, x1, x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result
    
    def compute_rbf(self, x1, x2, eps=1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1, x2, eps=1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
        return mmd
    
    def loss_function(self, z, mean, logv):
        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)

        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logv - mean ** 2 - logv.exp(), dim=1), dim=0)

        loss = (1. - self.alpha) * kld_loss + (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss
        return loss


    
                 
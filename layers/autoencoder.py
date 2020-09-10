from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self, hparams, dim_input, dim_hidden, noise_amount):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.noise_amount = noise_amount

        self.input = nn.Linear(self.dim_input, self.dim_input // 2)
        self.input2hidden = nn.Linear(self.dim_input // 2, self.dim_input // 4)
        self.hidden2bottleneck = nn.Linear(self.dim_input // 4, self.dim_hidden)
        self.bottleneck2hidden = nn.Linear(self.dim_hidden, self.dim_input // 4)
        self.hidden2output = nn.Linear(self.dim_input // 4, self.dim_input // 2)
        self.output = nn.Linear(self.dim_input // 2, self.dim_input)

    def encode(self, inputs):
        x = self.input(inputs)
        x = self.input2hidden(F.relu(x))
        return self.hidden2bottleneck(F.relu(x))

    def decode(self, hidden):
        x = self.bottleneck2hidden(hidden)
        x = self.hidden2output(F.relu(x))
        return self.output(F.relu(x))

    def forward(self, batch, with_noise=True):
        if with_noise is True:
            batch = self.add_noise(batch)
        hidden = self.encode(batch)
        output = self.decode(hidden)
        return torch.sigmoid(output), hidden

    def add_noise(self, batch):
        noisy_batch = batch.clone()
        noise_mask = torch.rand_like(noisy_batch) <= self.noise_amount
        noisy_batch[noise_mask] = torch.logical_not(noisy_batch[noise_mask]).float()
        return noisy_batch

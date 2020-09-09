from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F


FINGERPRINT_DIM = 2048


class Autoencoder(nn.Module):
    def __init__(self, hparams, output_dir):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.output_dir = output_dir
        self.dim_hidden = hparams.rnn_dim_state

        assert self.dim_hidden < FINGERPRINT_DIM // 4

        self.input = nn.Linear(FINGERPRINT_DIM, FINGERPRINT_DIM // 2)
        self.input2hidden = nn.Linear(FINGERPRINT_DIM // 2, FINGERPRINT_DIM // 4)
        self.hidden2bottleneck = nn.Linear(FINGERPRINT_DIM // 4, self.dim_hidden)
        self.bottleneck2hidden = nn.Linear(self.dim_hidden, FINGERPRINT_DIM // 4)
        self.hidden2output = nn.Linear(FINGERPRINT_DIM // 4, FINGERPRINT_DIM // 2)
        self.output = nn.Linear(FINGERPRINT_DIM // 2, FINGERPRINT_DIM)

    def encode(self, inputs):
        x = self.input(inputs)
        x = self.input2hidden(F.relu(x))
        return self.hidden2bottleneck(F.relu(x))

    def decode(self, hidden):
        x = self.bottleneck2hidden(hidden)
        x = self.hidden2output(F.relu(x))
        return self.output(F.relu(x))

    def forward(self, batch, denoise=True):
        if denoise is True:
            batch = self.add_noise(batch)
        hidden = self.encode(batch)
        output = self.decode(hidden)
        return torch.sigmoid(output), hidden

    def add_noise(self, batch):
        noisy_batch = batch.clone()
        noise_mask = torch.rand_like(noisy_batch) <= self.hparams.autoencoder_noise
        noisy_batch[noise_mask] = torch.logical_not(noisy_batch[noise_mask]).float()
        return noisy_batch

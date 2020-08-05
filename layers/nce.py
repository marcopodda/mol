""" adapted from https://github.com/mingen-pan/easy-to-use-NCE-RNN-for-Pytorch"""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from core.utils.vocab import Tokens


EPSILON = 1e-8
MAX_BATCH_SIZE_ALLOWED = 1024 * 1024


def safe_log(x):
    return torch.log(x + EPSILON)


class NCELoss(nn.Module):
    def __init__(self, hparams, Q):
        super().__init__()
        # the Q is prior model
        # the N is the number of vocabulary
        # the K is the noise sampling times
        self.hparams = hparams

        self.N = Q.size(0)
        self.K = hparams.noise_ratio

        self.register_buffer("Q", Q)
        self.register_buffer("mask", torch.zeros(self.N))
        self.register_buffer("ones", torch.ones(self.K + 1, dtype=torch.long))
        self.register_buffer("range", torch.arange(MAX_BATCH_SIZE_ALLOWED, dtype=torch.long))

        self.Z = nn.Parameter(torch.Tensor([1.0]))
        self.register_parameter("Z", self.Z)

        # self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, output, target):
        # #output is the RNN output, which is the input of loss function
        B = output.size(0)
        assert B <= MAX_BATCH_SIZE_ALLOWED

        output = output.view(-1, self.N)
        noise_idx = self.get_noise(B, target)
        idx = self.get_combined_idx(target, noise_idx)
        P_target, P_noise = self.get_prob(idx, output)
        Q_target, Q_noise = self.get_Q(idx)
        loss = self.nce_loss(P_target, P_noise, Q_noise, Q_target)
        return loss.mean()

    def get_Q(self, idx):
        prob_model = self.Q[idx.view(-1)].view(idx.size())
        target, noise = prob_model[:, 0], prob_model[:, 1:]
        return target, noise

    def get_prob(self, idx, scores):
        scores = self.get_scores(idx, scores)
        prob = (scores - self.Z).exp()
        target, noise = prob[:, 0], prob[:, 1:]
        return target, noise

    def get_scores(self, idx, scores):
        (B, N), K = scores.size(), idx.size(1)
        idx_increment = N * self.range[:B].view(B, 1)
        idx_increment = idx_increment * self.ones
        new_idx = idx_increment + idx
        new_scores = scores.view(-1).index_select(0, new_idx.view(-1))
        return new_scores.view(B, K)

    def get_noise(self, B, target):
        noise = torch.randint(self.N, size=(B, self.K))
        return noise.to(target.device)

    def get_combined_idx(self, target_idx, noise_idx):
        return torch.cat((target_idx.view(-1, 1), noise_idx), 1)

    def nce_loss(self, P_target, P_noise, Q_noise, Q_target):
        kQ_target, kQ_noise = self.K * Q_target, self.K * Q_noise
        model_loss = safe_log(P_target / (kQ_target + P_target))
        noise_loss = torch.sum(safe_log(kQ_noise / (kQ_noise + P_noise)), dim=-1)
        loss = -(model_loss.view(-1) + noise_loss.view(-1))
        return loss

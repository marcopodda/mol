import math

import torch
from torch import nn
from torch.nn import functional as F

from core.hparams import HParams

class Attention(nn.Module):
    def __init__(self, hparams, dim_input, dim_output):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.dim_input = dim_input
        self.dim_output = dim_output

        self.attn = nn.Linear(self.dim_input, self.dim_output)
        self.v = nn.Parameter(torch.rand(self.dim_output))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # encoder_outputs [B*T*H]
        seq_len = encoder_outputs.size(1)
        hidden = hidden[-1:].repeat(seq_len, 1, 1)  # [T*B*H]
        attn_energies = self.score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        hidden = hidden.transpose(1, 0)  # [B*T*H]
        attn_inputs = torch.cat([hidden, encoder_outputs], dim=2)  # [B*T*2H]
        attn_outputs = self.attn(attn_inputs)  # [B*T*2H]->[B*T*H]
        energy = F.relu(attn_outputs).transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

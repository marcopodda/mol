import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim_context):
        super().__init__()

        self.dim_context = dim_context
        self.attn = nn.Linear(dim_context*2, dim_context)
        self.v = nn.Parameter(torch.rand(dim_context))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, enc_outputs):
        attn_energies, seq_len = [], enc_outputs.size(1)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            score = self.score(hidden, enc_outputs[:,i,:])
            attn_energies.append(score)
        attn_energies = torch.cat(attn_energies, dim=1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(1, 0)
        # encoder_outputs = encoder_outputs  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        attn_inputs = torch.cat([hidden, encoder_outputs], dim=2)
        energy = F.relu(self.attn(attn_inputs)).transpose(1, 2) # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

    # def score(self, hidden, encoder_output):
    #     energy = self.attn(encoder_output)
    #     energy = energy.view(*hidden.size())
    #     energy = energy.transpose(2, 1)
    #     energy = torch.bmm(hidden, energy)

    #     return energy.squeeze(1)
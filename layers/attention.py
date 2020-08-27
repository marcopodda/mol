import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim_context):
        super().__init__()

        self.dim_context = dim_context
        self.attn = nn.Linear(dim_context, dim_context)

    def forward(self, hidden, enc_outputs):
        attn_energies, seq_len = [], enc_outputs.size(1)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            score = self.score(hidden, enc_outputs[:,i,:])
            attn_energies.append(score)
        attn_energies = torch.cat(attn_energies, dim=1)

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = energy.view(*hidden.size())
        energy = energy.transpose(2, 1)
        energy = torch.bmm(hidden, energy)

        return energy.squeeze(1)
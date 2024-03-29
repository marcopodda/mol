import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from core.utils.vocab import Tokens
from layers.misc import WeightDropGRU


class Attention(nn.Module):
    def __init__(self, dim_hidden, max_length):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.attn = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, hidden, encoder_outputs):
        attn_energies, seq_len = [], encoder_outputs.size(1)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            score = self.score(hidden, encoder_outputs[:,i,:])
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


class Decoder(nn.Module):
    def __init__(self, hparams, max_length, rnn_dropout, num_layers, dim_input, dim_hidden, dim_output):
        super().__init__()

        self.max_length = max_length
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout

        self.gru = WeightDropGRU(input_size=dim_input,
                                 hidden_size=dim_hidden,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 weight_dropout=rnn_dropout,
                                 dropout=rnn_dropout)

        self.attention = Attention(dim_hidden=dim_hidden, max_length=max_length)
        self.proj = nn.Linear(dim_hidden * 2, dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_output)

    def forward(self, x, hidden):
        rnn_output, hidden = self.gru(x, hidden)
        output = rnn_output.reshape(-1, rnn_output.size(2))
        logits = self.out(output).squeeze(1)
        return logits, hidden

    def forward_att(self, x, hidden, prev_context, enc_outputs):
        # Note: we run this one step at a time

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat([x, prev_context], dim=-1)
        rnn_output, hidden = self.gru(x, hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attention(rnn_output, enc_outputs)
        context = attn_weights.bmm(enc_outputs) # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        output = self.proj(torch.cat([rnn_output, context], dim=-1))
        output = rnn_output.reshape(-1, rnn_output.size(2))
        logits = self.out(output).squeeze(1)

        # Return final output, hidden state, and attention weights (for visualization)
        return logits, hidden, context, attn_weights

    def tie_weights(self, embedder):
        self.out.weight = embedder.weight

import torch
from torch import nn

from layers.embedder import GNN
from layers.rnn import WeightDropGRU


class Encoder(nn.Module):
    def __init__(self, hparams, rnn_dropout, num_layers, dim_input, dim_hidden):
        super().__init__()
        self.hparams = hparams
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout

        self.gru = WeightDropGRU(input_size=dim_input,
                          hidden_size=dim_hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=rnn_dropout,
                          weight_dropout=rnn_dropout,
                          bidirectional=True)

    def forward(self, x):
        x = x.unsqueeze(0) if x.ndim == 2 else x
        output, hidden = self.gru(x)

        batch_size = output.size(0)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.dim_hidden).sum(dim=1)
        output = output[:,:,:self.dim_hidden] + output[:,:,self.dim_hidden:]
        return output, hidden
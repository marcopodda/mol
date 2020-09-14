import torch
from torch import nn
from torch.nn import functional as F

from core.hparams import HParams

class Autoencoder(nn.Module):
    def __init__(self, hparams, dim_input, dim_hidden):
        super().__init__()

        self.hparams = HParams.load(hparams)
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden

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

    def forward(self, batch):
        hidden = self.encode(batch)
        output = self.decode(hidden)
        return output, hidden
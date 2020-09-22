from torch import nn
from torch.nn import functional as F

from core.hparams import HParams


class Autoencoder(nn.Module):
    def __init__(self, hparams, dim_input, dim_hidden):
        super().__init__()

        self.hparams = HParams.load(hparams)
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = self.hparams.rnn_num_layers

        self.input = nn.Linear(self.dim_input, self.dim_input // 2)
        self.bn_input = nn.BatchNorm1d(self.dim_input // 2)
        self.input2hidden = nn.Linear(self.dim_input // 2, self.dim_input // 4)
        self.bn_input2hidden = nn.BatchNorm1d(self.dim_input // 4)
        self.hidden2bottleneck = nn.Linear(self.dim_input // 4, self.dim_hidden)
        self.bn_hidden2bottleneck = nn.BatchNorm1d(self.dim_hidden)
        self.bottleneck2hidden = nn.Linear(self.dim_hidden, self.dim_input // 4)
        self.bn_bottleneck2hidden = nn.BatchNorm1d(self.dim_input // 4)
        self.hidden2output = nn.Linear(self.dim_input // 4, self.dim_input // 2)
        self.output = nn.Linear(self.dim_input // 2, self.dim_input)

    def encode(self, inputs):
        x = self.input(inputs)
        x = self.input2hidden(F.relu(self.bn_input(x)))
        return self.hidden2bottleneck(F.relu(self.bn_input2hidden(x)))

    def decode(self, hidden):
        x = self.bottleneck2hidden(F.relu(self.bn_hidden2bottleneck(hidden)))
        x = self.hidden2output(F.relu(self.bn_bottleneck2hidden(x)))
        return self.output(x)

    def forward(self, batch):
        hidden = self.encode(batch)
        output = self.decode(hidden)
        return output, hidden

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
        self.bn_input = nn.BatchNorm1d(self.input.out_features)

        self.input2hidden = nn.Linear(self.dim_input // 2, self.dim_input // 4)
        self.bn_input2hidden = nn.BatchNorm1d(self.input2hidden.out_features)

        self.hidden2bottleneck = nn.Linear(self.dim_input // 4, self.dim_hidden)
        self.bn_hidden2bottleneck = nn.BatchNorm1d(self.hidden2bottleneck.out_features)

        self.bottleneck2hidden = nn.Linear(self.dim_hidden, self.dim_input // 4)
        self.bn_bottleneck2hidden = nn.BatchNorm1d(self.bottleneck2hidden.out_features)

        self.hidden2output = nn.Linear(self.dim_input // 4, self.dim_input // 2)
        self.bn_hidden2output = nn.BatchNorm1d(self.hidden2output.out_features)

        self.output = nn.Linear(self.dim_input // 2, self.dim_input)

    def encode(self, inputs):
        if inputs.ndim == 3:
            inputs = inputs.squeeze(1)

        x = self.input(inputs)
        x = F.relu(self.bn_input(x))
        x = self.input2hidden(x)
        x = F.relu(self.bn_input2hidden(x))
        return self.hidden2bottleneck(x)

    def decode(self, hidden):
        x = self.bottleneck2hidden(hidden)
        x = F.relu(self.bn_bottleneck2hidden(x))
        x = self.hidden2output(x)
        x = F.relu(self.bn_hidden2output(x))
        return self.output(x)

    def forward(self, batch):
        hidden = self.encode(batch)
        output = self.decode(hidden)
        if hidden.ndim < 3:
            hidden = hidden.unsqueeze(0)
        return output, hidden

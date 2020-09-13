from torch import nn
from torch.nn import functional as F

from core.hparams import HParams


class Encoder(nn.Module):
    def __init__(self, hparams, num_layers, dim_input, dim_state, dropout, seq_length):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_state = dim_state
        self.dropout = dropout
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size=self.dim_input,
                          hidden_size=self.dim_state,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)

        self.out = nn.Linear(self.dim_state*self.seq_length*2, self.seq_length)

    def forward(self, x):
        x = x.unsqueeze(0) if x.ndim == 2 else x
        output, hidden = self.gru(x)
        batch_size = output.size(0)

        enc_output = output.reshape(batch_size, -1)
        logits = self.out(F.relu(enc_output))

        hidden = hidden.view(self.num_layers, 2, batch_size, self.dim_state).sum(dim=1)
        output = output[:, :, :self.dim_state] + output[:, :, self.dim_state:]
        return logits, output, hidden

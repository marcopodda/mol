from torch import nn


class Encoder(nn.Module):
    def __init__(self, hparams, num_layers, dim_input, dim_state, dropout):
        super().__init__()
        self.hparams = hparams

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_state = dim_state
        self.dropout = dropout

        self.gru = nn.GRU(input_size=self.dim_input,
                          hidden_size=self.dim_state,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)

    def forward(self, x):
        x = x.unsqueeze(0) if x.ndim == 2 else x
        output, hidden = self.gru(x)

        batch_size = output.size(0)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.dim_state).sum(dim=1)
        output = output[:, :, :self.dim_state] + output[:, :, self.dim_state:]
        return output, hidden

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.lin1 = nn.Linear(dim_input, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.lin2(x)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers=1):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers
        
        self.input_layer = nn.Linear(dim_input, dim_hidden)
        
        self.hidden_layers = nn.ModuleList([])
        for l in range(num_layers):
            layer = nn.Linear(dim_hidden, dim_hidden)
            self.hidden_layers.append(layer)
        
        self.output_layer = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
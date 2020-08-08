import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree

from core.datasets.features import ATOM_FDIM, BOND_FDIM
from layers.mlp import MLP


class GNNLayer(nn.Module):
    def __init__(self, hparams, dim_input):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = hparams.gnn_dim_hidden_edge
        self.dim_output = hparams.gnn_dim_hidden

        self.conv = NNConv(
            in_channels=self.dim_input,
            out_channels=self.dim_output,
            nn=MLP(
                dim_input=BOND_FDIM,
                dim_hidden=self.dim_hidden,
                dim_output=self.dim_input * self.dim_output
            )
        )

        self.bn = BatchNorm(self.dim_output)

    def forward(self, x, edge_index, edge_attr):
        outputs = self.conv(x, edge_index, edge_attr)
        return F.relu(outputs)


class GNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_layers = hparams.gnn_num_layers
        self.dim_input = ATOM_FDIM
        self.dim_hidden = hparams.gnn_dim_hidden
        self.dim_embed = hparams.gnn_dim_embed

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden
            self.layers.append(GNNLayer(hparams, dim_input))

        self.lin = nn.Linear(self.dim_hidden * self.num_layers, self.dim_embed)

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            outputs.append(global_add_pool(x, batch))

        outputs = torch.cat(outputs, dim=1)
        return self.lin(outputs)

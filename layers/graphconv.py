import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_add_pool, Set2Set
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree

from core.datasets.features import ATOM_FDIM, BOND_FDIM
from layers.mlp import MLP


class GNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_layers = hparams.gnn_num_layers
        self.dim_input = ATOM_FDIM
        self.dim_hidden = hparams.gnn_dim_hidden
        self.dim_embed = hparams.gnn_dim_embed

        self.convs = nn.ModuleList([])
        
        edge_net = MLP(
            dim_input=BOND_FDIM,
            dim_hidden=self.hparams.gnn_dim_hidden_edge,
            dim_output=BOND_FDIM * self.dim_hidden
        )
        
        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden
            conv = NNConv(in_channels=dim_input, out_channels=self.dim_hidden, nn=edge_net)
            self.convs.append(conv)

        self.bn = BatchNorm(self.dim_hidden, track_running_stats=False)
        self.readout = nn.Linear(self.dim_hidden, self.dim_embed)

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        outputs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.bn(F.relu(x))
        
        output = global_add_pool(x, batch)
        return self.readout(output)

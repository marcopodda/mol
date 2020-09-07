import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_add_pool, Set2Set
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
from torch_scatter import scatter_add

from core.datasets.features import ATOM_FDIM, BOND_FDIM
from layers.mlp import MLP


class GNN(nn.Module):
    def __init__(self, hparams, num_layers, dim_edge_embed, dim_hidden, dim_output):
        super().__init__()
        self.hparams = hparams
        
        self.dim_input = ATOM_FDIM
        self.dim_edge_embed = dim_edge_embed
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        self.num_layers = num_layers

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        
        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden
            
            edge_net = MLP(
                dim_input=BOND_FDIM,
                dim_hidden=self.dim_edge_embed,
                dim_output=dim_input * self.dim_hidden,
            )
            
            conv = NNConv(
                in_channels=dim_input, 
                out_channels=self.dim_hidden, 
                nn=edge_net,
                root_weight=False,
                bias=False)
            
            self.convs.append(conv)
            
            bn = nn.BatchNorm1d(self.dim_hidden, track_running_stats=False)
            self.bns.append(bn)

        if self.dim_output != self.dim_hidden:
            self.readout = nn.Linear(self.dim_hidden, self.dim_output)
        else:
            self.readout = None

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))
        
        x = global_add_pool(x, batch) 
        nodes_per_graph = scatter_add(torch.ones_like(batch), batch).view(-1, 1)
        output = self.readout(x) if self.dim_output != self.dim_hidden else x
        
        return output / nodes_per_graph
        

class GNNEmbedder(nn.Module):
    def __init__(self, hparams, num_layers, dim_edge_embed, dim_hidden, dim_output):
        super().__init__()
        self.hparams = hparams
        
        self.gnn = GNN(
            hparams=hparams,
            num_layers=num_layers,
            dim_edge_embed=dim_edge_embed,
            dim_hidden=dim_hidden,
            dim_output=dim_output)
    
    def forward(self, data, mat, input=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.gnn(x, edge_index, edge_attr, batch=data.frags_batch)
        
        cumsum = 0
        for i, l in enumerate(data.length):
            offset = list(range(1, l+1)) if input is True else list(range(l))
            mat[i,offset,:] = x[cumsum:cumsum+l,:]
            cumsum += l
        
        return mat
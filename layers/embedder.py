import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_add_pool
from torch_scatter import scatter_add

from core.hparams import HParams
from layers.mlp import MLP


class GNN(nn.Module):
    def __init__(self, hparams, num_layers, dim_input, dim_edge_features, dim_edge_embed, dim_hidden, dim_output):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_edge_features = dim_edge_features
        self.dim_edge_embed = dim_edge_embed
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden

            edge_net = MLP(
                hparams=self.hparams,
                dim_input=self.dim_edge_features,
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

    def embed_single(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))

        nodes_per_graph = scatter_add(torch.ones_like(batch), batch)
        nodes_per_graph = nodes_per_graph.repeat_interleave(nodes_per_graph.view(-1))
        output = global_add_pool(x / nodes_per_graph.view(-1, 1), batch)

        if self.readout is not None:
            output = self.readout(x)

        return output

    def forward(self, x, edge_index, edge_attr, frag_batch, graph_batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))

        nodes_per_graph = scatter_add(torch.ones_like(frag_batch), frag_batch)
        nodes_per_graph = nodes_per_graph.repeat_interleave(nodes_per_graph.view(-1))
        output = global_add_pool(x / nodes_per_graph.view(-1, 1), frag_batch)

        graph_output = global_add_pool(x / nodes_per_graph.view(-1, 1), graph_batch)

        if self.readout is not None:
            output = self.readout(x)

        # nodes_per_graph = scatter_add(torch.ones_like(batch), batch)
        # output = global_add_pool(x, batch)
        return output, graph_output


class Embedder(nn.Module):
    def __init__(self, hparams, num_layers, dim_input, dim_edge_features, dim_edge_embed, dim_hidden, dim_output):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_edge_features = dim_edge_features
        self.dim_edge_embed = dim_edge_embed
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.gnn = GNN(
            hparams=self.hparams,
            num_layers=self.num_layers,
            dim_input=self.dim_input,
            dim_edge_features=self.dim_edge_features,
            dim_edge_embed=self.dim_edge_embed,
            dim_hidden=self.dim_hidden,
            dim_output=self.dim_output)

    def forward(self, data, mat, input=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, bag_of_fragments = self.gnn(x, edge_index, edge_attr, frag_batch=data.frags_batch, graph_batch=data.batch)

        cumsum = 0
        for i, l in enumerate(data.length):
            offset = 1 if input is True else 0
            seq_element = x[cumsum:cumsum + l, :]
            mat[i, range(offset, offset + l), :] = seq_element
            cumsum += l

        return mat, bag_of_fragments

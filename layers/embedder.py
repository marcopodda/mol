import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, GINEConv, global_add_pool
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
        self.edge_nets = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden
            dim_output = self.dim_output if i == self.num_layers - 1 else self.dim_hidden

            conv = GINEConv(nn=MLP(
                hparams=self.hparams,
                dim_input=dim_input,
                dim_hidden=self.dim_hidden,
                dim_output=dim_output))
            self.convs.append(conv)

            bn = nn.BatchNorm1d(dim_output, track_running_stats=False)
            self.bns.append(bn)

            dim_output = self.dim_input if i == 0 else self.dim_hidden
            edge_net = nn.Linear(self.dim_edge_features, dim_output)
            self.edge_nets.append(edge_net)

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def embed_single(self, x, edge_index, edge_attr, batch):
        for conv, en, bn in zip(self.convs, self.edge_nets, self.bns):
            edge_attr = en(edge_attr)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))

        output = self.aggregate_nodes(x, batch)
        return output

    def forward(self, x, edge_index, edge_attr, frag_batch, graph_batch):
        for i, (conv, en, bn) in enumerate(zip(self.convs, self.edge_nets, self.bns)):
            print(edge_attr.size(), en.in_features, en.out_features)
            edge_attr = en(edge_attr)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))

        # aggregate each fragment in the sequence
        output = self.aggregate_nodes(x, frag_batch)

        # aggregate all fragments in the sequence into a bag of frags
        graph_output = self.aggregate_nodes(x, graph_batch)
        return output, graph_output

    def aggregate_nodes(self, nodes_repr, batch):
        nodes_per_graph = scatter_add(torch.ones_like(batch), batch)
        nodes_per_graph = nodes_per_graph.repeat_interleave(nodes_per_graph.view(-1))
        nodes_per_graph = torch.sqrt(nodes_per_graph.view(-1, 1).float())
        graph_repr = global_add_pool(nodes_repr / nodes_per_graph, batch)
        return graph_repr


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
        self.dropout = self.hparams.embedder_dropout

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
            mat[i, range(offset, l.item() + offset), :] = seq_element
            cumsum += l

        return mat, bag_of_fragments

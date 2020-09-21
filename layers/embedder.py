import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import NNConv, global_add_pool
from torch_scatter import scatter_add

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

from core.hparams import HParams
from layers.mlp import MLP


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)
    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn, eps=0., train_eps=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        # Node and edge feature dimensionalites need to match.
        # if isinstance(edge_index, Tensor):
        #     assert edge_attr is not None
        #     assert x[0].size(-1) == edge_attr.size(-1)
        # elif isinstance(edge_index, SparseTensor):
        #     assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        return F.relu(x_j)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


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
            dim_output = self.dim_output if i == self.num_layers - 1 else self.dim_hidden

            conv = GINEConv(nn=MLP(
                hparams=self.hparams,
                dim_input=dim_input + dim_edge_features,
                dim_hidden=self.dim_hidden,
                dim_output=dim_output + dim_edge_features))
            self.convs.append(conv)

            bn = nn.BatchNorm1d(dim_output, track_running_stats=False)
            self.bns.append(bn)

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def embed_single(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(F.relu(x))

        output = self.aggregate_nodes(x, batch)
        return output

    def forward(self, x, edge_index, edge_attr, frag_batch, graph_batch):
        for conv, bn in zip(self.convs, self.bns):
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

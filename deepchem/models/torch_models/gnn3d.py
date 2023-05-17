import copy
from typing import List

import dgl
import dgl.function as fn
import torch
from torch import nn
from torch.nn import functional as F

from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.pna_gnn import AtomEncoder
from deepchem.utils.graph_utils import fourier_encode_dist


class Net3DLayer(nn.Module):
    """
    Net3DLayer is a single layer of a 3D graph neural network based on the 3D Infomax architecture [1].

    This class expects a DGL graph with node features stored under the name 'feat' and edge features stored under the name 'd' (representing 3D distances). The edge features are updated by the message network and the node features are updated by the update network.

    Parameters
    ----------
    edge_dim : int
        The dimension of the edge features.
    hidden_dim : int
        The dimension of the hidden layers.
    reduce_func : str
        The reduce function to use for aggregating messages. Can be either 'sum' or 'mean'.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    mid_activation : str, optional (default='SiLU')
        The activation function to use in the network.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    update_net_layers : int, optional (default=2)
        The number of update network layers.

    References
    ----------
    .. [1] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).

    Examples
    --------
    >>> net3d_layer = Net3DLayer(edge_dim=3, hidden_dim=3)
    >>> graph = dgl.graph(([0, 1], [1, 2]))
    >>> graph.ndata['feat'] = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> graph.edata['d'] = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    >>> output = net3d_layer(graph)
    """

    def __init__(self,
                 edge_dim: int,
                 hidden_dim: int,
                 reduce_func: str = 'sum',
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 dropout: float = 0.0,
                 message_net_layers: int = 2,
                 update_net_layers: int = 2):
        super(Net3DLayer, self).__init__()

        self.message_network = nn.Sequential(
            MultilayerPerceptron(d_input=hidden_dim * 2 + edge_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,) *
                                 (message_net_layers - 1),
                                 batch_norm=batch_norm,
                                 batch_norm_momentum=batch_norm_momentum,
                                 dropout=dropout), torch.nn.SiLU())
        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supported: ', reduce_func)

        self.update_network = MultilayerPerceptron(
            d_input=hidden_dim,
            d_hidden=(hidden_dim,) * (update_net_layers - 1),
            d_output=hidden_dim,
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum)

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, input_graph: dgl.DGLGraph):
        # copy the input graph to avoid in-place operations
        graph = input_graph.local_var()
        graph.ndata['feat'] = input_graph.ndata['feat'].clone()
        graph.edata['d'] = input_graph.edata['d'].clone()

        graph.update_all(message_func=self.message_function,
                         reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)
        return graph

    def message_function(self, edges):
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d']], dim=-1)
        message = self.message_network(message_input)
        edges.data['d'] += message
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}


class Net3D(nn.Module):
    """
    Net3D is a 3D graph neural network that expects a DGL graph input with 3D coordinates stored under the name 'd' and node features stored under the name 'feat'. It is based on the 3D Infomax architecture [1].

    Parameters
    ----------
    hidden_dim : int
        The dimension of the hidden layers.
    target_dim : int
        The dimension of the output layer.
    readout_aggregators : List[str]
        A list of aggregator functions for the readout layer.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    node_wise_output_layers : int, optional (default=2)
        The number of output layers for each node.
    readout_batchnorm : bool, optional (default=True)
        Whether to use batch normalization in the readout layer.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    reduce_func : str, optional (default='sum')
        The reduce function to use for aggregating messages.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    propagation_depth : int, optional (default=4)
        The number of propagation layers in the network.
    readout_layers : int, optional (default=2)
        The number of readout layers in the network.
    readout_hidden_dim : int, optional (default=None)
        The dimension of the hidden layers in the readout network.
    fourier_encodings : int, optional (default=0)
        The number of Fourier encodings to use.
    activation : str, optional (default='SiLU')
        The activation function to use in the network.
    update_net_layers : int, optional (default=2)
        The number of update network layers.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    use_node_features : bool, optional (default=False)
        Whether to use node features as input.

    References
    ----------
    .. [1] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 readout_aggregators: List[str],
                 batch_norm=False,
                 node_wise_output_layers=2,
                #  readout_batchnorm=True,
                 batch_norm_momentum=0.1,
                 reduce_func='sum',
                 dropout=0.0,
                 propagation_depth: int = 4,
                 readout_layers: int = 2,
                 readout_hidden_dim=None,
                 fourier_encodings=4,
                 #  activation: str = 'SiLU',
                 update_net_layers=2,
                 message_net_layers=2,
                 use_node_features=False):
        super(Net3D, self).__init__()
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1  # originally 1 XXX

        self.edge_input = nn.Sequential(
            MultilayerPerceptron(d_input=edge_in_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,),
                                 batch_norm=True,
                                 batch_norm_momentum=batch_norm_momentum),
            torch.nn.SiLU())

        self.use_node_features = use_node_features
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim,
                           hidden_dim=hidden_dim,
                           batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum,
                           dropout=dropout,
                           reduce_func=reduce_func,
                           message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MultilayerPerceptron(
                d_input=hidden_dim,
                d_output=hidden_dim,
                d_hidden=(hidden_dim,),
                batch_norm=True,
                batch_norm_momentum=batch_norm_momentum)

        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators

        self.output = MultilayerPerceptron(
            d_input=hidden_dim * len(self.readout_aggregators),
            d_output=target_dim,
            d_hidden=(readout_hidden_dim,) *
            (readout_layers -
             1),  # -1 because the input layer is not considered a hidden layer
            batch_norm=False)

    def forward(self, graph: dgl.DGLGraph):
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(
                graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(
                graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            graph = mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [
            dgl.readout_nodes(graph, 'feat', op=aggr)
            for aggr in self.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}
import math
from math import pi as PI
import numpy as np
import itertools
import sympy as sym
from typing import Any, Tuple, Optional, Sequence, List, Union, Callable, Dict, TypedDict
from collections.abc import Sequence as SequenceCollection
try:
    import torch
    from torch import Tensor
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
from deepchem.models.torch_models.flows import Affine
try:
    from torch_geometric.utils import scatter
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops
    from torch_geometric.nn.models.dimenet_utils import bessel_basis, real_sph_harm
except ModuleNotFoundError:
    pass

from deepchem.utils.typing import OneOrMany, ActivationFn, ArrayLike
from deepchem.utils.pytorch_utils import get_activation, segment_sum, unsorted_segment_sum, unsorted_segment_max
from torch.nn import init as initializers


class MultilayerPerceptron(nn.Module):
    """A simple fully connected feed-forward network, otherwise known as a multilayer perceptron (MLP).

    Examples
    --------
    >>> model = MultilayerPerceptron(d_input=10, d_hidden=(2,3), d_output=2, dropout=0.0, activation_fn='relu')
    >>> x = torch.ones(2, 10)
    >>> out = model(x)
    >>> print(out.shape)
    torch.Size([2, 2])
    """

    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_hidden: Optional[tuple] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 activation_fn: Union[Callable, str] = 'relu',
                 skip_connection: bool = False,
                 weighted_skip: bool = True):
        """Initialize the model.

        Parameters
        ----------
        d_input: int
            the dimension of the input layer
        d_output: int
            the dimension of the output layer
        d_hidden: tuple
            the dimensions of the hidden layers
        dropout: float
            the dropout probability
        batch_norm: bool
            whether to use batch normalization
        batch_norm_momentum: float
            the momentum for batch normalization
        activation_fn: str
            the activation function to use in the hidden layers
        skip_connection: bool
            whether to add a skip connection from the input to the output
        weighted_skip: bool
            whether to add a weighted skip connection from the input to the output
        """
        super(MultilayerPerceptron, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.activation_fn = get_activation(activation_fn)
        self.model = nn.Sequential(*self.build_layers())
        self.skip = nn.Linear(d_input, d_output) if skip_connection else None
        self.weighted_skip = weighted_skip

    def build_layers(self):
        """
        Build the layers of the model, iterating through the hidden dimensions to produce a list of layers.
        """

        layer_list = []
        layer_dim = self.d_input
        if self.d_hidden is not None:
            for d in self.d_hidden:
                layer_list.append(nn.Linear(layer_dim, d))
                layer_list.append(self.dropout)
                if self.batch_norm:
                    layer_list.append(
                        nn.BatchNorm1d(d, momentum=self.batch_norm_momentum))
                layer_dim = d
        layer_list.append(nn.Linear(layer_dim, self.d_output))
        return layer_list

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        input = x
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_fn(
                    x
                )  # Done because activation_fn returns a torch.nn.functional
        if self.skip is not None:
            if not self.weighted_skip:
                return x + input
            else:
                return x + self.skip(input)
        else:
            return x


class CNNModule(nn.Module):
    """A 1, 2, or 3 dimensional convolutional network for either regression or classification.
    The network consists of the following sequence of layers:
    - A configurable number of convolutional layers
    - A global pooling layer (either max pool or average pool)
    - A final fully connected layer to compute the output
    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    convolution layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same output shape.  Wherever the output shape changes, a
    simple convolution layer will be used even if residual=True.
    Examples
    --------
    >>> model = CNNModule(n_tasks=5, n_features=8, dims=2, layer_filters=[3,8,8,16], kernel_size=3, n_classes = 7, mode='classification', uncertainty=False, padding='same')
    >>> x = torch.ones(2, 224, 224, 8)
    >>> x = model(x)
    >>> for tensor in x:
    ...    print(tensor.shape)
    torch.Size([2, 5, 7])
    torch.Size([2, 5, 7])
    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 dims: int,
                 layer_filters: List[int] = [100],
                 kernel_size: OneOrMany[int] = 5,
                 strides: OneOrMany[int] = 1,
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 pool_type: str = 'max',
                 mode: str = 'classification',
                 n_classes: int = 2,
                 uncertainty: bool = False,
                 residual: bool = False,
                 padding: Union[int, str] = 'valid') -> None:
        """Create a CNN.

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        dims: int
            the number of dimensions to apply convolutions over (1, 2, or 3)
        layer_filters: list
            the number of output filters for each convolutional layer in the network.
            The length of this list determines the number of layers.
        kernel_size: int, tuple, or list
            a list giving the shape of the convolutional kernel for each layer.  Each
            element may be either an int (use the same kernel width for every dimension)
            or a tuple (the kernel width along each dimension).  Alternatively this may
            be a single int or tuple instead of a list, in which case the same kernel
            shape is used for every layer.
        strides: int, tuple, or list
            a list giving the stride between applications of the  kernel for each layer.
            Each element may be either an int (use the same stride for every dimension)
            or a tuple (the stride along each dimension).  Alternatively this may be a
            single int or tuple instead of a list, in which case the same stride is
            used for every layer.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization
            of each layer.  The length of this list should equal len(layer_filters)+1,
            where the final element corresponds to the dense layer.  Alternatively this
            may be a single value instead of a list, in which case the same value is used
            for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The length of this
            list should equal len(layer_filters)+1, where the final element corresponds
            to the dense layer.  Alternatively this may be a single value instead of a
            list, in which case the same value is used for every layer.
        dropouts: list or float
            the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
        activation_fns: str or list
            the torch activation function to apply to each layer. The length of this list should equal
            len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer, 'relu' by default
        pool_type: str
            the type of pooling layer to use, either 'max' or 'average'
        mode: str
            Either 'classification' or 'regression'
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        residual: bool
            if True, the model will be composed of pre-activation residual blocks instead
            of a simple stack of convolutional layers.
        padding: str, int or tuple
            the padding to use for convolutional layers, either 'valid' or 'same'
        """

        super(CNNModule, self).__init__()

        if dims not in (1, 2, 3):
            raise ValueError('Number of dimensions must be 1, 2 or 3')

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        self.n_tasks = n_tasks
        self.n_features = n_features
        self.dims = dims
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        self.mode = mode
        self.layer_filters = layer_filters
        self.residual = residual

        n_layers = len(layer_filters)

        # PyTorch layers require input and output channels as parameter
        # if only one layer to make the model creating loop below work, multiply layer_filters wutg 2
        if len(layer_filters) == 1:
            layer_filters = layer_filters * 2

        if not isinstance(kernel_size, SequenceCollection):
            kernel_size = [kernel_size] * n_layers
        if not isinstance(strides, SequenceCollection):
            strides = [strides] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers

        self.activation_fns = [get_activation(f) for f in activation_fns]
        self.dropouts = dropouts

        if uncertainty:

            if mode != 'regression':
                raise ValueError(
                    "Uncertainty is only supported in regression mode")

            if any(d == 0.0 for d in dropouts):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty'
                )

        # Python tuples use 0 based indexing, dims defines number of dimension for convolutional operation
        ConvLayer = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.dims - 1]

        if pool_type == 'average':
            PoolLayer = (F.avg_pool1d, F.avg_pool2d,
                         F.avg_pool3d)[self.dims - 1]
        elif pool_type == 'max':
            PoolLayer = (
                F.max_pool1d,
                F.max_pool2d,  # type: ignore
                F.max_pool3d)[self.dims - 1]
        else:
            raise ValueError("pool_type must be either 'average' or 'max'")

        self.PoolLayer = PoolLayer
        self.layers = nn.ModuleList()

        in_shape = n_features

        for out_shape, size, stride, weight_stddev, bias_const in zip(
                layer_filters, kernel_size, strides, weight_init_stddevs,
                bias_init_consts):

            layer = ConvLayer(in_channels=in_shape,
                              out_channels=out_shape,
                              kernel_size=size,
                              stride=stride,
                              padding=padding,
                              dilation=1,
                              groups=1,
                              bias=True)

            nn.init.normal_(layer.weight, 0, weight_stddev)

            # initializing layer bias with nn.init gives mypy typecheck error
            # using the following workaround
            if layer.bias is not None:
                layer.bias = nn.Parameter(
                    torch.full(layer.bias.shape, bias_const))

            self.layers.append(layer)

            in_shape = out_shape

        self.classifier_ffn = nn.LazyLinear(self.n_tasks * self.n_classes)
        self.output_layer = nn.LazyLinear(self.n_tasks)
        self.uncertainty_layer = nn.LazyLinear(self.n_tasks)

    def forward(self, inputs: OneOrMany[torch.Tensor]) -> List[Any]:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input Tensor
        Returns
        -------
        torch.Tensor
            Output as per use case : regression/classification
        """
        if isinstance(inputs, torch.Tensor):
            x, dropout_switch = inputs, None
        else:
            x, dropout_switch = inputs

        x = torch.transpose(x, 1, -1)  # n h w c -> n c h w

        prev_layer = x

        for layer, activation_fn, dropout in zip(self.layers,
                                                 self.activation_fns,
                                                 self.dropouts):
            x = layer(x)

            if dropout > 0. and dropout_switch:
                x = F.dropout(x, dropout)

            # residual blocks can only be used when successive layers have the same output shape
            if self.residual and x.shape[1] == prev_layer.shape[1]:
                x = x + prev_layer

            if activation_fn is not None:
                x = activation_fn(x)

            prev_layer = x

        x = self.PoolLayer(x, kernel_size=x.size()[2:])

        outputs = []
        batch_size = x.shape[0]

        x = torch.reshape(x, (batch_size, -1))

        if self.mode == "classification":

            logits = self.classifier_ffn(x)
            logits = logits.view(batch_size, self.n_tasks, self.n_classes)
            output = F.softmax(logits, dim=2)
            outputs = [output, logits]

        else:
            output = self.output_layer(x)
            output = output.view(batch_size, self.n_tasks)

            if self.uncertainty:
                log_var = self.uncertainty_layer(x)
                log_var = log_var.view(batch_size, self.n_tasks, 1)
                var = torch.exp(log_var)
                outputs = [output, var, output, log_var]
            else:
                outputs = [output]

        return outputs


class ScaleNorm(nn.Module):
    """Apply Scale Normalization to input.

    The ScaleNorm layer first computes the square root of the scale, then computes the matrix/vector norm of the input tensor.
    The norm value is calculated as `sqrt(scale) / matrix norm`.
    Finally, the result is returned as `input_tensor * norm value`.

    This layer can be used instead of LayerNorm when a scaled version of the norm is required.
    Instead of performing the scaling operation (`scale / norm`) in a lambda-like layer, we are defining it within this layer to make prototyping more efficient.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import ScaleNorm
    >>> scale = 0.35
    >>> layer = ScaleNorm(scale)
    >>> input_tensor = torch.tensor([[1.269, 39.36], [0.00918, -9.12]])
    >>> output_tensor = layer(input_tensor)
    """

    def __init__(self, scale: float, eps: float = 1e-5):
        """Initialize a ScaleNorm layer.

        Parameters
        ----------
        scale: float
            Scale magnitude.
        eps: float
            Epsilon value. Default = 1e-5.
        """
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.scale / torch.norm(x, dim=-1,
                                       keepdim=True).clamp(min=self.eps)
        return x * norm


class MultiHeadedMATAttention(nn.Module):
    """First constructs an attention layer tailored to the Molecular Attention Transformer [1]_ and then converts it into Multi-Headed Attention.

    In Multi-Headed attention the attention mechanism multiple times parallely through the multiple attention heads.
    Thus, different subsequences of a given sequences can be processed differently.
    The query, key and value parameters are split multiple ways and each split is passed separately through a different attention head.
    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MultiHeadedMATAttention, MATEmbedding
    >>> import deepchem as dc
    >>> import torch
    >>> input_smile = "CC"
    >>> feat = dc.feat.MATFeaturizer()
    >>> input_smile = "CC"
    >>> out = feat.featurize(input_smile)
    >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
    >>> layer = MultiHeadedMATAttention(
    ...    dist_kernel='softmax',
    ...    lambda_attention=0.33,
    ...    lambda_distance=0.33,
    ...    h=16,
    ...    hsize=1024,
    ...    dropout_p=0.0)
    >>> op = MATEmbedding()(node)
    >>> output = layer(op, op, op, mask, adj, dist)
    """

    def __init__(self,
                 dist_kernel: str = 'softmax',
                 lambda_attention: float = 0.33,
                 lambda_distance: float = 0.33,
                 h: int = 16,
                 hsize: int = 1024,
                 dropout_p: float = 0.0,
                 output_bias: bool = True):
        """Initialize a multi-headed attention layer.
        Parameters
        ----------
        dist_kernel: str
            Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential.
        lambda_attention: float
            Constant to be multiplied with the attention matrix.
        lambda_distance: float
            Constant to be multiplied with the distance matrix.
        h: int
            Number of attention heads.
        hsize: int
            Size of dense layer.
        dropout_p: float
            Dropout probability.
        output_bias: bool
            If True, dense layers will use bias vectors.
        """
        super().__init__()
        if dist_kernel == "softmax":
            self.dist_kernel = lambda x: torch.softmax(-x, dim=-1)
        elif dist_kernel == "exp":
            self.dist_kernel = lambda x: torch.exp(-x)
        self.lambda_attention = lambda_attention
        self.lambda_distance = lambda_distance
        self.lambda_adjacency = 1.0 - self.lambda_attention - self.lambda_distance
        self.d_k = hsize // h
        self.h = h
        linear_layer = nn.Linear(hsize, hsize)
        self.linear_layers = nn.ModuleList([linear_layer for _ in range(3)])
        self.dropout_p = nn.Dropout(dropout_p)
        self.output_linear = nn.Linear(hsize, hsize, output_bias)

    def _single_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
            adj_matrix: torch.Tensor,
            distance_matrix: torch.Tensor,
            dropout_p: float = 0.0,
            eps: float = 1e-6,
            inf: float = 1e12) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defining and computing output for a single MAT attention layer.
        Parameters
        ----------
        query: torch.Tensor
            Standard query parameter for attention.
        key: torch.Tensor
            Standard key parameter for attention.
        value: torch.Tensor
            Standard value parameter for attention.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dist_matrix: torch.Tensor
            Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dropout_p: float
            Dropout probability.
        eps: float
            Epsilon value
        inf: float
            Value of infinity to be used.
        """
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2],
                                         1) == 0, -inf)
        p_attn = F.softmax(scores, dim=-1)

        adj_matrix = adj_matrix / (
            torch.sum(torch.tensor(adj_matrix), dim=-1).unsqueeze(2) + eps)

        if len(adj_matrix.shape) <= 3:
            p_adj = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        else:
            p_adj = adj_matrix.repeat(1, query.shape[1], 1, 1)

        distance_matrix = torch.tensor(distance_matrix).squeeze().masked_fill(
            mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)

        distance_matrix = self.dist_kernel(distance_matrix)

        p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
        p_weighted = self.dropout_p(p_weighted)

        return torch.matmul(p_weighted.float(), value.float()), p_attn

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor,
                adj_matrix: torch.Tensor,
                distance_matrix: torch.Tensor,
                dropout_p: float = 0.0,
                eps: float = 1e-6,
                inf: float = 1e12) -> torch.Tensor:
        """Output computation for the MultiHeadedAttention layer.
        Parameters
        ----------
        query: torch.Tensor
            Standard query parameter for attention.
        key: torch.Tensor
            Standard key parameter for attention.
        value: torch.Tensor
            Standard value parameter for attention.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dist_matrix: torch.Tensor
            Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dropout_p: float
            Dropout probability.
        eps: float
            Epsilon value
        inf: float
            Value of infinity to be used.
        """
        if mask is not None and len(mask.shape) <= 2:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x, _ = self._single_attention(query, key, value, mask, adj_matrix,
                                      distance_matrix, dropout_p, eps, inf)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)

        return self.output_linear(x)


class MATEncoderLayer(nn.Module):
    """Encoder layer for use in the Molecular Attention Transformer [1]_.

    The MATEncoder layer primarily consists of a self-attention layer (MultiHeadedMATAttention) and a feed-forward layer (PositionwiseFeedForward).
    This layer can be stacked multiple times to form an encoder.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from rdkit import Chem
    >>> import torch
    >>> import deepchem
    >>> from deepchem.models.torch_models.layers import MATEmbedding, MATEncoderLayer
    >>> input_smile = "CC"
    >>> feat = deepchem.feat.MATFeaturizer()
    >>> out = feat.featurize(input_smile)
    >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
    >>> layer = MATEncoderLayer()
    >>> op = MATEmbedding()(node)
    >>> output = layer(op, mask, adj, dist)
    """

    def __init__(self,
                 dist_kernel: str = 'softmax',
                 lambda_attention: float = 0.33,
                 lambda_distance: float = 0.33,
                 h: int = 16,
                 sa_hsize: int = 1024,
                 sa_dropout_p: float = 0.0,
                 output_bias: bool = True,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 ff_dropout_p: float = 0.0,
                 encoder_hsize: int = 1024,
                 encoder_dropout_p: float = 0.0):
        """Initialize a MATEncoder layer.

        Parameters
        ----------
        dist_kernel: str
            Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential, for the self-attention layer.
        lambda_attention: float
            Constant to be multiplied with the attention matrix in the self-attention layer.
        lambda_distance: float
            Constant to be multiplied with the distance matrix in the self-attention layer.
        h: int
            Number of attention heads for the self-attention layer.
        sa_hsize: int
            Size of dense layer in the self-attention layer.
        sa_dropout_p: float
            Dropout probability for the self-attention layer.
        output_bias: bool
            If True, dense layers will use bias vectors in the self-attention layer.
        d_input: int
            Size of input layer in the feed-forward layer.
        d_hidden: int
            Size of hidden layer in the feed-forward layer.
        d_output: int
            Size of output layer in the feed-forward layer.
        activation: str
            Activation function to be used in the feed-forward layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers in the feed-forward layer.
        dropout_p: float
            Dropout probability in the feeed-forward layer.
        encoder_hsize: int
            Size of Dense layer for the encoder itself.
        encoder_dropout_p: float
            Dropout probability for connections in the encoder layer.
        """
        super(MATEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedMATAttention(dist_kernel, lambda_attention,
                                                 lambda_distance, h, sa_hsize,
                                                 sa_dropout_p, output_bias)
        self.feed_forward = PositionwiseFeedForward(d_input, d_hidden, d_output,
                                                    activation, n_layers,
                                                    ff_dropout_p)
        layer = SublayerConnection(size=encoder_hsize,
                                   dropout_p=encoder_dropout_p)
        self.sublayer = nn.ModuleList([layer for _ in range(2)])
        self.size = encoder_hsize

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                adj_matrix: torch.Tensor,
                distance_matrix: torch.Tensor,
                sa_dropout_p: float = 0.0) -> torch.Tensor:
        """Output computation for the MATEncoder layer.

        In the MATEncoderLayer intialization, self.sublayer is defined as an nn.ModuleList of 2 layers. We will be passing our computation through these layers sequentially.
        nn.ModuleList is subscriptable and thus we can access it as self.sublayer[0], for example.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of a molecule.
        distance_matrix: torch.Tensor
            Distance matrix of a molecule.
        sa_dropout_p: float
            Dropout probability for the self-attention layer (MultiHeadedMATAttention).
        """
        x = self.sublayer[0](x,
                             self.self_attn(x,
                                            x,
                                            x,
                                            mask=mask,
                                            dropout_p=sa_dropout_p,
                                            adj_matrix=adj_matrix,
                                            distance_matrix=distance_matrix))
        return self.sublayer[1](x, self.feed_forward(x))


class SublayerConnection(nn.Module):
    """SublayerConnection layer based on the paper `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    The SublayerConnection normalizes and adds dropout to output tensor of an arbitary layer.
    It further adds a residual layer connection between the input of the arbitary layer and the dropout-adjusted layer output.

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import SublayerConnection
    >>> scale = 0.35
    >>> layer = SublayerConnection(2, 0.)
    >>> input_ar = torch.tensor([[1., 2.], [5., 6.]])
    >>> output = layer(input_ar, input_ar)
  """

    def __init__(self, size: int, dropout_p: float = 0.0):
        """Initialize a SublayerConnection Layer.

        Parameters
        ----------
        size: int
            Size of layer.
        dropout_p: float
            Dropout probability.
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout_p = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Output computation for the SublayerConnection layer.

        Takes an input tensor x, then adds the dropout-adjusted sublayer output for normalized x to it.
        This is done to add a residual connection followed by LayerNorm.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        output: torch.Tensor
            Layer whose normalized output will be added to x.
        """
        if x is None:
            return self.dropout_p(self.norm(output))
        return x + self.dropout_p(self.norm(output))


class PositionwiseFeedForward(nn.Module):
    """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_

    Each layer in the MAT encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
    This is done in addition to the SublayerConnection module.

    Note: This modified version of `PositionwiseFeedForward` class contains `dropout_at_input_no_act` condition to facilitate its use in defining
        the feed-forward (FFN) algorithm for the Directed Message Passing Neural Network (D-MPNN) [2]_

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    .. [2] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import PositionwiseFeedForward
    >>> feed_fwd_layer = PositionwiseFeedForward(d_input = 2, d_hidden = 2, d_output = 2, activation = 'relu', n_layers = 1, dropout_p = 0.1)
    >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
    >>> output_tensor = feed_fwd_layer(input_tensor)
  """

    def __init__(self,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden: int (same as d_input if d_output = 0)
            Size of hidden layer.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        super(PositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act

        if activation == 'relu':
            self.activation: Any = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == "linear":
            self.activation = lambda x: x

        self.n_layers: int = n_layers
        d_output = d_output if d_output != 0 else d_input
        d_hidden = d_hidden if d_hidden != 0 else d_input

        if n_layers == 1:
            self.linears: Any = [nn.Linear(d_input, d_output)]

        else:
            self.linears = [nn.Linear(d_input, d_hidden)] + [
                nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)
            ] + [nn.Linear(d_hidden, d_output)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output Computation for the PositionwiseFeedForward layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        if not self.n_layers:
            return x

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return self.linears[0](self.dropout_p[0](x))
            else:
                return self.dropout_p[0](self.activation(self.linears[0](x)))

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)


class MATEmbedding(nn.Module):
    """Embedding layer to create embedding for inputs.

    In an embedding layer, input is taken and converted to a vector representation for each input.
    In the MATEmbedding layer, an input tensor is processed through a dropout-adjusted linear layer and the resultant vector is returned.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MATEmbedding
    >>> layer = MATEmbedding(d_input = 3, d_output = 3, dropout_p = 0.2)
    >>> input_tensor = torch.tensor([1., 2., 3.])
    >>> output = layer(input_tensor)
    """

    def __init__(self,
                 d_input: int = 36,
                 d_output: int = 1024,
                 dropout_p: float = 0.0):
        """Initialize a MATEmbedding layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_output: int
            Size of output layer.
        dropout_p: float
            Dropout probability for layer.
        """
        super(MATEmbedding, self).__init__()
        self.linear_unit = nn.Linear(d_input, d_output)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computation for the MATEmbedding layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to be converted into a vector.
        """
        return self.dropout(self.linear_unit(x))


class MATGenerator(nn.Module):
    """MATGenerator defines the linear and softmax generator step for the Molecular Attention Transformer [1]_.

    In the MATGenerator, a Generator is defined which performs the Linear + Softmax generation step.
    Depending on the type of aggregation selected, the attention output layer performs different operations.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MATGenerator
    >>> layer = MATGenerator(hsize = 3, aggregation_type = 'mean', d_output = 1, n_layers = 1, dropout_p = 0.3, attn_hidden = 128, attn_out = 4)
    >>> input_tensor = torch.tensor([1., 2., 3.])
    >>> mask = torch.tensor([1., 1., 1.])
    >>> output = layer(input_tensor, mask)
    """

    def __init__(self,
                 hsize: int = 1024,
                 aggregation_type: str = 'mean',
                 d_output: int = 1,
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 attn_hidden: int = 128,
                 attn_out: int = 4):
        """Initialize a MATGenerator.

        Parameters
        ----------
        hsize: int
            Size of input layer.
        aggregation_type: str
            Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
        d_output: int
            Size of output layer.
        n_layers: int
            Number of layers in MATGenerator.
        dropout_p: float
            Dropout probability for layer.
        attn_hidden: int
            Size of hidden attention layer.
        attn_out: int
            Size of output attention layer.
        """
        super(MATGenerator, self).__init__()

        if aggregation_type == 'grover':
            self.att_net = nn.Sequential(
                nn.Linear(hsize, attn_hidden, bias=False),
                nn.Tanh(),
                nn.Linear(attn_hidden, attn_out, bias=False),
            )
            hsize *= attn_out

        if n_layers == 1:
            self.proj: Any = nn.Linear(hsize, d_output)

        else:
            self.proj = []

            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(hsize, attn_hidden))
                self.proj.append(nn.LeakyReLU(negative_slope=0.1))
                self.proj.append(nn.LayerNorm(attn_hidden))
                self.proj.append(nn.Dropout(dropout_p))
            self.proj.append(nn.Linear(attn_hidden, d_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computation for the MATGenerator layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        mask: torch.Tensor
            Mask for padding so that padded values do not get included in attention score calculation.
        """

        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask

        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum

        elif self.aggregation_type == 'grover':
            out_attn = self.att_net(out_masked)
            out_attn = out_attn.masked_fill(mask == 0, -1e9)
            out_attn = F.softmax(out_attn, dim=1)
            out_avg_pooling = torch.matmul(torch.transpose(out_attn, -1, -2),
                                           out_masked)
            out_avg_pooling = out_avg_pooling.view(out_avg_pooling.size(0), -1)

        elif self.aggregation_type == 'contextual':
            out_avg_pooling = x

        return self.proj(out_avg_pooling)


class GraphNetwork(torch.nn.Module):
    """Graph Networks

    A Graph Network [1]_ takes a graph as input and returns an updated graph
    as output. The output graph has same structure as input graph but it
    has updated node features, edge features and global state features.

    Parameters
    ----------
    n_node_features: int
        Number of features in a node
    n_edge_features: int
        Number of features in a edge
    n_global_features: int
        Number of global features
    is_undirected: bool, optional (default True)
        Directed or undirected graph
    residual_connection: bool, optional (default True)
        If True, the layer uses a residual connection during training

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import GraphNetwork as GN
    >>> n_nodes, n_node_features = 5, 10
    >>> n_edges, n_edge_features = 5, 2
    >>> n_global_features = 4
    >>> node_features = torch.randn(n_nodes, n_node_features)
    >>> edge_features = torch.randn(n_edges, n_edge_features)
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
    >>> global_features = torch.randn(1, n_global_features)
    >>> gn = GN(n_node_features=n_node_features, n_edge_features=n_edge_features, n_global_features=n_global_features)
    >>> node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)

    References
    ----------
    .. [1] Battaglia et al, Relational inductive biases, deep learning, and graph networks. https://arxiv.org/abs/1806.01261 (2018)
  """

    def __init__(self,
                 n_node_features: int = 32,
                 n_edge_features: int = 32,
                 n_global_features: int = 32,
                 is_undirected: bool = True,
                 residual_connection: bool = True):
        super().__init__()
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_global_features = n_global_features
        self.is_undirected = is_undirected
        self.residual_connection = residual_connection

        self.edge_models, self.node_models, self.global_models = torch.nn.ModuleList(
        ), torch.nn.ModuleList(), torch.nn.ModuleList()
        self.edge_models.append(
            nn.Linear(in_features=n_node_features * 2 + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.node_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.global_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))

        # Used for converting edges back to their original shape
        self.edge_dense = nn.Linear(in_features=32,
                                    out_features=n_edge_features)
        self.node_dense = nn.Linear(in_features=32,
                                    out_features=n_node_features)
        self.global_dense = nn.Linear(in_features=32,
                                      out_features=n_global_features)

    def reset_parameters(self) -> None:
        self.edge_dense.reset_parameters()
        self.node_dense.reset_parameters()
        self.global_dense.reset_parameters()
        for i in range(0, len(self.edge_models)):
            self.edge_models[i].reset_parameters()
        for i in range(0, len(self.node_models)):
            self.node_models[i].reset_parameters()
        for i in range(0, len(self.global_models)):
            self.global_models[i].reset_parameters()

    def _update_edge_features(self, node_features, edge_index, edge_features,
                              global_features, batch):
        src_index, dst_index = edge_index
        out = torch.cat((node_features[src_index], node_features[dst_index],
                         edge_features, global_features[batch]),
                        dim=1)
        assert out.shape[
            1] == self.n_node_features * 2 + self.n_edge_features + self.n_global_features
        for model in self.edge_models:
            out = model(out)
        return self.edge_dense(out)

    def _update_node_features(self, node_features, edge_index, edge_features,
                              global_features, batch):

        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter(src=edge_features,
                                             index=dst_index,
                                             dim=0,
                                             reduce='mean')
        out = torch.cat(
            (node_features, edge_features_mean_by_node, global_features[batch]),
            dim=1)
        for model in self.node_models:
            out = model(out)
        return self.node_dense(out)

    def _update_global_features(self, node_features, edge_features,
                                global_features, node_batch_map,
                                edge_batch_map):
        edge_features_mean = scatter(src=edge_features,
                                     index=edge_batch_map,
                                     dim=0,
                                     reduce='mean')
        node_features_mean = scatter(src=node_features,
                                     index=node_batch_map,
                                     dim=0,
                                     reduce='mean')
        out = torch.cat(
            (edge_features_mean, node_features_mean, global_features), dim=1)
        for model in self.global_models:
            out = model(out)
        return self.global_dense(out)

    def forward(
            self,
            node_features: Tensor,
            edge_index: Tensor,
            edge_features: Tensor,
            global_features: Tensor,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Output computation for a GraphNetwork

        Parameters
        ----------
        node_features: torch.Tensor
            Input node features of shape :math:`(|\mathcal{V}|, F_n)`
        edge_index: torch.Tensor
            Edge indexes of shape :math:`(2, |\mathcal{E}|)`
        edge_features: torch.Tensor
            Edge features of the graph, shape: :math:`(|\mathcal{E}|, F_e)`
        global_features: torch.Tensor
            Global features of the graph, shape: :math:`(F_g, 1)` where, :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the number of nodes and edges in the graph,
            :math:`F_n`, :math:`F_e`, :math:`F_g` denotes the number of node features, edge features and global state features respectively.
        batch: torch.LongTensor (optional, default: None)
            A vector that maps each node to its respective graph identifier. The attribute is used only when more than one graph are batched together during a single forward pass.
        """
        if batch is None:
            batch = node_features.new_zeros(node_features.size(0),
                                            dtype=torch.int64)

        node_features_copy, edge_features_copy, global_features_copy = node_features, edge_features, global_features
        if self.is_undirected is True:
            # holding bi-directional edges in case of undirected graphs
            edge_index = torch.cat((edge_index, edge_index.flip([0])), dim=1)
            edge_features_len = edge_features.shape[0]
            edge_features = torch.cat((edge_features, edge_features), dim=0)
        edge_batch_map = batch[edge_index[0]]
        edge_features = self._update_edge_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features,
                                                   edge_batch_map)
        node_features = self._update_node_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features, batch)
        global_features = self._update_global_features(node_features,
                                                       edge_features,
                                                       global_features, batch,
                                                       edge_batch_map)

        if self.is_undirected is True:
            # coonverting edge features to its original shape
            split = torch.split(edge_features,
                                [edge_features_len, edge_features_len])
            edge_features = (split[0] + split[1]) / 2

        if self.residual_connection:
            edge_features += edge_features_copy
            node_features += node_features_copy
            global_features += global_features_copy

        return node_features, edge_features, global_features

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(n_node_features={self.n_node_features}, n_edge_features={self.n_edge_features}, n_global_features={self.n_global_features}, is_undirected={self.is_undirected}, residual_connection={self.residual_connection})'
        )


class DMPNNEncoderLayer(nn.Module):
    """
    Encoder layer for use in the Directed Message Passing Neural Network (D-MPNN) [1]_.

    The role of the DMPNNEncoderLayer class is to generate molecule encodings in following steps:

    - Message passing phase
    - Get new atom hidden states and readout phase
    - Concatenate the global features


    Let the diagram given below represent a molecule containing 5 atoms (nodes) and 4 bonds (edges):-

    |   1 --- 5
    |   |
    |   2 --- 4
    |   |
    |   3

    Let the bonds from atoms 1->2 (**B[12]**) and 2->1 (**B[21]**) be considered as 2 different bonds.
    Hence, by considering the same for all atoms, the total number of bonds = 8.

    Let:

    - **atom features** : ``a1, a2, a3, a4, a5``
    - **hidden states of atoms** : ``h1, h2, h3, h4, h5``
    - **bond features bonds** : ``b12, b21, b23, b32, b24, b42, b15, b51``
    - **initial hidden states of bonds** : ``(0)h12, (0)h21, (0)h23, (0)h32, (0)h24, (0)h42, (0)h15, (0)h51``

    The hidden state of every bond is a function of the concatenated feature vector which contains
    concatenation of the **features of initial atom of the bond** and **bond features**.

    Example: ``(0)h21 = func1(concat(a2, b21))``

    .. note::
     Here func1 is ``self.W_i``

    **The Message passing phase**

    The goal of the message-passing phase is to generate **hidden states of all the atoms in the molecule**.

    The hidden state of an atom is **a function of concatenation of atom features and messages (at T depth)**.

    A message is a sum of **hidden states of bonds coming to the atom (at T depth)**.

    .. note::
     Depth refers to the number of iterations in the message passing phase (here, T iterations). After each iteration, the hidden states of the bonds are updated.


    Example:
    ``h1 = func3(concat(a1, m1))``

    .. note::
     Here func3 is ``self.W_o``.

     `m1` refers to the message coming to the atom.

    ``m1 = (T-1)h21 + (T-1)h51``
    (hidden state of bond 2->1 + hidden state of bond 5->1) (at T depth)

    for, depth T = 2:

    - the hidden states of the bonds @ 1st iteration will be => (0)h21, (0)h51
    - the hidden states of the bonds @ 2nd iteration will be => (1)h21, (1)h51

    The hidden states of the bonds in 1st iteration are already know.
    For hidden states of the bonds in 2nd iteration, we follow the criterion that:

    - hidden state of the bond is a function of **initial hidden state of bond**
    and **messages coming to that bond in that iteration**

    Example:
    ``(1)h21 = func2( (0)h21 , (1)m21 )``

    .. note::
     Here func2 is ``self.W_h``.

     `(1)m21` refers to the messages coming to that bond 2->1 in that 2nd iteration.

    Messages coming to a bond in an iteration is **a sum of hidden states of bonds (from previous iteration) coming to this bond**.

    Example:
    ``(1)m21 = (0)h32 + (0)h42``

    |   2 <--- 3
    |   ^
    |   |
    |   4

    **Computing the messages**

    .. code-block:: python

                             B0      B1      B2      B3      B4      B5      B6      B7      B8
        f_ini_atoms_bonds = [(0)h12, (0)h21, (0)h23, (0)h32, (0)h24, (0)h42, (0)h15, (0)h51, h(-1)]


    .. note::
     h(-1) is an empty array of the same size as other hidden states of bond states.

    .. code-block:: python

                    B0      B1      B2      B3      B4      B5      B6      B7       B8
        mapping = [ [-1,B7] [B3,B5] [B0,B5] [-1,-1] [B0,B3] [-1,-1] [B1,-1] [-1,-1]  [-1,-1] ]

    Later, the encoder will map the concatenated features from the ``f_ini_atoms_bonds``
    to ``mapping`` in each iteration upto Tth iteration.

    Next the encoder will sum-up the concat features within same bond index.

    .. code-block:: python

                        (1)m12           (1)m21           (1)m23              (1)m32          (1)m24           (1)m42           (1)m15          (1)m51            m(-1)
        message = [ [h(-1) + (0)h51] [(0)h32 + (0)h42] [(0)h12 + (0)h42] [h(-1) + h(-1)] [(0)h12 + (0)h32] [h(-1) + h(-1)] [(0)h21 + h(-1)] [h(-1) + h(-1)]  [h(-1) + h(-1)] ]

    Hence, this is how encoder can get messages for message-passing steps.

    **Get new atom hidden states and readout phase**

    Hence now for h1:

    .. code-block:: python

        h1 = func3(
                    concat(
                         a1,
                         [
                            func2( (0)h21 , (0)h32 + (0)h42 ) +
                            func2( (0)h51 , 0               )
                         ]
                        )
                 )

    Similarly, h2, h3, h4 and h5 are calculated.

    Next, all atom hidden states are concatenated to make a feature vector of the molecule:

    ``mol_encodings = [[h1, h2, h3, h4, h5]]``

    **Concatenate the global features**

    Let,
    ``global_features = [[gf1, gf2, gf3]]``
    This array contains molecule level features. In case of this example, it contains 3 global features.

    Hence after concatenation,

    ``mol_encodings = [[h1, h2, h3, h4, h5, gf1, gf2, gf3]]``
    (Final output of the encoder)

    References
    ----------
    .. [1] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from rdkit import Chem
    >>> import torch
    >>> import deepchem as dc
    >>> input_smile = "CC"
    >>> feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
    >>> graph = feat.featurize(input_smile)
    >>> from deepchem.models.torch_models.dmpnn import _MapperDMPNN
    >>> mapper = _MapperDMPNN(graph[0])
    >>> atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values
    >>> atom_features = torch.from_numpy(atom_features).float()
    >>> f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
    >>> atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
    >>> mapping = torch.from_numpy(mapping)
    >>> global_features = torch.from_numpy(global_features).float()
    >>> molecules_unbatch_key = len(atom_features)
    >>> layer = DMPNNEncoderLayer(d_hidden=2)
    >>> output = layer(atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features, molecules_unbatch_key)
    """

    def __init__(self,
                 use_default_fdim: bool = True,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 d_hidden: int = 300,
                 depth: int = 3,
                 bias: bool = False,
                 activation: str = 'relu',
                 dropout_p: float = 0.0,
                 aggregation: str = 'mean',
                 aggregation_norm: Union[int, float] = 100):
        """Initialize a DMPNNEncoderLayer layer.

        Parameters
        ----------
        use_default_fdim: bool
            If ``True``, ``self.atom_fdim`` and ``self.bond_fdim`` are initialized using values from the GraphConvConstants class. If ``False``, ``self.atom_fdim`` and ``self.bond_fdim`` are initialized from the values provided.
        atom_fdim: int
            Dimension of atom feature vector.
        bond_fdim: int
            Dimension of bond feature vector.
        d_hidden: int
            Size of hidden layer in the encoder layer.
        depth: int
            No of message passing steps.
        bias: bool
            If ``True``, dense layers will use bias vectors.
        activation: str
            Activation function to be used in the encoder layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
        dropout_p: float
            Dropout probability for the encoder layer.
        aggregation: str
            Aggregation type to be used in the encoder layer.
            Can choose between 'mean', 'sum', and 'norm'.
        aggregation_norm: Union[int, float]
            Value required if `aggregation` type is 'norm'.
        """
        super(DMPNNEncoderLayer, self).__init__()

        if use_default_fdim:
            from deepchem.feat.molecule_featurizers.dmpnn_featurizer import GraphConvConstants
            self.atom_fdim: int = GraphConvConstants.ATOM_FDIM
            self.concat_fdim: int = GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM
        else:
            self.atom_fdim = atom_fdim
            self.concat_fdim = atom_fdim + bond_fdim

        self.depth: int = depth
        self.aggregation: str = aggregation
        self.aggregation_norm: Union[int, float] = aggregation_norm

        if activation == 'relu':
            self.activation: nn.modules.activation.Module = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        self.dropout: nn.modules.dropout.Module = nn.Dropout(dropout_p)

        # Input
        self.W_i: nn.Linear = nn.Linear(self.concat_fdim, d_hidden, bias=bias)

        # Shared weight matrix across depths (default):
        # For messages hidden states
        self.W_h: nn.Linear = nn.Linear(d_hidden, d_hidden, bias=bias)

        # For atom hidden states
        self.W_o: nn.Linear = nn.Linear(self.atom_fdim + d_hidden, d_hidden)

    def _get_updated_atoms_hidden_state(
            self, atom_features: torch.Tensor, h_message: torch.Tensor,
            atom_to_incoming_bonds: torch.Tensor) -> torch.Tensor:
        """
        Method to compute atom hidden states.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        h_message: torch.Tensor
            Tensor containing hidden states of messages.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.

        Returns
        -------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        """
        messages_to_atoms: torch.Tensor = h_message[atom_to_incoming_bonds].sum(
            1)  # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self.W_o(
            torch.cat((atom_features, messages_to_atoms),
                      1))  # num_atoms x hidden_size
        atoms_hidden_states = self.activation(
            atoms_hidden_states)  # num_atoms x hidden_size
        atoms_hidden_states = self.dropout(
            atoms_hidden_states)  # num_atoms x hidden_size
        return atoms_hidden_states  # num_atoms x hidden_size

    def _readout(self, atoms_hidden_states: torch.Tensor,
                 molecules_unbatch_key: List) -> torch.Tensor:
        """
        Method to execute the readout phase. (compute molecules encodings from atom hidden states)

        Parameters
        ----------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        molecule_hidden_state: torch.Tensor
            Tensor containing molecule encodings.
        """
        mol_vecs: List = []
        atoms_hidden_states_split: Sequence[Tensor] = torch.split(
            atoms_hidden_states, molecules_unbatch_key)
        mol_vec: torch.Tensor
        for mol_vec in atoms_hidden_states_split:
            if self.aggregation == 'mean':
                mol_vec = mol_vec.sum(dim=0) / len(mol_vec)
            elif self.aggregation == 'sum':
                mol_vec = mol_vec.sum(dim=0)
            elif self.aggregation == 'norm':
                mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
            else:
                raise Exception("Invalid aggregation")
            mol_vecs.append(mol_vec)

        molecule_hidden_state: torch.Tensor = torch.stack(mol_vecs, dim=0)
        return molecule_hidden_state  # num_molecules x hidden_size

    def forward(self, atom_features: torch.Tensor,
                f_ini_atoms_bonds: torch.Tensor,
                atom_to_incoming_bonds: torch.Tensor, mapping: torch.Tensor,
                global_features: torch.Tensor,
                molecules_unbatch_key: List) -> torch.Tensor:
        """
        Output computation for the DMPNNEncoderLayer.

        Steps:

        - Get original bond hidden states from concatenation of initial atom and bond features. (``input``)
        - Get initial messages hidden states. (``message``)
        - Execute message passing step for ``self.depth - 1`` iterations.
        - Get atom hidden states using atom features and message hidden states.
        - Get molecule encodings.
        - Concatenate global molecular features and molecule encodings.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        f_ini_atoms_bonds: torch.Tensor
            Tensor containing concatenated feature vector which contains concatenation of initial atom and bond features.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.
        mapping: torch.Tensor
            Tensor containing the mapping that maps bond index to 'array of indices of the bonds'
            incoming at the initial atom of the bond (excluding the reverse bonds).
        global_features: torch.Tensor
            Tensor containing molecule features.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        output: torch.Tensor
            Tensor containing the encodings of the molecules.
        """
        input: torch.Tensor = self.W_i(
            f_ini_atoms_bonds)  # num_bonds x hidden_size
        message: torch.Tensor = self.activation(
            input)  # num_bonds x hidden_size

        for _ in range(1, self.depth):
            message = message[mapping].sum(1)  # num_bonds x hidden_size
            h_message: torch.Tensor = input + self.W_h(
                message)  # num_bonds x hidden_size
            h_message = self.activation(h_message)  # num_bonds x hidden_size
            h_message = self.dropout(h_message)  # num_bonds x hidden_size

        # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self._get_updated_atoms_hidden_state(
            atom_features, h_message, atom_to_incoming_bonds)

        # num_molecules x hidden_size
        output: torch.Tensor = self._readout(atoms_hidden_states,
                                             molecules_unbatch_key)

        # concat global features
        if global_features.size()[0] != 0:
            if len(global_features.shape) == 1:
                global_features = global_features.view(len(output), -1)
            output = torch.cat([output, global_features], dim=1)

        return output  # num_molecules x hidden_size


class InteratomicL2Distances(nn.Module):
    """Compute (squared) L2 Distances between atoms given neighbors.

    This class is the pytorch implementation of the original InteratomicL2Distances layer implemented in Keras.
    Pairwise distance (L2) is calculated between input atoms, given the number of neighbors to consider, along with the number of descriptors for every atom.

    Examples
    --------
    >>> atoms = 5
    >>> neighbors = 2
    >>> coords = np.random.rand(atoms, 3)
    >>> neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
    >>> layer = InteratomicL2Distances(atoms, neighbors, 3)
    >>> result = np.array(layer([coords, neighbor_list]))
    >>> result.shape
    (5, 2)

    """

    def __init__(self, N_atoms: int, M_nbrs: int, ndim: int, **kwargs):
        """Constructor for this layer.

        Parameters
        ----------
        N_atoms: int
            Number of atoms in the system total.
        M_nbrs: int
            Number of neighbors to consider when computing distances.
        n_dim:  int
            Number of descriptors for each atom.
        """
        super(InteratomicL2Distances, self).__init__(**kwargs)
        self.N_atoms = N_atoms
        self.M_nbrs = M_nbrs
        self.ndim = ndim

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(N_atoms={self.N_atoms}, M_nbrs={self.M_nbrs}, ndim={self.ndim})'
        )

    def forward(
        self, inputs: List[Union[torch.Tensor,
                                 List[Union[int, float]]]]) -> torch.Tensor:
        """Invokes this layer.

        Parameters
        ----------
        inputs: list
            Should be of form `inputs=[coords, nbr_list]` where `coords` is a
            tensor of shape `(None, N, 3)` and `nbr_list` is a list.

        Returns
        -------
        Tensor of shape `(N_atoms, M_nbrs)` with interatomic distances.
        """
        if len(inputs) != 2:
            raise ValueError("InteratomicDistances requires coords,nbr_list")
        coords, nbr_list = (torch.tensor(inputs[0]), torch.tensor(inputs[1]))
        N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
        # Shape (N_atoms, M_nbrs, ndim)
        nbr_coords = coords[nbr_list.long()]
        # Shape (N_atoms, M_nbrs, ndim)
        tiled_coords = torch.tile(torch.reshape(coords, (N_atoms, 1, ndim)),
                                  (1, M_nbrs, 1))
        # Shape (N_atoms, M_nbrs)
        return torch.sum((tiled_coords - nbr_coords)**2, dim=2)


class RealNVPLayer(nn.Module):
    """Real NVP Transformation Layer

    This class class is a constructor transformation layer used on a
    NormalizingFLow model. The Real Non-Preserving-Volumen (Real NVP) is a type
    of normalizing flow layer which gives advantages over this mainly because an
    ease to compute the inverse pass [realnvp1]_, this is to learn a target
    distribution.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> from torch.distributions import MultivariateNormal
    >>> from deepchem.models.torch_models.layers import RealNVPLayer
    >>> dim = 2
    >>> samples = 96
    >>> data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> tensor = data.sample(torch.Size((samples, dim)))

    >>> layers = 4
    >>> hidden_size = 16
    >>> masks = F.one_hot(torch.tensor([i % 2 for i in range(layers)])).float()
    >>> layers = nn.ModuleList([RealNVPLayer(mask, hidden_size) for mask in masks])

    >>> for layer in layers:
    ...   _, inverse_log_det_jacobian = layer.inverse(tensor)
    ...   inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
    >>> len(inverse_log_det_jacobian)
    96

    References
    ----------
    .. [realnvp1] Stimper, V., Schölkopf, B., & Hernández-Lobato, J. M. (2021). Resampling Base
    Distributions of Normalizing Flows. (2017). Retrieved from http://arxiv.org/abs/2110.15828
    """

    def __init__(self, mask: torch.Tensor, hidden_size: int) -> None:
        """
        Parameters
        -----------
        mask: torch.Tensor
            Tensor with zeros and ones and its size depende on the number of layers
            and dimenssions the user request.
        hidden_size: int
            The size of the outputs and inputs used on the internal nodes of the
            transformation layer.

        """
        super(RealNVPLayer, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.dim = len(mask)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim))

        self.scale = nn.Parameter(torch.Tensor(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim))

    def forward(self, x: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        This particular transformation is represented by the following function:
        y = x + (1 - x) * exp( s(x)) + t(x), where t and s needs an activation
        function. This class also returns the logarithm of the jacobians
        determinant which is useful when invert a transformation and compute
        the probability of the transformation.

        Parameters
        ----------
        x : Sequence
            Tensor sample with the initial distribution data which will pass into
            the normalizing algorithm

        Returns
        -------
        y : torch.Tensor
            Transformed tensor according to Real NVP layer with the shape of 'x'.
        log_det_jacobian : torch.Tensor
            Tensor which represents the info about the deviation of the initial
            and target distribution.

        """
        x_mask = x * self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)

        log_det_jacobian = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jacobian

    def inverse(self, y: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Inverse pass

        This class performs the inverse of the previous method (formward).
        Also, this metehod returns the logarithm of the jacobians determinant
        which is useful to compute the learneable features of target distribution.

        Parameters
        ----------
        y : Sequence
            Tensor sample with transformed distribution data which will be used in
            the normalizing algorithm inverse pass.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor according to Real NVP layer with the shape of 'y'.
        inverse_log_det_jacobian : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.

        """
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)

        inverse_log_det_jacobian = ((1 - self.mask) * -s).sum(-1)

        return x, inverse_log_det_jacobian


class NeighborList(nn.Module):
    """Computes a neighbor-list in PyTorch.

    Neighbor-lists (also called Verlet Lists) are a tool for grouping
    atoms which are close to each other spatially. This layer computes a
    Neighbor List from a provided tensor of atomic coordinates. You can
    think of this as a general "k-means" layer, but optimized for the
    case `k==3`.

    Examples
    --------
    >>> N_atoms = 5
    >>> start = 0
    >>> stop = 12
    >>> nbr_cutoff = 3
    >>> ndim = 3
    >>> M_nbrs = 2
    >>> coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
    >>> coords = torch.as_tensor(coords, dtype=torch.float)
    >>> layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
    ...                      stop)
    >>> result = layer(coords)
    >>> result.shape
    torch.Size([5, 2])

    TODO(rbharath): Make this layer support batching.
    """

    def __init__(self, N_atoms: int, M_nbrs: int, ndim: int,
                 nbr_cutoff: Union[int,
                                   float], start: int, stop: int, **kwargs):
        """
        Parameters
        ----------
        N_atoms: int
            Maximum number of atoms this layer will neighbor-list.
        M_nbrs: int
            Maximum number of spatial neighbors possible for atom.
        ndim: int
            Dimensionality of space atoms live in. (Typically 3D, but sometimes will
            want to use higher dimensional descriptors for atoms).
        nbr_cutoff: int or float
            Length in Angstroms (?) at which atom boxes are gridded.
        start: int
            Start of range for the box in which the locations of all grid points will be calculated in `self.get_cells()`.
        stop: int
            End of range for the box in which the locations of all grid points will be calculated in `self.get_cells()`.
        """
        super(NeighborList, self).__init__(**kwargs)
        self.N_atoms = N_atoms
        self.M_nbrs = M_nbrs
        self.ndim = ndim
        # Number of grid cells
        n_cells = int(((stop - start) / nbr_cutoff)**ndim)
        self.n_cells = n_cells
        self.nbr_cutoff = nbr_cutoff
        self.start = start
        self.stop = stop

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(N_atoms={self.N_atoms}, M_nbrs={self.M_nbrs}, ndim={self.ndim}, n_cells={self.n_cells}, nbr_cutoff={self.nbr_cutoff}, start={self.start}, stop={self.stop})'
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Invokes this layer.

        Parameters
        ----------
        inputs: torch.Tensor
            Shape (num_atoms, ndim)

        Returns
        -------
        neighbor_list: torch.Tensor
            Shape `(N_atoms, M_nbrs)`
        """
        if isinstance(inputs, SequenceCollection):
            if len(inputs) != 1:
                raise ValueError("NeighborList can only have one input")
            inputs = inputs[0]
        if len(inputs.shape) != 2:
            # TODO(rbharath): Support batching
            raise ValueError("Parent tensor must be (num_atoms, ndum)")
        return self.compute_nbr_list(inputs)

    def compute_nbr_list(self, coords: torch.Tensor) -> torch.Tensor:
        """Get closest neighbors for atoms.

        Needs to handle padding for atoms with no neighbors.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)

        Returns
        -------
        nbr_list: torch.Tensor
            Shape (N_atoms, M_nbrs) of atom indices
        """
        # Shape (n_cells, ndim)
        cells = self.get_cells()

        # List of length N_atoms, each element of different length uniques_i
        nbrs = self.get_atoms_in_nbrs(coords, cells)
        padding = torch.full((self.M_nbrs,), -1)
        padded_nbrs = [
            torch.concat([unique_nbrs, padding], 0) for unique_nbrs in nbrs
        ]

        # List of length N_atoms, each element of different length uniques_i
        # List of length N_atoms, each a tensor of shape
        # (uniques_i, ndim)
        nbr_coords = [
            torch.index_select(coords, 0, atom_nbrs) for atom_nbrs in nbrs
        ]

        # Add phantom atoms that exist far outside the box
        coord_padding = torch.full((self.M_nbrs, self.ndim),
                                   2 * self.stop).to(torch.float)
        padded_nbr_coords = [
            torch.cat([nbr_coord, coord_padding], 0) for nbr_coord in nbr_coords
        ]

        # List of length N_atoms, each of shape (1, ndim)
        atom_coords = torch.tensor_split(coords, self.N_atoms)

        # TODO(rbharath): How does distance need to be modified here to
        # account for periodic boundary conditions?
        # List of length N_atoms each of shape (M_nbrs)
        padded_dists = [
            torch.sum((atom_coord - padded_nbr_coord)**2, dim=-1)
            for (atom_coord,
                 padded_nbr_coord) in zip(atom_coords, padded_nbr_coords)
        ]

        padded_closest_nbrs = [
            torch.topk(padded_dist, k=self.M_nbrs, largest=False)[1]
            for padded_dist in padded_dists
        ]

        # N_atoms elts of size (M_nbrs,) each
        padded_neighbor_list = [
            torch.gather(padded_atom_nbrs, 0, padded_closest_nbr)
            for (padded_atom_nbrs,
                 padded_closest_nbr) in zip(padded_nbrs, padded_closest_nbrs)
        ]

        neighbor_list = torch.stack(padded_neighbor_list)

        return neighbor_list

    def get_atoms_in_nbrs(self, coords: torch.Tensor,
                          cells: torch.Tensor) -> List[torch.Tensor]:
        """Get the atoms in neighboring cells for each cells.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)
        cells: torch.Tensor

        Returns
        -------
        atoms_in_nbrs: List[torch.Tensor]
            (N_atoms, n_nbr_cells, M_nbrs)
        """
        # Shape (N_atoms, 1)
        cells_for_atoms = self.get_cells_for_atoms(coords, cells)

        # Find M_nbrs atoms closest to each cell
        # Shape (n_cells, M_nbrs)
        closest_atoms = self.get_closest_atoms(coords, cells)

        # Associate each cell with its neighbor cells. Assumes periodic boundary
        # conditions, so does wrapround. O(constant)
        # Shape (n_cells, n_nbr_cells)
        neighbor_cells = self.get_neighbor_cells(cells)

        # Shape (N_atoms, n_nbr_cells)
        neighbor_cells = torch.squeeze(
            torch.index_select(neighbor_cells, 0,
                               torch.squeeze(cells_for_atoms)))

        # Shape (N_atoms, n_nbr_cells, M_nbrs)
        atoms_in_nbrs = torch.index_select(closest_atoms, 0,
                                           neighbor_cells.flatten())

        # Shape (N_atoms, n_nbr_cells*M_nbrs)
        atoms_in_nbrs = torch.reshape(atoms_in_nbrs, [self.N_atoms, -1])

        # List of length N_atoms, each element length uniques_i
        nbrs_per_atom = torch.split(atoms_in_nbrs, self.N_atoms)

        uniques = [
            torch.unique(atom_nbrs, sorted=False)
            for atom_nbrs in nbrs_per_atom[0]
        ]

        # TODO(rbharath): FRAGILE! Uses fact that identity seems to be the first
        # element removed to remove self from list of neighbors. Need to verify
        # this holds more broadly or come up with robust alternative.
        uniques = [unique[1:] for unique in uniques]

        return uniques

    def get_closest_atoms(self, coords: torch.Tensor,
                          cells: torch.Tensor) -> torch.Tensor:
        """For each cell, find M_nbrs closest atoms.

        Let N_atoms be the number of atoms.

        Parameters
        ----------
        coords: torch.Tensor
            (N_atoms, ndim) shape.
        cells: torch.Tensor
            (n_cells, ndim) shape.

        Returns
        -------
        closest_inds: torch.Tensor
            Of shape (n_cells, M_nbrs)
        """
        N_atoms, n_cells, ndim, M_nbrs = (self.N_atoms, self.n_cells, self.ndim,
                                          self.M_nbrs)
        # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
        tiled_cells = torch.reshape(torch.tile(cells, (1, N_atoms)),
                                    (N_atoms * n_cells, ndim))

        # Shape (N_atoms*n_cells, ndim) after tile
        tiled_coords = torch.tile(coords, (n_cells, 1))

        # Shape (N_atoms*n_cells)
        coords_vec = torch.sum((tiled_coords - tiled_cells)**2, dim=-1)
        # Shape (n_cells, N_atoms)
        coords_norm = torch.reshape(coords_vec, (n_cells, N_atoms))

        # Find k atoms closest to this cell.
        # Tensor of shape (n_cells, M_nbrs)
        closest_inds = torch.topk(coords_norm, k=M_nbrs, largest=False)[1]

        return closest_inds

    def get_cells_for_atoms(self, coords: torch.Tensor,
                            cells: torch.Tensor) -> torch.Tensor:
        """Compute the cells each atom belongs to.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)
        cells: torch.Tensor
            (n_cells, ndim) shape.
        Returns
        -------
        cells_for_atoms: torch.Tensor
            Shape (N_atoms, 1)
        """
        N_atoms, n_cells, ndim = self.N_atoms, self.n_cells, self.ndim
        n_cells = int(n_cells)
        # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
        tiled_cells = torch.tile(cells, (N_atoms, 1))

        # Shape (N_atoms*n_cells, 1) after tile
        tiled_coords = torch.reshape(torch.tile(coords, (1, n_cells)),
                                     (n_cells * N_atoms, ndim))
        coords_vec = torch.sum((tiled_coords - tiled_cells)**2, dim=-1)
        coords_norm = torch.reshape(coords_vec, (N_atoms, n_cells))

        closest_inds = torch.topk(coords_norm, k=1, largest=False)[1]

        return closest_inds

    def _get_num_nbrs(self) -> int:
        """Get number of neighbors in current dimensionality space."""
        return 3**self.ndim

    def get_neighbor_cells(self, cells: torch.Tensor) -> torch.Tensor:
        """Compute neighbors of cells in grid.

        # TODO(rbharath): Do we need to handle periodic boundary conditions
        properly here?
        # TODO(rbharath): This doesn't handle boundaries well. We hard-code
        # looking for n_nbr_cells neighbors, which isn't right for boundary cells in
        # the cube.

        Parameters
        ----------
        cells: torch.Tensor
            (n_cells, ndim) shape.
        Returns
        -------
        nbr_cells: torch.Tensor
            (n_cells, n_nbr_cells)
        """
        ndim, n_cells = self.ndim, self.n_cells
        n_nbr_cells = self._get_num_nbrs()
        # Tile cells to form arrays of size (n_cells*n_cells, ndim)
        # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
        # Tile (a, a, a, b, b, b, etc.)
        tiled_centers = torch.reshape(torch.tile(cells, (1, n_cells)),
                                      (n_cells * n_cells, ndim))
        # Tile (a, b, c, a, b, c, ...)
        tiled_cells = torch.tile(cells, (n_cells, 1))

        coords_vec = torch.sum((tiled_centers - tiled_cells)**2, dim=-1)
        coords_norm = torch.reshape(coords_vec, (n_cells, n_cells))
        closest_inds = torch.topk(coords_norm, k=n_nbr_cells, largest=False)[1]

        return closest_inds

    def get_cells(self) -> torch.Tensor:
        """Returns the locations of all grid points in box.

        Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
        Then would return a list of length 20^3 whose entries would be
        [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

        Returns
        -------
        cells: torch.Tensor
          (n_cells, ndim) shape.
        """
        start, stop, nbr_cutoff = self.start, self.stop, self.nbr_cutoff
        mesh_args = [
            torch.arange(start, stop, nbr_cutoff) for _ in range(self.ndim)
        ]
        return torch.reshape(
            torch.permute(
                torch.stack(torch.meshgrid(*mesh_args, indexing='xy')),
                tuple(range(self.ndim, -1, -1))),
            (self.n_cells, self.ndim)).to(torch.float)


class LSTMStep(nn.Module):
    """Layer that performs a single step LSTM update.

    This is the Torch equivalent of the original implementation using Keras.
    """

    def __init__(self,
                 output_dim,
                 input_dim,
                 init_fn='xavier_uniform_',
                 inner_init_fn='orthogonal_',
                 activation_fn='tanh',
                 inner_activation_fn='hardsigmoid',
                 **kwargs):
        """
        Parameters
        ----------
        output_dim: int
            Dimensionality of output vectors.
        input_dim: int
            Dimensionality of input vectors.
        init_fn: str
            PyTorch initialization to use for W.
        inner_init_fn: str
            PyTorch initialization to use for U.
        activation_fn: str
            PyTorch activation to use for output.
        inner_activation_fn: str
            PyTorch activation to use for inner steps.
        """

        super(LSTMStep, self).__init__(**kwargs)
        self.init = init_fn
        self.inner_init = inner_init_fn
        self.output_dim = output_dim
        # No other forget biases supported right now.
        self.activation = activation_fn
        self.inner_activation = inner_activation_fn
        self.activation_fn = get_activation(activation_fn)
        self.inner_activation_fn = get_activation(inner_activation_fn)
        self.input_dim = input_dim
        self.build()

    def get_config(self):
        config = super(LSTMStep, self).get_config()
        config['output_dim'] = self.output_dim
        config['input_dim'] = self.input_dim
        config['init_fn'] = self.init
        config['inner_init_fn'] = self.inner_init
        config['activation_fn'] = self.activation
        config['inner_activation_fn'] = self.inner_activation
        return config

    def get_initial_states(self, input_shape):
        return [torch.zeros(input_shape), torch.zeros(input_shape)]

    def build(self):
        """Constructs learnable weights for this layer."""
        init = getattr(initializers, self.init)
        inner_init = getattr(initializers, self.inner_init)
        self.W = init(torch.empty(self.input_dim, 4 * self.output_dim))
        self.U = inner_init(torch.empty(self.output_dim, 4 * self.output_dim))

        self.b = torch.tensor(np.hstack(
            (np.zeros(self.output_dim), np.ones(self.output_dim),
             np.zeros(self.output_dim), np.zeros(self.output_dim))),
                              dtype=torch.float32)

    def forward(self, inputs):
        """Execute this layer on input tensors.

        Parameters
        ----------
        inputs: list
            List of three tensors (x, h_tm1, c_tm1). h_tm1 means "h, t-1".

        Returns
        -------
        list
            Returns h, [h, c]
        """
        x, h_tm1, c_tm1 = inputs
        x, h_tm1, c_tm1 = torch.tensor(x), torch.tensor(h_tm1), torch.tensor(
            c_tm1)

        z = torch.matmul(x, self.W) + torch.matmul(h_tm1, self.U) + self.b

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim:2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim:3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation_fn(z0)
        f = self.inner_activation_fn(z1)
        c = f * c_tm1 + i * self.activation_fn(z2)
        o = self.inner_activation_fn(z3)

        h = o * self.activation_fn(c)
        return h, [h, c]


class AtomicConvolution(nn.Module):
    """Implements the Atomic Convolutional transform, introduced in

    Gomes, Joseph, et al. "Atomic convolutional networks for predicting
    protein-ligand binding affinity." arXiv preprint arXiv:1703.10603
    (2017).

    At a high level, this transform performs a graph convolution
    on the nearest neighbors graph in 3D space.

    Examples
    --------
    >>> batch_size = 4
    >>> max_atoms = 5
    >>> max_neighbors = 2
    >>> dimensions = 3
    >>> radial_params = torch.tensor([[5.0, 2.0, 0.5], [10.0, 2.0, 0.5],
    ...                               [5.0, 1.0, 0.2]])
    >>> input1 = np.random.rand(batch_size, max_atoms, dimensions).astype(np.float32)
    >>> input2 = np.random.randint(max_atoms,
    ...                            size=(batch_size, max_atoms, max_neighbors))
    >>> input3 = np.random.randint(1, 10, size=(batch_size, max_atoms, max_neighbors))
    >>> layer = AtomicConvolution(radial_params=radial_params)
    >>> result = layer([input1, input2, input3])
    >>> result.shape
    torch.Size([4, 5, 3])
    """

    def __init__(self,
                 atom_types: Optional[Union[ArrayLike, torch.Tensor]] = None,
                 radial_params: Union[ArrayLike, torch.Tensor] = list(),
                 box_size: Optional[Union[ArrayLike, torch.Tensor]] = None,
                 **kwargs):
        """Initialize this layer.

        Parameters
        ----------
        atom_types : Union[ArrayLike, torch.Tensor], optional
            List of atom types.
        radial_params : Union[ArrayLike, torch.Tensor], optional
            List of radial params.
        box_size : Union[ArrayLike, torch.Tensor], optional
            Length must be equal to the number of features.
        """

        super(AtomicConvolution, self).__init__(**kwargs)
        self.atom_types = atom_types
        self.radial_params = radial_params

        if box_size is None or isinstance(box_size, torch.Tensor):
            self.box_size = box_size
        else:
            self.box_size = torch.tensor(box_size)

        vars = []
        for i in range(3):
            val = np.array([p[i] for p in self.radial_params]).reshape(
                (-1, 1, 1, 1))
            vars.append(torch.tensor(val, dtype=torch.float))
        self.rc = nn.Parameter(vars[0])
        self.rs = nn.Parameter(vars[1])
        self.re = nn.Parameter(vars[2])

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(atom_types={self.atom_types}, radial_params={self.radial_params}, box_size={self.box_size}, rc={self.rc}, rs={self.rs}, re={self.re})'
        )

    def forward(
            self, inputs: Sequence[Union[ArrayLike,
                                         torch.Tensor]]) -> torch.Tensor:
        """Invoke this layer.

        B, N, M, d, l = batch_size, max_num_atoms, max_num_neighbors, num_features, len(radial_params) * len(atom_types)

        Parameters
        ----------
        inputs: Sequence[Union[ArrayLike, torch.Tensor]]
            First input are the coordinates/features, of shape (B, N, d)
            Second input is the neighbor list, of shape (B, N, M)
            Third input are the atomic numbers of neighbor atoms, of shape (B, N, M)

        Returns
        -------
        torch.Tensor of shape (B, N, l)
            Output of atomic convolution layer.

        Raises
        ------
        ValueError
            When the length of `inputs` is not equal to 3.
        """
        if len(inputs) != 3:
            raise ValueError(
                f"`inputs` has to be of length 3, got: {len(inputs)}")

        X = torch.tensor(inputs[0])
        Nbrs = torch.tensor(inputs[1], dtype=torch.int64)
        Nbrs_Z = torch.tensor(inputs[2])

        B, N, d = X.shape
        M = Nbrs.shape[-1]

        D = self.distance_tensor(X, Nbrs, self.box_size, B, N, M, d)
        R = self.distance_matrix(D)
        R = torch.unsqueeze(R, 0)
        rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)

        if not self.atom_types:
            cond = torch.not_equal(Nbrs_Z, 0).to(torch.float).reshape(
                (1, -1, N, M))
            layer = torch.sum(cond * rsf, 3)
        else:
            # Sum the pairwise-interactions between atoms that are of `atom_type` and its neighbors for each atom type in `atom_types`.
            symmetries = []
            for atom_type in self.atom_types:
                cond = torch.eq(Nbrs_Z, atom_type).to(torch.float).reshape(
                    (1, -1, N, M))
                symmetries.append(torch.sum(cond * rsf, 3))
            layer = torch.concat(symmetries, 0)

        layer = torch.permute(layer, [1, 2, 0])
        var, mean = torch.var_mean(layer, [0, 2])
        var, mean = var.detach(), mean.detach()

        return F.batch_norm(layer, mean, var)

    def distance_tensor(self, X: torch.Tensor, Nbrs: torch.Tensor,
                        box_size: Union[torch.Tensor, None], B: int, N: int,
                        M: int, d: int) -> torch.Tensor:
        """Calculate distance tensor for a batch of molecules.

        B, N, M, d = batch_size, max_num_atoms, max_num_neighbors, num_features

        Parameters
        ----------
        X : torch.Tensor of shape (B, N, d)
            Coordinates/features.
        Nbrs : torch.Tensor of shape (B, N, M)
            Neighbor list.
        box_size : torch.Tensor
            Length must be equal to `d`.
        B : int
            Batch size
        N : int
            Maximum number of atoms
        M : int
            Maximum number of neighbors
        d : int
            Number of features

        Returns
        -------
        torch.Tensor of shape (B, N, M, d)
            Coordinates/features distance tensor.

        Raises
        ------
        ValueError
            When the length of `box_size` is not equal to `d`.
        """
        if box_size is not None and len(box_size) != d:
            raise ValueError("Length of `box_size` must be equal to `d`")

        flat_neighbors = torch.reshape(Nbrs, (-1, N * M))
        neighbor_coords = torch.stack(
            [X[b, flat_neighbors[b]] for b in range(B)])
        neighbor_coords = torch.reshape(neighbor_coords, (-1, N, M, d))
        D = neighbor_coords - torch.unsqueeze(X, 2)
        if box_size is not None:
            box_size = torch.reshape(box_size, (1, 1, 1, d))
            D -= torch.round(D / box_size) * box_size

        return D

    def distance_matrix(self, D: torch.Tensor) -> torch.Tensor:
        """Calculate a distance matrix, given a distance tensor.

        B, N, M, d = batch_size, max_num_atoms, max_num_neighbors, num_features

        Parameters
        ----------
        D : torch.Tensor of shape (B, N, M, d)
            Distance tensor

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Distance matrix.
        """
        return torch.sqrt(torch.sum(torch.mul(D, D), 3))

    def gaussian_distance_matrix(self, R: torch.Tensor, rs: torch.Tensor,
                                 re: torch.Tensor) -> torch.Tensor:
        """Calculate a Gaussian distance matrix.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rs : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix mean.
        re : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix width.

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Gaussian distance matrix.
        """
        return torch.exp(-re * (R - rs)**2)

    def radial_cutoff(self, R: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Calculate a radial cut-off matrix.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rc : torch.Tensor of shape (l, 1, 1, 1)
            Interaction cutoff (in angstrom).

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Radial cutoff matrix.
        """
        T = 0.5 * (torch.cos(np.pi * R / rc) + 1)
        E = torch.zeros_like(T)
        cond = torch.less_equal(R, rc)
        FC = torch.where(cond, T, E)

        return FC

    def radial_symmetry_function(self, R: torch.Tensor, rc: torch.Tensor,
                                 rs: torch.Tensor,
                                 re: torch.Tensor) -> torch.Tensor:
        """Calculate a radial symmetry function.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rc : torch.Tensor of shape (l, 1, 1, 1)
            Interaction cutoff (in angstrom).
        rs : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix mean.
        re : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix width.
        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Pre-summation radial symmetry function.
        """
        K = self.gaussian_distance_matrix(R, rs, re)
        FC = self.radial_cutoff(R, rc)

        return torch.mul(K, FC)


class AtomicConv(nn.Module):
    """
    Implements an Atomic Convolution Model.

    The atomic convolutional networks function as a variant of
    graph convolutions. The difference is that the "graph" here is
    the nearest neighbors graph in 3D space [1]. The AtomicConvModule
    leverages these connections in 3D space to train models that
    learn to predict energetic states starting from the spatial
    geometry of the model.

    References
    ----------
    .. [1] Gomes, Joseph, et al. "Atomic convolutional networks for predicting protein-ligand binding affinity." arXiv preprint arXiv:1703.10603 (2017).

    Examples
    --------
    >>> n_tasks = 1
    >>> frag1_num_atoms = 70
    >>> frag2_num_atoms = 634
    >>> complex_num_atoms = 701
    >>> max_num_neighbors = 12
    >>> batch_size = 24
    >>> atom_types = [
    ...     6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.,
    ...     -1.
    ... ]
    >>> radial = [[
    ...     1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    ...     8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
    ... ], [0.0, 4.0, 8.0], [0.4]]
    >>> layer_sizes = [32, 32, 16]
    >>> acnn_model = AtomicConv(n_tasks=n_tasks,
    ... frag1_num_atoms=frag1_num_atoms,
    ... frag2_num_atoms=frag2_num_atoms,
    ... complex_num_atoms=complex_num_atoms,
    ... max_num_neighbors=max_num_neighbors,
    ... batch_size=batch_size,
    ... atom_types=atom_types,
    ... radial=radial,
    ... layer_sizes=layer_sizes)
    """

    def __init__(self,
                 n_tasks: int,
                 frag1_num_atoms: int = 70,
                 frag2_num_atoms: int = 634,
                 complex_num_atoms: int = 701,
                 max_num_neighbors: int = 12,
                 batch_size: int = 24,
                 atom_types: Sequence[float] = [
                     6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35.,
                     53., -1.
                 ],
                 radial: Sequence[Sequence[float]] = [[
                     1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                     7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
                 ], [0.0, 4.0, 8.0], [0.4]],
                 layer_sizes=[100],
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = ['relu'],
                 init: str = 'trunc_normal_',
                 **kwargs) -> None:
        """
        Parameters
        ----------
        n_tasks: int
            number of tasks
        frag1_num_atoms: int
            Number of atoms in first fragment
        frag2_num_atoms: int
            Number of atoms in sec
        max_num_neighbors: int
            Maximum number of neighbors possible for an atom. Recall neighbors
            are spatial neighbors.
        atom_types: list
            List of atoms recognized by model. Atoms are indicated by their
            nuclear numbers.
        radial: list
            Radial parameters used in the atomic convolution transformation.
        layer_sizes: list
            the size of each dense layer in the network.  The length of
            this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight
            initialization of each layer.  The length of this list should
            equal len(layer_sizes).  Alternatively, this may be a single
            value instead of a list, where the same value is used
            for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer.  The
            length of this list should equal len(layer_sizes).
            Alternatively, this may be a single value instead of a list, where the same value is used for every layer.
        dropouts: list or float
            the dropout probability to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively, this may be a single value instead of a list, where the same value is used for every layer.
        activation_fns: list or object
            the Tensorflow activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively, this may be a single value instead of a list, where the
            same value is used for every layer.
        """
        super(AtomicConv, self).__init__()
        self.complex_num_atoms = complex_num_atoms
        self.frag1_num_atoms = frag1_num_atoms
        self.frag2_num_atoms = frag2_num_atoms
        self.max_num_neighbors = max_num_neighbors
        self.batch_size = batch_size
        self.atom_types = atom_types
        self.init = init
        self.n_tasks = n_tasks

        self.rp = [x for x in itertools.product(*radial)]

        frag1_X = np.random.rand(self.batch_size, self.frag1_num_atoms,
                                 3).astype(np.float32)
        frag1_nbrs = np.random.randint(frag1_num_atoms,
                                       size=(batch_size, frag1_num_atoms,
                                             max_num_neighbors))
        frag1_nbrs_z = np.random.randint(1,
                                         10,
                                         size=(batch_size, frag1_num_atoms,
                                               max_num_neighbors))
        frag1_z = torch.tensor((frag1_num_atoms,))

        frag2_X = np.random.rand(self.batch_size, self.frag2_num_atoms,
                                 3).astype(np.float32)
        frag2_nbrs = np.random.randint(frag2_num_atoms,
                                       size=(batch_size, frag2_num_atoms,
                                             max_num_neighbors))
        frag2_nbrs_z = np.random.randint(1,
                                         10,
                                         size=(batch_size, frag2_num_atoms,
                                               max_num_neighbors))
        frag2_z = torch.tensor((frag2_num_atoms,))

        complex_X = np.random.rand(self.batch_size, self.complex_num_atoms,
                                   3).astype(np.float32)
        complex_nbrs = np.random.randint(complex_num_atoms,
                                         size=(batch_size, complex_num_atoms,
                                               max_num_neighbors))
        complex_nbrs_z = np.random.randint(1,
                                           10,
                                           size=(batch_size, complex_num_atoms,
                                                 max_num_neighbors))
        complex_z = torch.tensor((complex_num_atoms,))

        flattener = nn.Flatten()
        self._frag1_conv = AtomicConvolution(
            atom_types=self.atom_types, radial_params=self.rp,
            box_size=None)([frag1_X, frag1_nbrs, frag1_nbrs_z])
        flattened1 = nn.Flatten()(self._frag1_conv)

        self._frag2_conv = AtomicConvolution(
            atom_types=self.atom_types, radial_params=self.rp,
            box_size=None)([frag2_X, frag2_nbrs, frag2_nbrs_z])
        flattened2 = flattener(self._frag2_conv)

        self._complex_conv = AtomicConvolution(
            atom_types=self.atom_types, radial_params=self.rp,
            box_size=None)([complex_X, complex_nbrs, complex_nbrs_z])
        flattened3 = flattener(self._complex_conv)

        concat = torch.cat((flattened1, flattened2, flattened3), dim=1)

        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        self.activation_fns = [get_activation(f) for f in activation_fns]
        self.dropouts = dropouts

        self.prev_layer = concat
        prev_size = concat.size(1)
        next_activation = None

        # Define the layers
        self.layers = nn.ModuleList()
        for size, weight_stddev, bias_const, dropout, activation_fn in zip(
                layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
                activation_fns):
            layer = self.prev_layer
            if next_activation is not None:
                layer = next_activation(layer)
            linear = nn.Linear(prev_size, size)
            nn.init.trunc_normal_(linear.weight, std=weight_stddev)
            nn.init.constant_(linear.bias, bias_const)
            self.layers.append(linear)

            prev_size = size
            next_activation = activation_fn

        # Create the final layers
        self.neural_fingerprint = self.prev_layer
        self.output = nn.Sequential(nn.Linear(prev_size, n_tasks),
                                    nn.Linear(n_tasks, 1))

    def forward(self, inputs: OneOrMany[torch.Tensor]):
        """
        Parameters
        ----------
        inputs: torch.Tensor
            Input Tensor
        Returns
        -------
        torch.Tensor
            Output for each label.
        """

        flattener = nn.Flatten()
        frag1_conv = AtomicConvolution(atom_types=self.atom_types,
                                       radial_params=self.rp,
                                       box_size=None)(
                                           [inputs[0], inputs[1], inputs[2]])
        flattened1 = nn.Flatten()(frag1_conv)

        frag2_conv = AtomicConvolution(atom_types=self.atom_types,
                                       radial_params=self.rp,
                                       box_size=None)(
                                           [inputs[4], inputs[5], inputs[6]])
        flattened2 = flattener(frag2_conv)

        complex_conv = AtomicConvolution(atom_types=self.atom_types,
                                         radial_params=self.rp,
                                         box_size=None)(
                                             [inputs[8], inputs[9], inputs[10]])
        flattened3 = flattener(complex_conv)

        inputs_x = torch.cat((flattened1, flattened2, flattened3), dim=1)

        for layer, activation_fn, dropout in zip(self.layers,
                                                 self.activation_fns,
                                                 self.dropouts):
            x = layer(inputs_x)

            if dropout > 0:
                x = F.dropout(x, dropout)

            if activation_fn is not None:
                x = activation_fn(x)

        outputs = []
        output = self.output(x)
        outputs = [output]

        return outputs


class CombineMeanStd(nn.Module):
    """Generate Gaussian noise.

    This is the Torch equivalent of the original implementation using Keras.
    """

    def __init__(self,
                 training_only: bool = False,
                 noise_epsilon: float = 1.0,
                 **kwargs):
        """Create a CombineMeanStd layer.

        This layer should have two inputs with the same shape, and its
        output also has the same shape.  Each element of the output is a
        Gaussian distributed random number whose mean is the corresponding
        element of the first input, and whose standard deviation is the
        corresponding element of the second input.

        Parameters
        ----------
        training_only: bool, optional (default False).
            if True, noise is only generated during training.  During
            prediction, the output is simply equal to the first input (that
            is, the mean of the distribution used during training).
        noise_epsilon: float, optional (default 1.0).
            The noise is scaled by this factor
        """
        super(CombineMeanStd, self).__init__(**kwargs)
        self.training_only = training_only
        self.noise_epsilon = noise_epsilon

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(training_only={self.training_only}, noise_epsilon={self.noise_epsilon})'
        )

    def forward(self,
                inputs: List[torch.Tensor],
                training: bool = True) -> torch.Tensor:
        """Invoke this layer.

        Parameters
        ----------
        inputs: List
            First element are the means for the random generated numbers.
            Second element are the standard deviations for the random generated numbers.
        training: bool, optional (default True).
            Specifies whether to generate noise.
            Noise is only added when training.

        Returns
        -------
        Tensor of Gaussian distributed random numbers: torch.Tensor
            Same shape as the means and standard deviations from `inputs`.
        """
        if len(inputs) != 2:
            raise ValueError("Must have two in_layers")

        mean_parent, std_parent = inputs[0], inputs[1]
        noise_scale = torch.tensor(training or not self.training_only,
                                   dtype=torch.float,
                                   device=mean_parent.device)
        sample_noise = torch.normal(0.0,
                                    self.noise_epsilon,
                                    mean_parent.shape,
                                    device=mean_parent.device)
        return mean_parent + noise_scale * std_parent * sample_noise


class GatedRecurrentUnit(nn.Module):
    """ Submodule for Message Passing """

    def __init__(self, n_hidden=100, init='xavier_uniform_', **kwargs):
        super(GatedRecurrentUnit, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.init = init
        init = getattr(initializers, self.init)
        self.Wz = init(torch.empty(n_hidden, n_hidden))
        self.Wr = init(torch.empty(n_hidden, n_hidden))
        self.Wh = init(torch.empty(n_hidden, n_hidden))
        self.Uz = init(torch.empty(n_hidden, n_hidden))
        self.Ur = init(torch.empty(n_hidden, n_hidden))
        self.Uh = init(torch.empty(n_hidden, n_hidden))
        self.bz = torch.zeros((n_hidden,))
        self.br = torch.zeros((n_hidden,))
        self.bh = torch.zeros((n_hidden,))

    def forward(self, inputs):
        sigmoid = get_activation('sigmoid')
        tanh = get_activation('tanh')
        h_tm1, x = inputs
        z = sigmoid(
            torch.matmul(x, self.Wz) + torch.matmul(h_tm1, self.Uz) + self.bz)
        r = sigmoid(
            torch.matmul(x, self.Wr) + torch.matmul(h_tm1, self.Ur) + self.br)
        h = (1 - z) * tanh(
            torch.matmul(x, self.Wh) + torch.matmul(h_tm1 * r, self.Uh) +
            self.bh) + z * x
        return h


class WeightedLinearCombo(nn.Module):
    """Compute a weighted linear combination of input layers, where the weight variables are trained.

    Examples
    --------
    >>> input1 = np.random.rand(5, 10).astype(np.float32)
    >>> input2 = np.random.rand(5, 10).astype(np.float32)
    >>> layer = WeightedLinearCombo(len([input1, input2]))
    >>> result = layer([input1, input2])
    >>> result.shape
    torch.Size([5, 10])
    """

    def __init__(self, num_inputs: int, std: float = 0.3, **kwargs):
        """

        Parameters
        ----------
        num_inputs: int
            Number of inputs given to `self.forward()`
            This is used to initialize the correct amount of weight variables to be trained.
        std: float
            The standard deviation for the normal distribution that is used to initialize the trainable weights.
        """
        super(WeightedLinearCombo, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.std = std
        self.input_weights = nn.Parameter(torch.empty(self.num_inputs))
        nn.init.normal_(self.input_weights, std=std)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(num_inputs={self.num_inputs}, std={self.std}, input_weights={self.input_weights})'
        )

    def forward(
        self, inputs: Sequence[Union[ArrayLike,
                                     torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        Parameters
        ----------
        inputs: Sequence[Union[ArrayLike, torch.Tensor]]
            The initial input layers.
            The length must be the same as `self.num_inputs`.

        Returns
        -------
        out_tensor: torch.Tensor or None
            The tensor containing the weighted linear combination.
        """
        out_tensor = None
        for in_tensor, w in zip(inputs, self.input_weights):
            in_tensor = torch.FloatTensor(in_tensor)
            if out_tensor is None:
                out_tensor = w * in_tensor
            else:
                out_tensor += w * in_tensor
        return out_tensor


class SetGather(nn.Module):
    """set2set gather layer for graph-based model

    Models using this layer must set `pad_batches=True`

    Torch Equivalent of Keras SetGather layer

    Parameters
    ----------
    M: int
        Number of LSTM steps
    batch_size: int
        Number of samples in a batch(all batches must have same size)
    n_hidden: int, optional
        number of hidden units in the passing phase

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models import layers
    >>> total_n_atoms = 4
    >>> n_atom_feat = 4
    >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)
    >>> atom_split = np.array([0, 0, 1, 1], dtype=np.int32)
    >>> gather = layers.SetGather(2, 2, n_hidden=4)
    >>> output_molecules = gather([atom_feat, atom_split])
    >>> print(output_molecules.shape)
    torch.Size([2, 8])

    """

    def __init__(self,
                 M: int,
                 batch_size: int,
                 n_hidden: int = 100,
                 init='orthogonal',
                 **kwargs):

        super(SetGather, self).__init__(**kwargs)
        self.M = M
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.init = init

        self.U = nn.Parameter(
            torch.Tensor(2 * self.n_hidden, 4 * self.n_hidden).normal_(mean=0.0,
                                                                       std=0.1))
        self.b = nn.Parameter(
            torch.cat((torch.zeros(self.n_hidden), torch.ones(self.n_hidden),
                       torch.zeros(self.n_hidden), torch.zeros(self.n_hidden))))
        self.built = True

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(M={self.M}, batch_size={self.batch_size}, n_hidden={self.n_hidden}, init={self.init})'
        )

    def forward(self, inputs: List) -> torch.Tensor:
        """Perform M steps of set2set gather,

        Detailed descriptions in: https://arxiv.org/abs/1511.06391

        Parameters
        ----------
        inputs: List
            This contains two elements.
            atom_features: np.ndarray
            atom_split: np.ndarray

        Returns
        -------
        q_star: torch.Tensor
            Final state of the model after all M steps.

        """
        atom_features, atom_split = inputs
        c = torch.zeros((self.batch_size, self.n_hidden))
        h = torch.zeros((self.batch_size, self.n_hidden))

        for i in range(self.M):
            q_expanded = h[atom_split]
            e = (torch.from_numpy(atom_features) * q_expanded).sum(dim=-1)
            e_mols = self._dynamic_partition(e, atom_split, self.batch_size)

            # Add another value(~-Inf) to prevent error in softmax
            e_mols = [
                torch.cat([e_mol, torch.tensor([-1000.])], dim=0)
                for e_mol in e_mols
            ]
            a = torch.cat([
                torch.nn.functional.softmax(e_mol[:-1], dim=0)
                for e_mol in e_mols
            ],
                          dim=0)

            r = scatter(torch.reshape(a, [-1, 1]) * atom_features,
                        torch.from_numpy(atom_split).long(),
                        dim=0)
            # Model using this layer must set `pad_batches=True`
            q_star = torch.cat([h, r], dim=1)
            h, c = self._LSTMStep(q_star, c)
        return q_star

    def _LSTMStep(self,
                  h: torch.Tensor,
                  c: torch.Tensor,
                  x=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """This methord performs a single step of a Long Short-Term Memory (LSTM) cell,

        Parameters
        ----------
        h: torch.Tensor
            The hidden state of the LSTM cell.
        c: torch.Tensor
            The cell state of the LSTM cell.

        Returns
        -------
        h_out: torch.Tensor
            The updated hidden state of the LSTM cell.
        c_out: torch.Tensor
            The updated cell state of the LSTM cell.

        """
        # z = torch.mm(h, self.U) + self.b
        z = F.linear(h.float().detach(),
                     self.U.float().T.detach(), self.b.detach())
        i = torch.sigmoid(z[:, :self.n_hidden])
        f = torch.sigmoid(z[:, self.n_hidden:2 * self.n_hidden])
        o = torch.sigmoid(z[:, 2 * self.n_hidden:3 * self.n_hidden])
        z3 = z[:, 3 * self.n_hidden:]
        c_out = f * c + i * torch.tanh(z3)
        h_out = o * torch.tanh(c_out)
        return h_out, c_out

    def _dynamic_partition(self, input_tensor: torch.Tensor,
                           partition_tensor: np.ndarray,
                           num_partitions: int) -> List[torch.Tensor]:
        """Partitions `data` into `num_partitions` tensors using indices from `partitions`.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The tensor to be partitioned.
        partition_tensor: np.ndarray
            A 1-D tensor whose size is equal to the first dimension of `input_tensor`.
        num_partitions: int
            The number of partitions to output.

        Returns
        -------
        partitions: List[torch.Tensor]
            A list of `num_partitions` `Tensor` objects with the same type as `data`.

        """
        # create a boolean mask for each partition
        partition_masks = [partition_tensor == i for i in range(num_partitions)]

        # partition the input tensor using the masks
        partitions = [input_tensor[mask] for mask in partition_masks]

        return partitions


class DTNNEmbedding(nn.Module):
    """DTNNEmbedding layer for DTNN model.

    Assign initial atomic descriptors. [1]_

    This layer creates 'n' number of embeddings as initial atomic descriptors. According to the required weight initializer and periodic_table_length (Total number of unique atoms).

    References
    ----------
    [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    Examples
    --------
    >>> from deepchem.models.torch_models import layers
    >>> import torch
    >>> layer = layers.DTNNEmbedding(30, 30, 'xavier_uniform_')
    >>> output = layer(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    >>> output.shape
    torch.Size([10, 30])

    """

    def __init__(self,
                 n_embedding: int = 30,
                 periodic_table_length: int = 30,
                 initalizer: str = 'xavier_uniform_',
                 **kwargs):
        """
        Parameters
        ----------
        n_embedding: int, optional
            Number of features for each atom
        periodic_table_length: int, optional
            Length of embedding, 83=Bi
        initalizer: str, optional
            Weight initialization for filters.
            Options: {xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_}

        """
        super(DTNNEmbedding, self).__init__(**kwargs)
        self.n_embedding = n_embedding
        self.periodic_table_length = periodic_table_length
        self.initalizer = initalizer  # Set weight initialization

        init_func: Callable = getattr(initializers, self.initalizer)
        self.embedding_list: nn.Parameter = nn.Parameter(
            init_func(
                torch.empty([self.periodic_table_length, self.n_embedding])))

    def __repr__(self) -> str:
        """Returns a string representing the configuration of the layer.

        Returns
        -------
        n_embedding: int, optional
            Number of features for each atom
        periodic_table_length: int, optional
            Length of embedding, 83=Bi
        initalizer: str, optional
            Weight initialization for filters.
            Options: {xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_}

        """
        return f'{self.__class__.__name__}(n_embedding={self.n_embedding}, periodic_table_length={self.periodic_table_length}, initalizer={self.initalizer})'

    def forward(self, inputs: torch.Tensor):
        """Returns Embeddings according to indices.

        Parameters
        ----------
        inputs: torch.Tensor
            Indices of Atoms whose embeddings are requested.

        Returns
        -------
        atom_embeddings: torch.Tensor
            Embeddings of atoms accordings to indices.

        """
        atom_number = inputs
        atom_enbeddings = torch.nn.functional.embedding(atom_number,
                                                        self.embedding_list)
        return atom_enbeddings


class MolGANConvolutionLayer(nn.Module):
    """
    Graph convolution layer used in MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    Not used directly, higher level layers like MolGANMultiConvolutionLayer use it.
    This layer performs basic convolution on one-hot encoded matrices containing
    atom and bond information. This layer also accepts three inputs for the case
    when convolution is performed more than once and results of previous convolution
    need to used. It was done in such a way to avoid creating another layer that
    accepts three inputs rather than two. The last input layer is so-called
    hidden_layer and it hold results of the convolution while first two are unchanged
    input tensors.

    Examples
    --------
    See: MolGANMultiConvolutionLayer for using in layers.

    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> units = 128

    >>> layer1 = MolGANConvolutionLayer(units=units, edges=edges, nodes=nodes, name='layer1')
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))
    >>> output = layer1([adjacency_tensor, node_tensor])

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: int,
                 nodes: int,
                 activation=torch.tanh,
                 dropout_rate: float = 0.0,
                 edges: int = 5,
                 name: str = "",
                 prev_shape: int = 0,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize this layer.

        Parameters
        ---------
        units: int
            Dimesion of dense layers used for convolution
        nodes: int
            Number of features in node tensor
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Dropout rate used by dropout layer
        edges: int, optional (default=5)
            How many dense layers to use in convolution.
            Typically equal to number of bond types used in the model.
        name: string, optional (default="")
            Name of the layer
        prev_shape: int, optional (default=0)
            Shape of the previous layer, used when more than two inputs are passed
        """
        super(MolGANConvolutionLayer, self).__init__()

        self.activation = activation
        self.dropout_rate: float = dropout_rate
        self.units: int = units
        self.edges: int = edges
        self.name: str = name
        self.nodes: int = nodes
        self.device = device
        # Case when >2 inputs are passed
        if prev_shape:
            self.dense1 = nn.ModuleList([
                nn.Linear(prev_shape + self.nodes, self.units)
                for _ in range(edges - 1)
            ])
        else:
            self.dense1 = nn.ModuleList(
                [nn.Linear(self.nodes, self.units) for _ in range(edges - 1)])
        self.dense2: nn.Linear = nn.Linear(nodes, self.units)
        self.dropout: nn.Dropout = nn.Dropout(self.dropout_rate)

    def __repr__(self) -> str:
        """
        Returns a string representing the configuration of the layer.

        Returns
        -------
        str
            String representation of the layer
        """
        return (
            f'{self.__class__.__name__}(Units={self.units}, Nodes={self.nodes}, Activation={self.activation}, Dropout_rate={self.droput_rate}, Edges={self.edges}, Name={self.name})'
        )

    def forward(
            self,
            inputs: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: list
            List of two input matrices, adjacency tensor and node features tensors
            in one-hot encoding format.

        Returns
        --------
        tuple(torch.Tensor,torch.Tensor,torch.Tensor)
            First and second are original input tensors
            Third is the result of convolution
        """
        ic: int = len(inputs)
        if ic < 2:
            raise ValueError(
                "MolGANConvolutionLayer requires at least two inputs: [adjacency_tensor, node_features_tensor]"
            )

        adjacency_tensor: torch.Tensor = inputs[0].to(self.device)
        node_tensor: torch.Tensor = inputs[1].to(self.device)

        if ic > 2:
            hidden_tensor: torch.Tensor = inputs[2]
            annotations = torch.cat((hidden_tensor, node_tensor), -1)
        else:
            annotations = node_tensor

        output_dense: torch.Tensor = torch.stack(
            [dense(annotations) for dense in self.dense1], 1)

        adj: torch.Tensor = adjacency_tensor.permute(0, 3, 1, 2)[:, 1:, :, :]

        output_mul: torch.Tensor = torch.matmul(adj, output_dense)
        output_sum: torch.Tensor = torch.sum(output_mul,
                                             dim=1) + self.dense2(node_tensor)
        output_act: torch.Tensor = self.activation(output_sum)
        output = self.dropout(output_act)
        return adjacency_tensor, node_tensor, output


class MolGANAggregationLayer(nn.Module):
    """
    Graph Aggregation layer used in MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    Performs aggregation on tensor resulting from convolution layers.
    Given its simple nature it might be removed in future and moved to
    MolGANEncoderLayer.


    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> units = 128

    >>> layer_1 = MolGANConvolutionLayer(units=units,nodes=nodes,edges=edges, name='layer1')
    >>> layer_2 = MolGANAggregationLayer(units=128, name='layer2')
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))
    >>> hidden_1 = layer_1([adjacency_tensor, node_tensor])
    >>> output = layer_2(hidden_1[2])

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: int = 128,
                 activation=torch.tanh,
                 dropout_rate: float = 0.0,
                 name: str = "",
                 prev_shape: int = 0,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize the layer

        Parameters
        ---------
        units: int, optional (default=128)
            Dimesion of dense layers used for aggregation
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        name: string, optional (default="")
            Name of the layer
        prev_shape: int, optional (default=0)
            Shape of the input tensor
        """

        super(MolGANAggregationLayer, self).__init__()
        self.units: int = units
        self.activation = activation
        self.dropout_rate: float = dropout_rate
        self.name: str = name
        self.device = device

        if prev_shape:
            self.d1 = nn.Linear(prev_shape, self.units)
            self.d2 = nn.Linear(prev_shape, self.units)
        else:
            self.d1 = nn.Linear(self.units, self.units)
            self.d2 = nn.Linear(self.units, self.units)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(units={self.units}, activation={self.activation}, dropout_rate={self.dropout_rate}, Name={self.name})"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: List
            Single tensor resulting from graph convolution layer

        Returns
        --------
        aggregation tensor: torch.Tensor
          Result of aggregation function on input convolution tensor.
        """
        inputs = inputs.to(self.device)
        i = torch.sigmoid(self.d1(inputs))
        j = self.activation(self.d2(inputs))
        output = torch.sum(i * j, dim=1)
        output = self.activation(output)
        output = self.dropout_layer(output)
        return output


class MolGANMultiConvolutionLayer(nn.Module):
    """
    Multiple pass convolution layer used in MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    It takes outputs of previous convolution layer and uses
    them as inputs for the next one.
    It simplifies the overall framework, but might be moved to
    MolGANEncoderLayer in the future in order to reduce number of layers.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> units = (128,64)

    >>> layer_1 = MolGANMultiConvolutionLayer(units=units, nodes=nodes, edges=edges, name='layer1')
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))
    >>> output = layer_1([adjacency_tensor, node_tensor])

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: Tuple = (128, 64),
                 nodes: int = 5,
                 activation=torch.tanh,
                 dropout_rate: float = 0.0,
                 edges: int = 5,
                 name: str = "",
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        """
        Initialize the layer

        Parameters
        ---------
        units: Tuple, optional (default=(128,64)), min_length=2
            ist of dimensions used by consecutive convolution layers.
            The more values the more convolution layers invoked.
        nodes: int, optional (default=5)
            Number of features in node tensor
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        edges: int, optional (default=5)
            Controls how many dense layers use for single convolution unit.
            Typically matches number of bond types used in the molecule.
        name: string, optional (default="")
            Name of the layer
        """

        super(MolGANMultiConvolutionLayer, self).__init__()
        if len(units) < 2:
            raise ValueError("units parameter must contain at least two values")

        self.nodes: int = nodes
        self.units: Tuple = units
        self.activation = activation
        self.dropout_rate: float = dropout_rate
        self.edges: int = edges
        self.name: str = name
        self.device = device

        self.first_convolution = MolGANConvolutionLayer(
            units=self.units[0],
            nodes=self.nodes,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            edges=self.edges,
            device=self.device)
        self.gcl = nn.ModuleList([
            MolGANConvolutionLayer(units=u,
                                   nodes=self.nodes,
                                   activation=self.activation,
                                   dropout_rate=self.dropout_rate,
                                   edges=self.edges,
                                   prev_shape=self.units[count],
                                   device=self.device)
            for count, u in enumerate(self.units[1:])
        ])

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(units={self.units}, nodes={self.nodes}, activation={self.activation}, dropout_rate={self.dropout_rate}), edges={self.edges}, Name={self.name})"

    def forward(self, inputs: List) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: list
            List of two input matrices, adjacency tensor and node features tensors
            in one-hot encoding format.

        Returns
        --------
        convolution tensor: torch.Tensor
            Result of input tensors going through convolution a number of times.
        """

        adjacency_tensor = inputs[0].to(self.device)
        node_tensor = inputs[1].to(self.device)

        tensors = self.first_convolution([adjacency_tensor, node_tensor])

        # Loop over the remaining convolution layers
        for layer in self.gcl:
            # Apply the current layer to the outputs from the previous layer
            tensors = layer(tensors)

        _, _, hidden_tensor = tensors

        return hidden_tensor


class MolGANEncoderLayer(nn.Module):
    """
    Main learning layer used by MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    It role is to further simplify model.
    This layer can be manually built by stacking graph convolution layers
    followed by graph aggregation.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> dropout_rate = 0.0
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))

    >>> graph = MolGANEncoderLayer(units = [(128,64),128], dropout_rate= dropout_rate, edges=edges, nodes=nodes)([adjacency_tensor,node_tensor])
    >>> dense = nn.Linear(128,128)(graph)
    >>> dense = torch.tanh(dense)
    >>> dense = nn.Dropout(dropout_rate)(dense)
    >>> dense = nn.Linear(128,64)(dense)
    >>> dense = torch.tanh(dense)
    >>> dense = nn.Dropout(dropout_rate)(dense)
    >>> output = nn.Linear(64,1)(dense)

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: List = [(128, 64), 128],
                 activation: Callable = torch.tanh,
                 dropout_rate: float = 0.0,
                 edges: int = 5,
                 nodes: int = 5,
                 name: str = "",
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        """
        Initialize the layer

        Parameters
        ----------
        units: List, optional (default=[(128,64),128])
            List of dimensions used by consecutive convolution layers.
            The more values the more convolution layers invoked.
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        edges: int, optional (default=5)
            Controls how many dense layers use for single convolution unit.
            Typically matches number of bond types used in the molecule.
        nodes: int, optional (default=5)
            Number of features in node tensor
        name: string, optional (default="")
            Name of the layer
        """

        super(MolGANEncoderLayer, self).__init__()
        if len(units) != 2:
            raise ValueError("units parameter must contain two values")
        self.graph_convolution_units, self.auxiliary_units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.edges = edges
        self.nodes = nodes
        self.device = device

        self.multi_graph_convolution_layer = MolGANMultiConvolutionLayer(
            units=self.graph_convolution_units,
            nodes=self.nodes,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            edges=self.edges,
            device=self.device)
        self.graph_aggregation_layer = MolGANAggregationLayer(
            units=self.auxiliary_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            prev_shape=self.graph_convolution_units[-1] + nodes,
            device=self.device)

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(graph_convolution_units={self.graph_convolution_units}, auxiliary_units={self.auxiliary_units}, activation={self.activation}, dropout_rate={self.dropout_rate}), edges={self.edges})"

    def forward(self, inputs: List) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: list
            List of two input matrices, adjacency tensor and node features tensors
            in one-hot encoding format.

        Returns
        --------
        encoder tensor: tf.Tensor
            Tensor that been through number of convolutions followed
            by aggregation.
        """

        output = self.multi_graph_convolution_layer(inputs)

        node_tensor = inputs[1]

        if len(inputs) > 2:
            hidden_tensor = inputs[2]
            annotations = torch.cat((output, hidden_tensor, node_tensor), -1)
        else:
            _, node_tensor = inputs
            annotations = torch.cat((output, node_tensor), -1)

        output = self.graph_aggregation_layer(annotations)
        return output


class DTNNStep(nn.Module):
    """DTNNStep Layer for DTNN model.

    Encodes the atom's interaction with other atoms according to distance relationships. [1]_

    This Layer implements the Eq (7) from DTNN Paper. Then sums them up to get the final output using Eq (6) from DTNN Paper.

    Eq (7): V_ij = tanh[W_fc . ((W_cf . C_j + b_cf) * (W_df . d_ij + b_df))]

    Eq (6): C_i = C_i + sum(V_ij)

    Here : '.'=Matrix Multiplication , '*'=Multiplication

    References
    ----------
    [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    Examples
    --------
    >>> from deepchem.models.torch_models import layers
    >>> import torch
    >>> embedding_layer = layers.DTNNEmbedding(4, 4)
    >>> emb = embedding_layer(torch.Tensor([0,1,2,3]).to(torch.int64))
    >>> step_layer = layers.DTNNStep(4, 6, 8)
    >>> output_torch = step_layer([
    ...     torch.Tensor(emb),
    ...     torch.Tensor([0, 1, 2, 3, 4, 5]).to(torch.float32),
    ...     torch.Tensor([1]).to(torch.int64),
    ...     torch.Tensor([[1]]).to(torch.int64)
    ... ])
    >>> output_torch.shape
    torch.Size([2, 4, 4])

    """

    def __init__(self,
                 n_embedding: int = 30,
                 n_distance: int = 100,
                 n_hidden: int = 60,
                 initializer: str = 'xavier_uniform_',
                 activation='tanh',
                 **kwargs):
        """
        Parameters
        ----------
        n_embedding: int, optional
            Number of features for each atom
        n_distance: int, optional
            granularity of distance matrix
        n_hidden: int, optional
            Number of nodes in hidden layer
        initializer: str, optional
            Weight initialization for filters.
            Options: {xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_}
        activation: str, optional
            Activation function applied

        """
        super(DTNNStep, self).__init__(**kwargs)
        self.n_embedding = n_embedding
        self.n_distance = n_distance
        self.n_hidden = n_hidden
        self.initializer = initializer  # Set weight initialization
        self.activation = activation  # Get activations
        self.activation_fn = get_activation(self.activation)

        init_func: Callable = getattr(initializers, self.initializer)

        self.W_cf = nn.Parameter(
            init_func(torch.empty([self.n_embedding, self.n_hidden])))
        self.W_df = nn.Parameter(
            init_func(torch.empty([self.n_distance, self.n_hidden])))
        self.W_fc = nn.Parameter(
            init_func(torch.empty([self.n_hidden, self.n_embedding])))
        self.b_cf = nn.Parameter(torch.zeros(size=[
            self.n_hidden,
        ]))
        self.b_df = nn.Parameter(torch.zeros(size=[
            self.n_hidden,
        ]))

    def __repr__(self):
        """Returns a string representing the configuration of the layer.

        Returns
        -------
        n_embedding: int, optional
            Number of features for each atom
        n_distance: int, optional
            granularity of distance matrix
        n_hidden: int, optional
            Number of nodes in hidden layer
        initializer: str, optional
            Weight initialization for filters.
            Options: {xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_}
        activation: str, optional
            Activation function applied

        """
        return f'{self.__class__.__name__}(n_embedding={self.n_embedding}, n_distance={self.n_distance}, n_hidden={self.n_hidden}, initializer={self.initializer}, activation={self.activation})'

    def forward(self, inputs):
        """Executes the equations and Returns the intraction vector of the atom with other atoms.

        Parameters
        ----------
        inputs: torch.Tensor
            List of Tensors having atom_features, distance, distance_membership_i, distance_membership_j.

        Returns
        -------
        interaction_vector: torch.Tensor
            interaction of the atom with other atoms based on distance and distance_membership.

        """
        atom_features = inputs[0]
        distance = inputs[1]
        distance_membership_i = inputs[2]
        distance_membership_j = inputs[3]
        distance_hidden = torch.matmul(distance, self.W_df) + self.b_df
        atom_features_hidden = torch.matmul(atom_features,
                                            self.W_cf) + self.b_cf
        outputs = torch.mul(
            distance_hidden,
            torch.embedding(atom_features_hidden, distance_membership_j))

        # for atom i in a molecule m, this step multiplies together distance info of atom pair(i,j)
        # and embeddings of atom j(both gone through a hidden layer)
        outputs = torch.matmul(outputs, self.W_fc)
        outputs = self.activation_fn(outputs)

        output_ii = torch.mul(self.b_df, atom_features_hidden)
        output_ii = torch.matmul(output_ii, self.W_fc)
        output_ii = self.activation_fn(output_ii)

        # for atom i, sum the influence from all other atom j in the molecule
        intraction_vector = scatter(outputs, distance_membership_i,
                                    dim=0) - output_ii + atom_features
        return intraction_vector


class DTNNGather(nn.Module):
    """DTNNGather Layer for DTNN Model.

    Predict Molecular Energy using atom_features and atom_membership. [1]_

    This Layer gathers the inputs got from the step layer according to atom_membership and calulates the total Molecular Energy.

    References
    ----------
    [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.

    Examples
    --------
    >>> from deepchem.models.torch_models import layers as layers_torch
    >>> import torch
    >>> gather_layer_torch = layers_torch.DTNNGather(3, 3, [10])
    >>> result = gather_layer_torch([torch.Tensor([[3, 2, 1]]).to(torch.float32), torch.Tensor([0]).to(torch.int64)])
    >>> result.shape
    torch.Size([1, 3])

    """

    def __init__(self,
                 n_embedding=30,
                 n_outputs=100,
                 layer_sizes=[100],
                 output_activation=True,
                 initializer='xavier_uniform_',
                 activation='tanh',
                 **kwargs):
        """
        Parameters
        ----------
        n_embedding: int, optional
            Number of features for each atom
        n_outputs: int, optional
            Number of features for each molecule(output)
        layer_sizes: list of int, optional(default=[100])
            Structure of hidden layer(s)
        initializer: str, optional
            Weight initialization for filters.
        activation: str, optional
            Activation function applied

        """

        super(DTNNGather, self).__init__(**kwargs)
        self.n_embedding = n_embedding
        self.n_outputs = n_outputs
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation
        self.initializer = initializer  # Set weight initialization
        self.activation = activation  # Get activations
        self.activation_fn = get_activation(self.activation)

        self.W_list = nn.ParameterList()
        self.b_list = nn.ParameterList()

        init_func: Callable = getattr(initializers, self.initializer)

        prev_layer_size = self.n_embedding
        for i, layer_size in enumerate(self.layer_sizes):
            self.W_list.append(
                nn.Parameter(
                    init_func(torch.empty([prev_layer_size, layer_size]))))
            self.b_list.append(nn.Parameter(torch.zeros(size=[
                layer_size,
            ])))
            prev_layer_size = layer_size
        self.W_list.append(
            nn.Parameter(
                init_func(torch.empty([prev_layer_size, self.n_outputs]))))
        self.b_list.append(nn.Parameter(torch.zeros(size=[
            self.n_outputs,
        ])))

    def __repr__(self):
        """Returns a string representing the configuration of the layer.

        Returns
        ----------
        n_embedding: int, optional
            Number of features for each atom
        n_outputs: int, optional
            Number of features for each molecule(output)
        layer_sizes: list of int, optional(default=[1000])
            Structure of hidden layer(s)
        initializer: str, optional
            Weight initialization for filters.
        activation: str, optional
            Activation function applied

        """
        return f'{self.__class__.__name__}(n_embedding={self.n_embedding}, n_outputs={self.n_outputs}, layer_sizes={self.layer_sizes}, output_activation={self.output_activation}, initializer={self.initializer}, activation={self.activation})'

    def forward(self, inputs):
        """Executes the equation and Returns Molecular Energies according to atom_membership.

        Parameters
        ----------
        inputs: torch.Tensor
            List of Tensor containing atom_features and atom_membership

        Returns
        -------
        molecular_energies: torch.Tensor
            Tensor containing the Molecular Energies according to atom_membership.

        """
        output = inputs[0]
        atom_membership = inputs[1]

        for i, W in enumerate(self.W_list[:-1]):
            output = torch.matmul(output, W) + self.b_list[i]
            output = self.activation_fn(output)
        output = torch.matmul(output, self.W_list[-1]) + self.b_list[-1]
        if self.output_activation:
            output = self.activation_fn(output)
        return scatter(output, atom_membership)


class EdgeNetwork(nn.Module):
    """The EdgeNetwork module is a PyTorch submodule designed for message passing in graph neural networks.

    Examples
    --------
    >>> pair_features = torch.rand((4, 2), dtype=torch.float32)
    >>> atom_features = torch.rand((5, 2), dtype=torch.float32)
    >>> atom_to_pair = []
    >>> n_atoms = 2
    >>> start = 0
    >>> C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
    >>> atom_to_pair.append(np.transpose(np.array([C1.flatten() + start, C0.flatten() + start])))
    >>> atom_to_pair = torch.Tensor(atom_to_pair)
    >>> atom_to_pair = torch.squeeze(atom_to_pair.to(torch.int64), dim=0)
    >>> inputs = [pair_features, atom_features, atom_to_pair]
    >>> n_pair_features = 2
    >>> n_hidden = 2
    >>> init = 'xavier_uniform_'
    >>> layer = EdgeNetwork(n_pair_features, n_hidden, init)
    >>> result = layer(inputs)
    >>> result.shape[1]
    2
    """

    def __init__(self,
                 n_pair_features: int = 8,
                 n_hidden: int = 100,
                 init: str = 'xavier_uniform_',
                 **kwargs):
        """Initalises a EdgeNetwork Layer

        Parameters
        ----------
        n_pair_features: int, optional
            The length of the pair features vector.
        n_hidden: int, optional
            number of hidden units in the passing phase
        init: str, optional
            Initialization function to be used in the message passing layer.
        """

        super(EdgeNetwork, self).__init__(**kwargs)
        self.n_pair_features: int = n_pair_features
        self.n_hidden: int = n_hidden
        self.init: str = init

        init_func: Callable = getattr(initializers, self.init)
        self.W: torch.Tensor = init_func(
            torch.empty([self.n_pair_features, self.n_hidden * self.n_hidden]))
        self.b: torch.Tensor = torch.zeros((self.n_hidden * self.n_hidden,))
        self.built: bool = True

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(n_pair_features:{self.n_pair_features},n_hidden:{self.n_hidden},init:{self.init})'
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: List[torch.Tensor]
            The length of atom_to_pair should be same as n_pair_features.
        Returns
        -------
        result: torch.Tensor
            Tensor containing the mapping of the edge vector to a d × d matrix, where d denotes the dimension of the internal hidden representation of each node in the graph.
        """
        pair_features: torch.Tensor
        atom_features: torch.Tensor
        atom_to_pair: torch.Tensor
        pair_features, atom_features, atom_to_pair = inputs

        A: torch.Tensor = torch.add(torch.matmul(pair_features, self.W), self.b)
        A = torch.reshape(A, (-1, self.n_hidden, self.n_hidden))
        out: torch.Tensor = torch.unsqueeze(atom_features[atom_to_pair[:, 1]],
                                            dim=2)
        out_squeeze: torch.Tensor = torch.squeeze(torch.matmul(A, out), dim=2)
        ind: torch.Tensor = atom_to_pair[:, 0]

        result: torch.Tensor = segment_sum(out_squeeze, ind)

        return result


class WeaveLayer(nn.Module):
    """This class implements the core Weave convolution from the Google graph convolution paper [1]_
    This is the Torch equivalent of the original implementation using Keras.

    This model contains atom features and bond features
    separately.Here, bond features are also called pair features.
    There are 2 types of transformation, atom->atom, atom->pair, pair->atom, pair->pair that this model implements.

    Examples
    --------
    This layer expects 4 inputs in a list of the form `[atom_features,
    pair_features, pair_split, atom_to_pair]`. We'll walk through the structure
    of these inputs. Let's start with some basic definitions.

    >>> import deepchem as dc
    >>> import numpy as np

    Suppose you have a batch of molecules

    >>> smiles = ["CCC", "C"]

    Note that there are 4 atoms in total in this system. This layer expects its input molecules to be batched together.

    >>> total_n_atoms = 4

    Let's suppose that we have a featurizer that computes `n_atom_feat` features per atom.

    >>> n_atom_feat = 75

    Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
    n_atom_feat)` of atomic features. For simplicity, let's just go with a
    random such matrix.

    >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

    Let's suppose we have `n_pair_feat` pairwise features

    >>> n_pair_feat = 14

    For each molecule, we compute a matrix of shape `(n_atoms*n_atoms,n_pair_feat)` of pairwise features for each pair of atoms in the molecule.
    Let's construct this conceptually for our example.

    >>> pair_feat = [np.random.rand(3*3, n_pair_feat), np.random.rand(1*1,n_pair_feat)]
    >>> pair_feat = np.concatenate(pair_feat, axis=0)
    >>> pair_feat.shape
    (10, 14)

    `pair_split` is an index into `pair_feat` which tells us which atom each row belongs to. In our case, we hve

    >>> pair_split = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

    That is, the first 9 entries belong to "CCC" and the last entry to "C". The
    final entry `atom_to_pair` goes in a little more in-depth than `pair_split`
    and tells us the precise pair each pair feature belongs to. In our case

    >>> atom_to_pair = np.array([[0, 0],
    ...                          [0, 1],
    ...                          [0, 2],
    ...                          [1, 0],
    ...                          [1, 1],
    ...                          [1, 2],
    ...                          [2, 0],
    ...                          [2, 1],
    ...                          [2, 2],
    ...                          [3, 3]])

    Let's now define the actual layer

    >>> layer = WeaveLayer()

    And invoke it

    >>> [A, P] = layer([atom_feat, pair_feat, pair_split, atom_to_pair])

    The weave layer produces new atom/pair features. Let's check their shapes

    >>> A = A.detach().numpy()
    >>> A.shape
    (4, 50)
    >>> P = P.detach().numpy()
    >>> P.shape
    (10, 50)

    The 4 is `total_num_atoms` and the 10 is the total number of pairs. Where
    does `50` come from? It's from the default arguments `n_atom_input_feat` and
    `n_pair_input_feat`.

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.
    """

    def __init__(self,
                 n_atom_input_feat: int = 75,
                 n_pair_input_feat: int = 14,
                 n_atom_output_feat: int = 50,
                 n_pair_output_feat: int = 50,
                 n_hidden_AA: int = 50,
                 n_hidden_PA: int = 50,
                 n_hidden_AP: int = 50,
                 n_hidden_PP: int = 50,
                 update_pair: bool = True,
                 init_: str = 'xavier_uniform_',
                 activation: str = 'relu',
                 batch_normalize: bool = True,
                 **kwargs):
        """
    Parameters
    ----------
    n_atom_input_feat: int, optional (default 75)
      Number of features for each atom in input.
    n_pair_input_feat: int, optional (default 14)
      Number of features for each pair of atoms in input.
    n_atom_output_feat: int, optional (default 50)
      Number of features for each atom in output.
    n_pair_output_feat: int, optional (default 50)
      Number of features for each pair of atoms in output.
    n_hidden_AA: int, optional (default 50)
      Number of units(convolution depths) in corresponding hidden layer
    n_hidden_PA: int, optional (default 50)
      Number of units(convolution depths) in corresponding hidden layer
    n_hidden_AP: int, optional (default 50)
      Number of units(convolution depths) in corresponding hidden layer
    n_hidden_PP: int, optional (default 50)
      Number of units(convolution depths) in corresponding hidden layer
    update_pair: bool, optional (default True)
      Whether to calculate for pair features,
      could be turned off for last layer
    init: str, optional (default 'xavier_uniform_')
      Weight initialization for filters.
    activation: str, optional (default 'relu')
      Activation function applied
    batch_normalize: bool, optional (default True)
      If this is turned on, apply batch normalization before applying
      activation functions on convolutional layers.
    """
        super(WeaveLayer, self).__init__(**kwargs)
        self.init: str = init_  # Set weight initialization
        self.activation: str = activation  # Get activations
        self.activation_fn: torch.nn.Module = get_activation(activation)
        self.update_pair: bool = update_pair  # last weave layer does not need to update
        self.n_hidden_AA: int = n_hidden_AA
        self.n_hidden_PA: int = n_hidden_PA
        self.n_hidden_AP: int = n_hidden_AP
        self.n_hidden_PP: int = n_hidden_PP
        self.n_hidden_A: int = n_hidden_AA + n_hidden_PA
        self.n_hidden_P: int = n_hidden_AP + n_hidden_PP
        self.batch_normalize: bool = batch_normalize

        self.n_atom_input_feat: int = n_atom_input_feat
        self.n_pair_input_feat: int = n_pair_input_feat
        self.n_atom_output_feat: int = n_atom_output_feat
        self.n_pair_output_feat: int = n_pair_output_feat

        # Construct internal trainable weights
        init = getattr(initializers, self.init)
        # Weight matrix and bias matrix required to compute new atom layer from the previous atom layer
        self.W_AA: torch.Tensor = init(
            torch.empty(self.n_atom_input_feat, self.n_hidden_AA))
        self.b_AA: torch.Tensor = torch.zeros((self.n_hidden_AA,))
        self.AA_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_hidden_AA,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True)

        # Weight matrix and bias matrix required to compute new atom layer from the previous pair layer
        self.W_PA: torch.Tensor = init(
            torch.empty(self.n_pair_input_feat, self.n_hidden_PA))
        self.b_PA: torch.Tensor = torch.zeros((self.n_hidden_PA,))
        self.PA_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_hidden_PA,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True)

        self.W_A: torch.Tensor = init(
            torch.empty(self.n_hidden_A, self.n_atom_output_feat))
        self.b_A: torch.Tensor = torch.zeros((self.n_atom_output_feat,))
        self.A_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_atom_output_feat,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True)

        if self.update_pair:
            # Weight matrix and bias matrix required to compute new pair layer from the previous atom layer
            self.W_AP: torch.Tensor = init(
                torch.empty(self.n_atom_input_feat * 2, self.n_hidden_AP))
            self.b_AP: torch.Tensor = torch.zeros((self.n_hidden_AP,))
            self.AP_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_hidden_AP,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True)
            # Weight matrix and bias matrix required to compute new pair layer from the previous pair layer
            self.W_PP: torch.Tensor = init(
                torch.empty(self.n_pair_input_feat, self.n_hidden_PP))
            self.b_PP: torch.Tensor = torch.zeros((self.n_hidden_PP,))
            self.PP_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_hidden_PP,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True)

            self.W_P: torch.Tensor = init(
                torch.empty(self.n_hidden_P, self.n_pair_output_feat))
            self.b_P: torch.Tensor = torch.zeros((self.n_pair_output_feat,))
            self.P_bn: nn.BatchNorm1d = nn.BatchNorm1d(
                num_features=self.n_pair_output_feat,
                eps=1e-3,
                momentum=0.99,
                affine=True,
                track_running_stats=True)
        self.built = True

    def __repr__(self) -> str:
        """
    Returns a string representation of the object.

    Returns:
    -------
    str: A string that contains the class name followed by the values of its instance variable.
    """
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(n_atom_input_feat:{self.n_atom_input_feat},n_pair_input_feat:{self.n_pair_input_feat},n_atom_output_feat:{self.n_atom_output_feat},n_pair_output_feat:{self.n_pair_output_feat},n_hidden_AA:{self.n_hidden_AA},n_hidden_PA:{self.n_hidden_PA},n_hidden_AP:{self.n_hidden_AP},n_hidden_PP:{self.n_hidden_PP},batch_normalize:{self.batch_normalize},update_pair:{self.update_pair},init:{self.init},activation:{self.activation})'
        )

    def forward(
        self, inputs: List[Union[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray]]
    ) -> List[Union[torch.Tensor, torch.Tensor]]:
        """
        Creates weave tensors.

        Parameters
        ----------
        inputs: List[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
          Should contain 4 tensors [atom_features, pair_features, pair_split,
          atom_to_pair]

        Returns
        -------
        List[Union[torch.Tensor, torch.Tensor]]
          A: Atom features tensor of shape `(total_num_atoms,atom feature size)`

          P: Pair features tensor of shape `(total num of pairs,bond feature size)`
        """
        # Converting the input to torch tensors
        atom_features: torch.Tensor = torch.tensor(inputs[0])
        pair_features: torch.Tensor = torch.tensor(inputs[1])

        pair_split: torch.Tensor = torch.tensor(inputs[2])
        atom_to_pair: torch.Tensor = torch.tensor(inputs[3])

        activation = self.activation_fn

        # AA is a tensor with shape[total_num_atoms,n_hidden_AA]
        AA: torch.Tensor = torch.matmul(atom_features.type(torch.float32),
                                        self.W_AA) + self.b_AA
        if self.batch_normalize:
            self.AA_bn.eval()
            AA = self.AA_bn(AA)
        AA = activation(AA)
        # PA is a tensor with shape[total number of pairs,n_hidden_PA]
        PA: torch.Tensor = torch.matmul(pair_features.type(torch.float32),
                                        self.W_PA) + self.b_PA
        if self.batch_normalize:
            self.PA_bn.eval()
            PA = self.PA_bn(PA)
        PA = activation(PA)

        # Split the PA tensor according to the 'pair_split' tensor
        t_grp: Dict[Tensor, Tensor] = {}
        idx: int = 0
        for i, s_id in enumerate(pair_split):
            s_id = s_id.item()
            if s_id in t_grp:
                t_grp[s_id] = t_grp[s_id] + PA[idx]
            else:
                t_grp[s_id] = PA[idx]
            idx = i + 1

            lst = list(t_grp.values())
            tensor = torch.stack(lst)
        PA = tensor

        A: torch.Tensor = torch.matmul(torch.concat([AA, PA], 1),
                                       self.W_A) + self.b_A
        if self.batch_normalize:
            self.A_bn.eval()
            A = self.A_bn(A)
        A = activation(A)

        if self.update_pair:
            # Note that AP_ij and AP_ji share the same self.AP_bn batch
            # normalization
            AP_ij: torch.Tensor = torch.matmul(
                torch.reshape(atom_features[atom_to_pair],
                              [-1, 2 * self.n_atom_input_feat]).type(
                                  torch.float32), self.W_AP) + self.b_AP
            if self.batch_normalize:
                self.AP_bn.eval()
                AP_ij = self.AP_bn(AP_ij)
            AP_ij = activation(AP_ij)
            AP_ji: torch.Tensor = torch.matmul(
                torch.reshape(atom_features[torch.flip(atom_to_pair, [1])],
                              [-1, 2 * self.n_atom_input_feat]).type(
                                  torch.float32), self.W_AP) + self.b_AP
            if self.batch_normalize:
                self.AP_bn.eval()
                AP_ji = self.AP_bn(AP_ji)
            AP_ji = activation(AP_ji)
            # PP is a tensor with shape [total number of pairs,n_hidden_PP]
            PP: torch.Tensor = torch.matmul(pair_features.type(torch.float32),
                                            self.W_PP) + self.b_PP
            if self.batch_normalize:
                self.PP_bn.eval()
                PP = self.PP_bn(PP)
            PP = activation(PP)
            P: torch.Tensor = torch.matmul(
                torch.concat([AP_ij + AP_ji, PP], 1).type(torch.float32),
                self.W_P) + self.b_P
            if self.batch_normalize:
                self.P_bn.eval()
                P = self.P_bn(P)
            P = activation(P)
        else:
            P = pair_features

        return [A, P]


class WeaveGather(nn.Module):
    """Implements the weave-gathering section of weave convolutions.
    This is the Torch equivalent of the original implementation using Keras.

    Implements the gathering layer from [1]_. The weave gathering layer gathers
    per-atom features to create a molecule-level fingerprint in a weave
    convolutional network. This layer can also performs Gaussian histogram
    expansion as detailed in [1]_. Note that the gathering function here is
    simply addition as in [1]_>

    Examples
    --------
    This layer expects 2 inputs in a list of the form `[atom_features,
    pair_features]`. We'll walk through the structure
    of these inputs. Let's start with some basic definitions.

    >>> import deepchem as dc
    >>> import numpy as np

    Suppose you have a batch of molecules

    >>> smiles = ["CCC", "C"]

    Note that there are 4 atoms in total in this system. This layer expects its
    input molecules to be batched together.

    >>> total_n_atoms = 4

    Let's suppose that we have `n_atom_feat` features per atom.

    >>> n_atom_feat = 75

    Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
    n_atom_feat)` of atomic features. For simplicity, let's just go with a
    random such matrix.

    >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

    We then need to provide a mapping of indices to the atoms they belong to. In
    ours case this would be

    >>> atom_split = np.array([0, 0, 0, 1])

    Let's now define the actual layer

    >>> gather = WeaveGather(batch_size=2, n_input=n_atom_feat)
    >>> output_molecules = gather([atom_feat, atom_split])
    >>> len(output_molecules)
    2

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.
    """

    def __init__(self,
                 batch_size: int,
                 n_input: int = 128,
                 gaussian_expand: bool = True,
                 compress_post_gaussian_expansion: bool = False,
                 init_: str = 'xavier_uniform_',
                 activation: str = 'tanh',
                 **kwargs):
        """
        Parameters
        ----------
        batch_size: int
            number of molecules in a batch
        n_input: int, optional (default 128)
            number of features for each input molecule
        gaussian_expand: boolean, optional (default True)
            Whether to expand each dimension of atomic features by gaussian histogram
        compress_post_gaussian_expansion: bool, optional (default False)
            If True, compress the results of the Gaussian expansion back to the
            original dimensions of the input by using a linear layer with specified
            activation function. Note that this compression was not in the original
            paper, but was present in the original DeepChem implementation so is
            left present for backwards compatibility.
        init: str, optional (default 'xavier_uniform_')
            Weight initialization for filters if `compress_post_gaussian_expansion`
            is True.
        activation: str, optional (default 'tanh')
            Activation function applied for filters if
            `compress_post_gaussian_expansion` is True.
        """
        super(WeaveGather, self).__init__(**kwargs)
        self.n_input: int = n_input
        self.batch_size: int = batch_size
        self.gaussian_expand: bool = gaussian_expand
        self.compress_post_gaussian_expansion: bool = compress_post_gaussian_expansion
        self.init: str = init_  # Set weight initialization
        self.activation: str = activation  # Get activations
        self.activation_fn: torch.nn.Module = get_activation(activation)

        if self.compress_post_gaussian_expansion:
            init = getattr(initializers, self.init)
            self.W: torch.Tensor = init(
                torch.empty([self.n_input * 11, self.n_input]))
            self.b: torch.Tensor = torch.zeros((self.n_input,))
        self.built = True

    def __repr__(self):
        """
        Returns a string representation of the object.

        Returns:
        -------
        str: A string that contains the class name followed by the values of its instance variable.
        """
        return (
            f'{self.__class__.__name__}(batch_size:{self.batch_size},n_input:{self.n_input},gaussian_expand:{self.gaussian_expand},init:{self.init},activation:{self.activation},compress_post_gaussian_expansion:{self.compress_post_gaussian_expansion})'
        )

    def forward(self, inputs: List[Union[np.ndarray,
                                         np.ndarray]]) -> torch.Tensor:
        """Creates weave tensors.

        Parameters
        ----------
        inputs: List[Union[np.ndarray,np.ndarray]]
            Should contain 2 tensors [atom_features, atom_split]

        Returns
        -------
        output_molecules: torch.Tensor
            Each entry in this list is of shape `(self.n_inputs,)`

        """
        outputs: torch.Tensor = torch.tensor(inputs[0])
        atom_split: torch.Tensor = torch.tensor(inputs[1])

        if self.gaussian_expand:
            outputs = self.gaussian_histogram(outputs)

        t_grp: Dict[Tensor, Tensor] = {}
        idx: int = 0
        for i, s_id in enumerate(atom_split):
            s_id = s_id.item()
            if s_id in t_grp:
                t_grp[s_id] = t_grp[s_id] + outputs[idx]
            else:
                t_grp[s_id] = outputs[idx]
            idx = i + 1

            lst = list(t_grp.values())
            tensor = torch.stack(lst)
        output_molecules: torch.Tensor = tensor

        if self.compress_post_gaussian_expansion:
            output_molecules = torch.matmul(
                output_molecules.type(torch.float32), self.W) + self.b
            output_molecules = self.activation_fn(output_molecules)

        return output_molecules

    def gaussian_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Expands input into a set of gaussian histogram bins.

        Parameters
        ----------
        x: torch.Tensor
            Of shape `(N, n_feat)`

        Examples
        --------
        This method uses 11 bins spanning portions of a Gaussian with zero mean
        and unit standard deviation.

        >>> gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170),
        ...                         (-0.739, 0.134), (-0.468, 0.118),
        ...                         (-0.228, 0.114), (0., 0.114),
        ...                         (0.228, 0.114), (0.468, 0.118),
        ...                         (0.739, 0.134), (1.080, 0.170),
        ...                         (1.645, 0.283)]

        We construct a Gaussian at `gaussian_memberships[i][0]` with standard
        deviation `gaussian_memberships[i][1]`. Each feature in `x` is assigned
        the probability of falling in each Gaussian, and probabilities are
        normalized across the 11 different Gaussians.

        Returns
        -------
        outputs: torch.Tensor
            Of shape `(N, 11*n_feat)`
        """
        import torch.distributions as dist
        gaussian_memberships: List[Tuple[float, float]] = [(-1.645, 0.283),
                                                           (-1.080, 0.170),
                                                           (-0.739, 0.134),
                                                           (-0.468, 0.118),
                                                           (-0.228, 0.114),
                                                           (0., 0.114),
                                                           (0.228, 0.114),
                                                           (0.468, 0.118),
                                                           (0.739, 0.134),
                                                           (1.080, 0.170),
                                                           (1.645, 0.283)]

        distributions: List[dist.Normal] = [
            dist.Normal(torch.tensor(p[0]), torch.tensor(p[1]))
            for p in gaussian_memberships
        ]
        dist_max: List[torch.Tensor] = [
            distributions[i].log_prob(torch.tensor(
                gaussian_memberships[i][0])).exp() for i in range(11)
        ]

        outputs: List[torch.Tensor] = [
            distributions[i].log_prob(torch.tensor(x)).exp() / dist_max[i]
            for i in range(11)
        ]
        output: torch.Tensor = torch.stack(outputs, dim=2)
        output = output / torch.sum(output, dim=2, keepdim=True)
        output = output.view(-1, self.n_input * 11)
        return output


class _MXMNetEnvelope(torch.nn.Module):
    """
    A PyTorch module implementing an envelope function. This is a helper class for MXMNetSphericalBasisLayer and MXMNetBesselBasisLayer to be used in MXMNet Model.

    The envelope function is defined as follows:
    env(x) = 1 / x + a * x^e + b * x^(e+1) + c * x^(e+2)        if x < 1
    env(x) = 0                                                  if x >= 1

    where
    'x' is the input tensor
    'e' is the exponent parameter
    'a' = -(e + 1) * (e + 2) / 2
    'b' = e * (e + 2)
    'c' = -e * (e + 1) / 2

    Examples
    --------
    >>> env = _MXMNetEnvelope(exponent=2)
    >>> input_tensor = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> output = env(input_tensor)
    >>> output.shape
    torch.Size([4])
    """

    def __init__(self, exponent: int):
        """
        Parameters
        ----------
        exponent: float
            The exponent 'e' used in the envelope function.
        """
        super(_MXMNetEnvelope, self).__init__()
        self.e: int = exponent
        self.a: float = -(self.e + 1) * (self.e + 2) / 2
        self.b: float = self.e * (self.e + 2)
        self.c: float = -self.e * (self.e + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the envelope function for the input tensor 'x'.

        Parameters
        ----------
        x: torch.Tensor
            The Input tensor

        Returns
        -------
        output: torch.Tensor
            The tensor containing the computed envelope values for each element of 'x'.
        """
        e: int = self.e
        a: float = self.a
        b: float = self.b
        c: float = self.c

        x_pow_p0: torch.Tensor = x.pow(e)
        x_pow_p1: torch.Tensor = x_pow_p0 * x
        env_val: torch.Tensor = 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x

        zero: torch.Tensor = torch.zeros_like(x)
        output: torch.Tensor = torch.where(x < 1, env_val, zero)
        return output


try:

    class MXMNetGlobalMessagePassing(MessagePassing):
        """This class implements the Global Message Passing Layer from the Molecular Mechanics-Driven Graph Neural Network
        with Multiplex Graph for Molecular Structures(MXMNet) paper [1]_.

        This layer consists of two message passing steps and an update step between them.

        Let:
            - **x_i** : ``The node to be updated``
            - **h_i** : ``The hidden state of x_i``
            - **x_j** : ``The neighbour node connected to x_i by edge e_ij``
            - **h_j** : ``The hidden state of x_j``
            - **W** : ``The edge weights``
            - **m_ij** : ``The message between x_i and x_j``
            - **h_j (self_loop)** : ``The set of hidden states of atom features``
            - **mlp** : ``MultilayerPerceptron``
            - **res** : ``ResidualBlock``

        **In each message passing step**

            .. code-block:: python

                m_ij = mlp1([h_i || h_j || e_ij])*(e_ij W)

            **To handle self loops**

                .. code-block:: python

                    m_ij = m_ij + h_j(self_loop)

        **In each update step**

            .. code-block:: python

                hm_j = res1(sum(m_ij))
                h_j_new = mlp2(hm_j) + h_j
                h_j_new = res2(h_j_new)
                h_j_new = res3(h_j_new)

        .. note::
        Message passing and message aggregation(sum) is handled by ``propagate()``.

        References
        ----------
        .. [1] Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures. https://arxiv.org/pdf/2011.07457.pdf


        Examples
        --------
        The provided example demonstrates how to use the GlobalMessagePassing layer by creating an instance, passing input tensors (node_features, edge_attributes, edge_indices) through it, and checking the shape of the output.

        Initializes variables and creates a configuration dictionary with specific values.

        >>> dim = 1
        >>> node_features = torch.tensor([[0.8343], [1.2713], [1.2713], [1.2713], [1.2713]])
        >>> edge_attributes = torch.tensor([[1.0004], [1.0004], [1.0005], [1.0004], [1.0004],[-0.2644], [-0.2644], [-0.2644], [1.0004],[-0.2644], [-0.2644], [-0.2644], [1.0005],[-0.2644], [-0.2644], [-0.2644], [1.0004],[-0.2644], [-0.2644], [-0.2644]])
        >>> edge_indices = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])
        >>> out = MXMNetGlobalMessagePassing(dim)
        >>> output = out(node_features, edge_attributes, edge_indices)
        >>> output.shape
        torch.Size([5, 1])

        """

        def __init__(self,
                     dim: int,
                     activation_fn: Union[Callable, str] = 'silu'):
            """Initializes the MXMNETGlobalMessagePassing layer.

            Parameters
            -----------
            dim: int
                The dimension of the input and output features.
            """

            super(MXMNetGlobalMessagePassing, self).__init__()
            activation_fn = get_activation(activation_fn)

            self.h_mlp: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim, d_output=dim, activation_fn=activation_fn)

            self.res1: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim,
                d_hidden=(dim,),
                d_output=dim,
                activation_fn=activation_fn,
                skip_connection=True,
                weighted_skip=False)
            self.res2: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim,
                d_hidden=(dim,),
                d_output=dim,
                activation_fn=activation_fn,
                skip_connection=True,
                weighted_skip=False)
            self.res3: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim,
                d_hidden=(dim,),
                d_output=dim,
                activation_fn=activation_fn,
                skip_connection=True,
                weighted_skip=False)

            self.mlp: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim, d_output=dim, activation_fn=activation_fn)

            self.x_edge_mlp: MultilayerPerceptron = MultilayerPerceptron(
                d_input=dim * 3, d_output=dim, activation_fn=activation_fn)
            self.linear: nn.Linear = nn.Linear(dim, dim, bias=False)

        def forward(self, node_features: torch.Tensor,
                    edge_attributes: torch.Tensor,
                    edge_indices: torch.Tensor) -> torch.Tensor:
            """
            Performs the forward pass of the GlobalMessagePassing layer.

            Parameters
            -----------
            node_features: torch.Tensor
                The input node features tensor of shape (num_nodes, feature_dim).
            edge_attributes: torch.Tensor
                The input edge attribute tensor of shape (num_edges, attribute_dim).
            edge_indices: torch.Tensor
                The input edge index tensor of shape (2, num_edges).

            Returns
            --------
            torch.Tensor
                The updated node features tensor after message passing of shape (num_nodes, feature_dim).
            """
            edge_indices, _ = add_self_loops(edge_indices,
                                             num_nodes=node_features.size(0))

            residual_node_features: torch.Tensor = node_features

            # Integrate the Cross Layer Mapping inside the Global Message Passing
            node_features = self.h_mlp(node_features)

            # Message Passing operation
            node_features = self.propagate(edge_indices,
                                           x=node_features,
                                           num_nodes=node_features.size(0),
                                           edge_attr=edge_attributes)

            # Update function f_u
            node_features = self.res1(node_features)
            node_features = self.mlp(node_features) + residual_node_features
            node_features = self.res2(node_features)
            node_features = self.res3(node_features)

            # Message Passing operation
            node_features = self.propagate(edge_indices,
                                           x=node_features,
                                           num_nodes=node_features.size(0),
                                           edge_attr=edge_attributes)

            return node_features

        def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                    edge_attr: torch.Tensor) -> torch.Tensor:
            """Constructs messages to be passed along the edges in the graph.

            Parameters
            -----------
            x_i: torch.Tensor
                The source node features tensor of shape (num_edges+num_nodes, feature_dim).
            x_j: torch.Tensor
                The target node features tensor of shape (num_edges+num_nodes, feature_dim).
            edge_attributes: torch.Tensor
                The edge attribute tensor of shape (num_edges, attribute_dim).

            Returns
            --------
            torch.Tensor
                The constructed messages tensor.
            """
            num_edge: int = edge_attr.size()[0]

            x_edge: torch.Tensor = torch.cat(
                (x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
            x_edge = self.x_edge_mlp(x_edge)

            x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]),
                            dim=0)

            return x_j

except:
    pass


class MXMNetBesselBasisLayer(torch.nn.Module):
    """This layer implements a basis layer for the MXMNet model using Bessel functions.
    The basis layer is used to model radial symmetry in molecular systems.

    The output of the layer is given by:
    output = envelope(dist / cutoff) * (freq * dist / cutoff).sin()

    Examples
    --------
    >>> radial_layer = MXMNetBesselBasisLayer(num_radial=2, cutoff=2.0, envelope_exponent=2)
    >>> distances = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> output = radial_layer(distances)
    >>> output.shape
    torch.Size([4, 2])
    """

    def __init__(self,
                 num_radial: int,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        """Initialize the MXMNet Bessel Basis Layer.

        Parameters
        ----------
        num_radial: int
            The number of radial basis functions to use.
        cutoff: float, optional (default 5.0)
            The radial cutoff distance used to scale the distances.
        envelope_exponent: int, optional (default 5)
            The exponent of the envelope function.
        """

        super(MXMNetBesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope: _MXMNetEnvelope = _MXMNetEnvelope(envelope_exponent)
        self.freq: torch.Tensor = torch.nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initialize the learnable parameters of the MXMNet Bessel Basis Layer.

        The 'freq' tensor, representing the frequencies of the Bessel functions, is set up with initial values proportional to π (PI) and becomes a learnable parameter.

        The 'freq' tensor will be updated during the training process to optimize the performance of the MXMNet model for the specific task it is being trained on.
        """

        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Compute the output of the MXMNet Bessel Basis Layer.

        Parameters
        ----------
        dist: torch.Tensor
            The input tensor representing the pairwise distances between atoms.

        Returns
        -------
        output: torch.Tensor
            The output tensor representing the radial basis functions applied to the input distances.
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        output: torch.Tensor = self.envelope(dist) * (self.freq * dist).sin()
        return output


class VariationalRandomizer(nn.Module):
    """Add random noise to the embedding and include a corresponding loss.

    This adds random noise to the encoder, and also adds a constraint term to
    the loss that forces the embedding vector to have a unit Gaussian distribution.
    We can then pick random vectors from a Gaussian distribution, and the output
    sequences should follow the same distribution as the training data.

    We can use this layer with an AutoEncoder, which makes it a Variational
    AutoEncoder. The constraint term in the loss is initially set to 0, so the
    optimizer just tries to minimize the reconstruction loss. Once it has made
    reasonable progress toward that, the constraint term can be gradually turned
    back on. The range of steps over which this happens is configured by modifying
    the annealing_start_step and annealing final_step parameter.

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import VariationalRandomizer
    >>> import torch
    >>> embedding_dimension = 512
    >>> batch_size = 100
    >>> annealing_start_step = 1000
    >>> annealing_final_step = 2000
    >>> embedding_shape = (batch_size, embedding_dimension)
    >>> embeddings = torch.rand(embedding_shape)
    >>> global_step = torch.tensor([100])
    >>> layer = VariationalRandomizer(embedding_dimension, annealing_start_step, annealing_final_step)
    >>> output = layer([embeddings, global_step])
    >>> output.shape
    torch.Size([100, 512])

    References
    ----------
    .. [1] Samuel R. Bowman et al., "Generating Sentences from a Continuous Space"

    """

    def __init__(self, embedding_dimension: int, annealing_start_step: int,
                 annealing_final_step: int, **kwargs):
        """Initialize the VariationalRandomizer layer.

        Parameters
        ----------
        embedding_dimension: int
            The dimension of the embedding.
        annealing_start_step: int
            the step (that is, batch) at which to begin turning on the constraint
            term for KL cost annealing.
        annealing_final_step: int
            the step (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """

        super(VariationalRandomizer, self).__init__(**kwargs)
        self._embedding_dimension = embedding_dimension
        self._annealing_final_step = annealing_final_step
        self._annealing_start_step = annealing_start_step
        self.dense_mean = nn.Linear(embedding_dimension,
                                    embedding_dimension,
                                    bias=False)
        self.dense_stddev = nn.Linear(embedding_dimension,
                                      embedding_dimension,
                                      bias=False)
        self.combine = CombineMeanStd(training_only=True)
        self.loss_list: List = list()

    def __repr__(self) -> str:
        """Returns a string representing the configuration of the layer.

        Returns
        -------
        embedding_dimension: int
            The dimension of the embedding.
        annealing_start_step: int
            The step (that is, batch) at which to begin turning on the constraint
            term for KL cost annealing.
        annealing_final_step: int
            The step (that is, batch) at which to finish turning on the constraint
            term for KL cost annealing.

        """
        return f'{self.__class__.__name__}(embedding_dimension={self.embedding_dimension}, annealing_start_step={self.annealing_start_step}, annealing_final_step={self.annealing_final_step})'

    def forward(self, inputs: List[torch.Tensor], training=True):
        """Returns the Variationally Randomized Embedding.

        Parameters
        ----------
        inputs: List[torch.Tensor]
            A list of two tensors, the first of which is the input to the layer
            and the second of which is the global step.
        training: bool, optional (default True)
            Whether to use the layer in training mode or inference mode.

        Returns
        -------
        embedding: torch.Tensor
            The embedding tensor.

        """
        input, global_step = inputs
        embedding_mean = self.dense_mean(input)
        embedding_stddev = self.dense_stddev(input)
        embedding = self.combine([embedding_mean, embedding_stddev],
                                 training=training)
        mean_sq = embedding_mean * embedding_mean
        stddev_sq = embedding_stddev * embedding_stddev
        kl = mean_sq + stddev_sq - torch.log(stddev_sq + 1e-20) - 1
        anneal_steps = self._annealing_final_step - self._annealing_start_step
        if anneal_steps > 0:
            current_step = global_step.to(
                torch.float32) - self._annealing_start_step
            anneal_frac = torch.maximum(torch.tensor(0.0),
                                        current_step) / anneal_steps
            kl_scale = torch.minimum(torch.tensor(1.0),
                                     anneal_frac * anneal_frac)
        else:
            kl_scale = torch.tensor(1.0)
        self.add_loss(0.5 * kl_scale * torch.mean(kl))
        return embedding

    def add_loss(self, loss):
        """Add a loss term to the layer.

        Parameters
        ----------
        loss: torch.Tensor
            The loss tensor to add to the layer.

        """
        self.loss_list.append(loss)


class EncoderRNN(nn.Module):
    """Encoder Layer for SeqToSeq Model.

    It takes input sequences and converts them into a fixed-size context vector
    called the "embedding". This vector contains all relevant information from
    the input sequence. This context vector is then used by the decoder to
    generate the output sequence and can also be used as a representation of the
    input sequence for other Models.

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import EncoderRNN
    >>> import torch
    >>> embedding_dimensions = 7
    >>> num_input_token = 4
    >>> n_layers = 9
    >>> input = torch.tensor([[1, 0, 2, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    >>> layer = EncoderRNN(num_input_token, embedding_dimensions, n_layers)
    >>> emb, hidden = layer(input)
    >>> emb.shape
    torch.Size([3, 5, 7])

    References
    ----------
    .. [1] Sutskever et al., "Sequence to Sequence Learning with Neural Networks"

    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout_p: float = 0.1,
                 **kwargs):
        """Initialize the EncoderRNN layer.

        Parameters
        ----------
        input_size: int
            The number of expected features.
        hidden_size: int
            The number of features in the hidden state.
        dropout_p: float (default 0.1)
            The dropout probability to use during training.

        """
        super(EncoderRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def __repr__(self) -> str:
        """Returns a string representing the configuration of the layer.

        Returns
        -------
        input_size: int
            Number of expected features.
        hidden_size: int
            Number of features in the hidden state.
        dropout_p: float (default 0.1)
            Dropout probability to use during training.

        """
        return f'{self.__class__.__name__}(input_size={self.input_size}, hidden_size={self.hidden_size}, dropout_p={self.dropout_p})'

    def forward(self, input: torch.Tensor):
        """Returns Embeddings according to provided sequences.

        Parameters
        ----------
        input: torch.Tensor
            Batch of input sequences.

        Returns
        -------
        output: torch.Tensor
            Batch of Embeddings.
        hidden: torch.Tensor
            Batch of hidden states.

        """
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden[-1]


class DecoderRNN(nn.Module):
    """Decoder Layer for SeqToSeq Model.

    The decoder transforms the embedding vector into the output sequence.
    It is trained to predict the next token in the sequence given the previous
    tokens in the sequence. It uses the context vector from the encoder to
    help generate the correct token in the sequence.

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import DecoderRNN
    >>> import torch
    >>> embedding_dimensions = 512
    >>> num_output_tokens = 7
    >>> max_length = 10
    >>> batch_size = 100
    >>> n_layers = 2
    >>> layer = DecoderRNN(embedding_dimensions, num_output_tokens, n_layers, max_length, batch_size)
    >>> embeddings = torch.randn(batch_size, embedding_dimensions)
    >>> output, hidden = layer([embeddings, None])
    >>> output.shape
    torch.Size([100, 10, 7])

    References
    ----------
    .. [1] Sutskever et al., "Sequence to Sequence Learning with Neural Networks"

    """

    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 n_layers: int,
                 max_length: int,
                 batch_size: int,
                 step_activation: str = "relu",
                 **kwargs):
        """Initialize the DecoderRNN layer.

        Parameters
        ----------
        hidden_size: int
            Number of features in the hidden state.
        output_size: int
            Number of expected features.
        max_length: int
            Maximum length of the sequence.
        batch_size: int
            Batch size of the input.
        step_activation: str (default "relu")
            Activation function to use after every step.

        """
        super(DecoderRNN, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.act = get_activation("softmax")
        self.step_act = get_activation(step_activation)
        self.MAX_LENGTH = max_length
        self.batch_size = batch_size

    def __repr__(self) -> str:
        """Returns a string representing the configuration of the layer.

        Returns
        -------
        hidden_size: int
            Number of features in the hidden state.
        output_size: int
            Number of expected features.
        max_length: int
            Maximum length of the sequence.
        batch_size: int
            Batch size of the input.
        step_activation: str (default "relu")
            Activation function to use after every step.

        """
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size}, output_size={self.output_size}, max_length={self.max_length}, batch_size={self.batch_size})'

    def forward(self, inputs: List[torch.Tensor]):
        """
        Parameters
        ----------
        inputs: List[torch.Tensor]
            A list of tensor containg encoder_hidden and target_tensor.

        Returns
        -------
        decoder_outputs: torch.Tensor
            Predicted output sequences.
        decoder_hidden: torch.Tensor
            Hidden state of the decoder.

        """
        encoder_hidden, target_tensor = inputs
        decoder_input = torch.ones(self.batch_size,
                                   1,
                                   dtype=torch.long,
                                   device=encoder_hidden.device)
        decoder_hidden = torch.stack(self.n_layers * [encoder_hidden])
        decoder_outputs = []

        for i in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden = self.step(decoder_input,
                                                       decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:,
                                              i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1).detach()  # detach from history as input

        decoder_output = torch.cat(decoder_outputs, dim=1)
        decoder_output = self.act(decoder_output, dim=-1)
        return decoder_output, decoder_hidden

    def step(self, input, hidden):
        output = self.embedding(input)
        output = self.step_act(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class FerminetElectronFeature(torch.nn.Module):
    """
    A Pytorch Module implementing the ferminet's electron features interaction layer _[1]. This is a helper class for the Ferminet model.

    The layer consists of 2 types of linear layers - v for the one elctron features and w for the two electron features. The number and dimensions
    of each layer depends on the number of atoms and electrons in the molecule system.

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Examples
    --------
    >>> import deepchem as dc
    >>> electron_layer = dc.models.torch_models.layers.FerminetElectronFeature([32,32,32],[16,16,16], 4, 8, 10, [5,5])
    >>> one_electron_test = torch.randn(8, 10, 4*4)
    >>> two_electron_test = torch.randn(8, 10, 10, 4)
    >>> one, two = electron_layer.forward(one_electron_test, two_electron_test)
    >>> one.size()
    torch.Size([8, 10, 32])
    >>> two.size()
    torch.Size([8, 10, 10, 16])
    """

    def __init__(self, n_one: List[int], n_two: List[int], no_of_atoms: int,
                 batch_size: int, total_electron: int, spin: List[int]):
        """
        Parameters
        ----------
        n_one: List[int]
            List of integer values containing the dimensions of each n_one layer's output
        n_two: List[int]
            List of integer values containing the dimensions of each n_one layer's output
        no_of_atoms: int:
            Value containing the number of atoms in the molecule system
        batch_size: int
            Value containing the number of batches for the input provided
        total_electron: int
            Value containing the total number of electrons in the molecule system
        spin: List[int]
            List data structure in the format of [number of up-spin electrons, number of down-spin electrons]
        v: torch.nn.ModuleList
            torch ModuleList containing the linear layer with the n_one layer's dimension size.
        w: torch.nn.ModuleList
            torch ModuleList containing the linear layer with the n_two layer's dimension size.
        layer_size: int
            Value containing the number of n_one and n_two layers
        """

        super(FerminetElectronFeature, self).__init__()
        self.n_one = n_one
        self.n_two = n_two
        self.no_of_atoms = no_of_atoms
        self.batch_size = batch_size
        self.total_electron = total_electron
        self.spin = spin

        self.v: torch.nn.ModuleList = torch.nn.ModuleList()
        self.w: torch.nn.ModuleList = torch.nn.ModuleList()
        self.layer_size: int = len(self.n_one)

        # Initializing the first layer (first layer has different dims than others)
        self.v.append(
            nn.Linear(8 + 3 * 4 * self.no_of_atoms, self.n_one[0], bias=True))
        #filling the weights with xavier uniform method for the linear weights and random assignment for the bias
        torch.nn.init.xavier_uniform_(self.v[0].weight)
        self.v[0].bias.data = (torch.randn(size=(self.v[0].weight.shape[0],)))

        self.w.append(nn.Linear(4, self.n_two[0], bias=True))
        torch.nn.init.xavier_uniform_(self.w[0].weight)
        self.w[0].bias.data = (torch.randn(size=(self.w[0].weight.shape[0],)))

        for i in range(1, self.layer_size):
            self.v.append(
                nn.Linear(3 * self.n_one[i - 1] + 2 * self.n_two[i - 1],
                          n_one[i],
                          bias=True))
            torch.nn.init.xavier_uniform_(self.v[i].weight)
            self.v[i].bias.data = (torch.randn(
                size=(self.v[i].weight.shape[0],)))

            self.w.append(nn.Linear(self.n_two[i - 1], self.n_two[i],
                                    bias=True))
            torch.nn.init.xavier_uniform_(self.w[i].weight)
            self.w[i].bias.data = (torch.randn(
                size=(self.w[i].weight.shape[0],)))

        self.projection_module = nn.ModuleList()
        self.projection_module.append(
            nn.Linear(
                4 * self.no_of_atoms,
                n_one[0],
                bias=False,
            ))
        self.projection_module.append(nn.Linear(4, n_two[0], bias=False))
        torch.nn.init.xavier_uniform_(self.projection_module[0].weight)

        torch.nn.init.xavier_uniform_(self.projection_module[1].weight)

    def forward(self, one_electron: torch.Tensor, two_electron: torch.Tensor):
        """
        Parameters
        ----------
        one_electron: torch.Tensor
            The one electron feature which has the shape (batch_size, number of electrons, number of atoms * 4). Here the last dimension contains
            the electron's distance from each of the atom as a vector concatenated with norm of that vector.
        two_electron: torch.Tensor
            The two electron feature which has the shape (batch_size, number of electrons, number of electron , 4). Here the last dimension contains
            the electron's distance from the other electrons as a vector concatenated with norm of that vector.

        Returns
        -------
        one_electron: torch.Tensor
            The one electron feature after passing through the layer which has the shape (batch_size, number of electrons, n_one shape).
        two_electron: torch.Tensor
            The two electron feature after passing through the layer which has the shape (batch_size, number of electrons, number of electron , n_two shape).
            The two electron feature after passing through the layer which has the shape (batch_size, number of electrons, number of electron , n_two shape).
        """
        for l in range(self.layer_size):
            # Calculating one-electron feature's average
            g_one_up: torch.Tensor = torch.mean(
                one_electron[:, :self.spin[0], :], dim=-2)
            g_one_down: torch.Tensor = torch.mean(
                one_electron[:, self.spin[0]:, :], dim=-2)
            # temporary lists containing each electron's embeddings which will be torch.stack on the end
            one_electron_tmp = []
            two_electron_tmp = []
            for i in range(self.total_electron):
                # Calculating two-electron feature's average
                g_two_up: torch.Tensor = torch.mean(
                    two_electron[:, i, :self.spin[0], :], dim=1)
                g_two_down: torch.Tensor = torch.mean(
                    two_electron[:, i, self.spin[0]:, :], dim=1)
                f: torch.Tensor = torch.cat((one_electron[:, i, :], g_one_up,
                                             g_one_down, g_two_up, g_two_down),
                                            dim=1)
                if l == 0 or (self.n_one[l] != self.n_one[l - 1]) or (
                        self.n_two[l] != self.n_two[l - 1]):
                    one_electron_tmp.append((torch.tanh(self.v[l](f))) +
                                            self.projection_module[0]
                                            (one_electron[:, i, :]))
                    two_electron_tmp.append(
                        (torch.tanh(self.w[l](two_electron[:, i, :, :]))) +
                        self.projection_module[1](two_electron[:, i, :, :]))
                else:
                    one_electron_tmp.append(
                        (torch.tanh(self.v[l](f)) + one_electron[:, i, :]))
                    two_electron_tmp.append(
                        (torch.tanh(self.w[l](two_electron[:, i, :, :])) +
                         two_electron[:, i, :, :]))
            one_electron = torch.stack(one_electron_tmp, dim=1)
            two_electron = torch.stack(two_electron_tmp, dim=1)

        return one_electron, two_electron


class FerminetEnvelope(torch.nn.Module):
    """
    A Pytorch Module implementing the ferminet's envlope layer _[1], which is used to calculate the spin up and spin down orbital values.
    This is a helper class for the Ferminet model.
    The layer consists of 4 types of parameter lists - envelope_w, envelope_g, sigma and pi, which helps to calculate the orbital vlaues.

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Examples
    --------
    >>> import deepchem as dc
    >>> import torch
    >>> envelope_layer = dc.models.torch_models.layers.FerminetEnvelope([32, 32, 32], [16, 16, 16], 10, 8, [5, 5], 5, 16)
    >>> one_electron = torch.randn(8, 10, 32)
    >>> one_electron_permuted = torch.randn(8, 10, 5, 3)
    >>> psi, psi_up, psi_down = envelope_layer.forward(one_electron, one_electron_permuted)
    >>> psi.size()
    torch.Size([8])
    >>> psi_up.size()
    torch.Size([8, 16, 5, 5])
    >>> psi_down.size()
    torch.Size([8, 16, 5, 5])
    """

    def __init__(self, n_one: List[int], n_two: List[int], total_electron: int,
                 batch_size: int, spin: List[int], no_of_atoms: int,
                 determinant: int):
        """
        Parameters
        ----------
        n_one: List[int]
            List of integer values containing the dimensions of each n_one layer's output
        n_two: List[int]
            List of integer values containing the dimensions of each n_one layer's output
        total_electron: int
            Value containing the total number of electrons in the molecule system
        batch_size: int
            Value containing the number of batches for the input provided
        spin: List[int]
            List data structure in the format of [number of up-spin electrons, number of down-spin electrons]
        no_of_atoms: int
            Value containing the number of atoms in the molecule system
        determinant: int
            The number of determinants to be incorporated in the post-HF solution.
        envelope_w: torch.nn.ParameterList
            torch ParameterList containing the torch Tensor with n_one layer's dimension size.
        envelope_g: torch.nn.ParameterList
            torch ParameterList containing the torch Tensor with the unit dimension size, which acts as bias.
        sigma: torch.nn.ParameterList
            torch ParameterList containing the torch Tensor with the unit dimension size.
        pi: torch.nn.ParameterList
            torch ParameterList containing the linear layer with the n_two layer's dimension size.
        layer_size: int
            Value containing the number of n_one and n_two layers
        """

        super(FerminetEnvelope, self).__init__()
        self.n_one = n_one
        self.n_two = n_two
        self.total_electron = total_electron
        self.batch_size = batch_size
        self.spin = spin
        self.no_of_atoms = no_of_atoms
        self.determinant = determinant

        self.layer_size: int = len(self.n_one)

        self.envelope_w = torch.nn.ParameterList()
        self.envelope_g = torch.nn.ParameterList()
        self.sigma = torch.nn.ParameterList()
        self.pi = torch.nn.ParameterList()
        self.wdet = torch.nn.ParameterList()

        # initialized weights with torch.zeros, torch.eye and using xavier init.
        for i in range(self.determinant):
            self.wdet.append(torch.nn.init.normal_(torch.zeros(1)).squeeze(0))
            for j in range(self.total_electron):
                self.envelope_w.append(
                    (torch.nn.init.normal_(torch.zeros(n_one[-1], 1),) /
                     math.sqrt(n_one[-1])).squeeze(-1))
                self.envelope_g.append(
                    (torch.nn.init.normal_(torch.zeros(1))).squeeze(0))
                for k in range(self.no_of_atoms):
                    self.pi.append((torch.zeros(1)))
                    self.sigma.append(torch.eye(3))

    def forward(self, one_electron: torch.Tensor,
                one_electron_vector_permuted: torch.Tensor):
        """
        Parameters
        ----------
        one_electron: torch.Tensor
            Torch tensor which is output from FerminElectronFeature layer in the shape of (batch_size, number of elctrons, n_one layer size).
        one_electron_vector_permuted: torch.Tensor
            Torch tensor which is shape permuted vector of the original one_electron vector tensor. shape of the tensor should be (batch_size, number of atoms, number of electrons, 3).

        Returns
        -------
        psi_up: torch.Tensor
            Torch tensor with a scalar value containing the sampled wavefunction value for each batch.
        """
        psi = torch.zeros(self.batch_size)
        psi_up = []
        psi_down = []
        for k in range(self.determinant):
            # temporary list to stack upon electrons axis at the end
            det = []
            for i in range(self.spin[0]):
                one_d_index = (k * (self.total_electron)) + i
                for j in range(self.spin[0]):
                    det.append(((torch.sum(
                        (self.envelope_w[one_d_index] * one_electron[:, j, :]) +
                        self.envelope_g[one_d_index],
                        dim=1)) * torch.sum(torch.exp(-torch.abs(
                            torch.norm(one_electron_vector_permuted[:, j, :, :]
                                       @ self.sigma[one_d_index],
                                       dim=2))) * self.pi[one_d_index].T,
                                            dim=1)))
            psi_up.append(
                torch.reshape(torch.stack(det, dim=1),
                              (self.batch_size, self.spin[0], self.spin[0])))

            det = []
            for i in range(self.spin[0], self.spin[0] + self.spin[1]):
                one_d_index = (k * (self.total_electron)) + i
                for j in range(self.spin[0], self.spin[0] + self.spin[1]):
                    det.append(((torch.sum(
                        (self.envelope_w[one_d_index] * one_electron[:, j, :]) +
                        self.envelope_g[one_d_index],
                        dim=1)) * torch.sum(torch.exp(-torch.abs(
                            torch.norm(one_electron_vector_permuted[:, j, :, :]
                                       @ self.sigma[one_d_index],
                                       dim=2))) * self.pi[one_d_index].T,
                                            dim=1)))
            psi_down.append(
                torch.reshape(torch.stack(det, dim=1),
                              (self.batch_size, self.spin[1], self.spin[1])))

            d_down = torch.det(psi_down[-1])
            d_up = torch.det(psi_up[-1])
            det_full = d_up * d_down
            psi = psi + self.wdet[k] * det_full
        psi_matrix_up = torch.stack(psi_up, dim=1)
        psi_matrix_down = torch.stack(psi_down, dim=1)
        return psi, psi_matrix_up, psi_matrix_down


class MXMNetLocalMessagePassing(nn.Module):
    """
    The MXMNetLocalMessagePassing class defines a local message passing layer used in the MXMNet model [1]_.
    This layer integrates cross-layer mappings inside the local message passing, allowing for the transformation
    of input tensors representing pairwise distances and angles between atoms in a molecular system.
    The layer aggregates information using message passing and updates atom representations accordingly.
    The 3-step message passing scheme is proposed in the paper [1]_.

    1. Step 1 contains Message Passing 1 that captures the two-hop angles and related pairwise distances to update edge-level embeddings {mji}.
    2. Step 2 contains Message Passing 2 that captures the one-hop angles and related pairwise distances to further update {mji}.
    3. Step 3 finally aggregates {mji} to update the node-level embedding hi.

    These steps in the t-th iteration can be formulated as follows:

    Let:
        - **mlp** : ``MultilayerPerceptron``
        - **res** : ``ResidualBlock``
        - **h** : ``node_features``
        - **m** : ``message with radial basis function``
        - **idx_kj**: ``Tensor containing indices for the k and j atoms``
        - **x_i** : ``The node to be updated``
        - **h_i** : ``The hidden state of x_i``
        - **x_j** : ``The neighbour node connected to x_i by edge e_ij``
        - **h_j** : ``The hidden state of x_j``
        - **rbf** : ``Input tensor representing radial basis functions``
        - **sbf** : ``Input tensor representing the spherical basis functions``
        - **idx_jj** : ``Tensor containing indices for the j and j' where j' is other neighbours of i``

    Step 1: Message Passing 1

        .. code-block:: python

            m = [h[i] || h[j] || rbf]
            m_kj = mlp_kj(m[idx_kj]) * (rbf*W) * mlp_sbf1(sbf1)
            m_ji = mlp_ji_1(m) + reduce_sum(m_kj)

    Step 2: Message Passing 2

        .. code-block:: python

            m_ji = mlp_jj(m_ji[idx_jj]) * (rbf*W) * mlp_sbf2(sbf2)
            m_ji = mlp_ji_2(m_ji) + reduce_sum(m_ji)

    Step 3: Aggregation and Update

        **In each aggregation step**

        .. code-block:: python

            m = reduce_sum(m_ji*(rbf*W))

        **In each update step**

        .. code-block:: python

            hm_i = res1(m)
            h_i_new = mlp2(hm_i) + h_i
            h_i_new = res2(h_i_new)
            h_i_new = res3(h_i_new)

    References
    ----------
    .. [1] Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures. https://arxiv.org/pdf/2011.07457
    Examples
    --------
    >>> dim = 1
    >>> h = torch.tensor([[0.8343], [1.2713], [1.2713], [1.2713], [1.2713]])
    >>> rbf = torch.tensor([[-0.2628], [-0.2628], [-0.2628], [-0.2628],
    ...                     [-0.2629], [-0.2629], [-0.2628], [-0.2628]])
    >>> sbf1 = torch.tensor([[-0.2767], [-0.2767], [-0.2767], [-0.2767],
    ...                      [-0.2767], [-0.2767], [-0.2767], [-0.2767],
    ...                      [-0.2767], [-0.2767], [-0.2767], [-0.2767]])
    >>> sbf2 = torch.tensor([[-0.0301], [-0.0301], [-0.1483], [-0.1486], [-0.1484],
    ...                      [-0.0301], [-0.1483], [-0.0301], [-0.1485], [-0.1483],
    ...                      [-0.0301], [-0.1486], [-0.1485], [-0.0301], [-0.1486],
    ...                      [-0.0301], [-0.1484], [-0.1483], [-0.1486], [-0.0301]])
    >>> idx_kj = torch.tensor([3, 5, 7, 1, 5, 7, 1, 3, 7, 1, 3, 5])
    >>> idx_ji_1 = torch.tensor([0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6])
    >>> idx_jj = torch.tensor([0, 1, 3, 5, 7, 2, 1, 3, 5, 7, 4, 1, 3, 5, 7, 6, 1, 3, 5, 7])
    >>> idx_ji_2 = torch.tensor([0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 7, 7, 7, 7])
    >>> edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4],
    ...                           [1, 0, 2, 0, 3, 0, 4, 0]])
    >>> out = MXMNetLocalMessagePassing(dim, activation_fn='silu')
    >>> output = out(h,
    ...             rbf,
    ...             sbf1,
    ...             sbf2,
    ...             idx_kj,
    ...             idx_ji_1,
    ...             idx_jj,
    ...             idx_ji_2,
    ...             edge_index)
    >>> output[0].shape
    torch.Size([5, 1])
    >>> output[1].shape
    torch.Size([5, 1])
    """

    def __init__(self, dim: int, activation_fn: Union[Callable, str] = 'silu'):
        """Initializes the MXMNetLocalMessagePassing layer.

        Parameters
        ----------
        dim : int
            The dimension of the input and output tensors for the local message passing layer.
        activation_fn : Union[Callable, str], optional (default: 'silu')
            The activation function to be used in the multilayer perceptrons (MLPs) within the layer.
        """
        super(MXMNetLocalMessagePassing, self).__init__()

        activation_fn = get_activation(activation_fn)
        self.h_mlp: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim, d_output=dim, activation_fn=activation_fn)
        self.mlp_kj: MultilayerPerceptron = MultilayerPerceptron(
            d_input=3 * dim, d_output=dim, activation_fn=activation_fn)
        self.mlp_ji_1: MultilayerPerceptron = MultilayerPerceptron(
            d_input=3 * dim, d_output=dim, activation_fn=activation_fn)
        self.mlp_ji_2: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim, d_output=dim, activation_fn=activation_fn)
        self.mlp_jj: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim, d_output=dim, activation_fn=activation_fn)

        self.mlp_sbf1: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim,),
            d_output=dim,
            activation_fn=activation_fn)
        self.mlp_sbf2: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim,),
            d_output=dim,
            activation_fn=activation_fn)

        self.res1: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim,),
            d_output=dim,
            activation_fn=activation_fn,
            skip_connection=True,
            weighted_skip=False)
        self.res2: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim,),
            d_output=dim,
            activation_fn=activation_fn,
            skip_connection=True,
            weighted_skip=False)
        self.res3: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim,),
            d_output=dim,
            activation_fn=activation_fn,
            skip_connection=True,
            weighted_skip=False)

        self.lin_rbf1: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.lin_rbf2: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.lin_rbf_out: nn.Linear = nn.Linear(dim, dim, bias=False)

        self.mlp: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim, d_output=dim, activation_fn=activation_fn)
        self.out_mlp: MultilayerPerceptron = MultilayerPerceptron(
            d_input=dim,
            d_hidden=(dim, dim),
            d_output=dim,
            activation_fn=activation_fn)
        self.out_W: nn.Linear = nn.Linear(dim, 1)

    def forward(self, node_features: torch.Tensor, rbf: torch.Tensor,
                sbf1: torch.Tensor, sbf2: torch.Tensor, idx_kj: torch.Tensor,
                idx_ji_1: torch.Tensor, idx_jj: torch.Tensor,
                idx_ji_2: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward method performs the computation for the MXMNetLocalMessagePassing Layer.
        This method processes the input tensors representing atom features, radial basis functions (RBF), and spherical basis functions (SBF) using message passing over the molecular graph. The message passing updates the atom representations, and the resulting tensor represents the updated atom feature after local message passing.

        Parameters
        ----------
        node_features : torch.Tensor
            Input tensor representing atom features.
        rbf : torch.Tensor
            Input tensor representing radial basis functions.
        sbf1 : torch.Tensor
            Input tensor representing the first set of spherical basis functions.
        sbf2 : torch.Tensor
            Input tensor representing the second set of spherical basis functions.
        idx_kj : torch.Tensor
            Tensor containing indices for the k and j atoms involved in each interaction.
        idx_ji_1 : torch.Tensor
            Tensor containing indices for the j and i atoms involved in the first message passing step.
        idx_jj : torch.Tensor
            Tensor containing indices for the j and j' atoms involved in the second message passing step.
        idx_ji_2 : torch.Tensor
            Tensor containing indices for the j and i atoms involved in the second message passing step.
        edge_index : torch.Tensor
            Tensor containing the edge indices of the molecular graph, with shape (2, M), where M is the number of edges.

        Returns
        -------
        node_features: torch.Tensor
            Updated atom representations after local message passing.
        output: torch.Tensor
            Output tensor representing a fixed-size representation, with shape (N, 1).
        """

        residual_node_features: torch.Tensor = node_features

        # Integrate the Cross Layer Mapping inside the Local Message Passing
        node_features = self.h_mlp(node_features)

        # Message Passing 1
        j, i = edge_index
        m: torch.Tensor = torch.cat([node_features[i], node_features[j], rbf],
                                    dim=-1)

        m_kj: torch.Tensor = self.mlp_kj(m)
        m_kj = m_kj * self.lin_rbf1(rbf)
        m_kj = m_kj[idx_kj] * self.mlp_sbf1(sbf1)
        m_kj = scatter(m_kj, idx_ji_1, dim=0, dim_size=m.size(0), reduce='add')

        m_ji_1: torch.Tensor = self.mlp_ji_1(m)

        m = m_ji_1 + m_kj

        # Message Passing 2
        m_jj: torch.Tensor = self.mlp_jj(m)
        m_jj = m_jj * self.lin_rbf2(rbf)
        m_jj = m_jj[idx_jj] * self.mlp_sbf2(sbf2)
        m_jj = scatter(m_jj, idx_ji_2, dim=0, dim_size=m.size(0), reduce='add')

        m_ji_2: torch.Tensor = self.mlp_ji_2(m)

        m = m_ji_2 + m_jj

        # Aggregation
        m = self.lin_rbf_out(rbf) * m
        node_features = scatter(m,
                                i,
                                dim=0,
                                dim_size=node_features.size(0),
                                reduce='add')

        # Update function f_u
        node_features = self.res1(node_features)
        node_features = self.mlp(node_features) + residual_node_features
        node_features = self.res2(node_features)
        node_features = self.res3(node_features)

        # Output Module
        out: torch.Tensor = self.out_mlp(node_features)
        output: torch.Tensor = self.out_W(out)

        return node_features, output


class MXMNetSphericalBasisLayer(torch.nn.Module):
    """It takes pairwise distances and angles between atoms as input and combines radial basis functions with spherical harmonic
    functions to generate a fixed-size representation that captures both radial and orientation information. This type of
    representation is commonly used in molecular modeling and simulations to capture the behavior of atoms and molecules in
    chemical systems.

    Inside the initialization, Bessel basis functions and real spherical harmonic functions are generated.
    The Bessel basis functions capture the radial information, and the spherical harmonic functions capture the orientation information.
    These functions are generated based on the provided num_spherical and num_radial parameters.

    Examples
    --------
    >>> dist = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> angle = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> idx_kj = torch.tensor([0, 1, 2, 3])
    >>> spherical_layer = MXMNetSphericalBasisLayer(envelope_exponent=2, num_spherical=2, num_radial=2, cutoff=2.0)
    >>> output = spherical_layer(dist, angle, idx_kj)
    >>> output.shape
    torch.Size([4, 4])
    """

    def __init__(self,
                 num_spherical: int,
                 num_radial: int,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        """Initialize the MXMNetSphericalBasisLayer.

        Parameters
        ----------
        num_spherical: int
            The number of spherical harmonic functions to use. These functions capture orientation information related to atom positions.
        num_radial: int
            The number of radial basis functions to use. These functions capture information about pairwise distances between atoms.
        cutoff: float, optional (default 5.0)
            The cutoff distance for the radial basis functions. It specifies the distance beyond which the interactions are ignored.
        envelope_exponent: int, optional (default 5)
            The exponent for the envelope function. It controls the degree of damping for the radial basis functions.
        """
        super(MXMNetSphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_spherical: int = num_spherical
        self.num_radial: int = num_radial
        self.cutoff: float = cutoff
        self.envelope: _MXMNetEnvelope = _MXMNetEnvelope(envelope_exponent)

        bessel_forms: List = bessel_basis(num_spherical, num_radial)
        sph_harm_forms: List[List[str]] = real_sph_harm(num_spherical)
        self.sph_funcs: List = []
        self.bessel_funcs: List = []
        x: Any
        theta: Any
        x, theta = sym.symbols('x theta')
        modules: Dict = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1: Any = sym.lambdify([theta], sph_harm_forms[i][0],
                                         modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph: Any = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel: Any = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist: torch.Tensor, angle: torch.Tensor,
                idx_kj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MXMNetSphericalBasisLayer.

        Parameters
        ----------
        dist: torch.Tensor
            Input tensor representing pairwise distances between atoms.
        angle: torch.Tensor
            Input tensor representing pairwise angles between atoms.
        idx_kj: torch.Tensor
            Tensor containing indices for the k and j atoms.

        Returns
        -------
        output: torch.Tensor
            The output tensor containing the fixed-size representation.
        """
        dist = dist / self.cutoff
        rbf: torch.Tensor = torch.stack([f(dist) for f in self.bessel_funcs],
                                        dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf: torch.Tensor = torch.stack([f(angle) for f in self.sph_funcs],
                                        dim=1)
        n: int = self.num_spherical
        k: int = self.num_radial

        output: torch.Tensor = (rbf[idx_kj].view(-1, n, k) *
                                cbf.view(-1, n, 1)).view(-1, n * k)
        return output


class HighwayLayer(torch.nn.Module):
    """
    Highway layer from "Training Very Deep Networks" [1]

    y = H(x) * T(x) + x * C(x), where

    H(x): 1-layer neural network with non-linear activation
    T(x): 1-layer neural network with sigmoid activation
    C(X): 1 - T(X); As per the original paper

    The output will be of the same dimension as the input

    References
    ----------
    .. [1] Srivastava et al., "Training Very Deep Networks".https://arxiv.org/abs/1507.06228

    Examples
    --------
    >>> x = torch.randn(16, 20)
    >>> highway_layer = HighwayLayer(d_input=x.shape[1])
    >>> y = highway_layer(x)
    >>> x.shape
    torch.Size([16, 20])
    >>> y.shape
    torch.Size([16, 20])
    """

    def __init__(self,
                 d_input: int,
                 activation_fn: Union[Callable, str] = 'relu'):
        """
        Initializes the HighwayLayer.

        Parameters
        ----------
            d_input: int
                the dimension of the input layer
            activation_fn: str
                the activation function to use for H(x)
        """

        super(HighwayLayer, self).__init__()
        self.d_input = d_input
        self.activation_fn = get_activation(activation_fn)
        self.sigmoid_fn = get_activation('sigmoid')

        self.H = nn.Linear(d_input, d_input)
        self.T = nn.Linear(d_input, d_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HighwayLayer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of dimension (,input_dim).

        Returns
        -------
        output: torch.Tensor
            Output tensor of dimension (,input_dim)
        """

        H_out = self.activation_fn(self.H(x))
        T_out = self.sigmoid_fn(self.T(x))
        output = H_out * T_out + x * (1 - T_out)

        return output


class GraphConv(nn.Module):
    """Graph Convolutional Layers

    This layer implements the graph convolution introduced in [1]_.  The graph
    convolution combines per-node feature vectures in a nonlinear fashion with
    the feature vectors for neighboring nodes.  This "blends" information in
    local neighborhoods of a graph.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import deepchem.models.torch_models.layers as torch_layers
    >>> out_channels = 2
    >>> n_atoms = 4  # In CCC and C, there are 4 atoms
    >>> raw_smiles = ['CCC', 'C']
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    >>> featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    >>> mols = featurizer.featurize(mols)
    >>> multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    >>> atom_features = torch.from_numpy(multi_mol.get_atom_features().astype(np.float32))
    >>> degree_slice = torch.from_numpy(multi_mol.deg_slice)
    >>> membership = torch.from_numpy(multi_mol.membership)
    >>> deg_adjs = [torch.from_numpy(i) for i in multi_mol.get_deg_adjacency_lists()[1:]]
    >>> args = [atom_features, degree_slice, membership] + deg_adjs
    >>> layer = torch_layers.GraphConv(out_channels, number_input_features=atom_features.shape[-1])
    >>> result = layer(args)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([4, 2])
    >>> num_deg = 2 * layer.max_degree + (1 - layer.min_degree)
    >>> num_deg
    21

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints."
        Advances in neural information processing systems. 2015. https://arxiv.org/abs/1509.09292

  """

    def __init__(self,
                 out_channel: int,
                 number_input_features: int,
                 min_deg: int = 0,
                 max_deg: int = 10,
                 activation_fn: Optional[Callable] = None,
                 **kwargs):
        """Initialize a graph convolutional layer.

        Parameters
        ----------
        out_channel: int
            The number of output channels per graph node.
        number_input_features: int
            The number of input features.
        min_deg: int, optional (default 0)
            The minimum allowed degree for each graph node.
        max_deg: int, optional (default 10)
            The maximum allowed degree for each graph node. Note that this
            is set to 10 to handle complex molecules (some organometallic
            compounds have strange structures). If you're using this for
            non-molecular applications, you may need to set this much higher
            depending on your dataset.
        activation_fn: function
            A nonlinear activation function to apply. If you're not sure,
            `torch.nn.ReLU` is probably a good default for your application.
        """
        super(GraphConv, self).__init__(**kwargs)
        self.out_channel: int = out_channel
        self.min_degree: int = min_deg
        self.max_degree: int = max_deg
        self.number_input_features: int = number_input_features
        self.activation_fn: Optional[Callable] = activation_fn

        # Generate the nb_affine weights and biases
        num_deg: int = 2 * self.max_degree + (1 - self.min_degree)
        self.W_list: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                getattr(initializers,
                        'xavier_uniform_')(torch.empty(number_input_features,
                                                       self.out_channel)))
            for k in range(num_deg)
        ])
        self.b_list: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                getattr(initializers, 'zeros_')(torch.empty(self.out_channel,)))
            for k in range(num_deg)
        ])
        self.built = True

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
        -------
        str: A string that contains the class name followed by the values of its instance variable.
        """
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(out_channel:{self.out_channel},min_deg:{self.min_deg},max_deg:{self.max_deg},activation_fn:{self.activation_fn})'
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        The forward pass combines per-node feature vectors in a nonlinear fashion with
        the feature vectors for neighboring nodes.
        Parameters
        ----------
        inputs: List[torch.Tensor]
        Should contain atom features and arrays describing graph topology

        Returns:
        -------
        torch.Tensor
          Combined atom features
        """

        # Extract atom_features
        atom_features: torch.Tensor = inputs[0]

        # Extract graph topology
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        W = iter(self.W_list)
        b = iter(self.b_list)

        # Sum all neighbors using adjacency matrix
        deg_summed: List[np.ndarray] = self.sum_neigh(atom_features,
                                                      deg_adj_lists)

        # Get collection of modified atom features
        new_rel_atoms_collection = []

        split_features: Tuple[torch.Tensor,
                              ...] = torch.split(atom_features,
                                                 (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            # Obtain relevant atoms for this degree
            rel_atoms: torch.Tensor = torch.from_numpy(deg_summed[deg - 1])

            # Get self atoms
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            # Apply hidden affine to relevant atoms and append
            rel_out: torch.Tensor = torch.matmul(rel_atoms.type(torch.float32),
                                                 next(W)) + next(b)
            self_out: torch.Tensor = torch.matmul(
                self_atoms.type(torch.float32), next(W)) + next(b)
            out: torch.Tensor = rel_out + self_out
            new_rel_atoms_collection.append(
                torch.from_numpy(out.detach().numpy()))

        # Determine the min_deg=0 case
        if self.min_degree == 0:
            self_atoms = split_features[0]

            # Only use the self layer
            out = torch.matmul(self_atoms.type(torch.float32),
                               next(W)) + next(b)
            new_rel_atoms_collection.insert(
                0, torch.from_numpy(out.detach().numpy()))

        # Combine all atoms back into the list
        atom_features = torch.concat(new_rel_atoms_collection, 0)

        if self.activation_fn is not None:
            atom_features = self.activation_fn(atom_features)

        return atom_features

    def sum_neigh(self, atoms: torch.Tensor, deg_adj_lists) -> List[np.ndarray]:
        """Store the summed atoms by degree"""
        deg_summed = []

        for deg in range(1, self.max_degree + 1):
            gathered_atoms: torch.Tensor = atoms[deg_adj_lists[deg - 1]]
            # Sum along neighbors as well as self, and store
            summed_atoms: torch.Tensor = torch.sum(gathered_atoms, 1)
            deg_summed.append(summed_atoms.detach().numpy())

        return deg_summed


class GraphPool(nn.Module):
    """A GraphPool gathers data from local neighborhoods of a graph.

    This layer does a max-pooling over the feature vectors of atoms in a
    neighborhood. You can think of this layer as analogous to a max-pooling
    layer for 2D convolutions but which operates on graphs instead. This
    technique is described in [1]_.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import deepchem.models.torch_models.layers as torch_layers
    >>> n_atoms = 4  # In CCC and C, there are 4 atoms
    >>> raw_smiles = ['CCC', 'C']
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    >>> featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    >>> mols = featurizer.featurize(mols)
    >>> multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    >>> atom_features = torch.from_numpy(multi_mol.get_atom_features().astype(np.float32))
    >>> degree_slice = torch.from_numpy(multi_mol.deg_slice)
    >>> membership = torch.from_numpy(multi_mol.membership)
    >>> deg_adjs = [torch.from_numpy(i) for i in multi_mol.get_deg_adjacency_lists()[1:]]
    >>> args = [atom_features, degree_slice, membership] + deg_adjs
    >>> result = torch_layers.GraphPool()(args)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([4, 75])

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015. https://arxiv.org/abs/1509.09292

    """

    def __init__(self, min_degree: int = 0, max_degree: int = 10, **kwargs):
        """Initialize this layer

        Parameters
        ----------
        min_deg: int, optional (default 0)
            The minimum allowed degree for each graph node.
        max_deg: int, optional (default 10)
            The maximum allowed degree for each graph node. Note that this
            is set to 10 to handle complex molecules (some organometallic
            compounds have strange structures). If you're using this for
            non-molecular applications, you may need to set this much higher
            depending on your dataset.
        """
        super(GraphPool, self).__init__(**kwargs)
        self.min_degree: int = min_degree
        self.max_degree: int = max_degree

    def get_config(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
        -------
        str: A string that contains the class name followed by the values of its instance variable.
        """
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(min_degree:{self.min_degree},max_degree:{self.max_degree})'
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        The forward pass performs max-pooling over the feature vectors of atoms in a neighborhood.

        Parameters
        ----------
        inputs: List[np.ndarray]
        Should contain atom features and arrays describing graph topology.

        Returns:
        -------
        torch.Tensor
        """
        atom_features: torch.Tensor = inputs[0]
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        # Perform the mol gather
        deg_maxed = []

        split_features: Tuple[torch.Tensor,
                              ...] = torch.split(atom_features,
                                                 (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            # Get self atoms
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            if deg_adj_lists[deg - 1].shape[0] == 0:
                # There are no neighbors of this degree, so just create an empty tensor directly.
                maxed_atoms: torch.Tensor = torch.zeros(
                    (0, self_atoms.shape[-1]))
                deg_maxed.append(maxed_atoms)
            else:
                # Expand dims
                self_atoms = torch.unsqueeze(self_atoms, 1)

                # always deg-1 for deg_adj_lists
                gathered_atoms: torch.Tensor = atom_features[deg_adj_lists[deg -
                                                                           1]]
                gathered_atoms = torch.concat([self_atoms, gathered_atoms], 1)

                max_atoms: tuple = torch.max(gathered_atoms, 1)
                deg_maxed.append(max_atoms[0])

        if self.min_degree == 0:
            self_atoms = split_features[0]
            deg_maxed.insert(0, self_atoms)

        return torch.concat(deg_maxed, 0)


class GraphGather(nn.Module):
    """A GraphGather layer pools node-level feature vectors to create a graph feature vector.

    Many graph convolutional networks manipulate feature vectors per
    graph-node. For a molecule for example, each node might represent an
    atom, and the network would manipulate atomic feature vectors that
    summarize the local chemistry of the atom. However, at the end of
    the application, we will likely want to work with a molecule level
    feature representation. The `GraphGather` layer creates a graph level
    feature vector by combining all the node-level feature vectors.

    One subtlety about this layer is that it depends on the
    `batch_size`. This is done for internal implementation reasons. The
    `GraphConv`, and `GraphPool` layers pool all nodes from all graphs
    in a batch that's being processed. The `GraphGather` reassembles
    these jumbled node feature vectors into per-graph feature vectors.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import deepchem.models.torch_models.layers as torch_layers
    >>> batch_size = 2
    >>> raw_smiles = ['CCC', 'C']
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    >>> featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    >>> mols = featurizer.featurize(mols)
    >>> multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    >>> atom_features = torch.from_numpy(multi_mol.get_atom_features().astype(np.float32))
    >>> degree_slice = torch.from_numpy(multi_mol.deg_slice)
    >>> membership = torch.from_numpy(multi_mol.membership)
    >>> deg_adjs = [torch.from_numpy(i) for i in multi_mol.get_deg_adjacency_lists()[1:]]
    >>> args = [atom_features, degree_slice, membership] + deg_adjs
    >>> result = torch_layers.GraphGather(batch_size)(args)
    >>> type(result)
    <class 'torch.Tensor'>
    >>> result.shape
    torch.Size([2, 150])

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015. https://arxiv.org/abs/1509.09292
    """

    def __init__(self,
                 batch_size: int,
                 activation_fn: Optional[Callable] = None,
                 **kwargs):
        """Initialize this layer.

        Parameters
        ---------
        batch_size: int
            The batch size for this layer. Note that the layer's behavior
            changes depending on the batch size.
        activation_fn: function
            A nonlinear activation function to apply. If you're not sure,
            `relu` is probably a good default for your application.
        """

        super(GraphGather, self).__init__(**kwargs)
        self.batch_size: int = batch_size
        self.activation_fn: Optional[Callable] = activation_fn

    def get_config(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
        -------
        str: A string that contains the class name followed by the values of its instance variable.
        """
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(batch_size:{self.batch_size},activation_fn:{self.activation_fn})'
        )

    def forward(self, inputs: List[torch.Tensor]):
        """Invoking this layer.

        Parameters
        ----------
        inputs: List[torch.Tensor]
            This list should consist of `inputs = [atom_features, deg_slice,
            membership, deg_adj_list placeholders...]`. These are all
            tensors that are created/process by `GraphConv` and `GraphPool`

        Returns:
        -------
        torch.Tensor
        """
        atom_features: torch.Tensor = inputs[0]

        # Extract graph topology
        membership: torch.Tensor = inputs[2].to(torch.int64)

        assert self.batch_size > 1, "graph_gather requires batches larger than 1"

        sparse_reps: torch.Tensor = unsorted_segment_sum(
            atom_features, membership, self.batch_size)
        max_reps: torch.Tensor = unsorted_segment_max(atom_features, membership,
                                                      self.batch_size)
        mol_features: torch.Tensor = torch.concat([sparse_reps, max_reps], 1)

        if self.activation_fn is not None:
            mol_features = self.activation_fn(mol_features)
        return mol_features


class EquivariantLinear(nn.Module):
    """
    An equivariant linear layer for transforming feature tensors.

    This layer is designed for 3D atomic or molecular data, handling per-atom features
    such as charges or embeddings. It ensures transformations respect equivariance
    properties, making it suitable for tasks involving atomic coordinates and related
    features.

    Parameters
    ----------
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.

    Example
    -------
    >>> layer = EquivariantLinear(4, 8)
    >>> x = torch.randn(3, 4)  # Default dtype is torch.float32
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 8])
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the equivariant linear layer.

        Parameters
        ----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.
        """
        super(EquivariantLinear, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a linear transformation.

        Parameters
        ----------
        x: torch.Tensor
            Input feature tensor of shape `(B, N, in_features)`.

        Returns
        -------
        output: torch.Tensor
            Transformed feature tensor of shape `(B, N, out_features)`.
        """
        # Current implementation work for int features (dist)
        output = torch.matmul(x, self.weight) + self.bias
        return output


class SphericalHarmonics:
    """
    Custom computation of spherical harmonics up to a specified degree.

    Spherical harmonics are implemented to capture rotationally equivariant features
    based on interatomic relative positions.

    Parameters
    ----------
    max_degree: int
        Maximum degree of the spherical harmonics.

    Example
    -------
    >>> sh = SphericalHarmonics(max_degree=2)
    >>> relative_positions = torch.randn(3, 5, 5, 3)
    >>> result = sh.compute(relative_positions)
    >>> result.shape
    torch.Size([3, 5, 5, 9])
    """

    def __init__(self, max_degree: int) -> None:
        """
        Initialize the custom spherical harmonics calculator.

        Parameters
        ----------
        max_degree: int
            Maximum degree of the spherical harmonics.
        """
        self.max_degree = max_degree

    def compute_legendre_polynomials(self, l: int, m: int,
                                     x: torch.Tensor) -> torch.Tensor:
        """
        Compute the associated Legendre polynomial.

        Parameters
        ----------
        l: int
            Degree of the polynomial.
        m: int
            Order of the polynomial.
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Computed Legendre polynomial values.

        Example
        -------
        >>> sh = SphericalHarmonics(max_degree=2)
        >>> x = torch.tensor(0.5)
        >>> sh.compute_legendre_polynomials(1, 0, x)
        tensor(0.5000)
        """
        l_tensor = torch.tensor(l, dtype=x.dtype, device=x.device)
        m_tensor = torch.tensor(m, dtype=x.dtype, device=x.device)

        if m == 0:
            return (x**l_tensor)
        elif m > 0:
            return ((1 - x**2).sqrt()**
                    m_tensor) * self.compute_legendre_polynomials(l, m - 1, x)
        else:
            return (-1)**m * self.compute_legendre_polynomials(l, -m, x)

    def compute_spherical_harmonics(self, l: int, m: int, theta: torch.Tensor,
                                    phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the spherical harmonics Y_l^m(theta, phi).

        Parameters
        ----------
        l: int
            Degree of the spherical harmonics.
        m: int
            Order of the spherical harmonics.
        theta: torch.Tensor
            Polar angles in radians.
        phi: torch.Tensor
            Azimuthal angles in radians.

        Returns
        -------
        torch.Tensor
            Spherical harmonics values.

        Example
        -------
        >>> sh = SphericalHarmonics(max_degree=2)
        >>> theta = torch.tensor(0.5)
        >>> phi = torch.tensor(1.0)
        >>> sh.compute_spherical_harmonics(1, 0, theta, phi)
        tensor(0.4288+0.j)
        """
        l_tensor = torch.tensor(l, dtype=theta.dtype, device=theta.device)
        m_tensor = torch.tensor(m, dtype=theta.dtype, device=theta.device)

        legendre = self.compute_legendre_polynomials(l, m, torch.cos(theta))

        normalization = torch.sqrt(
            (2 * l_tensor + 1) /
            (4 * torch.tensor(math.pi, dtype=theta.dtype, device=theta.device))
            * torch.exp(
                torch.lgamma(l_tensor - torch.abs(m_tensor) + 1) -
                torch.lgamma(l_tensor + torch.abs(m_tensor) + 1)))
        return normalization * legendre * torch.exp(1j * m_tensor * phi)

    def compute(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute all spherical harmonics for relative positions.

        Parameters
        ----------
        relative_positions: torch.Tensor
            Tensor of shape `(B, N, N, 3)` representing relative positions.

        Returns
        -------
        torch.Tensor
            Spherical harmonics tensor of shape `(B, N, N, SH_dim)`.

        Example
        -------
        >>> sh = SphericalHarmonics(max_degree=1)
        >>> rel_positions = torch.randn(1, 3, 3, 3)
        >>> sh.compute(rel_positions).shape
        torch.Size([1, 3, 3, 4])
        """
        r = relative_positions.norm(dim=-1, keepdim=True) + 1e-6
        theta = torch.acos(
            torch.clamp(relative_positions[..., 2] / r.squeeze(-1), -1.0, 1.0))
        phi = torch.atan2(relative_positions[..., 1], relative_positions[...,
                                                                         0])

        spherical_harmonics = []
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                sh_lm = self.compute_spherical_harmonics(l, m, theta, phi)
                spherical_harmonics.append(sh_lm.real)

        return torch.stack(spherical_harmonics,
                           dim=-1).reshape(*relative_positions.shape[:-1], -1)


class SE3Attention(nn.Module):
    """
    SE(3) Attention Module with Spherical Harmonics.
    
    This module is designed for 3D atomic or molecular data, using spherical harmonics
    to compute rotationally equivariant attention based on interatomic distances and
    relative positions. It ensures SE(3)-equivariance for both feature and coordinate updates.

    Parameters
    ----------
    embed_dim: int
        Dimensionality of feature embeddings.
    num_heads: int
        Number of attention heads.
    sh_max_degree: int
        Maximum degree of spherical harmonics.

    Example
    -------
    >>> layer = SE3Attention(embed_dim=64, num_heads=4, sh_max_degree=2)
    >>> x = torch.randn(1, 10, 64)  # Default dtype torch.float32
    >>> coords = torch.randn(1, 10, 3)  # Default dtype torch.float32
    >>> features, coords = layer(x, coords)
    >>> features.shape, coords.shape
    (torch.Size([1, 10, 64]), torch.Size([1, 10, 3]))
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 sh_max_degree: int = 2) -> None:
        """
        Initialize the SE(3) Attention Module.

        Parameters
        ----------
        embed_dim: int
            Dimensionality of feature embeddings.
        num_heads: int
            Number of attention heads.
        sh_max_degree: int, optional
            Maximum degree of spherical harmonics. Default is 2.
        """
        super(SE3Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sh_max_degree = sh_max_degree

        # Linear transformations
        self.query = EquivariantLinear(embed_dim, embed_dim)
        self.key = EquivariantLinear(embed_dim, embed_dim)
        self.value = EquivariantLinear(embed_dim, embed_dim)
        self.out = EquivariantLinear(embed_dim, embed_dim)
        self.coord_linear = EquivariantLinear(embed_dim, 3)

        # Spherical harmonics
        self.sh_computer = SphericalHarmonics(max_degree=sh_max_degree)
        self.sh_projection = nn.Linear((sh_max_degree + 1)**2,
                                       embed_dim // num_heads)

    def compute_spherical_harmonics(
            self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distances and spherical harmonics for relative positions.

        Parameters
        ----------
        coords: torch.Tensor
            Input coordinates tensor of shape `(B, N, 3)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Pairwise distances of shape `(B, N, N, 1)` and spherical harmonics of shape `(B, N, N, SH_dim)`.

        Example
        -------
        >>> coords = torch.randn(1, 10, 3)
        >>> layer = SE3Attention(embed_dim=64, num_heads=4, sh_max_degree=2)
        >>> dist, sh = layer.compute_spherical_harmonics(coords)
        >>> dist.shape, sh.shape
        (torch.Size([1, 10, 10, 1]), torch.Size([1, 10, 10, 16]))
        """
        relative_positions = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = relative_positions.norm(dim=-1, keepdim=True)
        sh = self.sh_computer.compute(relative_positions)

        sh = self.sh_projection(sh)
        return dist, sh

    def forward(self, x: torch.Tensor,
                coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention computation and coordinate updates.

        Parameters
        ----------
        x: torch.Tensor
            Input feature tensor of shape `(B, N, embed_dim)`.
        coords: torch.Tensor
            Input coordinate tensor of shape `(B, N, 3)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated feature tensor of shape `(B, N, embed_dim)` and updated coordinate tensor of shape `(B, N, 3)`.

        Example
        -------
        >>> layer = SE3Attention(embed_dim=64, num_heads=4, sh_max_degree=2)
        >>> x = torch.randn(1, 10, 64)
        >>> coords = torch.randn(1, 10, 3)
        >>> features, coords = layer(x, coords)
        >>> features.shape, coords.shape
        (torch.Size([1, 10, 64]), torch.Size([1, 10, 3]))
        """
        dist, sh = self.compute_spherical_harmonics(coords)

        B, N, _ = x.shape
        q = self.query(x).view(B, N, self.num_heads, -1)
        k = self.key(x).view(B, N, self.num_heads, -1)
        v = self.value(x).view(B, N, self.num_heads, -1)

        # attention weights
        attn_weights = torch.einsum('bnm,bnhd,bmhd->bhnm', dist.squeeze(-1), q,
                                    k)
        sh_weights = torch.einsum('bnmd,bnhd->bhnm', sh, q)
        attn_weights += sh_weights

        attn_weights = F.softmax(attn_weights, dim=-1)

        context = torch.einsum('bhnm,bmhd->bnhd', attn_weights,
                               v).reshape(B, N, -1)
        x = self.out(context)

        # update coordinates
        coords_update = self.coord_linear(context)
        coords_update = coords_update / (
            coords_update.norm(dim=-1, keepdim=True) + 1e-6)
        coords = coords + 0.01 * coords_update

        return x, coords


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity (inner product) between two tensors.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of shape `(B, N, P)` representing the first set of vectors.
    y: torch.Tensor
        Input tensor of shape `(B, M, P)` representing the second set of vectors.

    Returns
    -------
    torch.Tensor
        Cosine similarity tensor of shape `(B, N, M)` where each entry represents the cosine similarity
        between vectors from `x` and `y`.

    Examples
    --------
    The cosine similarity between two equivalent vectors will be 1. The cosine
    similarity between two equivalent tensors (tensors where all the elements are
    the same) will be a tensor of 1s. In this scenario, if the input tensors `x` and
    `y` are each of shape `(n,p)`, where each element in `x` and `y` is the same, then
    the output tensor would be a tensor of shape `(n,n)` with 1 in every entry.

    >>> import deepchem as dc
    >>> import numpy as np
    >>> import deepchem.models.torch_models.layers as torch_layers
    >>> x = torch.ones((6, 4), dtype=torch.float32)
    >>> y_same = torch.ones((6, 4), dtype=torch.float32)
    >>> cos_sim_same = torch_layers.cosine_dist(x, y_same)

    `x` and `y_same` are the same tensor (equivalent at every element, in this
    case 1). As such, the pairwise inner product of the rows in `x` and `y` will
    always be 1. The output tensor will be of shape (6,6).

    >>> diff = cos_sim_same - torch.ones((6, 6), dtype=torch.float32)
    >>> np.allclose(0.0, diff.sum().item(), atol=1e-05)
    True
    >>> cos_sim_same.shape
    torch.Size([6, 6])

    The cosine similarity between two orthogonal vectors will be 0 (by definition).
    If every row in `x` is orthogonal to every row in `y`, then the output will be a
    tensor of 0s. In the following example, each row in the tensor `x1` is orthogonal
    to each row in `x2` because they are halves of an identity matrix.

    >>> identity_tensor = torch.eye(512, dtype=torch.float32)
    >>> x1 = identity_tensor[0:256,:]
    >>> x2 = identity_tensor[256:512,:]
    >>> cos_sim_orth = torch_layers.cosine_dist(x1, x2)

    Each row in `x1` is orthogonal to each row in `x2`. As such, the pairwise inner
    product of the rows in `x1` and `x2` will always be 0. Furthermore, because the
    shape of the input tensors are both of shape `(256,512)`, the output tensor will
    be of shape `(256,256)`.

    >>> np.allclose(0.0, cos_sim_orth.sum().item(), atol=1e-05)
    True
    >>> cos_sim_orth.shape
    torch.Size([256, 256])

    """

    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(-1, -2))
    return cosine_similarity

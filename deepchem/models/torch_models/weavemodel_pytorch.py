import numpy as np
from collections.abc import Sequence as SequenceCollection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
from typing import List, Tuple, Iterable, Optional, Callable, Union, Sequence
from deepchem.data import Dataset
from deepchem.metrics import to_one_hot
from deepchem.utils.typing import OneOrMany, ActivationFn
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel
import deepchem.models.torch_models.layers as torch_layers
from deepchem.utils.pytorch_utils import get_activation


class Weave(nn.Module):
    """
    A graph convolutional network(GCN) for either classification or regression.
    The network consists of the following sequence of layers:

    - Weave feature modules

    - Final convolution

    - Weave Gather Layer

    - A fully connected layer

    - A Softmax layer

    Example
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> featurizer = dc.feat.WeaveFeaturizer()
    >>> X = featurizer(["C", "CC"])
    >>> y = np.array([1, 0])
    >>> batch_size = 2
    >>> weavemodel = dc.models.torch_models.WeaveModel(n_tasks=1,n_weave=2, fully_connected_layer_sizes=[2000, 1000],mode="classification",batch_size=batch_size)
    >>> atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = weavemodel.compute_features_on_batch(X)
    >>> model = dc.models.torch_models.Weave(n_tasks=1,n_weave=2,fully_connected_layer_sizes=[2000, 1000],mode="classification")
    >>> input_data = [atom_feat, pair_feat, pair_split, atom_split, atom_to_pair]
    >>> output = model(input_data)

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.
    """

    def __init__(
        self,
        n_tasks: int,
        n_atom_feat: OneOrMany[int] = 75,
        n_pair_feat: OneOrMany[int] = 14,
        n_hidden: int = 50,
        n_graph_feat: int = 128,
        n_weave: int = 2,
        fully_connected_layer_sizes: List[int] = [2000, 100],
        conv_weight_init_stddevs: OneOrMany[float] = 0.03,
        weight_init_stddevs: OneOrMany[float] = 0.01,
        bias_init_consts: OneOrMany[float] = 0.0,
        dropouts: OneOrMany[float] = 0.25,
        final_conv_activation_fn=F.tanh,
        activation_fns: OneOrMany[ActivationFn] = 'relu',
        batch_normalize: bool = True,
        gaussian_expand: bool = True,
        compress_post_gaussian_expansion: bool = False,
        mode: str = "classification",
        n_classes: int = 2,
        batch_size: int = 100,
    ):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_atom_feat: int, optional (default 75)
            Number of features per atom. Note this is 75 by default and should be 78
            if chirality is used by `WeaveFeaturizer`.
        n_pair_feat: int, optional (default 14)
            Number of features per pair of atoms.
        n_hidden: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_graph_feat: int, optional (default 128)
            Number of output features for each molecule(graph)
        n_weave: int, optional (default 2)
            The number of weave layers in this model.
        fully_connected_layer_sizes: list (default `[2000, 100]`)
            The size of each dense layer in the network.  The length of
            this list determines the number of layers.
        conv_weight_init_stddevs: list or float (default 0.03)
            The standard deviation of the distribution to use for weight
            initialization of each convolutional layer. The length of this lisst
            should equal `n_weave`. Alternatively, this may be a single value instead
            of a list, in which case the same value is used for each layer.
        weight_init_stddevs: list or float (default 0.01)
            The standard deviation of the distribution to use for weight
            initialization of each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float (default 0.0)
            The value to initialize the biases in each fully connected layer.  The
            length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in
            which case the same value is used for every layer.
        dropouts: list or float (default 0.25)
            The dropout probablity to use for each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        final_conv_activation_fn: Optional[ActivationFn] (default `F.tanh`)
            The activation funcntion to apply to the final
            convolution at the end of the weave convolutions. If `None`, then no
            activate is applied (hence linear).
        activation_fns: str (default `relu`)
            The activation function to apply to each fully connected layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a
            single value instead of a list, in which case the same value is used for
            every layer.
        batch_normalize: bool, optional (default True)
            If this is turned on, apply batch normalization before applying
            activation functions on convolutional and fully connected layers.
        gaussian_expand: boolean, optional (default True)
            Whether to expand each dimension of atomic features by gaussian
            histogram
        compress_post_gaussian_expansion: bool, optional (default False)
            If True, compress the results of the Gaussian expansion back to the
            original dimensions of the input.
        mode: str (default "classification")
            Either "classification" or "regression" for type of model.
        n_classes: int (default 2)
            Number of classes to predict (only used in classification mode)
        batch_size: int (default 100)
            Batch size used by this model for training.
        """
        super(Weave, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        if not isinstance(n_atom_feat, SequenceCollection):
            n_atom_feat = [n_atom_feat] * n_weave
        if not isinstance(n_pair_feat, SequenceCollection):
            n_pair_feat = [n_pair_feat] * n_weave
        n_layers = len(fully_connected_layer_sizes)
        if not isinstance(conv_weight_init_stddevs, SequenceCollection):
            conv_weight_init_stddevs = [conv_weight_init_stddevs] * n_weave
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        self.n_tasks: int = n_tasks
        self.n_atom_feat: OneOrMany[int] = n_atom_feat
        self.n_pair_feat: OneOrMany[int] = n_pair_feat
        self.n_hidden: int = n_hidden
        self.n_graph_feat: int = n_graph_feat
        self.mode: str = mode
        self.n_classes: int = n_classes
        self.n_layers: int = n_layers
        self.fully_connected_layer_sizes: List[
            int] = fully_connected_layer_sizes
        self.weight_init_stddevs: OneOrMany[float] = weight_init_stddevs
        self.bias_init_consts: OneOrMany[float] = bias_init_consts
        self.dropouts: Sequence[float] = dropouts
        self.activation_fns: OneOrMany[ActivationFn] = [
            get_activation(i) for i in activation_fns
        ]
        self.batch_normalize: bool = batch_normalize
        self.n_weave: int = n_weave

        torch.manual_seed(22)
        self.layers: nn.ModuleList = nn.ModuleList()
        for ind in range(n_weave):
            n_atom: int = self.n_atom_feat[ind]
            n_pair: int = self.n_pair_feat[ind]
            if ind < n_weave - 1:
                n_atom_next: int = self.n_atom_feat[ind + 1]
                n_pair_next: int = self.n_pair_feat[ind + 1]
            else:
                n_atom_next = n_hidden
                n_pair_next = n_hidden
            weave_layer = torch_layers.WeaveLayer(
                n_atom_input_feat=n_atom,
                n_pair_input_feat=n_pair,
                n_atom_output_feat=n_atom_next,
                n_pair_output_feat=n_pair_next,
                batch_normalize=batch_normalize)
            nn.init.trunc_normal_(weave_layer.W_AA,
                                  0,
                                  std=conv_weight_init_stddevs[ind])
            nn.init.trunc_normal_(weave_layer.W_PA,
                                  0,
                                  std=conv_weight_init_stddevs[ind])
            nn.init.trunc_normal_(weave_layer.W_A,
                                  0,
                                  std=conv_weight_init_stddevs[ind])
            if weave_layer.update_pair:
                nn.init.trunc_normal_(weave_layer.W_AP,
                                      0,
                                      std=conv_weight_init_stddevs[ind])
                nn.init.trunc_normal_(weave_layer.W_PP,
                                      0,
                                      std=conv_weight_init_stddevs[ind])
                nn.init.trunc_normal_(weave_layer.W_P,
                                      0,
                                      std=conv_weight_init_stddevs[ind])
            self.layers.append(weave_layer)

        self.dense1: nn.Linear = nn.Linear(n_hidden, self.n_graph_feat)
        self.dense1_act = final_conv_activation_fn
        self.dense1_bn: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=self.n_graph_feat,
            eps=1e-3,
            momentum=0.99,
            affine=True,
            track_running_stats=True)

        self.weave_gather = torch_layers.WeaveGather(
            batch_size,
            n_input=self.n_graph_feat,
            gaussian_expand=gaussian_expand,
            compress_post_gaussian_expansion=compress_post_gaussian_expansion)

        if n_layers > 0:
            self.layers2: nn.ModuleList = nn.ModuleList()
            in_size = self.n_graph_feat * 11
            for ind, layer_size, weight_stddev, bias_const, dropout, activation_fn in zip(
                [0, 1], fully_connected_layer_sizes, weight_init_stddevs,
                    bias_init_consts, dropouts, self.activation_fns):
                self.layer: nn.Linear = nn.Linear(in_size, layer_size)
                nn.init.trunc_normal_(self.layer.weight, 0, std=weight_stddev)
                if self.layer.bias is not None:
                    self.layer.bias = nn.Parameter(
                        torch.full(self.layer.bias.shape, bias_const))
                self.layer.layer_bn = nn.BatchNorm1d(num_features=layer_size,
                                                     eps=1e-3,
                                                     momentum=0.99,
                                                     affine=True,
                                                     track_running_stats=True)
                self.layer.weight_stddev = weight_stddev
                self.layer.bias_const = bias_const
                self.layer.dropout = nn.Dropout(dropout)
                self.layer.layer_act = activation_fn
                self.layers2.append(self.layer)
                in_size = layer_size

        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            self.layer_2 = nn.Linear(fully_connected_layer_sizes[1],
                                     n_tasks * n_classes)

        else:
            self.layer_2 = nn.Linear(fully_connected_layer_sizes[1], n_tasks)

    def forward(self, inputs: OneOrMany[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        inputs: OneOrMany[torch.Tensor]
            Should contain 5 tensors [atom_features, pair_features, pair_split, atom_split, atom_to_pair]

        Returns
        -------
        List[torch.Tensor]
            Output as per use case : regression/classification
        """
        input1: List[np.ndarray] = [
            np.array(inputs[0]),
            np.array(inputs[1]),
            np.array(inputs[2]),
            np.array(inputs[4])
        ]
        for ind in range(self.n_weave):
            weave_layer_ind_A, weave_layer_ind_P = self.layers[ind](input1)
            input1 = [
                weave_layer_ind_A, weave_layer_ind_P,
                np.array(inputs[2]),
                np.array(inputs[4])
            ]

        dense1: torch.Tensor = self.dense1(weave_layer_ind_A)
        dense1 = self.dense1_act(dense1)
        if self.batch_normalize:
            self.dense1_bn.eval()
            dense1 = self.dense1_bn(dense1)

        weave_gather: torch.Tensor = self.weave_gather([dense1, inputs[3]])
        if self.n_layers > 0:
            input_layer: torch.Tensor = weave_gather
            for ind, dropout in zip([0, 1], self.dropouts):
                dense2 = self.layers2[ind]
                layer = self.layers2[ind](input_layer)
                if dropout > 0.0:
                    dense2.dropout.eval()
                    layer = dense2.dropout(layer)
                if self.batch_normalize:
                    dense2.layer_bn.eval()
                    layer = dense2.layer_bn(layer)
                layer = dense2.layer_act(layer)
                input_layer = layer
            output: torch.Tensor = input_layer
        else:
            output = weave_gather

        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits: torch.Tensor = torch.reshape(self.layer_2(output),
                                                 (-1, n_tasks, n_classes))
            output = F.softmax(logits, dim=2)
            outputs: List[torch.Tensor] = [output, logits]
        else:
            output = self.layer_2(output)
            outputs = [output]

        return outputs


class WeaveModel(TorchModel):
    """Implements Google-style Weave Graph Convolutions

    This model implements the Weave style graph convolutions
    from [1]_.

    The biggest difference between WeaveModel style convolutions
    and GraphConvModel style convolutions is that Weave
    convolutions model bond features explicitly. This has the
    side effect that it needs to construct a NxN matrix
    explicitly to model bond interactions. This may cause
    scaling issues, but may possibly allow for better modeling
    of subtle bond effects.

    Note that [1]_ introduces a whole variety of different architectures for
    Weave models. The default settings in this class correspond to the W2N2
    variant from [1]_ which is the most commonly used variant..

    Examples
    --------

    Here's an example of how to fit a `WeaveModel` on a tiny sample dataset.

    >>> import numpy as np
    >>> import deepchem as dc
    >>> featurizer = dc.feat.WeaveFeaturizer()
    >>> X = featurizer(["C", "CC"])
    >>> y = np.array([1, 0])
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> model = dc.models.torch_models.WeaveModel(n_tasks=1, n_weave=2, fully_connected_layer_sizes=[2000, 1000], mode="classification")
    >>> loss = model.fit(dataset)

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.

    """

    def __init__(self,
                 n_tasks: int,
                 n_atom_feat: OneOrMany[int] = 75,
                 n_pair_feat: OneOrMany[int] = 14,
                 n_hidden: int = 50,
                 n_graph_feat: int = 128,
                 n_weave: int = 2,
                 fully_connected_layer_sizes: List[int] = [2000, 100],
                 conv_weight_init_stddevs: OneOrMany[float] = 0.03,
                 weight_init_stddevs: OneOrMany[float] = 0.01,
                 bias_init_consts: OneOrMany[float] = 0.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = "l2",
                 dropouts: OneOrMany[float] = 0.25,
                 final_conv_activation_fn: Optional[ActivationFn] = F.tanh,
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 batch_normalize: bool = True,
                 gaussian_expand: bool = True,
                 compress_post_gaussian_expansion: bool = False,
                 mode: str = "classification",
                 n_classes: int = 2,
                 batch_size: int = 100,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_atom_feat: int, optional (default 75)
            Number of features per atom. Note this is 75 by default and should be 78
            if chirality is used by `WeaveFeaturizer`.
        n_pair_feat: int, optional (default 14)
            Number of features per pair of atoms.
        n_hidden: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_graph_feat: int, optional (default 128)
            Number of output features for each molecule(graph)
        n_weave: int, optional (default 2)
            The number of weave layers in this model.
        fully_connected_layer_sizes: list (default `[2000, 100]`)
            The size of each dense layer in the network.  The length of
            this list determines the number of layers.
        conv_weight_init_stddevs: list or float (default 0.03)
            The standard deviation of the distribution to use for weight
            initialization of each convolutional layer. The length of this lisst
            should equal `n_weave`. Alternatively, this may be a single value instead
            of a list, in which case the same value is used for each layer.
        weight_init_stddevs: list or float (default 0.01)
            The standard deviation of the distribution to use for weight
            initialization of each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float (default 0.0)
            The value to initialize the biases in each fully connected layer.  The
            length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in
            which case the same value is used for every layer.
        weight_decay_penalty: float (default 0.0)
            The magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str (default "l2")
            The type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float (default 0.25)
            The dropout probablity to use for each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        final_conv_activation_fn: Optional[ActivationFn] (default `F.tanh`)
            The activation funcntion to apply to the final
            convolution at the end of the weave convolutions. If `None`, then no
            activate is applied (hence linear).
        activation_fns: str (default `relu`)
            The activation function to apply to each fully connected layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a
            single value instead of a list, in which case the same value is used for
            every layer.
        batch_normalize: bool, optional (default True)
            If this is turned on, apply batch normalization before applying
            activation functions on convolutional and fully connected layers.
        gaussian_expand: boolean, optional (default True)
            Whether to expand each dimension of atomic features by gaussian
            histogram
        compress_post_gaussian_expansion: bool, optional (default False)
            If True, compress the results of the Gaussian expansion back to the
            original dimensions of the input.
        mode: str (default "classification")
            Either "classification" or "regression" for type of model.
        n_classes: int (default 2)
            Number of classes to predict (only used in classification mode)
        batch_size: int (default 100)
            Batch size used by this model for training.
        """
        self.mode: str = mode
        self.model = Weave(
            n_tasks=n_tasks,
            n_atom_feat=n_atom_feat,
            n_pair_feat=n_pair_feat,
            n_hidden=n_hidden,
            n_graph_feat=n_graph_feat,
            n_weave=n_weave,
            fully_connected_layer_sizes=fully_connected_layer_sizes,
            conv_weight_init_stddevs=conv_weight_init_stddevs,
            weight_init_stddevs=weight_init_stddevs,
            bias_init_consts=bias_init_consts,
            dropouts=dropouts,
            final_conv_activation_fn=final_conv_activation_fn,
            activation_fns=activation_fns,
            batch_normalize=batch_normalize,
            gaussian_expand=gaussian_expand,
            compress_post_gaussian_expansion=compress_post_gaussian_expansion,
            mode=mode,
            n_classes=n_classes,
            batch_size=batch_size)

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        regularization_loss: Optional[Callable]
        if weight_decay_penalty != 0.0:
            weights = [layer.weight for layer in self.model.layers2]
            if weight_decay_penalty_type == 'l1':
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.abs(w).sum() for w in weights]))
            else:
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None

        loss: Union[SoftmaxCrossEntropy, L2Loss]

        if self.mode == 'classification':
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:
            output_types = ['prediction']
            loss = L2Loss()

        super(WeaveModel,
              self).__init__(self.model,
                             loss=loss,
                             output_types=output_types,
                             batch_size=batch_size,
                             regularization_loss=regularization_loss,
                             **kwargs)

    def compute_features_on_batch(self, X_b):
        """Compute tensors that will be input into the model from featurized representation.

        The featurized input to `WeaveModel` is instances of `WeaveMol` created by
        `WeaveFeaturizer`. This method converts input `WeaveMol` objects into
        tensors used by the Keras implementation to compute `WeaveModel` outputs.

        Parameters
        ----------
        X_b: np.ndarray
            A numpy array with dtype=object where elements are `WeaveMol` objects.

        Returns
        -------
        atom_feat: np.ndarray
            Of shape `(N_atoms, N_atom_feat)`.
        pair_feat: np.ndarray
            Of shape `(N_pairs, N_pair_feat)`. Note that `N_pairs` will depend on
            the number of pairs being considered. If `max_pair_distance` is
            `None`, then this will be `N_atoms**2`. Else it will be the number
            of pairs within the specifed graph distance.
        pair_split: np.ndarray
            Of shape `(N_pairs,)`. The i-th entry in this array will tell you the
            originating atom for this pair (the "source"). Note that pairs are
            symmetric so for a pair `(a, b)`, both `a` and `b` will separately be
            sources at different points in this array.
        atom_split: np.ndarray
            Of shape `(N_atoms,)`. The i-th entry in this array will be the molecule
            with the i-th atom belongs to.
        atom_to_pair: np.ndarray
            Of shape `(N_pairs, 2)`. The i-th row in this array will be the array
            `[a, b]` if `(a, b)` is a pair to be considered. (Note by symmetry, this
            implies some other row will contain `[b, a]`.
        """
        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        start = 0
        for im, mol in enumerate(X_b):
            n_atoms = mol.get_num_atoms()
            # pair_edges is of shape (2, N)
            pair_edges = mol.get_pair_edges()
            # number of atoms in each molecule
            atom_split.extend([im] * n_atoms)
            # index of pair features
            C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
            atom_to_pair.append(pair_edges.T + start)
            # Get starting pair atoms
            pair_starts = pair_edges.T[:, 0]
            # number of pairs for each atom
            pair_split.extend(pair_starts + start)
            start = start + n_atoms

            # atom features
            atom_feat.append(mol.get_atom_features())
            # pair features
            pair_feat.append(mol.get_pair_features())

        return (np.concatenate(atom_feat, axis=0),
                np.concatenate(pair_feat, axis=0), np.array(pair_split),
                np.array(atom_split), np.concatenate(atom_to_pair, axis=0))

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Convert a dataset into the tensors needed for learning.

        Parameters
        ----------
        dataset: `dc.data.Dataset`
            Dataset to convert
        epochs: int, optional (Default 1)
            Number of times to walk over `dataset`
        mode: str, optional (Default 'fit')
            Ignored in this implementation.
        deterministic: bool, optional (Default True)
            Whether the dataset should be walked in a deterministic fashion
        pad_batches: bool, optional (Default True)
            If true, each returned batch will have size `self.batch_size`.

        Returns
        -------
        Iterator which walks over the batches
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if y_b is not None:
                    if self.model.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(),
                                         self.model.n_classes).reshape(
                                             -1, self.model.n_tasks,
                                             self.model.n_classes)
                inputs = self.compute_features_on_batch(X_b)
                yield (inputs, [y_b], [w_b])

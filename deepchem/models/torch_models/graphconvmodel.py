from collections.abc import Sequence as SequenceCollection
import deepchem as dc
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
from typing import List, Union, Callable, Any
from deepchem.data import Dataset
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.utils.typing import OneOrMany
import deepchem.models.torch_models.layers as torch_layers
from deepchem.utils.pytorch_utils import get_activation
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy


class TrimGraphOutput(nn.Module):
    """Trim the output to the correct number of samples.

    GraphGather always outputs fixed size batches.  This layer trims the output
    to the number of samples that were in the actual input tensors.
    """

    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def forward(self, inputs):
        n_samples = torch.squeeze(inputs[1])
        return inputs[0][0:n_samples]


class _GraphConvTorchModel(nn.Module):
    """
    Graph Convolutional Models.

    This class implements the graph convolutional model from the
    following paper [1]_. These graph convolutions start with a per-atom set of
    descriptors for each atom in a molecule, then combine and recombine these
    descriptors over convolutional layers.
    following [1]_.

    All arguments have the same meaning as in GraphConvModel.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models import _GraphConvTorchModel
    >>> batch_size = 10
    >>> out_channels = 2
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
    >>> args = [atom_features, degree_slice, membership, torch.tensor(2)] + deg_adjs
    >>> model = _GraphConvTorchModel(out_channels, graph_conv_layers=[64, 64], number_input_features=[atom_features.shape[-1], 64],dense_layer_size=128,dropout=0.0,mode="classification",number_atom_features=75,n_classes=2,batch_normalize=False,uncertainty=False,batch_size=batch_size)
    >>> result = model(args)
    >>> len(result)
    3

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints."
        Advances in neural information processing systems. 2015. https://arxiv.org/abs/1509.09292
        """

    def __init__(self,
                 n_tasks: int,
                 number_input_features: List[int],
                 graph_conv_layers: List[int] = [64, 64],
                 dense_layer_size: int = 128,
                 dropout=0.0,
                 mode: str = "classification",
                 number_atom_features: int = 75,
                 n_classes: int = 2,
                 batch_normalize: bool = True,
                 uncertainty: bool = False,
                 batch_size: int = 100):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        number_input_features: int
            Number of input features to each of the Graph Conv Layer
        graph_conv_layers: list of int
            Width of channels for the Graph Convolution Layers
        dense_layer_size: int
            Width of channels for Atom Level Dense Layer after GraphPool
        dropout: list or float
            the dropout probablity to use for each layer.  The length of this list
            should equal len(graph_conv_layers)+1 (one value for each convolution
            layer, and one for the dense layer).  Alternatively this may be a single
            value instead of a list, in which case the same value is used for every
            layer.
        mode: str
            Either "classification" or "regression"
        number_atom_features: int
            75 is the default number of atom features created, but
            this can vary if various options are passed to the
            function atom_features in graph_features
        n_classes: int
            the number of classes to predict (only used in classification mode)
        batch_normalize: True
            if True, apply batch normalization to model
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        super(_GraphConvTorchModel, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        self.n_tasks: int = n_tasks
        self.n_classes: int = n_classes
        self.mode: str = mode
        self.uncertainty: bool = uncertainty

        if not isinstance(dropout, SequenceCollection):
            dropout = [dropout] * (len(graph_conv_layers) + 1)
        if len(dropout) != len(graph_conv_layers) + 1:
            raise ValueError('Wrong number of dropout probabilities provided')
        if uncertainty:
            if mode != "regression":
                raise ValueError(
                    "Uncertainty is only supported in regression mode")
            if any(d == 0.0 for d in dropout):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty'
                )

        self.graph_convs: nn.ModuleList = nn.ModuleList([
            torch_layers.GraphConv(layer_size,
                                   input_size,
                                   activation_fn=get_activation('relu'))
            for layer_size, input_size in zip(graph_conv_layers,
                                              number_input_features)
        ])

        self.batch_norms: nn.ModuleList = nn.ModuleList([
            nn.BatchNorm1d(num_features=num_features,
                           eps=1e-3,
                           momentum=0.99,
                           affine=True,
                           track_running_stats=True)
            if batch_normalize else nn.Identity()
            for num_features in graph_conv_layers
        ])
        self.batch_norms.append(
            nn.BatchNorm1d(num_features=dense_layer_size,
                           eps=1e-3,
                           momentum=0.99,
                           affine=True,
                           track_running_stats=True) if batch_normalize else nn.
            Identity())
        self.dropouts: nn.ModuleList = nn.ModuleList([
            nn.Dropout(rate) if rate > 0.0 else nn.Identity()
            for rate in dropout
        ])
        self.graph_pools: nn.ModuleList = nn.ModuleList(
            [torch_layers.GraphPool() for _ in graph_conv_layers])
        self.dense: nn.Linear = nn.Linear(64, dense_layer_size)
        self.dense_act = F.relu
        self.graph_gather = torch_layers.GraphGather(
            batch_size=batch_size, activation_fn=get_activation('tanh'))
        self.trim = TrimGraphOutput()
        if self.mode == 'classification':
            self.reshape_dense: nn.Linear = nn.Linear(dense_layer_size * 2,
                                                      n_tasks * n_classes)
        else:
            self.regression_dense: nn.Linear = nn.Linear(
                dense_layer_size * 2, n_tasks)
            if self.uncertainty:
                self.uncertainty_dense: nn.Linear = nn.Linear(
                    dense_layer_size * 2, n_tasks)
                self.uncertainty_trim = TrimGraphOutput()

    def forward(self,
                inputs: OneOrMany[torch.Tensor],
                training=False) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        inputs: OneOrMany[torch.Tensor]
        Should contain tensors [atom_features, degree_slice, membership, n_samples] and deg_adjs

        Returns
        -------
        List[torch.Tensor]
            Output as per use case : regression/classification
        """
        atom_features: torch.Tensor = inputs[0]
        degree_slice: torch.Tensor = inputs[1]
        membership: torch.Tensor = inputs[2].to(torch.int64)
        n_samples: torch.Tensor = inputs[3].to(torch.int64)
        deg_adjs: List[torch.Tensor] = [
            deg_adj.to(torch.int64) for deg_adj in inputs[4:]
        ]

        in_layer: torch.Tensor = atom_features
        for i in range(len(self.graph_convs)):
            gc_in: List[torch.Tensor] = [in_layer, degree_slice, membership
                                        ] + deg_adjs
            gc1: torch.Tensor = self.graph_convs[i](gc_in)
            if self.batch_norms[i] is not None:
                gc1 = self.batch_norms[i](gc1)
            if training and self.dropouts[i] is not None:
                gc1 = self.dropouts[i](gc1)
            gp_in: List[torch.Tensor] = [gc1, degree_slice, membership
                                        ] + deg_adjs
            in_layer = self.graph_pools[i](gp_in)
        dense: torch.Tensor = self.dense(in_layer)
        denseact: torch.Tensor = self.dense_act(dense)
        if self.batch_norms[-1] is not None:
            denseact = self.batch_norms[-1](denseact)
        if training and self.dropouts[-1] is not None:
            denseact = self.dropouts[-1](denseact)
        neural_fingerprint: torch.Tensor = self.graph_gather(
            [denseact, degree_slice, membership] + deg_adjs)
        if self.mode == 'classification':
            logits: torch.Tensor = torch.reshape(
                self.reshape_dense(neural_fingerprint),
                (-1, self.n_tasks, self.n_classes))
            logits = self.trim([logits, n_samples])
            output: torch.Tensor = F.softmax(logits, dim=2)
            outputs: List[torch.Tensor] = [output, logits, neural_fingerprint]
        else:
            output = self.regression_dense(neural_fingerprint)
            output = self.trim([output, n_samples])
            if self.uncertainty:
                log_var: torch.Tensor = self.uncertainty_dense(
                    neural_fingerprint)
                log_var = self.uncertainty_trim([log_var, n_samples])
                var: torch.Tensor = torch.exp(log_var)
                outputs = [output, var, output, log_var, neural_fingerprint]
            else:
                outputs = [output, neural_fingerprint]

        return outputs


class GraphConvModel(TorchModel):
    """Graph Convolutional Models.

    This class implements the graph convolutional model from the
    following paper [1]_. These graph convolutions start with a per-atom set of
    descriptors for each atom in a molecule, then combine and recombine these
    descriptors over convolutional layers.
    following [1]_.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models import GraphConvModel
    >>> featurizer = dc.feat.ConvMolFeaturizer()
    >>> tasks = ["outcome"]
    >>> mols = ["C", "CO", "CC"]
    >>> X = featurizer(mols)
    >>> y = np.array([0, 1, 0])
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
    >>> batch_size = 10
    >>> model = GraphConvModel(len(tasks), number_input_features=[75, 64], batch_size=batch_size, batch_normalize=False, mode='classification')
    >>> loss = model.fit(dataset, nb_epoch=20)

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015.
    """

    def __init__(self,
                 n_tasks: int,
                 number_input_features: Union[List[int], int],
                 graph_conv_layers: List[int] = [64, 64],
                 dense_layer_size: int = 128,
                 dropout: float = 0.0,
                 mode: str = "classification",
                 number_atom_features: int = 75,
                 n_classes: int = 2,
                 batch_size: int = 100,
                 batch_normalize: bool = True,
                 uncertainty: bool = False,
                 **kwargs):
        """The wrapper class for graph convolutions.

        Note that since the underlying _GraphConvKerasModel class is
        specified using imperative subclassing style, this model
        cannout make predictions for arbitrary outputs.

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        number_input_features: list of int
            Number of input features to each of the Graph Conv Layer
        graph_conv_layers: list of int
            Width of channels for the Graph Convolution Layers
        dense_layer_size: int
            Width of channels for Atom Level Dense Layer after GraphPool
        dropout: list or float
            the dropout probablity to use for each layer.  The length of this list
            should equal len(graph_conv_layers)+1 (one value for each convolution
            layer, and one for the dense layer).  Alternatively this may be a single
            value instead of a list, in which case the same value is used for every
            layer.
        mode: str
            Either "classification" or "regression"
        number_atom_features: int
            75 is the default number of atom features created, but
            this can vary if various options are passed to the
            function atom_features in graph_features
        n_classes: int
            the number of classes to predict (only used in classification mode)
        batch_normalize: True
            if True, apply batch normalization to model
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        self.mode: str = mode
        self.n_tasks: int = n_tasks
        self.n_classes: int = n_classes
        self.batch_size: int = batch_size
        self.uncertainty: bool = uncertainty
        if isinstance(number_input_features, int):
            number_input_features = [number_input_features
                                    ] + graph_conv_layers[:-1]
        model = _GraphConvTorchModel(
            n_tasks,
            graph_conv_layers=graph_conv_layers,
            number_input_features=number_input_features,
            dense_layer_size=dense_layer_size,
            dropout=dropout,
            mode=mode,
            number_atom_features=number_atom_features,
            n_classes=n_classes,
            batch_normalize=batch_normalize,
            uncertainty=uncertainty,
            batch_size=batch_size)
        loss: Union[SoftmaxCrossEntropy, L2Loss, Callable[[Any, Any, Any], Any]]
        if mode == "classification":
            output_types = ['prediction', 'loss', 'embedding']
            loss = SoftmaxCrossEntropy()
        else:
            if self.uncertainty:
                output_types = [
                    'prediction', 'variance', 'loss', 'loss', 'embedding'
                ]

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_pytorch_shapes_consistent(
                        outputs[0], labels[0])
                    losses = torch.square(output - labels) / torch.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        shape = tuple(w.shape)
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = torch.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return torch.mean(losses * w)
            else:
                output_types = ['prediction', 'embedding']
                loss = L2Loss()
        super(GraphConvModel, self).__init__(model,
                                             loss,
                                             output_types=output_types,
                                             batch_size=batch_size,
                                             **kwargs)

    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          mode: str = 'fit',
                          deterministic: bool = True,
                          pad_batches: bool = True):
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
                if y_b is not None and self.mode == 'classification' and not (
                        mode == 'predict'):
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(X_b.shape[0])
                inputs = [
                    multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples
                ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
                yield (inputs, [y_b], [w_b])

from collections.abc import Sequence as SequenceCollection

import deepchem as dc
import numpy as np
import tensorflow as tf

from typing import List, Union, Tuple, Iterable, Dict, Optional
from deepchem.utils.typing import OneOrMany, LossFn, ActivationFn
from deepchem.data import Dataset, pad_features
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization


class TrimGraphOutput(tf.keras.layers.Layer):
    """Trim the output to the correct number of samples.

    GraphGather always outputs fixed size batches.  This layer trims the output
    to the number of samples that were in the actual input tensors.
    """

    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def call(self, inputs):
        n_samples = tf.squeeze(inputs[1])
        return inputs[0][0:n_samples]


class WeaveModel(KerasModel):
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
    >>> model = dc.models.WeaveModel(n_tasks=1, n_weave=2, fully_connected_layer_sizes=[2000, 1000], mode="classification")
    >>> loss = model.fit(dataset)

    Note
    ----
    In general, the use of batch normalization can cause issues with NaNs. If
    you're having trouble with NaNs while using this model, consider setting
    `batch_normalize_kwargs={"trainable": False}` or turning off batch
    normalization entirely with `batch_normalize=False`.

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
                 final_conv_activation_fn: Optional[ActivationFn] = tf.nn.tanh,
                 activation_fns: OneOrMany[ActivationFn] = tf.nn.relu,
                 batch_normalize: bool = True,
                 batch_normalize_kwargs: Dict = {
                   
                   
                 },
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
        final_conv_activation_fn: Optional[ActivationFn] (default `tf.nn.tanh`)
            The Tensorflow activation funcntion to apply to the final
            convolution at the end of the weave convolutions. If `None`, then no
            activate is applied (hence linear).
        activation_fns: list or object (default `tf.nn.relu`)
            The Tensorflow activation function to apply to each fully connected layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a
            single value instead of a list, in which case the same value is used for
            every layer.
        batch_normalize: bool, optional (default True)
            If this is turned on, apply batch normalization before applying
            activation functions on convolutional and fully connected layers.
        batch_normalize_kwargs: Dict, optional (default `{"renorm"=True, "fused": False}`)
            Batch normalization is a complex layer which has many potential
            argumentswhich change behavior. This layer accepts user-defined
            parameters which are passed to all `BatchNormalization` layers in
            `WeaveModel`, `WeaveLayer`, and `WeaveGather`.
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
        if not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers
        if weight_decay_penalty != 0.0:
            if weight_decay_penalty_type == 'l1':
                regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
            else:
                regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
        else:
            regularizer = None

        self.n_tasks = n_tasks
        self.n_atom_feat = n_atom_feat
        self.n_pair_feat = n_pair_feat
        self.n_hidden = n_hidden
        self.n_graph_feat = n_graph_feat
        self.mode = mode
        self.n_classes = n_classes

        # Build the model.
        atom_features = Input(shape=(self.n_atom_feat[0],))
        pair_features = Input(shape=(self.n_pair_feat[0],))
        pair_split = Input(shape=tuple(), dtype=tf.int32)
        atom_split = Input(shape=tuple(), dtype=tf.int32)
        atom_to_pair = Input(shape=(2,), dtype=tf.int32)
        inputs = [atom_features, pair_features, pair_split, atom_to_pair]
        for ind in range(n_weave):
            n_atom = self.n_atom_feat[ind]
            n_pair = self.n_pair_feat[ind]
            if ind < n_weave - 1:
                n_atom_next = self.n_atom_feat[ind + 1]
                n_pair_next = self.n_pair_feat[ind + 1]
            else:
                n_atom_next = n_hidden
                n_pair_next = n_hidden
            weave_layer_ind_A, weave_layer_ind_P = layers.WeaveLayer(
                n_atom_input_feat=n_atom,
                n_pair_input_feat=n_pair,
                n_atom_output_feat=n_atom_next,
                n_pair_output_feat=n_pair_next,
                init=tf.keras.initializers.TruncatedNormal(
                    stddev=conv_weight_init_stddevs[ind]),
                batch_normalize=batch_normalize)(inputs)
            inputs = [
                weave_layer_ind_A, weave_layer_ind_P, pair_split, atom_to_pair
            ]
        # Final atom-layer convolution. Note this differs slightly from the paper
        # since we use a tanh activation as default. This seems necessary for numerical
        # stability.
        dense1 = Dense(self.n_graph_feat,
                       activation=final_conv_activation_fn)(weave_layer_ind_A)
        if batch_normalize:
            dense1 = BatchNormalization(**batch_normalize_kwargs)(dense1)
        weave_gather = layers.WeaveGather(
            batch_size,
            n_input=self.n_graph_feat,
            gaussian_expand=gaussian_expand,
            compress_post_gaussian_expansion=compress_post_gaussian_expansion)(
                [dense1, atom_split])

        if n_layers > 0:
            # Now fully connected layers
            input_layer = weave_gather
            for layer_size, weight_stddev, bias_const, dropout, activation_fn in zip(
                    fully_connected_layer_sizes, weight_init_stddevs,
                    bias_init_consts, dropouts, activation_fns):
                layer = Dense(
                    layer_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=weight_stddev),
                    bias_initializer=tf.constant_initializer(value=bias_const),
                    kernel_regularizer=regularizer)(input_layer)
                if dropout > 0.0:
                    layer = Dropout(rate=dropout)(layer)
                if batch_normalize:
                    # Should this allow for training?
                    layer = BatchNormalization(**batch_normalize_kwargs)(layer)
                layer = Activation(activation_fn)(layer)
                input_layer = layer
            output = input_layer
        else:
            output = weave_gather

        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape(
                (n_tasks, n_classes))(Dense(n_tasks * n_classes)(output))
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            loss: Loss = SoftmaxCrossEntropy()
        else:
            output = Dense(n_tasks)(output)
            outputs = [output]
            output_types = ['prediction']
            loss = L2Loss()
        model = tf.keras.Model(inputs=[
            atom_features, pair_features, pair_split, atom_split, atom_to_pair
        ],
                               outputs=outputs)
        super(WeaveModel, self).__init__(model,
                                         loss,
                                         output_types=output_types,
                                         batch_size=batch_size,
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
                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)
                inputs = self.compute_features_on_batch(X_b)
                yield (inputs, [y_b], [w_b])


class DTNNModel(KerasModel):
    """Deep Tensor Neural Networks

    This class implements deep tensor neural networks as first defined in [1]_

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Quantum-chemical insights from deep
        tensor neural networks." Nature communications 8.1 (2017): 1-8.
    """

    def __init__(self,
                 n_tasks,
                 n_embedding=30,
                 n_hidden=100,
                 n_distance=100,
                 distance_min=-1,
                 distance_max=18,
                 output_activation=True,
                 mode="regression",
                 dropout=0.0,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_embedding: int, optional
            Number of features per atom.
        n_hidden: int, optional
            Number of features for each molecule after DTNNStep
        n_distance: int, optional
            granularity of distance matrix
            step size will be (distance_max-distance_min)/n_distance
        distance_min: float, optional
            minimum distance of atom pairs, default = -1 Angstorm
        distance_max: float, optional
            maximum distance of atom pairs, default = 18 Angstorm
        mode: str
            Only "regression" is currently supported.
        dropout: float
            the dropout probablity to use.
        """
        if mode not in ['regression']:
            raise ValueError("Only 'regression' mode is currently supported")
        self.n_tasks = n_tasks
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_distance = n_distance
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.step_size = (distance_max - distance_min) / n_distance
        self.steps = np.array(
            [distance_min + i * self.step_size for i in range(n_distance)])
        self.steps = np.expand_dims(self.steps, 0)
        self.output_activation = output_activation
        self.mode = mode
        self.dropout = dropout

        # Build the model.

        atom_number = Input(shape=tuple(), dtype=tf.int32)
        distance = Input(shape=(self.n_distance,))
        atom_membership = Input(shape=tuple(), dtype=tf.int32)
        distance_membership_i = Input(shape=tuple(), dtype=tf.int32)
        distance_membership_j = Input(shape=tuple(), dtype=tf.int32)

        dtnn_embedding = layers.DTNNEmbedding(
            n_embedding=self.n_embedding)(atom_number)
        if self.dropout > 0.0:
            dtnn_embedding = Dropout(rate=self.dropout)(dtnn_embedding)
        dtnn_layer1 = layers.DTNNStep(n_embedding=self.n_embedding,
                                      n_distance=self.n_distance)([
                                          dtnn_embedding, distance,
                                          distance_membership_i,
                                          distance_membership_j
                                      ])
        if self.dropout > 0.0:
            dtnn_layer1 = Dropout(rate=self.dropout)(dtnn_layer1)
        dtnn_layer2 = layers.DTNNStep(n_embedding=self.n_embedding,
                                      n_distance=self.n_distance)([
                                          dtnn_layer1, distance,
                                          distance_membership_i,
                                          distance_membership_j
                                      ])
        if self.dropout > 0.0:
            dtnn_layer2 = Dropout(rate=self.dropout)(dtnn_layer2)
        dtnn_gather = layers.DTNNGather(
            n_embedding=self.n_embedding,
            layer_sizes=[self.n_hidden],
            n_outputs=self.n_tasks,
            output_activation=self.output_activation)(
                [dtnn_layer2, atom_membership])
        if self.dropout > 0.0:
            dtnn_gather = Dropout(rate=self.dropout)(dtnn_gather)

        n_tasks = self.n_tasks
        output = Dense(n_tasks)(dtnn_gather)
        model = tf.keras.Model(inputs=[
            atom_number, distance, atom_membership, distance_membership_i,
            distance_membership_j
        ],
                               outputs=[output])
        super(DTNNModel, self).__init__(model, L2Loss(), **kwargs)

    def compute_features_on_batch(self, X_b):
        """Computes the values for different Feature Layers on given batch

        A tf.py_func wrapper is written around this when creating the
        input_fn for tf.Estimator

        """
        distance = []
        atom_membership = []
        distance_membership_i = []
        distance_membership_j = []
        num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))
        atom_number = [
            np.round(
                np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]),
                         1 / 2.4)).astype(int) for i in range(len(num_atoms))
        ]
        start = 0
        for im, molecule in enumerate(atom_number):
            distance_matrix = np.outer(
                molecule, molecule) / X_b[im, :num_atoms[im], :num_atoms[im]]
            np.fill_diagonal(distance_matrix, -100)
            distance.append(np.expand_dims(distance_matrix.flatten(), 1))
            atom_membership.append([im] * num_atoms[im])
            membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
            membership_i = membership.flatten(order='F')
            membership_j = membership.flatten()
            distance_membership_i.append(membership_i + start)
            distance_membership_j.append(membership_j + start)
            start = start + num_atoms[im]

        atom_number = np.concatenate(atom_number).astype(np.int32)
        distance = np.concatenate(distance, axis=0)
        gaussian_dist = np.exp(-np.square(distance - self.steps) /
                               (2 * self.step_size**2))
        gaussian_dist = gaussian_dist.astype(np.float32)
        atom_mem = np.concatenate(atom_membership).astype(np.int32)
        dist_mem_i = np.concatenate(distance_membership_i).astype(np.int32)
        dist_mem_j = np.concatenate(distance_membership_j).astype(np.int32)

        features = [
            atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j
        ]

        return features

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                yield (self.compute_features_on_batch(X_b), [y_b], [w_b])


class DAGModel(KerasModel):
    """Directed Acyclic Graph models for molecular property prediction.

    This model is based on the following paper:

    Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. "Deep architectures and deep learning in chemoinformatics: the prediction of aqueous solubility for drug-like molecules." Journal of chemical information and modeling 53.7 (2013): 1563-1575.

    The basic idea for this paper is that a molecule is usually
    viewed as an undirected graph. However, you can convert it to
    a series of directed graphs. The idea is that for each atom,
    you make a DAG using that atom as the vertex of the DAG and
    edges pointing "inwards" to it. This transformation is
    implemented in
    `dc.trans.transformers.DAGTransformer.UG_to_DAG`.

    This model accepts ConvMols as input, just as GraphConvModel
    does, but these ConvMol objects must be transformed by
    dc.trans.DAGTransformer.

    As a note, performance of this model can be a little
    sensitive to initialization. It might be worth training a few
    different instantiations to get a stable set of parameters.
    """

    def __init__(self,
                 n_tasks,
                 max_atoms=50,
                 n_atom_feat=75,
                 n_graph_feat=30,
                 n_outputs=30,
                 layer_sizes=[100],
                 layer_sizes_gather=[100],
                 dropout=None,
                 mode="classification",
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        max_atoms: int, optional
            Maximum number of atoms in a molecule, should be defined based on dataset.
        n_atom_feat: int, optional
            Number of features per atom.
        n_graph_feat: int, optional
            Number of features for atom in the graph.
        n_outputs: int, optional
            Number of features for each molecule.
        layer_sizes: list of int, optional
            List of hidden layer size(s) in the propagation step:
            length of this list represents the number of hidden layers,
            and each element is the width of corresponding hidden layer.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability, applied after each propagation step and gather step.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.max_atoms = max_atoms
        self.n_atom_feat = n_atom_feat
        self.n_graph_feat = n_graph_feat
        self.n_outputs = n_outputs
        self.layer_sizes = layer_sizes
        self.layer_sizes_gather = layer_sizes_gather
        self.dropout = dropout
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        if uncertainty:
            if mode != "regression":
                raise ValueError(
                    "Uncertainty is only supported in regression mode")
            if dropout is None or dropout == 0.0:
                raise ValueError(
                    'Dropout must be included to predict uncertainty')

        # Build the model.

        atom_features = Input(shape=(self.n_atom_feat,))
        parents = Input(shape=(self.max_atoms, self.max_atoms), dtype=tf.int32)
        calculation_orders = Input(shape=(self.max_atoms,), dtype=tf.int32)
        calculation_masks = Input(shape=(self.max_atoms,), dtype=tf.bool)
        membership = Input(shape=tuple(), dtype=tf.int32)
        n_atoms = Input(shape=tuple(), dtype=tf.int32)
        dag_layer1 = layers.DAGLayer(n_graph_feat=self.n_graph_feat,
                                     n_atom_feat=self.n_atom_feat,
                                     max_atoms=self.max_atoms,
                                     layer_sizes=self.layer_sizes,
                                     dropout=self.dropout,
                                     batch_size=batch_size)([
                                         atom_features, parents,
                                         calculation_orders, calculation_masks,
                                         n_atoms
                                     ])
        dag_gather = layers.DAGGather(
            n_graph_feat=self.n_graph_feat,
            n_outputs=self.n_outputs,
            max_atoms=self.max_atoms,
            layer_sizes=self.layer_sizes_gather,
            dropout=self.dropout)([dag_layer1, membership])
        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape(
                (n_tasks, n_classes))(Dense(n_tasks * n_classes)(dag_gather))
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:
            output = Dense(n_tasks)(dag_gather)
            if self.uncertainty:
                log_var = Dense(n_tasks)(dag_gather)
                var = Activation(tf.exp)(log_var)
                outputs = [output, var, output, log_var]
                output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_tf_shapes_consistent(
                        outputs[0], labels[0])
                    output, labels = dc.models.losses._ensure_float(
                        output, labels)
                    losses = tf.square(output - labels) / tf.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        if tf.is_tensor(w):
                            shape = tuple(w.shape.as_list())
                        else:
                            shape = w.shape
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = tf.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return tf.reduce_mean(losses * w) + sum(self.model.losses)
            else:
                outputs = [output]
                output_types = ['prediction']
                loss = L2Loss()
        model = tf.keras.Model(
            inputs=[
                atom_features,
                parents,
                calculation_orders,
                calculation_masks,
                membership,
                n_atoms,  # dropout_switch
            ],
            outputs=outputs)
        super(DAGModel, self).__init__(model,
                                       loss,
                                       output_types=output_types,
                                       batch_size=batch_size,
                                       **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        """Convert a dataset into the tensors needed for learning"""
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                if y_b is not None and self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)

                atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
                n_atoms = sum(atoms_per_mol)
                start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

                atoms_all = []
                # calculation orders for a batch of molecules
                parents_all = []
                calculation_orders = []
                calculation_masks = []
                membership = []
                for idm, mol in enumerate(X_b):
                    # padding atom features vector of each molecule with 0
                    atoms_all.append(mol.get_atom_features())
                    parents = mol.parents
                    parents_all.extend(parents)
                    calculation_index = np.array(parents)[:, :, 0]
                    mask = np.array(calculation_index - self.max_atoms,
                                    dtype=bool)
                    calculation_orders.append(calculation_index +
                                              start_index[idm])
                    calculation_masks.append(mask)
                    membership.extend([idm] * atoms_per_mol[idm])
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)

                yield ([
                    np.concatenate(atoms_all, axis=0),
                    np.stack(parents_all, axis=0),
                    np.concatenate(calculation_orders, axis=0),
                    np.concatenate(calculation_masks, axis=0),
                    np.array(membership),
                    np.array(n_atoms), dropout
                ], [y_b], [w_b])


class _GraphConvKerasModel(tf.keras.Model):

    def __init__(self,
                 n_tasks,
                 graph_conv_layers,
                 dense_layer_size=128,
                 dropout=0.0,
                 mode="classification",
                 number_atom_features=75,
                 n_classes=2,
                 batch_normalize=True,
                 uncertainty=False,
                 batch_size=100):
        """An internal keras model class.

        The graph convolutions use a nonstandard control flow so the
        standard Keras functional API can't support them. We instead
        use the imperative "subclassing" API to implement the graph
        convolutions.

        All arguments have the same meaning as in GraphConvModel.
        """
        super(_GraphConvKerasModel, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        self.mode = mode
        self.uncertainty = uncertainty

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

        self.graph_convs = [
            layers.GraphConv(layer_size, activation_fn=tf.nn.relu)
            for layer_size in graph_conv_layers
        ]
        self.batch_norms = [
            BatchNormalization(fused=False) if batch_normalize else None
            for _ in range(len(graph_conv_layers) + 1)
        ]
        self.dropouts = [
            Dropout(rate=rate) if rate > 0.0 else None for rate in dropout
        ]
        self.graph_pools = [layers.GraphPool() for _ in graph_conv_layers]
        self.dense = Dense(dense_layer_size, activation=tf.nn.relu)
        self.graph_gather = layers.GraphGather(batch_size=batch_size,
                                               activation_fn=tf.nn.tanh)
        self.trim = TrimGraphOutput()
        if self.mode == 'classification':
            self.reshape_dense = Dense(n_tasks * n_classes)
            self.reshape = Reshape((n_tasks, n_classes))
            self.softmax = Softmax()
        else:
            self.regression_dense = Dense(n_tasks)
            if self.uncertainty:
                self.uncertainty_dense = Dense(n_tasks)
                self.uncertainty_trim = TrimGraphOutput()
                self.uncertainty_activation = Activation(tf.exp)

    def call(self, inputs, training=False):
        atom_features = inputs[0]
        degree_slice = tf.cast(inputs[1], dtype=tf.int32)
        membership = tf.cast(inputs[2], dtype=tf.int32)
        n_samples = tf.cast(inputs[3], dtype=tf.int32)
        deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]

        in_layer = atom_features
        for i in range(len(self.graph_convs)):
            gc_in = [in_layer, degree_slice, membership] + deg_adjs
            gc1 = self.graph_convs[i](gc_in)
            if self.batch_norms[i] is not None:
                gc1 = self.batch_norms[i](gc1, training=training)
            if training and self.dropouts[i] is not None:
                gc1 = self.dropouts[i](gc1, training=training)
            gp_in = [gc1, degree_slice, membership] + deg_adjs
            in_layer = self.graph_pools[i](gp_in)
        dense = self.dense(in_layer)
        if self.batch_norms[-1] is not None:
            dense = self.batch_norms[-1](dense, training=training)
        if training and self.dropouts[-1] is not None:
            dense = self.dropouts[-1](dense, training=training)
        neural_fingerprint = self.graph_gather(
            [dense, degree_slice, membership] + deg_adjs)
        if self.mode == 'classification':
            logits = self.reshape(self.reshape_dense(neural_fingerprint))
            logits = self.trim([logits, n_samples])
            output = self.softmax(logits)
            outputs = [output, logits, neural_fingerprint]
        else:
            output = self.regression_dense(neural_fingerprint)
            output = self.trim([output, n_samples])
            if self.uncertainty:
                log_var = self.uncertainty_dense(neural_fingerprint)
                log_var = self.uncertainty_trim([log_var, n_samples])
                var = self.uncertainty_activation(log_var)
                outputs = [output, var, output, log_var, neural_fingerprint]
            else:
                outputs = [output, neural_fingerprint]

        return outputs


class GraphConvModel(KerasModel):
    """Graph Convolutional Models.

    This class implements the graph convolutional model from the
    following paper [1]_. These graph convolutions start with a per-atom set of
    descriptors for each atom in a molecule, then combine and recombine these
    descriptors over convolutional layers.
    following [1]_.


    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015.
    """

    def __init__(self,
                 n_tasks: int,
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
        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        model = _GraphConvKerasModel(n_tasks,
                                     graph_conv_layers=graph_conv_layers,
                                     dense_layer_size=dense_layer_size,
                                     dropout=dropout,
                                     mode=mode,
                                     number_atom_features=number_atom_features,
                                     n_classes=n_classes,
                                     batch_normalize=batch_normalize,
                                     uncertainty=uncertainty,
                                     batch_size=batch_size)
        if mode == "classification":
            output_types = ['prediction', 'loss', 'embedding']
            loss: Union[Loss, LossFn] = SoftmaxCrossEntropy()
        else:
            if self.uncertainty:
                output_types = [
                    'prediction', 'variance', 'loss', 'loss', 'embedding'
                ]

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_tf_shapes_consistent(
                        outputs[0], labels[0])
                    output, labels = dc.models.losses._ensure_float(
                        output, labels)
                    losses = tf.square(output - labels) / tf.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        if tf.is_tensor(w):
                            shape = tuple(w.shape.as_list())
                        else:
                            shape = w.shape
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = tf.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return tf.reduce_mean(losses * w) + sum(self.model.losses)
            else:
                output_types = ['prediction', 'embedding']
                loss = L2Loss()
        super(GraphConvModel, self).__init__(model,
                                             loss,
                                             output_types=output_types,
                                             batch_size=batch_size,
                                             **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
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


class MPNNModel(KerasModel):
    """ Message Passing Neural Network,

    Message Passing Neural Networks [1]_ treat graph convolutional
    operations as an instantiation of a more general message
    passing schem. Recall that message passing in a graph is when
    nodes in a graph send each other "messages" and update their
    internal state as a consequence of these messages.

    Ordering structures in this model are built according to [2]_

    References
    ----------
    .. [1] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.
        "Neural Message Passing for Quantum Chemistry." ICML 2017.
    .. [2] Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters:
        Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).
    """

    def __init__(self,
                 n_tasks,
                 n_atom_feat=70,
                 n_pair_feat=8,
                 n_hidden=100,
                 T=5,
                 M=10,
                 mode="regression",
                 dropout=0.0,
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_atom_feat: int, optional
            Number of features per atom.
        n_pair_feat: int, optional
            Number of features per pair of atoms.
        n_hidden: int, optional
            Number of units(convolution depths) in corresponding hidden layer
        n_graph_feat: int, optional
            Number of output features for each molecule(graph)
        dropout: float
            the dropout probablity to use.
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.n_atom_feat = n_atom_feat
        self.n_pair_feat = n_pair_feat
        self.n_hidden = n_hidden
        self.T = T
        self.M = M
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        if uncertainty:
            if mode != "regression":
                raise ValueError(
                    "Uncertainty is only supported in regression mode")
            if dropout == 0.0:
                raise ValueError(
                    'Dropout must be included to predict uncertainty')

        # Build the model.

        atom_features = Input(shape=(self.n_atom_feat,))
        pair_features = Input(shape=(self.n_pair_feat,))
        atom_split = Input(shape=tuple(), dtype=tf.int32)
        atom_to_pair = Input(shape=(2,), dtype=tf.int32)
        n_samples = Input(shape=tuple(), dtype=tf.int32)

        message_passing = layers.MessagePassing(self.T,
                                                message_fn='enn',
                                                update_fn='gru',
                                                n_hidden=self.n_hidden)([
                                                    atom_features,
                                                    pair_features, atom_to_pair
                                                ])

        atom_embeddings = Dense(self.n_hidden)(message_passing)

        mol_embeddings = layers.SetGather(self.M,
                                          batch_size,
                                          n_hidden=self.n_hidden)(
                                              [atom_embeddings, atom_split])

        dense1 = Dense(2 * self.n_hidden, activation=tf.nn.relu)(mol_embeddings)

        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape(
                (n_tasks, n_classes))(Dense(n_tasks * n_classes)(dense1))
            logits = TrimGraphOutput()([logits, n_samples])
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            loss = SoftmaxCrossEntropy()
        else:
            output = Dense(n_tasks)(dense1)
            output = TrimGraphOutput()([output, n_samples])
            if self.uncertainty:
                log_var = Dense(n_tasks)(dense1)
                log_var = TrimGraphOutput()([log_var, n_samples])
                var = Activation(tf.exp)(log_var)
                outputs = [output, var, output, log_var]
                output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_tf_shapes_consistent(
                        outputs[0], labels[0])
                    output, labels = dc.models.losses._ensure_float(
                        output, labels)
                    losses = tf.square(output - labels) / tf.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        if tf.is_tensor(w):
                            shape = tuple(w.shape.as_list())
                        else:
                            shape = w.shape
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = tf.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return tf.reduce_mean(losses * w) + sum(self.model.losses)
            else:
                outputs = [output]
                output_types = ['prediction']
                loss = L2Loss()
        model = tf.keras.Model(inputs=[
            atom_features, pair_features, atom_split, atom_to_pair, n_samples
        ],
                               outputs=outputs)
        super(MPNNModel, self).__init__(model,
                                        loss,
                                        output_types=output_types,
                                        batch_size=batch_size,
                                        **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                n_samples = np.array(X_b.shape[0])
                X_b = pad_features(self.batch_size, X_b)
                if y_b is not None and self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)

                atom_feat = []
                pair_feat = []
                atom_split = []
                atom_to_pair = []
                pair_split = []
                start = 0
                for im, mol in enumerate(X_b):
                    n_atoms = mol.get_num_atoms()
                    # number of atoms in each molecule
                    atom_split.extend([im] * n_atoms)
                    # index of pair features
                    C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
                    atom_to_pair.append(
                        np.transpose(
                            np.array(
                                [C1.flatten() + start,
                                 C0.flatten() + start])))
                    # number of pairs for each atom
                    pair_split.extend(C1.flatten() + start)
                    start = start + n_atoms

                    # atom features
                    atom_feat.append(mol.get_atom_features())
                    # pair features
                    pair_feat.append(
                        np.reshape(mol.get_pair_features(),
                                   (n_atoms * n_atoms, self.n_pair_feat)))

                inputs = [
                    np.concatenate(atom_feat, axis=0),
                    np.concatenate(pair_feat, axis=0),
                    np.array(atom_split),
                    np.concatenate(atom_to_pair, axis=0), n_samples
                ]
                yield (inputs, [y_b], [w_b])


#################### Deprecation warnings for renamed TensorGraph models ####################  # noqa: E266

import warnings  # noqa: E402

TENSORGRAPH_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class GraphConvTensorGraph(GraphConvModel):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("GraphConvTensorGraph",
                                           "GraphConvModel"), FutureWarning)

        super(GraphConvTensorGraph, self).__init__(*args, **kwargs)


class WeaveTensorGraph(WeaveModel):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("WeaveTensorGraph", "WeaveModel"),
            FutureWarning)

        super(WeaveModel, self).__init__(*args, **kwargs)


class DTNNTensorGraph(DTNNModel):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("DTNNTensorGraph", "DTNNModel"),
            FutureWarning)

        super(DTNNModel, self).__init__(*args, **kwargs)


class DAGTensorGraph(DAGModel):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("DAGTensorGraph", "DAGModel"),
            FutureWarning)

        super(DAGModel, self).__init__(*args, **kwargs)


class MPNNTensorGraph(MPNNModel):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("MPNNTensorGraph", "MPNNModel"),
            FutureWarning)

        super(MPNNModel, self).__init__(*args, **kwargs)

from typing import List, Optional, Sequence


class Loss:
    """A loss function for use in training models."""

    def _compute_tf_loss(self, output, labels):
        """Compute the loss function for TensorFlow tensors.

        The inputs are tensors containing the model's outputs and the labels for a
        batch.  The return value should be a tensor of shape (batch_size) or
        (batch_size, tasks) containing the value of the loss function on each
        sample or sample/task.

        Parameters
        ----------
        output: tensor
            the output of the model
        labels: tensor
            the expected output

        Returns
        -------
        The value of the loss function on each sample or sample/task pair
        """
        raise NotImplementedError("Subclasses must implement this")

    def _create_pytorch_loss(self):
        """Create a PyTorch loss function."""
        raise NotImplementedError("Subclasses must implement this")


class L1Loss(Loss):
    """The absolute difference between the true and predicted values."""

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.abs(output - labels)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.nn.functional.l1_loss(output, labels, reduction='none')

        return loss


class HuberLoss(Loss):
    """Modified version of L1 Loss, also known as Smooth L1 loss.
    Less sensitive to small errors, linear for larger errors.
    Huber loss is generally better for cases where are are both large outliers as well as small, as compared to the L1 loss.
    By default, Delta = 1.0 and reduction = 'none'.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        return tf.keras.losses.Huber(reduction='none')(output, labels)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.nn.functional.smooth_l1_loss(output,
                                                      labels,
                                                      reduction='none')

        return loss


class L2Loss(Loss):
    """The squared difference between the true and predicted values."""

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.square(output - labels)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.nn.functional.mse_loss(output,
                                                labels,
                                                reduction='none')

        return loss


class HingeLoss(Loss):
    """The hinge loss function.

    The 'output' argument should contain logits, and all elements of 'labels'
    should equal 0 or 1.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        return tf.keras.losses.hinge(labels, output)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.mean(torch.clamp(1 - labels * output, min=0), dim=-1)

        return loss


class SquaredHingeLoss(Loss):
    """The Squared Hinge loss function.

    Defined as the square of the hinge loss between y_true and y_pred. The Squared Hinge Loss is differentiable.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        return tf.keras.losses.SquaredHinge(reduction='none')(labels, output)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.mean(torch.pow(
                torch.max(1 - torch.mul(labels, output), torch.tensor(0.0)), 2),
                              dim=-1)

        return loss


class PoissonLoss(Loss):
    """The Poisson loss function is defined as the mean of the elements of y_pred - (y_true * log(y_pred) for an input of (y_true, y_pred).
    Poisson loss is generally used for regression tasks where the data follows the poisson
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        loss = tf.keras.losses.Poisson(reduction='auto')
        return loss(labels, output)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.mean(output - labels * torch.log(output))

        return loss


class BinaryCrossEntropy(Loss):
    """The cross entropy between pairs of probabilities.

    The arguments should each have shape (batch_size) or (batch_size, tasks) and
    contain probabilities.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.keras.losses.binary_crossentropy(labels, output)

    def _create_pytorch_loss(self):
        import torch
        bce = torch.nn.BCELoss(reduction='none')

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.mean(bce(output, labels), dim=-1)

        return loss


class CategoricalCrossEntropy(Loss):
    """The cross entropy between two probability distributions.

    The arguments should each have shape (batch_size, classes) or
    (batch_size, tasks, classes), and represent a probability distribution over
    classes.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.keras.losses.categorical_crossentropy(labels, output)

    def _create_pytorch_loss(self):
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return -torch.sum(labels * torch.log(output), dim=-1)

        return loss


class SigmoidCrossEntropy(Loss):
    """The cross entropy between pairs of probabilities.

    The arguments should each have shape (batch_size) or (batch_size, tasks).  The
    labels should be probabilities, while the outputs should be logits that are
    converted to probabilities using a sigmoid function.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels, output)

    def _create_pytorch_loss(self):
        import torch
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return bce(output, labels)

        return loss


class SoftmaxCrossEntropy(Loss):
    """The cross entropy between two probability distributions.

    The arguments should each have shape (batch_size, classes) or
    (batch_size, tasks, classes).  The labels should be probabilities, while the
    outputs should be logits that are converted to probabilities using a softmax
    function.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf
        output, labels = _make_tf_shapes_consistent(output, labels)
        output, labels = _ensure_float(output, labels)
        return tf.nn.softmax_cross_entropy_with_logits(labels, output)

    def _create_pytorch_loss(self):
        import torch
        ls = torch.nn.LogSoftmax(dim=-1)

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return -torch.sum(labels * ls(output), dim=-1)

        return loss


class SparseSoftmaxCrossEntropy(Loss):
    """The cross entropy between two probability distributions.

    The labels should have shape (batch_size) or (batch_size, tasks), and be
    integer class labels.  The outputs have shape (batch_size, classes) or
    (batch_size, tasks, classes) and be logits that are converted to probabilities
    using a softmax function.
    """

    def _compute_tf_loss(self, output, labels):
        import tensorflow as tf

        if len(labels.shape) == len(output.shape):
            labels = tf.squeeze(labels, axis=-1)

        labels = tf.cast(labels, tf.int32)

        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, output)

    def _create_pytorch_loss(self):
        import torch
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        def loss(output, labels):
            # Convert (batch_size, tasks, classes) to (batch_size, classes, tasks)
            # CrossEntropyLoss only supports (batch_size, classes, tasks)
            # This is for API consistency
            if len(output.shape) == 3:
                output = output.permute(0, 2, 1)

            if len(labels.shape) == len(output.shape):
                labels = labels.squeeze(-1)
            return ce_loss(output, labels.long())

        return loss


class VAE_ELBO(Loss):
    """The Variational AutoEncoder loss, KL Divergence Regularize + marginal log-likelihood.

    This losses based on _[1].
    ELBO(Evidence lower bound) lexically replaced Variational lower bound.
    BCE means marginal log-likelihood, and KLD means KL divergence with normal distribution.
    Added hyper parameter 'kl_scale' for KLD.

    The logvar and mu should have shape (batch_size, hidden_space).
    The x and reconstruction_x should have (batch_size, attribute).
    The kl_scale should be float.

    Examples
    --------
    Examples for calculating loss using constant tensor.

    batch_size = 2,
    hidden_space = 2,
    num of original attribute = 3
    >>> import numpy as np
    >>> import torch
    >>> import tensorflow as tf
    >>> logvar = np.array([[1.0,1.3],[0.6,1.2]])
    >>> mu = np.array([[0.2,0.7],[1.2,0.4]])
    >>> x = np.array([[0.9,0.4,0.8],[0.3,0,1]])
    >>> reconstruction_x = np.array([[0.8,0.3,0.7],[0.2,0,0.9]])

    Case tensorflow
    >>> VAE_ELBO()._compute_tf_loss(tf.constant(logvar), tf.constant(mu), tf.constant(x), tf.constant(reconstruction_x))
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.70165154, 0.76238271])>

    Case pytorch
    >>> (VAE_ELBO()._create_pytorch_loss())(torch.tensor(logvar), torch.tensor(mu), torch.tensor(x), torch.tensor(reconstruction_x))
    tensor([0.7017, 0.7624], dtype=torch.float64)


    References
    ----------
    .. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

    """

    def _compute_tf_loss(self, logvar, mu, x, reconstruction_x, kl_scale=1):
        import tensorflow as tf
        x, reconstruction_x = _make_tf_shapes_consistent(x, reconstruction_x)
        x, reconstruction_x = _ensure_float(x, reconstruction_x)
        BCE = tf.keras.losses.binary_crossentropy(x, reconstruction_x)
        KLD = VAE_KLDivergence()._compute_tf_loss(logvar, mu)
        return BCE + kl_scale * KLD

    def _create_pytorch_loss(self):
        import torch
        bce = torch.nn.BCELoss(reduction='none')

        def loss(logvar, mu, x, reconstruction_x, kl_scale=1):
            x, reconstruction_x = _make_pytorch_shapes_consistent(
                x, reconstruction_x)
            BCE = torch.mean(bce(reconstruction_x, x), dim=-1)
            KLD = (VAE_KLDivergence()._create_pytorch_loss())(logvar, mu)
            return BCE + kl_scale * KLD

        return loss


class VAE_KLDivergence(Loss):
    """The KL_divergence between hidden distribution and normal distribution.

    This loss represents KL divergence losses between normal distribution(using parameter of distribution)
    based on  _[1].

    The logvar should have shape (batch_size, hidden_space) and each term represents
    standard deviation of hidden distribution. The mean shuold have
    (batch_size, hidden_space) and each term represents mean of hidden distribtuon.

    Examples
    --------
    Examples for calculating loss using constant tensor.

    batch_size = 2,
    hidden_space = 2,
    >>> import numpy as np
    >>> import torch
    >>> import tensorflow as tf
    >>> logvar = np.array([[1.0,1.3],[0.6,1.2]])
    >>> mu = np.array([[0.2,0.7],[1.2,0.4]])

    Case tensorflow
    >>> VAE_KLDivergence()._compute_tf_loss(tf.constant(logvar), tf.constant(mu))
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.17381787, 0.51425203])>

    Case pytorch
    >>> (VAE_KLDivergence()._create_pytorch_loss())(torch.tensor(logvar), torch.tensor(mu))
    tensor([0.1738, 0.5143], dtype=torch.float64)

    References
    ----------
    .. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

    """

    def _compute_tf_loss(self, logvar, mu):
        import tensorflow as tf
        logvar, mu = _make_tf_shapes_consistent(logvar, mu)
        logvar, mu = _ensure_float(logvar, mu)
        return 0.5 * tf.reduce_mean(
            tf.square(mu) + tf.square(logvar) -
            tf.math.log(1e-20 + tf.square(logvar)) - 1, -1)

    def _create_pytorch_loss(self):
        import torch

        def loss(logvar, mu):
            logvar, mu = _make_pytorch_shapes_consistent(logvar, mu)
            return 0.5 * torch.mean(
                torch.square(mu) + torch.square(logvar) -
                torch.log(1e-20 + torch.square(logvar)) - 1, -1)

        return loss


class ShannonEntropy(Loss):
    """The ShannonEntropy of discrete-distribution.

    This loss represents shannon entropy based on _[1].

    The inputs should have shape (batch size, num of variable) and represents
    probabilites distribution.

    Examples
    --------
    Examples for calculating loss using constant tensor.

    batch_size = 2,
    num_of variable = variable,
    >>> import numpy as np
    >>> import torch
    >>> import tensorflow as tf
    >>> inputs = np.array([[0.7,0.3],[0.9,0.1]])

    Case tensorflow
    >>> ShannonEntropy()._compute_tf_loss(tf.constant(inputs))
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.30543215, 0.16254149])>

    Case pytorch
    >>> (ShannonEntropy()._create_pytorch_loss())(torch.tensor(inputs))
    tensor([0.3054, 0.1625], dtype=torch.float64)

    References
    ----------
    .. [1] Chen, Ricky Xiaofeng. "A Brief Introduction to Shannon’s Information Theory." arXiv preprint arXiv:1612.09316 (2016).

    """

    def _compute_tf_loss(self, inputs):
        import tensorflow as tf
        # extended one of probabilites to binary distribution
        if inputs.shape[-1] == 1:
            inputs = tf.concat([inputs, 1 - inputs], axis=-1)
        return tf.reduce_mean(-inputs * tf.math.log(1e-20 + inputs), -1)

    def _create_pytorch_loss(self):
        import torch

        def loss(inputs):
            # extended one of probabilites to binary distribution
            if inputs.shape[-1] == 1:
                inputs = torch.cat((inputs, 1 - inputs), dim=-1)
            return torch.mean(-inputs * torch.log(1e-20 + inputs), -1)

        return loss


class GlobalMutualInformationLoss(Loss):
    """
    Global-global encoding loss (comparing two full graphs).

    Compares the encodings of two molecular graphs and returns the loss between them based on the measure specified.
    The encodings are generated by two separate encoders in order to maximize the mutual information between the two encodings.

    Parameters
    ----------
    global_enc: torch.Tensor
        Features from a graph convolutional encoder.
    global_enc2: torch.Tensor
        Another set of features from a graph convolutional encoder.
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD', 'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average_loss: bool
        Whether to average the loss over the batch

    Returns
    -------
    loss: torch.Tensor
        Measure of mutual information between the encodings of the two graphs.

    References
    ----------
    .. [1] F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Maximization.” arXiv, Jan. 17, 2020. http://arxiv.org/abs/1908.01000

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem.models.losses as losses
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.infograph import InfoGraphEncoder
    >>> from deepchem.models.torch_models.layers import MultilayerPerceptron
    >>> graph_list = []
    >>> for i in range(3):
    ...     node_features = np.random.rand(5, 10)
    ...     edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    ...     edge_features = np.random.rand(5, 5)
    ...     graph_list.append(GraphData(node_features, edge_index, edge_features))
    >>> batch = BatchGraphData(graph_list).numpy_to_torch()
    >>> num_feat = 10
    >>> edge_dim = 5
    >>> dim = 4
    >>> encoder = InfoGraphEncoder(num_feat, edge_dim, dim)
    >>> encoding, feature_map = encoder(batch)
    >>> g_enc = MultilayerPerceptron(2 * dim, dim)(encoding)
    >>> g_enc2 = MultilayerPerceptron(2 * dim, dim)(encoding)
    >>> globalloss = losses.GlobalMutualInformationLoss()
    >>> loss = globalloss._create_pytorch_loss()(g_enc, g_enc2).detach().numpy()
    """

    def _create_pytorch_loss(self, measure='JSD', average_loss=True):
        import torch

        def loss(global_enc, global_enc2):
            device = global_enc.device
            num_graphs = global_enc.shape[0]
            pos_mask = torch.eye(num_graphs).to(device)
            neg_mask = 1 - pos_mask

            res = torch.mm(global_enc, global_enc2.t())

            E_pos = get_positive_expectation(res * pos_mask, measure,
                                             average_loss)
            E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
            E_neg = get_negative_expectation(res * neg_mask, measure,
                                             average_loss)
            E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

            return E_neg - E_pos

        return loss


class LocalMutualInformationLoss(Loss):
    """
    Local-global encoding loss (comparing a subgraph to the full graph).

    Compares the encodings of two molecular graphs and returns the loss between them based on the measure specified.
    The encodings are generated by two separate encoders in order to maximize the mutual information between the two encodings.

    Parameters
    ----------
    local_enc: torch.Tensor
        Features from a graph convolutional encoder.
    global_enc: torch.Tensor
        Another set of features from a graph convolutional encoder.
    batch_graph_index: graph_index: np.ndarray or torch.tensor, dtype int
        This vector indicates which graph the node belongs with shape [num_nodes,]. Only present in BatchGraphData, not in GraphData objects.
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD', 'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average_loss: bool
        Whether to average the loss over the batch

    Returns
    -------
    loss: torch.Tensor
        Measure of mutual information between the encodings of the two graphs.

    References
    ----------
    .. [1] F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Maximization.” arXiv, Jan. 17, 2020. http://arxiv.org/abs/1908.01000

    Example
    -------
    >>> import numpy as np
    >>> import deepchem.models.losses as losses
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.infograph import InfoGraphEncoder
    >>> from deepchem.models.torch_models.layers import MultilayerPerceptron
    >>> graph_list = []
    >>> for i in range(3):
    ...     node_features = np.random.rand(5, 10)
    ...     edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    ...     edge_features = np.random.rand(5, 5)
    ...     graph_list.append(GraphData(node_features, edge_index, edge_features))

    >>> batch = BatchGraphData(graph_list).numpy_to_torch()
    >>> num_feat = 10
    >>> edge_dim = 5
    >>> dim = 4
    >>> encoder = InfoGraphEncoder(num_feat, edge_dim, dim)
    >>> encoding, feature_map = encoder(batch)
    >>> g_enc = MultilayerPerceptron(2 * dim, dim)(encoding)
    >>> l_enc = MultilayerPerceptron(dim, dim)(feature_map)
    >>> localloss = losses.LocalMutualInformationLoss()
    >>> loss = localloss._create_pytorch_loss()(l_enc, g_enc, batch.graph_index).detach().numpy()
    """

    def _create_pytorch_loss(self, measure='JSD', average_loss=True):

        import torch

        def loss(local_enc, global_enc, batch_graph_index):
            device = local_enc.device
            num_graphs = global_enc.shape[0]
            num_nodes = local_enc.shape[0]

            pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
            neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
            for nodeidx, graphidx in enumerate(batch_graph_index):
                pos_mask[nodeidx][graphidx] = 1.
                neg_mask[nodeidx][graphidx] = 0.

            res = torch.mm(local_enc, global_enc.t())

            E_pos = get_positive_expectation(res * pos_mask, measure,
                                             average_loss)
            E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
            E_neg = get_negative_expectation(res * neg_mask, measure,
                                             average_loss)
            E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

            return E_neg - E_pos

        return loss


def get_positive_expectation(p_samples, measure='JSD', average_loss=True):
    """Computes the positive part of a divergence / difference.

    Parameters
    ----------
    p_samples: torch.Tensor
        Positive samples.
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD', 'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average: bool
        Average the result over samples.

    Returns
    -------
    Ep: torch.Tensor
        Positive part of the divergence / difference.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.losses import get_positive_expectation
    >>> p_samples = torch.tensor([0.5, 1.0, -0.5, -1.0])
    >>> measure = 'JSD'
    >>> result = get_positive_expectation(p_samples, measure)
    """
    import math

    import torch

    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = -torch.nn.functional.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - torch.nn.functional.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples**2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise ValueError('Unknown measure: {}'.format(measure))

    if average_loss:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure='JSD', average_loss=True):
    """Computes the negative part of a divergence / difference.

    Parameters
    ----------
    q_samples: torch.Tensor
        Negative samples.
    measure: str

    average: bool
        Average the result over samples.

    Returns
    -------
    Ep: torch.Tensor
        Negative part of the divergence / difference.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.losses import get_negative_expectation
    >>> q_samples = torch.tensor([0.5, 1.0, -0.5, -1.0])
    >>> measure = 'JSD'
    >>> result = get_negative_expectation(q_samples, measure)
    """
    import math

    import torch
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = torch.nn.functional.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = torch.nn.functional.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples**2) + 1.)**2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise ValueError('Unknown measure: {}'.format(measure))

    if average_loss:
        return Eq.mean()
    else:
        return Eq


def log_sum_exp(x, axis=None):
    """Log sum exp function.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor
    axis: int
        Axis to perform sum over

    Returns
    -------
    y: torch.Tensor
        Log sum exp of x

    """
    import torch
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


class GroverPretrainLoss(Loss):
    """GroverPretrainLoss

    The Grover Pretraining consists learning of atom embeddings and bond embeddings for
    a molecule. To this end, the learning consists of three tasks:
        1. Learning of atom vocabulary from atom embeddings and bond embeddings
        2. Learning of bond vocabulary from atom embeddings and bond embeddings
        3. Learning to predict functional groups from atom embedings readout and bond embeddings readout
    The loss function accepts atom vocabulary labels, bond vocabulary labels and functional group
    predictions produced by Grover model during pretraining as a dictionary and applies negative
    log-likelihood loss for atom vocabulary and bond vocabulary predictions and Binary Cross Entropy
    loss for functional group prediction and sums these to get overall loss.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.losses import GroverPretrainLoss
    >>> loss = GroverPretrainLoss()
    >>> loss_fn = loss._create_pytorch_loss()
    >>> batch_size = 3
    >>> output_dim = 10
    >>> fg_size = 8
    >>> atom_vocab_task_target = torch.ones(batch_size).type(torch.int64)
    >>> bond_vocab_task_target = torch.ones(batch_size).type(torch.int64)
    >>> fg_task_target = torch.ones(batch_size, fg_size)
    >>> atom_vocab_task_atom_pred = torch.zeros(batch_size, output_dim)
    >>> bond_vocab_task_atom_pred = torch.zeros(batch_size, output_dim)
    >>> atom_vocab_task_bond_pred = torch.zeros(batch_size, output_dim)
    >>> bond_vocab_task_bond_pred = torch.zeros(batch_size, output_dim)
    >>> fg_task_atom_from_atom = torch.zeros(batch_size, fg_size)
    >>> fg_task_atom_from_bond = torch.zeros(batch_size, fg_size)
    >>> fg_task_bond_from_atom = torch.zeros(batch_size, fg_size)
    >>> fg_task_bond_from_bond = torch.zeros(batch_size, fg_size)
    >>> result = loss_fn(atom_vocab_task_atom_pred, atom_vocab_task_bond_pred,
    ...     bond_vocab_task_atom_pred, bond_vocab_task_bond_pred, fg_task_atom_from_atom,
    ...     fg_task_atom_from_bond, fg_task_bond_from_atom, fg_task_bond_from_bond,
    ...     atom_vocab_task_target, bond_vocab_task_target, fg_task_target)

    Reference
    ---------
    .. [1] Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def _create_pytorch_loss(self):
        import torch
        import torch.nn as nn

        def loss(atom_vocab_task_atom_pred: torch.Tensor,
                 atom_vocab_task_bond_pred: torch.Tensor,
                 bond_vocab_task_atom_pred: torch.Tensor,
                 bond_vocab_task_bond_pred: torch.Tensor,
                 fg_task_atom_from_atom: torch.Tensor,
                 fg_task_atom_from_bond: torch.Tensor,
                 fg_task_bond_from_atom: torch.Tensor,
                 fg_task_bond_from_bond: torch.Tensor,
                 atom_vocab_task_target: torch.Tensor,
                 bond_vocab_task_target: torch.Tensor,
                 fg_task_target: torch.Tensor,
                 weights: Optional[List[Sequence]] = None,
                 dist_coff=0.1):
            """
            Parameters
            ----------
            atom_vocab_task_atom_pred: torch.Tensor
                Atom vocabulary prediction from atom embedding
            atom_vocab_task_bond_pred: torch.Tensor
                Atom vocabulary prediction from bond embedding
            bond_vocab_task_atom_pred: torch.Tensor
                Bond vocabulary prediction from atom embedding
            bond_vocab_task_bond_pred: torch.Tensor
                Bond vocabulary prediction from bond embedding
            fg_task_atom_from_atom: torch.Tensor
                Functional group prediction from atom embedding readout generated from atom embedding
            fg_task_atom_from_bond: torch.Tensor
                Functional group prediction from atom embedding readout generated from bond embedding
            fg_task_bond_from_atom: torch.Tensor
                Functional group prediction from bond embedding readout generated from atom embedding
            fg_task_bond_from_bond: torch.Tensor
                Functional group prediction from bond embedding readout generated from bond embedding
            atom_vocab_task_target: torch.Tensor
                Targets for atom vocabulary prediction
            bond_vocab_task_target: torch.Tensor
                Targets for bond vocabulary prediction
            fg_task_target: torch.Tensor
                Targets for functional groups
            dist_coff: float, default 0.1
                Loss term weight for weighting closeness between embedding generated from atom hidden state and bond hidden state in atom vocabulary and bond vocabulary prediction tasks.

            Returns
            -------
            loss: torch.Tensor
                loss value
            """
            av_task_loss = nn.NLLLoss(reduction="mean")  # same for av and bv
            fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")
            av_task_dist_loss = nn.MSELoss(reduction="mean")
            fg_task_dist_loss = nn.MSELoss(reduction="mean")

            sigmoid = nn.Sigmoid()

            av_atom_loss = av_task_loss(atom_vocab_task_atom_pred,
                                        atom_vocab_task_target)
            av_bond_loss = av_task_loss(atom_vocab_task_bond_pred,
                                        atom_vocab_task_target)
            bv_atom_loss = av_task_loss(bond_vocab_task_atom_pred,
                                        bond_vocab_task_target)
            bv_bond_loss = av_task_loss(bond_vocab_task_bond_pred,
                                        bond_vocab_task_target)

            fg_atom_from_atom_loss = fg_task_loss(fg_task_atom_from_atom,
                                                  fg_task_target)
            fg_atom_from_bond_loss = fg_task_loss(fg_task_atom_from_bond,
                                                  fg_task_target)
            fg_bond_from_atom_loss = fg_task_loss(fg_task_bond_from_atom,
                                                  fg_task_target)
            fg_bond_from_bond_loss = fg_task_loss(fg_task_bond_from_bond,
                                                  fg_task_target)

            av_dist_loss = av_task_dist_loss(atom_vocab_task_atom_pred,
                                             atom_vocab_task_bond_pred)
            bv_dist_loss = av_task_dist_loss(bond_vocab_task_atom_pred,
                                             bond_vocab_task_bond_pred)

            fg_atom_dist_loss = fg_task_dist_loss(
                sigmoid(fg_task_atom_from_atom),
                sigmoid(fg_task_atom_from_bond))
            fg_bond_dist_loss = fg_task_dist_loss(
                sigmoid(fg_task_bond_from_atom),
                sigmoid(fg_task_bond_from_bond))

            av_bv_loss = av_atom_loss + av_bond_loss + bv_atom_loss + bv_bond_loss
            fg_loss = fg_atom_from_atom_loss + fg_atom_from_bond_loss + fg_bond_from_atom_loss + fg_bond_from_bond_loss
            fg_dist_loss = fg_atom_dist_loss + fg_bond_dist_loss

            # NOTE The below comment is from original source code
            # dist_loss = av_dist_loss + bv_dist_loss + fg_dist_loss
            # return av_loss + fg_loss + dist_coff * dist_loss
            overall_loss = av_bv_loss + fg_loss + dist_coff * (
                av_dist_loss + bv_dist_loss + fg_dist_loss)

            # return overall_loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss
            # We just return overall_loss since TorchModel can handle only a single loss
            return overall_loss

        return loss


class EdgePredictionLoss(Loss):
    """
    EdgePredictionLoss is an unsupervised graph edge prediction loss function that calculates the loss based on the similarity between node embeddings for positive and negative edge pairs. This loss function is designed for graph neural networks and is particularly useful for pre-training tasks.

    This loss function encourages the model to learn node embeddings that can effectively distinguish between true edges (positive samples) and false edges (negative samples) in the graph.

    The loss is computed by comparing the similarity scores (dot product) of node embeddings for positive and negative edge pairs. The goal is to maximize the similarity for positive pairs and minimize it for negative pairs.

    To use this loss function, the input must be a BatchGraphData object transformed by the negative_edge_sampler. The loss function takes the node embeddings and the input graph data (with positive and negative edge pairs) as inputs and returns the edge prediction loss.

    Examples
    --------
    >>> from deepchem.models.losses import EdgePredictionLoss
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.gnn import negative_edge_sampler
    >>> import torch
    >>> import numpy as np
    >>> emb_dim = 8
    >>> num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
    >>> num_node_features, num_edge_features = 32, 32
    >>> edge_index_list = [
    ...     np.array([[0, 1], [1, 2]]),
    ...     np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
    ...     np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
    ... ]
    >>> graph_list = [
    ...     GraphData(node_features=np.random.random_sample(
    ...         (num_nodes_list[i], num_node_features)),
    ...               edge_index=edge_index_list[i],
    ...               edge_features=np.random.random_sample(
    ...                   (num_edge_list[i], num_edge_features)),
    ...               node_pos_features=None) for i in range(len(num_edge_list))
    ... ]
    >>> batched_graph = BatchGraphData(graph_list)
    >>> batched_graph = batched_graph.numpy_to_torch()
    >>> neg_sampled = negative_edge_sampler(batched_graph)
    >>> embedding = np.random.random((sum(num_nodes_list), emb_dim))
    >>> embedding = torch.from_numpy(embedding)
    >>> loss_func = EdgePredictionLoss()._create_pytorch_loss()
    >>> loss = loss_func(embedding, neg_sampled)

    References
    ----------
    .. [1] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).
    """

    def _create_pytorch_loss(self):
        import torch
        self.criterion = torch.nn.BCEWithLogitsLoss()

        def loss(node_emb, inputs):
            positive_score = torch.sum(node_emb[inputs.edge_index[0, ::2]] *
                                       node_emb[inputs.edge_index[1, ::2]],
                                       dim=1)
            negative_score = torch.sum(node_emb[inputs.negative_edge_index[0]] *
                                       node_emb[inputs.negative_edge_index[1]],
                                       dim=1)

            edge_pred_loss = self.criterion(
                positive_score,
                torch.ones_like(positive_score)) + self.criterion(
                    negative_score, torch.zeros_like(negative_score))
            return edge_pred_loss

        return loss


class GraphNodeMaskingLoss(Loss):
    """
    GraphNodeMaskingLoss is an unsupervised graph node masking loss function that calculates the loss based on the predicted node labels and true node labels. This loss function is designed for graph neural networks and is particularly useful for pre-training tasks.

    This loss function encourages the model to learn node embeddings that can effectively predict the masked node labels in the graph.

    The loss is computed using the CrossEntropyLoss between the predicted node labels and the true node labels.

    To use this loss function, the input must be a BatchGraphData object transformed by the mask_nodes function. The loss function takes the predicted node labels, predicted edge labels, and the input graph data (with masked node labels) as inputs and returns the node masking loss.

    Parameters
    ----------
    pred_node: torch.Tensor
        Predicted node labels
    pred_edge: Optional(torch.Tensor)
        Predicted edge labels
    inputs: BatchGraphData
        Input graph data with masked node and edge labels

    Examples
    --------
    >>> from deepchem.models.losses import GraphNodeMaskingLoss
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.gnn import mask_nodes
    >>> import torch
    >>> import numpy as np
    >>> num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
    >>> num_node_features, num_edge_features = 32, 32
    >>> edge_index_list = [
    ...     np.array([[0, 1], [1, 2]]),
    ...     np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
    ...     np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
    ... ]
    >>> graph_list = [
    ...     GraphData(node_features=np.random.random_sample(
    ...         (num_nodes_list[i], num_node_features)),
    ...               edge_index=edge_index_list[i],
    ...               edge_features=np.random.random_sample(
    ...                   (num_edge_list[i], num_edge_features)),
    ...               node_pos_features=None) for i in range(len(num_edge_list))
    ... ]
    >>> batched_graph = BatchGraphData(graph_list)
    >>> batched_graph = batched_graph.numpy_to_torch()
    >>> masked_graph = mask_nodes(batched_graph, 0.1)
    >>> pred_node = torch.randn((sum(num_nodes_list), num_node_features))
    >>> pred_edge = torch.randn((sum(num_edge_list), num_edge_features))
    >>> loss_func = GraphNodeMaskingLoss()._create_pytorch_loss()
    >>> loss = loss_func(pred_node[masked_graph.masked_node_indices],
    ...                  pred_edge[masked_graph.connected_edge_indices], masked_graph)

    References
    ----------
    .. [1] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).
    """

    def _create_pytorch_loss(self, mask_edge=True):
        import torch
        self.mask_edge = mask_edge
        self.criterion = torch.nn.CrossEntropyLoss()

        def loss(pred_node, pred_edge, inputs):

            # loss for nodes
            loss = self.criterion(pred_node, inputs.mask_node_label)

            if self.mask_edge:
                loss += self.criterion(pred_edge, inputs.mask_edge_label)
            return loss

        return loss


class GraphEdgeMaskingLoss(Loss):
    """
    GraphEdgeMaskingLoss is an unsupervised graph edge masking loss function that calculates the loss based on the predicted edge labels and true edge labels. This loss function is designed for graph neural networks and is particularly useful for pre-training tasks.

    This loss function encourages the model to learn node embeddings that can effectively predict the masked edge labels in the graph.

    The loss is computed using the CrossEntropyLoss between the predicted edge labels and the true edge labels.

    To use this loss function, the input must be a BatchGraphData object transformed by the mask_edges function. The loss function takes the predicted edge labels and the true edge labels as inputs and returns the edge masking loss.

    Parameters
    ----------
    pred_edge: torch.Tensor
        Predicted edge labels.
    inputs: BatchGraphData
        Input graph data (with masked edge labels).

    Examples
    --------
    >>> from deepchem.models.losses import GraphEdgeMaskingLoss
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.gnn import mask_edges
    >>> import torch
    >>> import numpy as np
    >>> num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
    >>> num_node_features, num_edge_features = 32, 32
    >>> edge_index_list = [
    ...     np.array([[0, 1], [1, 2]]),
    ...     np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
    ...     np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
    ... ]
    >>> graph_list = [
    ...     GraphData(node_features=np.random.random_sample(
    ...         (num_nodes_list[i], num_node_features)),
    ...               edge_index=edge_index_list[i],
    ...               edge_features=np.random.random_sample(
    ...                   (num_edge_list[i], num_edge_features)),
    ...               node_pos_features=None) for i in range(len(num_edge_list))
    ... ]
    >>> batched_graph = BatchGraphData(graph_list)
    >>> batched_graph = batched_graph.numpy_to_torch()
    >>> masked_graph = mask_edges(batched_graph, .1)
    >>> pred_edge = torch.randn((sum(num_edge_list), num_edge_features))
    >>> loss_func = GraphEdgeMaskingLoss()._create_pytorch_loss()
    >>> loss = loss_func(pred_edge[masked_graph.masked_edge_idx], masked_graph)

    References
    ----------
    .. [1] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).
    """

    def _create_pytorch_loss(self):
        import torch
        self.criterion = torch.nn.CrossEntropyLoss()

        def loss(pred_edge, inputs):
            # converting the binary classification to multiclass classification
            labels = torch.argmax(inputs.mask_edge_label, dim=1)
            loss = self.criterion(pred_edge, labels)
            return loss

        return loss


class DeepGraphInfomaxLoss(Loss):
    """
    Loss that maximizes mutual information between local node representations and a pooled global graph representation. This is to encourage nearby nodes to have similar embeddings.

    Parameters
    ----------
    positive_score: torch.Tensor
        Positive score. This score measures the similarity between the local node embeddings (`node_emb`) and the global graph representation (`positive_expanded_summary_emb`) derived from the same graph.
        The goal is to maximize this score, as it indicates that the local node embeddings and the global graph representation are highly correlated, capturing the mutual information between them.
    negative_score: torch.Tensor
        Negative score. This score measures the similarity between the local node embeddings (`node_emb`) and the global graph representation (`negative_expanded_summary_emb`) derived from a different graph (shifted by one position in this case).
        The goal is to minimize this score, as it indicates that the local node embeddings and the global graph representation from different graphs are not correlated, ensuring that the model learns meaningful representations that are specific to each graph.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from deepchem.feat.graph_data import GraphData
    >>> from torch_geometric.nn import global_mean_pool
    >>> from deepchem.models.losses import DeepGraphInfomaxLoss
    >>> x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    >>> edge_index = np.array([[0, 1, 2, 0, 3], [1, 0, 1, 3, 2]])
    >>> graph_index = np.array([0, 0, 1, 1])
    >>> data = GraphData(node_features=x, edge_index=edge_index, graph_index=graph_index).numpy_to_torch()
    >>> graph_infomax_loss = DeepGraphInfomaxLoss()._create_pytorch_loss()
    >>> # Initialize node_emb randomly
    >>> num_nodes = data.num_nodes
    >>> embedding_dim = 8
    >>> node_emb = torch.randn(num_nodes, embedding_dim)
    >>> # Compute the global graph representation
    >>> summary_emb = global_mean_pool(node_emb, data.graph_index)
    >>> # Compute positive and negative scores
    >>> positive_score = torch.matmul(node_emb, summary_emb.t())
    >>> negative_score = torch.matmul(node_emb, summary_emb.roll(1, dims=0).t())
    >>> loss = graph_infomax_loss(positive_score, negative_score)

    References
    ----------
    .. [1] Veličković, P. et al. Deep Graph Infomax. Preprint at https://doi.org/10.48550/arXiv.1809.10341 (2018).

    """

    def _create_pytorch_loss(self):
        import torch
        self.criterion = torch.nn.BCEWithLogitsLoss()

        def loss(positive_score, negative_score):

            return self.criterion(
                positive_score,
                torch.ones_like(positive_score)) + self.criterion(
                    negative_score, torch.zeros_like(negative_score))

        return loss


class GraphContextPredLoss(Loss):
    """
    GraphContextPredLoss is a loss function designed for graph neural networks that aims to predict the context of a node given its substructure. The context of a node is essentially the ring of nodes around it outside of an inner k1-hop diameter and inside an outer k2-hop diameter.

    This loss compares the representation of a node's neighborhood with the representation of the node's context. It then uses negative sampling to compare the representation of the node's neighborhood with the representation of a random node's context.

    Parameters
    ----------
    mode: str
        The mode of the model. It can be either "cbow" (continuous bag of words) or "skipgram".
    neg_samples: int
        The number of negative samples to use for negative sampling.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.losses import GraphContextPredLoss
    >>> substruct_rep = torch.randn(4, 8)
    >>> overlapped_node_rep = torch.randn(8, 8)
    >>> context_rep = torch.randn(4, 8)
    >>> neg_context_rep = torch.randn(2 * 4, 8)
    >>> overlapped_context_size = torch.tensor([2, 2, 2, 2])
    >>> mode = "cbow"
    >>> neg_samples = 2
    >>> graph_context_pred_loss = GraphContextPredLoss()._create_pytorch_loss(mode, neg_samples)
    >>> loss = graph_context_pred_loss(substruct_rep, overlapped_node_rep, context_rep, neg_context_rep, overlapped_context_size)
    """

    def _create_pytorch_loss(self, mode, neg_samples):
        import torch

        from deepchem.models.torch_models.gnn import cycle_index
        self.mode = mode
        self.neg_samples = neg_samples
        self.criterion = torch.nn.BCEWithLogitsLoss()

        def loss(substruct_rep, overlapped_node_rep, context_rep,
                 neg_context_rep, overlap_size):

            if self.mode == "cbow":
                # positive context prediction is the dot product of substructure representation and true context representation
                pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
                # negative context prediction is the dot product of substructure representation and negative (random) context representation.
                pred_neg = torch.sum(substruct_rep.repeat(
                    (self.neg_samples, 1)) * neg_context_rep,
                                     dim=1)

            elif self.mode == "skipgram":
                expanded_substruct_rep = torch.cat(
                    [substruct_rep[i].repeat((i, 1)) for i in overlap_size],
                    dim=0)
                # positive substructure prediction is the dot product of expanded substructure representation and true overlapped node representation.
                pred_pos = torch.sum(expanded_substruct_rep *
                                     overlapped_node_rep,
                                     dim=1)

                # shift indices of substructures to create negative examples
                shifted_expanded_substruct_rep = []
                for j in range(self.neg_samples):
                    shifted_substruct_rep = substruct_rep[cycle_index(
                        len(substruct_rep), j + 1)]
                    shifted_expanded_substruct_rep.append(
                        torch.cat([
                            shifted_substruct_rep[i].repeat((i, 1))
                            for i in overlap_size
                        ],
                                  dim=0))

                shifted_expanded_substruct_rep = torch.cat(
                    shifted_expanded_substruct_rep, dim=0)
                # negative substructure prediction is the dot product of shifted expanded substructure representation and true overlapped node representation.
                pred_neg = torch.sum(shifted_expanded_substruct_rep *
                                     overlapped_node_rep.repeat(
                                         (self.neg_samples, 1)),
                                     dim=1)

            else:
                raise ValueError(
                    "Invalid mode. Must be either cbow or skipgram.")

            # Compute the loss for positive and negative context representations
            loss_pos = self.criterion(
                pred_pos.double(),
                torch.ones(len(pred_pos)).to(pred_pos.device).double())
            loss_neg = self.criterion(
                pred_neg.double(),
                torch.zeros(len(pred_neg)).to(pred_neg.device).double())

            # The final loss is the sum of positive and negative context losses
            loss = loss_pos + self.neg_samples * loss_neg
            return loss

        return loss


class DensityProfileLoss(Loss):
    """
    Loss for the density profile entry type for Quantum Chemistry calculations.
    It is an integration of the squared difference between ground truth and calculated
    values, at all spaces in the integration grid.

    Examples
    --------
    >>> from deepchem.models.losses import DensityProfileLoss
    >>> import torch
    >>> volume = torch.Tensor([2.0])
    >>> output = torch.Tensor([3.0])
    >>> labels = torch.Tensor([4.0])
    >>> loss = (DensityProfileLoss()._create_pytorch_loss(volume))(output, labels)
    >>> # Generating volume tensor for an entry object:
    >>> from deepchem.feat.dft_data import DFTEntry
    >>> e_type = 'dens'
    >>> true_val = 0
    >>> systems =[{'moldesc': 'H 0.86625 0 0; F -0.86625 0 0','basis' : '6-311++G(3df,3pd)'}]
    >>> dens_entry_for_HF = DFTEntry.create(e_type, true_val, systems)
    >>> grid = (dens_entry_for_HF).get_integration_grid()

    The 6-311++G(3df,3pd) basis for atomz 1 does not exist, but we will download it
    Downloaded to /usr/share/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/01.gaussian94
    The 6-311++G(3df,3pd) basis for atomz 9 does not exist, but we will download it
    Downloaded to /usr/share/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/09.gaussian94

    >>> volume = grid.get_dvolume()

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/deepchem/deepchem/blob/0bc3139bb99ae7700ba2325a6756e33b6c327842/deepchem/models/dft/dftxc.py
    """

    def _create_pytorch_loss(self, volume):
        """
        Parameters
        ----------
        volume: torch.Tensor
            Shape of the tensor depends on the molecule/crystal and the integration grid
        """
        import torch

        def loss(output, labels):
            output, labels = _make_pytorch_shapes_consistent(output, labels)
            return torch.sum((labels - output)**2 * volume)

        return loss


class NTXentMultiplePositives(Loss):
    """
    This is a modification of the NTXent loss function from Chen [1]_. This loss is designed for contrastive learning of molecular representations, comparing the similarity of a molecule's latent representation to positive and negative samples.

    The modifications proposed in [2]_ enable multiple conformers to be used as positive samples.

    This loss function is designed for graph neural networks and is particularly useful for unsupervised pre-training tasks.

    Parameters
    ----------
    norm : bool, optional (default=True)
        Whether to normalize the similarity matrix.
    tau : float, optional (default=0.5)
        Temperature parameter for the similarity matrix.
    uniformity_reg : float, optional (default=0)
        Regularization weight for the uniformity loss.
    variance_reg : float, optional (default=0)
        Regularization weight for the variance loss.
    covariance_reg : float, optional (default=0)
        Regularization weight for the covariance loss.
    conformer_variance_reg : float, optional (default=0)
        Regularization weight for the conformer variance loss.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.losses import NTXentMultiplePositives
    >>> z1 = torch.randn(4, 8)
    >>> z2 = torch.randn(4 * 3, 8)
    >>> ntxent_loss = NTXentMultiplePositives(norm=True, tau=0.5)
    >>> loss_fn = ntxent_loss._create_pytorch_loss()
    >>> loss = loss_fn(z1, z2)

    References
    ----------
    .. [1] Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. A Simple Framework for Contrastive Learning of Visual Representations. Preprint at https://doi.org/10.48550/arXiv.2002.05709 (2020).

    .. [2] Stärk, H. et al. 3D Infomax improves GNNs for Molecular Property Prediction. Preprint at https://doi.org/10.48550/arXiv.2110.04126 (2022).
    """

    def __init__(self,
                 norm: bool = True,
                 tau: float = 0.5,
                 uniformity_reg=0,
                 variance_reg=0,
                 covariance_reg=0,
                 conformer_variance_reg=0) -> None:
        super(NTXentMultiplePositives, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

    def _create_pytorch_loss(self):
        import torch
        from torch import Tensor

        def std_loss(x: Tensor) -> Tensor:
            """
            Compute the standard deviation loss.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            loss : torch.Tensor
                The standard deviation loss.
            """
            std = torch.sqrt(x.var(dim=0) + 1e-04)
            return torch.mean(torch.relu(1 - std))

        def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
            """
            Compute the uniformity loss.

            Parameters
            ----------
            x1 : torch.Tensor
                First input tensor.
            x2 : torch.Tensor
                Second input tensor.
            t : int, optional (default=2)
                Exponent for the squared Euclidean distance.

            Returns
            -------
            loss : torch.Tensor
                The uniformity loss.
            """
            sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
            uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
            sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
            uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
            return (uniformity_x1 + uniformity_x2) / 2

        def cov_loss(x: Tensor) -> Tensor:
            """
            Compute the covariance loss.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            loss : torch.Tensor
                The covariance loss.
            """
            batch_size, metric_dim = x.size()
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (batch_size - 1)
            off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim +
                                                   1)[:, 1:].flatten()
            return off_diag_cov.pow_(2).sum() / metric_dim

        def loss(z1: Tensor, z2: Tensor) -> Tensor:
            """
            Compute the NTXentMultiplePositives loss.

            Parameters
            ----------
            z1 : torch.Tensor
                First input tensor with shape (batch_size, metric_dim).
            z2 : torch.Tensor
                Second input tensor with shape (batch_size * num_conformers, metric_dim).

            Returns
            -------
            loss : torch.Tensor
                The NTXentMultiplePositives loss.
            """
            batch_size, metric_dim = z1.size()
            z2 = z2.view(batch_size, -1,
                         metric_dim)  # [batch_size, num_conformers, metric_dim]
            z2 = z2.view(batch_size, -1,
                         metric_dim)  # [batch_size, num_conformers, metric_dim]

            sim_matrix = torch.einsum(
                'ik,juk->iju', z1,
                z2)  # [batch_size, batch_size, num_conformers]

            if self.norm:
                z1_abs = z1.norm(dim=1)
                z2_abs = z2.norm(dim=2)
                sim_matrix = sim_matrix / torch.einsum('i,ju->iju', z1_abs,
                                                       z2_abs)

            sim_matrix = torch.exp(
                sim_matrix /
                self.tau)  # [batch_size, batch_size, num_conformers]

            sim_matrix = sim_matrix.sum(dim=2)  # [batch_size, batch_size]
            pos_sim = torch.diagonal(sim_matrix)  # [batch_size]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = -torch.log(loss).mean()

            if self.variance_reg > 0:
                loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
            if self.conformer_variance_reg > 0:
                std = torch.sqrt(z2.var(dim=1) + 1e-04)
                std_conf_loss = torch.mean(torch.relu(1 - std))
                loss += self.conformer_variance_reg * std_conf_loss
            if self.covariance_reg > 0:
                loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
            if self.uniformity_reg > 0:
                loss += self.uniformity_reg * uniformity_loss(z1, z2)
            return loss

        return loss


def _make_tf_shapes_consistent(output, labels):
    """Try to make inputs have the same shape by adding dimensions of size 1."""
    import tensorflow as tf
    shape1 = output.shape
    shape2 = labels.shape
    len1 = len(shape1)
    len2 = len(shape2)
    if len1 == len2:
        return (output, labels)
    if isinstance(shape1, tf.TensorShape):
        shape1 = tuple(shape1.as_list())
    if isinstance(shape2, tf.TensorShape):
        shape2 = tuple(shape2.as_list())
    if len1 > len2 and all(i == 1 for i in shape1[len2:]):
        for i in range(len1 - len2):
            labels = tf.expand_dims(labels, -1)
        return (output, labels)
    if len2 > len1 and all(i == 1 for i in shape2[len1:]):
        for i in range(len2 - len1):
            output = tf.expand_dims(output, -1)
        return (output, labels)
    raise ValueError(
        "Incompatible shapes for outputs and labels: %s versus %s" %
        (str(shape1), str(shape2)))


def _make_pytorch_shapes_consistent(output, labels):
    """Try to make inputs have the same shape by adding dimensions of size 1."""
    import torch
    shape1 = output.shape
    shape2 = labels.shape
    len1 = len(shape1)
    len2 = len(shape2)
    if len1 == len2:
        return (output, labels)
    shape1 = tuple(shape1)
    shape2 = tuple(shape2)
    if len1 > len2 and all(i == 1 for i in shape1[len2:]):
        for i in range(len1 - len2):
            labels = torch.unsqueeze(labels, -1)
        return (output, labels)
    if len2 > len1 and all(i == 1 for i in shape2[len1:]):
        for i in range(len2 - len1):
            output = torch.unsqueeze(output, -1)
        return (output, labels)
    raise ValueError(
        "Incompatible shapes for outputs and labels: %s versus %s" %
        (str(shape1), str(shape2)))


def _ensure_float(output, labels):
    """Make sure the outputs and labels are both floating point types."""
    import tensorflow as tf
    if output.dtype not in (tf.float32, tf.float64):
        output = tf.cast(output, tf.float32)
    if labels.dtype not in (tf.float32, tf.float64):
        labels = tf.cast(labels, tf.float32)
    return (output, labels)

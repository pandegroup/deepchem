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
    return torch.nn.L1Loss(reduction='none')


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
    return torch.nn.SmoothL1Loss(reduction='none')


class L2Loss(Loss):
  """The squared difference between the true and predicted values."""

  def _compute_tf_loss(self, output, labels):
    import tensorflow as tf
    output, labels = _make_tf_shapes_consistent(output, labels)
    output, labels = _ensure_float(output, labels)
    return tf.square(output - labels)

  def _create_pytorch_loss(self):
    import torch
    return torch.nn.MSELoss(reduction='none')


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
      return torch.mean(
          torch.pow(
              torch.maximum(1 - torch.multiply(labels, output),
                            torch.tensor(0)), 2),
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
    ls = torch.nn.LogSoftmax(dim=1)

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
      x, reconstruction_x = _make_pytorch_shapes_consistent(x, reconstruction_x)
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
    #extended one of probabilites to binary distribution
    if inputs.shape[-1] == 1:
      inputs = tf.concat([inputs, 1 - inputs], axis=-1)
    return tf.reduce_mean(-inputs * tf.math.log(1e-20 + inputs), -1)

  def _create_pytorch_loss(self):
    import torch

    def loss(inputs):
      #extended one of probabilites to binary distribution
      if inputs.shape[-1] == 1:
        inputs = torch.cat((inputs, 1 - inputs), dim=-1)
      return torch.mean(-inputs * torch.log(1e-20 + inputs), -1)

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
  raise ValueError("Incompatible shapes for outputs and labels: %s versus %s" %
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
  raise ValueError("Incompatible shapes for outputs and labels: %s versus %s" %
                   (str(shape1), str(shape2)))


def _ensure_float(output, labels):
  """Make sure the outputs and labels are both floating point types."""
  import tensorflow as tf
  if output.dtype not in (tf.float32, tf.float64):
    output = tf.cast(output, tf.float32)
  if labels.dtype not in (tf.float32, tf.float64):
    labels = tf.cast(labels, tf.float32)
  return (output, labels)

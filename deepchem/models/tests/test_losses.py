import deepchem.models.losses as losses
import unittest
import numpy as np

try:
  import tensorflow as tf
  has_tensorflow = True
except:
  has_tensorflow = False

try:
  import torch
  has_pytorch = True
except:
  has_pytorch = False


class TestLosses(unittest.TestCase):
  """Test loss functions."""

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_l1_loss_tf(self):
    """Test L1Loss."""
    loss = losses.L1Loss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [[0.1, 0.2], [0.6, 0.6]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_l1_loss_pytorch(self):
    """Test L1Loss."""
    loss = losses.L1Loss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [[0.1, 0.2], [0.6, 0.6]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_huber_loss_tf(self):
    """Test HuberLoss."""
    loss = losses.HuberLoss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    result = np.mean(loss._compute_tf_loss(outputs, labels).numpy())
    expected = 0.67125
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_huber_loss_pytorch(self):
    """Test HuberLoss."""
    loss = losses.HuberLoss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    result = np.mean(loss._create_pytorch_loss()(outputs, labels).numpy())
    expected = 0.67125
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_l2_loss_tf(self):
    """Test L2Loss."""
    loss = losses.L2Loss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [[0.1**2, 0.2**2], [0.6**2, 0.6**2]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_l2_loss_pytorch(self):
    """Test L2Loss."""
    loss = losses.L2Loss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [[0.1**2, 0.2**2], [0.6**2, 0.6**2]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_hinge_loss_tf(self):
    """Test HingeLoss."""
    loss = losses.HingeLoss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [np.mean([0.9, 1.8]), np.mean([1.4, 0.4])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_hinge_loss_pytorch(self):
    """Test HingeLoss."""
    loss = losses.HingeLoss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [np.mean([0.9, 1.8]), np.mean([1.4, 0.4])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_squared_hinge_loss_tf(self):
    """Test SquaredHingeLoss."""
    loss = losses.SquaredHingeLoss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [np.mean([0.8100, 3.2400]), np.mean([1.9600, 0.1600])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_squared_hinge_loss_pytorch(self):
    """Test SquaredHingeLoss."""
    loss = losses.SquaredHingeLoss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [np.mean([0.8100, 3.2400]), np.mean([1.9600, 0.1600])]
    assert np.allclose(expected, result)

  def test_poisson_loss_tf(self):
    """Test PoissonLoss."""
    loss = losses.PoissonLoss()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = 0.75986
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_poisson_loss_pytorch(self):
    """Test PoissonLoss."""
    loss = losses.PoissonLoss()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = 0.75986
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_binary_cross_entropy_tf(self):
    """Test BinaryCrossEntropy."""
    loss = losses.BinaryCrossEntropy()
    outputs = tf.constant([[0.1, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [
        -np.mean([np.log(0.9), np.log(0.8)]),
        -np.mean([np.log(0.4), np.log(0.4)])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_binary_cross_entropy_pytorch(self):
    """Test BinaryCrossEntropy."""
    loss = losses.BinaryCrossEntropy()
    outputs = torch.tensor([[0.1, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [
        -np.mean([np.log(0.9), np.log(0.8)]),
        -np.mean([np.log(0.4), np.log(0.4)])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_categorical_cross_entropy_tf(self):
    """Test CategoricalCrossEntropy."""
    loss = losses.CategoricalCrossEntropy()
    outputs = tf.constant([[0.2, 0.8], [0.4, 0.6]])
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    expected = [-np.log(0.8), -np.log(0.4)]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_categorical_cross_entropy_pytorch(self):
    """Test CategoricalCrossEntropy."""
    loss = losses.CategoricalCrossEntropy()
    outputs = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    expected = [-np.log(0.8), -np.log(0.4)]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_sigmoid_cross_entropy_tf(self):
    """Test SigmoidCrossEntropy."""
    loss = losses.SigmoidCrossEntropy()
    y = [[0.1, 0.8], [0.4, 0.6]]
    outputs = tf.constant(y)
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    sigmoid = 1.0 / (1.0 + np.exp(-np.array(y)))
    expected = [[-np.log(1 - sigmoid[0, 0]), -np.log(sigmoid[0, 1])],
                [-np.log(sigmoid[1, 0]), -np.log(1 - sigmoid[1, 1])]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_sigmoid_cross_entropy_pytorch(self):
    """Test SigmoidCrossEntropy."""
    loss = losses.SigmoidCrossEntropy()
    y = [[0.1, 0.8], [0.4, 0.6]]
    outputs = torch.tensor(y)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    sigmoid = 1.0 / (1.0 + np.exp(-np.array(y)))
    expected = [[-np.log(1 - sigmoid[0, 0]), -np.log(sigmoid[0, 1])],
                [-np.log(sigmoid[1, 0]), -np.log(1 - sigmoid[1, 1])]]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_softmax_cross_entropy_tf(self):
    """Test SoftmaxCrossEntropy."""
    loss = losses.SoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = tf.constant(y)
    labels = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_softmax_cross_entropy_pytorch(self):
    """Test SoftmaxCrossEntropy."""
    loss = losses.SoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = torch.tensor(y)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_sparse_softmax_cross_entropy_tf(self):
    """Test SparseSoftmaxCrossEntropy."""
    loss = losses.SparseSoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = tf.constant(y)
    labels = tf.constant([1, 0])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

    labels = tf.constant([[1], [0]])
    result = loss._compute_tf_loss(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_sparse_softmax_cross_entropy_pytorch(self):
    """Test SparseSoftmaxCrossEntropy."""
    loss = losses.SparseSoftmaxCrossEntropy()
    y = np.array([[0.1, 0.8], [0.4, 0.6]])
    outputs = torch.tensor(y)
    labels = torch.tensor([1, 0])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

    labels = torch.tensor([[1], [0]])
    result = loss._create_pytorch_loss()(outputs, labels).numpy()
    softmax = np.exp(y) / np.expand_dims(np.sum(np.exp(y), axis=1), 1)
    expected = [-np.log(softmax[0, 1]), -np.log(softmax[1, 0])]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_VAE_ELBO_tf(self):
    """."""
    loss = losses.VAE_ELBO()
    logvar = tf.constant([[1.0, 1.3], [0.6, 1.2]])
    mu = tf.constant([[0.2, 0.7], [1.2, 0.4]])
    x = tf.constant([[0.9, 0.4, 0.8], [0.3, 0, 1]])
    reconstruction_x = tf.constant([[0.8, 0.3, 0.7], [0.2, 0, 0.9]])
    result = loss._compute_tf_loss(logvar, mu, x, reconstruction_x).numpy()
    expected = [
        0.5 * np.mean([
            0.04 + 1.0 - np.log(1e-20 + 1.0) - 1,
            0.49 + 1.69 - np.log(1e-20 + 1.69) - 1
        ]) - np.mean(
            np.array([0.9, 0.4, 0.8]) * np.log([0.8, 0.3, 0.7]) +
            np.array([0.1, 0.6, 0.2]) * np.log([0.2, 0.7, 0.3])),
        0.5 * np.mean([
            1.44 + 0.36 - np.log(1e-20 + 0.36) - 1,
            0.16 + 1.44 - np.log(1e-20 + 1.44) - 1
        ]) - np.mean(
            np.array([0.3, 0, 1]) * np.log([0.2, 1e-20, 0.9]) +
            np.array([0.7, 1, 0]) * np.log([0.8, 1, 0.1]))
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_VAE_ELBO_pytorch(self):
    """."""
    loss = losses.VAE_ELBO()
    logvar = torch.tensor([[1.0, 1.3], [0.6, 1.2]])
    mu = torch.tensor([[0.2, 0.7], [1.2, 0.4]])
    x = torch.tensor([[0.9, 0.4, 0.8], [0.3, 0, 1]])
    reconstruction_x = torch.tensor([[0.8, 0.3, 0.7], [0.2, 0, 0.9]])
    result = loss._create_pytorch_loss()(logvar, mu, x,
                                         reconstruction_x).numpy()
    expected = [
        0.5 * np.mean([
            0.04 + 1.0 - np.log(1e-20 + 1.0) - 1,
            0.49 + 1.69 - np.log(1e-20 + 1.69) - 1
        ]) - np.mean(
            np.array([0.9, 0.4, 0.8]) * np.log([0.8, 0.3, 0.7]) +
            np.array([0.1, 0.6, 0.2]) * np.log([0.2, 0.7, 0.3])),
        0.5 * np.mean([
            1.44 + 0.36 - np.log(1e-20 + 0.36) - 1,
            0.16 + 1.44 - np.log(1e-20 + 1.44) - 1
        ]) - np.mean(
            np.array([0.3, 0, 1]) * np.log([0.2, 1e-20, 0.9]) +
            np.array([0.7, 1, 0]) * np.log([0.8, 1, 0.1]))
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_VAE_KLDivergence_tf(self):
    """."""
    loss = losses.VAE_KLDivergence()
    logvar = tf.constant([[1.0, 1.3], [0.6, 1.2]])
    mu = tf.constant([[0.2, 0.7], [1.2, 0.4]])
    result = loss._compute_tf_loss(logvar, mu).numpy()
    expected = [
        0.5 * np.mean([
            0.04 + 1.0 - np.log(1e-20 + 1.0) - 1,
            0.49 + 1.69 - np.log(1e-20 + 1.69) - 1
        ]), 0.5 * np.mean([
            1.44 + 0.36 - np.log(1e-20 + 0.36) - 1,
            0.16 + 1.44 - np.log(1e-20 + 1.44) - 1
        ])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_VAE_KLDivergence_pytorch(self):
    """."""
    loss = losses.VAE_KLDivergence()
    logvar = torch.tensor([[1.0, 1.3], [0.6, 1.2]])
    mu = torch.tensor([[0.2, 0.7], [1.2, 0.4]])
    result = loss._create_pytorch_loss()(logvar, mu).numpy()
    expected = [
        0.5 * np.mean([
            0.04 + 1.0 - np.log(1e-20 + 1.0) - 1,
            0.49 + 1.69 - np.log(1e-20 + 1.69) - 1
        ]), 0.5 * np.mean([
            1.44 + 0.36 - np.log(1e-20 + 0.36) - 1,
            0.16 + 1.44 - np.log(1e-20 + 1.44) - 1
        ])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_tensorflow, 'TensorFlow is not installed')
  def test_ShannonEntropy_tf(self):
    """."""
    loss = losses.ShannonEntropy()
    inputs = tf.constant([[0.7, 0.3], [0.9, 0.1]])
    result = loss._compute_tf_loss(inputs).numpy()
    expected = [
        -np.mean([0.7 * np.log(0.7), 0.3 * np.log(0.3)]),
        -np.mean([0.9 * np.log(0.9), 0.1 * np.log(0.1)])
    ]
    assert np.allclose(expected, result)

  @unittest.skipIf(not has_pytorch, 'PyTorch is not installed')
  def test_ShannonEntropy_pytorch(self):
    """."""
    loss = losses.ShannonEntropy()
    inputs = torch.tensor([[0.7, 0.3], [0.9, 0.1]])
    result = loss._create_pytorch_loss()(inputs).numpy()
    expected = [
        -np.mean([0.7 * np.log(0.7), 0.3 * np.log(0.3)]),
        -np.mean([0.9 * np.log(0.9), 0.1 * np.log(0.1)])
    ]
    assert np.allclose(expected, result)

import logging
import numpy as np
import tensorflow as tf

from deepchem.utils.save import log
from deepchem.models import KerasModel, layers
from deepchem.models.losses import SigmoidCrossEntropy
from deepchem.trans import undo_transforms
from tensorflow.keras.layers import Input, Layer, Activation, Concatenate, Lambda


class IRVLayer(Layer):
  """ Core layer of IRV classifier, architecture described in:
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2750043/
  """

  def __init__(self, n_tasks, K, penalty, **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    K: int
      Number of nearest neighbours used in classification
    """
    self.n_tasks = n_tasks
    self.K = K
    self.penalty = penalty
    super(IRVLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.V = tf.Variable(tf.constant([0.01, 1.]), name="vote", dtype=tf.float32)
    self.W = tf.Variable(tf.constant([1., 1.]), name="w", dtype=tf.float32)
    self.b = tf.Variable(tf.constant([0.01]), name="b", dtype=tf.float32)
    self.b2 = tf.Variable(tf.constant([0.01]), name="b2", dtype=tf.float32)

  def call(self, inputs):
    K = self.K
    outputs = []
    for count in range(self.n_tasks):
      # Similarity values
      similarity = inputs[:, 2 * K * count:(2 * K * count + K)]
      # Labels for all top K similar samples
      ys = tf.cast(inputs[:, (2 * K * count + K):2 * K * (count + 1)], tf.int32)

      R = self.b + self.W[0] * similarity + self.W[1] * tf.constant(
          np.arange(K) + 1, dtype=tf.float32)
      R = tf.sigmoid(R)
      z = tf.reduce_sum(R * tf.gather(self.V, ys), axis=1) + self.b2
      outputs.append(tf.reshape(z, shape=[-1, 1]))
    loss = (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.V) + tf.nn.l2_loss(
        self.b) + tf.nn.l2_loss(self.b2)) * self.penalty
    self.add_loss(loss)
    return tf.concat(outputs, axis=1)


class Slice(Layer):
  """ Choose a slice of input on the last axis given order,
  Suppose input x has two dimensions,
  output f(x) = x[:, slice_num:slice_num+1]
  """

  def __init__(self, slice_num, axis=1, **kwargs):
    """
    Parameters
    ----------
    slice_num: int
      index of slice number
    axis: int
      axis id
    """
    self.slice_num = slice_num
    self.axis = axis
    super(Slice, self).__init__(**kwargs)

  def call(self, inputs):
    slice_num = self.slice_num
    axis = self.axis
    return tf.slice(inputs, [0] * axis + [slice_num], [-1] * axis + [1])


class MultitaskIRVClassifier(KerasModel):

  def __init__(self,
               n_tasks,
               K=10,
               penalty=0.0,
               mode="classification",
               **kwargs):
    """Initialize MultitaskIRVClassifier

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    K: int
      Number of nearest neighbours used in classification
    penalty: float
      Amount of penalty (l2 or l1 applied)

    """
    self.n_tasks = n_tasks
    self.K = K
    self.n_features = 2 * self.K * self.n_tasks
    self.penalty = penalty
    mol_features = Input(shape=(self.n_features,))
    predictions = IRVLayer(self.n_tasks, self.K, self.penalty)(mol_features)
    logits = []
    outputs = []
    for task in range(self.n_tasks):
      task_output = Slice(task, 1)(predictions)
      sigmoid = Activation(tf.sigmoid)(task_output)
      logits.append(task_output)
      outputs.append(sigmoid)
    outputs = layers.Stack(axis=1)(outputs)
    outputs2 = Lambda(lambda x: 1 - x)(outputs)
    outputs = [
        Concatenate(axis=2)([outputs2, outputs]), logits[0]
        if len(logits) == 1 else Concatenate(axis=1)(logits)
    ]
    model = tf.keras.Model(inputs=[mol_features], outputs=outputs)
    super(MultitaskIRVClassifier, self).__init__(
        model,
        SigmoidCrossEntropy(),
        output_types=['prediction', 'loss'],
        **kwargs)

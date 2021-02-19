import sys

import logging
import deepchem as dc
from deepchem.models import KerasModel
from deepchem.models.layers import AtomicConvolution
from deepchem.models.losses import L2Loss
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Concatenate 
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, Lambda

import numpy as np
import tensorflow as tf
import itertools
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import KerasActivationFn, LossFn, OneOrMany

logger = logging.getLogger(__name__)

class AtomicConvScore(Layer):
  """The scoring function used by the atomic convolution models."""

  def __init__(self, atom_types, layer_sizes, **kwargs):
    super(AtomicConvScore, self).__init__(**kwargs)
    self.atom_types = atom_types
    self.layer_sizes = layer_sizes

  def build(self, input_shape):
    self.type_weights = []
    self.type_biases = []
    self.output_weights = []
    self.output_biases = []
    n_features = int(input_shape[0][-1])
    layer_sizes = self.layer_sizes
    num_layers = len(layer_sizes)
    weight_init_stddevs = [1 / np.sqrt(x) for x in layer_sizes]
    bias_init_consts = [0.0] * num_layers
    for ind, atomtype in enumerate(self.atom_types):
      prev_layer_size = n_features
      self.type_weights.append([])
      self.type_biases.append([])
      self.output_weights.append([])
      self.output_biases.append([])
      for i in range(num_layers):
        weight, bias = initializeWeightsBiases(
            prev_layer_size=prev_layer_size,
            size=layer_sizes[i],
            weights=tf.random.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            biases=tf.constant(
                value=bias_init_consts[i], shape=[layer_sizes[i]]))
        self.type_weights[ind].append(weight)
        self.type_biases[ind].append(bias)
        prev_layer_size = layer_sizes[i]
      weight, bias = initializeWeightsBiases(prev_layer_size, 1)
      self.output_weights[ind].append(weight)
      self.output_biases[ind].append(bias)

  def call(self, inputs):
    frag1_layer, frag2_layer, complex_layer, frag1_z, frag2_z, complex_z = inputs
    atom_types = self.atom_types
    num_layers = len(self.layer_sizes)

    def atomnet(current_input, atomtype):
      prev_layer = current_input
      for i in range(num_layers):
        #layer = tf.nn.bias_add(
        #    tf.matmul(prev_layer, self.type_weights[atomtype][i]),
        #    self.type_biases[atomtype][i])
        #layer = tf.nn.relu(layer)
        layer = Dense(100)(prev_layer)
        prev_layer = layer

      #output_layer = tf.squeeze(
      #    tf.nn.bias_add(
      #        tf.matmul(prev_layer, self.output_weights[atomtype][0]),
      #        self.output_biases[atomtype][0]))
      print("self.output_weights[atomtype][0].shape")
      print(self.output_weights[atomtype][0].shape)
      output_layer = Dense(self.output_weights[atomtype][0].shape[0])(prev_layer)
      return output_layer

    frag1_zeros = tf.zeros_like(frag1_z, dtype=tf.float32)
    frag2_zeros = tf.zeros_like(frag2_z, dtype=tf.float32)
    complex_zeros = tf.zeros_like(complex_z, dtype=tf.float32)

    frag1_atomtype_energy = []
    frag2_atomtype_energy = []
    complex_atomtype_energy = []

    for ind, atomtype in enumerate(atom_types):
      frag1_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag1_layer)
      frag2_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag2_layer)
      complex_outputs = tf.map_fn(lambda x: atomnet(x, ind), complex_layer)

      cond = tf.equal(frag1_z, atomtype)
      frag1_atomtype_energy.append(tf.where(cond, frag1_outputs, frag1_zeros))
      cond = tf.equal(frag2_z, atomtype)
      frag2_atomtype_energy.append(tf.where(cond, frag2_outputs, frag2_zeros))
      cond = tf.equal(complex_z, atomtype)
      complex_atomtype_energy.append(
          tf.where(cond, complex_outputs, complex_zeros))

    frag1_outputs = tf.add_n(frag1_atomtype_energy)
    frag2_outputs = tf.add_n(frag2_atomtype_energy)
    complex_outputs = tf.add_n(complex_atomtype_energy)

    frag1_energy = tf.reduce_sum(frag1_outputs, 1)
    frag2_energy = tf.reduce_sum(frag2_outputs, 1)
    complex_energy = tf.reduce_sum(complex_outputs, 1)
    binding_energy = complex_energy - (frag1_energy + frag2_energy)
    return tf.expand_dims(binding_energy, axis=1)


class AtomicConvModel(KerasModel):
  """Implements an Atomic Convolution Model.

  Implements the atomic convolutional networks as introduced in

  Gomes, Joseph, et al. "Atomic convolutional networks for predicting protein-ligand binding affinity." arXiv preprint arXiv:1703.10603 (2017).

  The atomic convolutional networks function as a variant of
  graph convolutions. The difference is that the "graph" here is
  the nearest neighbors graph in 3D space. The AtomicConvModel
  leverages these connections in 3D space to train models that
  learn to predict energetic state starting from the spatial
  geometry of the model.
  """

  def __init__(self,
               n_tasks: int,
               frag1_num_atoms: int =70,
               frag2_num_atoms: int =634,
               complex_num_atoms: int=701,
               max_num_neighbors: int=12,
               batch_size: int=24,
               atom_types: Sequence[float]=[
                   6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35.,
                   53., -1.
               ],
               radial: Sequence[Sequence[float]]=[[
                   1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                   7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
               ], [0.0, 4.0, 8.0], [0.4]],
               #layer_sizes=[32, 32, 16],
               layer_sizes=[100],
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = "l2",
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[KerasActivationFn] = tf.nn.relu,
               residual: bool = False,
               learning_rate=0.001,
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
      equal len(layer_sizes).  Alternatively this may be a single
      value instead of a list, in which case the same value is used
      for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The
      length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in
      which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of dense layers.
    learning_rate: float
      Learning rate for the model.
    """
    # TODO: Turning off queue for now. Safe to re-activate?
    self.complex_num_atoms = complex_num_atoms
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.batch_size = batch_size
    self.atom_types = atom_types

    rp = [x for x in itertools.product(*radial)]
    frag1_X = Input(shape=(frag1_num_atoms, 3))
    frag1_nbrs = Input(shape=(frag1_num_atoms, max_num_neighbors))
    frag1_nbrs_z = Input(shape=(frag1_num_atoms, max_num_neighbors))
    frag1_z = Input(shape=(frag1_num_atoms,))

    frag2_X = Input(shape=(frag2_num_atoms, 3))
    frag2_nbrs = Input(shape=(frag2_num_atoms, max_num_neighbors))
    frag2_nbrs_z = Input(shape=(frag2_num_atoms, max_num_neighbors))
    frag2_z = Input(shape=(frag2_num_atoms,))

    complex_X = Input(shape=(complex_num_atoms, 3))
    complex_nbrs = Input(shape=(complex_num_atoms, max_num_neighbors))
    complex_nbrs_z = Input(shape=(complex_num_atoms, max_num_neighbors))
    complex_z = Input(shape=(complex_num_atoms,))

    self._frag1_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([frag1_X, frag1_nbrs, frag1_nbrs_z])
    flattened1 = Flatten()(self._frag1_conv)

    self._frag2_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([frag2_X, frag2_nbrs, frag2_nbrs_z])
    flattened2 = Flatten()(self._frag2_conv)

    self._complex_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([complex_X, complex_nbrs, complex_nbrs_z])
    flattened3 = Flatten()(self._complex_conv)

    concat = Concatenate()([flattened1, flattened2, flattened3])

    n_layers = len(layer_sizes)
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
    #score = AtomicConvScore(self.atom_types, layer_sizes)([
    #    self._frag1_conv, self._frag2_conv, self._complex_conv, frag1_z,
    #    frag2_z, complex_z
    #])
    #print("score")
    #print(score)
    prev_layer = concat
    #dropout_switch = Input(shape=tuple())
    prev_size = concat.shape[0] 
    next_activation = None
    ## Add the dense layers

    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = prev_layer
      if next_activation is not None:
        layer = Activation(next_activation)(layer)
      print("size")
      print(size)
      #layer = Dense(100)(layer)
      layer = Dense(
          size,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(layer)
      if dropout > 0.0:
        layer = Dropout(rate=dropout)(layer)
      if residual and prev_size == size:
        prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
      else:
        prev_layer = layer
      prev_size = size
      next_activation = activation_fn
      if next_activation is not None:
        prev_layer = Activation(activation_fn)(prev_layer)
    self.neural_fingerprint = prev_layer
    output = Reshape((n_tasks, 1))(Dense(
        n_tasks,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=weight_init_stddevs[-1]),
        bias_initializer=tf.constant_initializer(
            value=bias_init_consts[-1]))(prev_layer))
    loss: Union[dc.models.losses.Loss, LossFn]
    #prev_layer = Dense(100)(prev_layer)
    #output = Dense(1)(prev_layer)
    #print("output")
    #print(output)
    #loss = dc.models.losses.L2Loss()

    model = tf.keras.Model(
        inputs=[
            frag1_X, frag1_nbrs, frag1_nbrs_z, frag1_z, frag2_X, frag2_nbrs,
            frag2_nbrs_z, frag2_z, complex_X, complex_nbrs, complex_nbrs_z,
            complex_z
        ],
        #outputs=score)
        outputs=output)
    super(AtomicConvModel, self).__init__(
        model, L2Loss(), batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    batch_size = self.batch_size

    def replace_atom_types(z):

      def place_holder(i):
        if i in self.atom_types:
          return i
        return -1

      return np.array([place_holder(x) for x in z])

    for epoch in range(epochs):
      for ind, (F_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              batch_size, deterministic=True, pad_batches=pad_batches)):
        N = self.complex_num_atoms
        N_1 = self.frag1_num_atoms
        N_2 = self.frag2_num_atoms
        M = self.max_num_neighbors

        batch_size = F_b.shape[0]
        num_features = F_b[0][0].shape[1]
        frag1_X_b = np.zeros((batch_size, N_1, num_features))
        for i in range(batch_size):
          frag1_X_b[i] = F_b[i][0]

        frag2_X_b = np.zeros((batch_size, N_2, num_features))
        for i in range(batch_size):
          frag2_X_b[i] = F_b[i][3]

        complex_X_b = np.zeros((batch_size, N, num_features))
        for i in range(batch_size):
          complex_X_b[i] = F_b[i][6]

        frag1_Nbrs = np.zeros((batch_size, N_1, M))
        frag1_Z_b = np.zeros((batch_size, N_1))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][2])
          frag1_Z_b[i] = z
        frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
        for atom in range(N_1):
          for i in range(batch_size):
            atom_nbrs = F_b[i][1].get(atom, "")
            frag1_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

        frag2_Nbrs = np.zeros((batch_size, N_2, M))
        frag2_Z_b = np.zeros((batch_size, N_2))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][5])
          frag2_Z_b[i] = z
        frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
        for atom in range(N_2):
          for i in range(batch_size):
            atom_nbrs = F_b[i][4].get(atom, "")
            frag2_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

        complex_Nbrs = np.zeros((batch_size, N, M))
        complex_Z_b = np.zeros((batch_size, N))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][8])
          complex_Z_b[i] = z
        complex_Nbrs_Z = np.zeros((batch_size, N, M))
        for atom in range(N):
          for i in range(batch_size):
            atom_nbrs = F_b[i][7].get(atom, "")
            complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

        inputs = [
            frag1_X_b, frag1_Nbrs, frag1_Nbrs_Z, frag1_Z_b, frag2_X_b,
            frag2_Nbrs, frag2_Nbrs_Z, frag2_Z_b, complex_X_b, complex_Nbrs,
            complex_Nbrs_Z, complex_Z_b
        ]
        y_b = np.reshape(y_b, newshape=(batch_size, 1))
        yield (inputs, [y_b], [w_b])

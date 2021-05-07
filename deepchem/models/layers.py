# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection
from typing import Callable, Dict, List, Tuple
from tensorflow.keras import activations, initializers, backend
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, Activation


class InteratomicL2Distances(tf.keras.layers.Layer):
  """Compute (squared) L2 Distances between atoms given neighbors.

  This class computes pairwise distances between its inputs.

  Examples
  --------
  >>> import numpy as np
  >>> import deepchem as dc
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

  def get_config(self) -> Dict:
    """Returns config dictionary for this layer."""
    config = super(InteratomicL2Distances, self).get_config()
    config['N_atoms'] = self.N_atoms
    config['M_nbrs'] = self.M_nbrs
    config['ndim'] = self.ndim
    return config

  def call(self, inputs):
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
    coords, nbr_list = (inputs[0], inputs[1])
    N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
    # Shape (N_atoms, M_nbrs, ndim)
    nbr_coords = tf.gather(coords, nbr_list)
    # Shape (N_atoms, M_nbrs, ndim)
    tiled_coords = tf.tile(
        tf.reshape(coords, (N_atoms, 1, ndim)), (1, M_nbrs, 1))
    # Shape (N_atoms, M_nbrs)
    return tf.reduce_sum((tiled_coords - nbr_coords)**2, axis=2)


class GraphConv(tf.keras.layers.Layer):
  """Graph Convolutional Layers

  This layer implements the graph convolution introduced in [1]_.  The graph
  convolution combines per-node feature vectures in a nonlinear fashion with
  the feature vectors for neighboring nodes.  This "blends" information in
  local neighborhoods of a graph.

  References
  ----------
  .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints." Advances in neural information processing systems. 2015. https://arxiv.org/abs/1509.09292

  """

  def __init__(self,
               out_channel: int,
               min_deg: int = 0,
               max_deg: int = 10,
               activation_fn: Callable = None,
               **kwargs):
    """Initialize a graph convolutional layer.

    Parameters
    ----------
    out_channel: int
      The number of output channels per graph node.
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
      `tf.nn.relu` is probably a good default for your application.
    """
    super(GraphConv, self).__init__(**kwargs)
    self.out_channel = out_channel
    self.min_degree = min_deg
    self.max_degree = max_deg
    self.activation_fn = activation_fn

  def build(self, input_shape):
    # Generate the nb_affine weights and biases
    num_deg = 2 * self.max_degree + (1 - self.min_degree)
    self.W_list = [
        self.add_weight(
            name='kernel' + str(k),
            shape=(int(input_shape[0][-1]), self.out_channel),
            initializer='glorot_uniform',
            trainable=True) for k in range(num_deg)
    ]
    self.b_list = [
        self.add_weight(
            name='bias' + str(k),
            shape=(self.out_channel,),
            initializer='zeros',
            trainable=True) for k in range(num_deg)
    ]
    self.built = True

  def get_config(self):
    config = super(GraphConv, self).get_config()
    config['out_channel'] = self.out_channel
    config['min_deg'] = self.min_degree
    config['max_deg'] = self.max_degree
    config['activation_fn'] = self.activation_fn
    return config

  def call(self, inputs):

    # Extract atom_features
    atom_features = inputs[0]

    # Extract graph topology
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    W = iter(self.W_list)
    b = iter(self.b_list)

    # Sum all neighbors using adjacency matrix
    deg_summed = self.sum_neigh(atom_features, deg_adj_lists)

    # Get collection of modified atom features
    new_rel_atoms_collection = (self.max_degree + 1 - self.min_degree) * [None]

    split_features = tf.split(atom_features, deg_slice[:, 1])
    for deg in range(1, self.max_degree + 1):
      # Obtain relevant atoms for this degree
      rel_atoms = deg_summed[deg - 1]

      # Get self atoms
      self_atoms = split_features[deg - self.min_degree]

      # Apply hidden affine to relevant atoms and append
      rel_out = tf.matmul(rel_atoms, next(W)) + next(b)
      self_out = tf.matmul(self_atoms, next(W)) + next(b)
      out = rel_out + self_out

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Determine the min_deg=0 case
    if self.min_degree == 0:
      self_atoms = split_features[0]

      # Only use the self layer
      out = tf.matmul(self_atoms, next(W)) + next(b)

      new_rel_atoms_collection[0] = out

    # Combine all atoms back into the list
    atom_features = tf.concat(axis=0, values=new_rel_atoms_collection)

    if self.activation_fn is not None:
      atom_features = self.activation_fn(atom_features)

    return atom_features

  def sum_neigh(self, atoms, deg_adj_lists):
    """Store the summed atoms by degree"""
    deg_summed = self.max_degree * [None]

    # Tensorflow correctly processes empty lists when using concat
    for deg in range(1, self.max_degree + 1):
      gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
      # Sum along neighbors as well as self, and store
      summed_atoms = tf.reduce_sum(gathered_atoms, 1)
      deg_summed[deg - 1] = summed_atoms

    return deg_summed


class GraphPool(tf.keras.layers.Layer):
  """A GraphPool gathers data from local neighborhoods of a graph.

  This layer does a max-pooling over the feature vectors of atoms in a
  neighborhood. You can think of this layer as analogous to a max-pooling
  layer for 2D convolutions but which operates on graphs instead. This
  technique is described in [1]_.

  References
  ----------
  .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
  learning molecular fingerprints." Advances in neural information processing
  systems. 2015. https://arxiv.org/abs/1509.09292

  """

  def __init__(self, min_degree=0, max_degree=10, **kwargs):
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
    self.min_degree = min_degree
    self.max_degree = max_degree

  def get_config(self):
    config = super(GraphPool, self).get_config()
    config['min_degree'] = self.min_degree
    config['max_degree'] = self.max_degree
    return config

  def call(self, inputs):
    atom_features = inputs[0]
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    # Perform the mol gather
    # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_degree, self.min_degree)

    deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

    # Tensorflow correctly processes empty lists when using concat

    split_features = tf.split(atom_features, deg_slice[:, 1])
    for deg in range(1, self.max_degree + 1):
      # Get self atoms
      self_atoms = split_features[deg - self.min_degree]

      if deg_adj_lists[deg - 1].shape[0] == 0:
        # There are no neighbors of this degree, so just create an empty tensor directly.
        maxed_atoms = tf.zeros((0, self_atoms.shape[-1]))
      else:
        # Expand dims
        self_atoms = tf.expand_dims(self_atoms, 1)

        # always deg-1 for deg_adj_lists
        gathered_atoms = tf.gather(atom_features, deg_adj_lists[deg - 1])
        gathered_atoms = tf.concat(axis=1, values=[self_atoms, gathered_atoms])

        maxed_atoms = tf.reduce_max(gathered_atoms, 1)
      deg_maxed[deg - self.min_degree] = maxed_atoms

    if self.min_degree == 0:
      self_atoms = split_features[0]
      deg_maxed[0] = self_atoms

    return tf.concat(axis=0, values=deg_maxed)


class GraphGather(tf.keras.layers.Layer):
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

  References
  ----------
  .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
  learning molecular fingerprints." Advances in neural information processing
  systems. 2015. https://arxiv.org/abs/1509.09292
  """

  def __init__(self, batch_size, activation_fn=None, **kwargs):
    """Initialize this layer.

    Parameters
    ---------
    batch_size: int
      The batch size for this layer. Note that the layer's behavior
      changes depending on the batch size.
    activation_fn: function
      A nonlinear activation function to apply. If you're not sure,
      `tf.nn.relu` is probably a good default for your application.
    """

    super(GraphGather, self).__init__(**kwargs)
    self.batch_size = batch_size
    self.activation_fn = activation_fn

  def get_config(self):
    config = super(GraphGather, self).get_config()
    config['batch_size'] = self.batch_size
    config['activation_fn'] = self.activation_fn
    return config

  def call(self, inputs):
    """Invoking this layer.

    Parameters
    ----------
    inputs: list
      This list should consist of `inputs = [atom_features, deg_slice,
      membership, deg_adj_list placeholders...]`. These are all
      tensors that are created/process by `GraphConv` and `GraphPool`
    """
    atom_features = inputs[0]

    # Extract graph topology
    membership = inputs[2]

    assert self.batch_size > 1, "graph_gather requires batches larger than 1"

    sparse_reps = tf.math.unsorted_segment_sum(atom_features, membership,
                                               self.batch_size)
    max_reps = tf.math.unsorted_segment_max(atom_features, membership,
                                            self.batch_size)
    mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])

    if self.activation_fn is not None:
      mol_features = self.activation_fn(mol_features)
    return mol_features


class MolGANConvolutionLayer(tf.keras.layers.Layer):
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

  Example
  --------
  See: MolGANMultiConvolutionLayer for using in layers.

  >>> from tensorflow.keras import Model
  >>> from tensorflow.keras.layers import Input
  >>> vertices = 9
  >>> nodes = 5
  >>> edges = 5
  >>> units = 128

  >>> layer1 = MolGANConvolutionLayer(units=units,edges=edges)
  >>> layer2 = MolGANConvolutionLayer(units=units,edges=edges)
  >>> adjacency_tensor= Input(shape=(vertices, vertices, edges))
  >>> node_tensor = Input(shape=(vertices,nodes))
  >>> hidden1 = layer1([adjacency_tensor,node_tensor])
  >>> output = layer2(hidden1)
  >>> model = Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])

  References
  ----------
  .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
  for small molecular graphs", https://arxiv.org/abs/1805.11973
  """

  def __init__(self,
               units: int,
               activation: Callable = activations.tanh,
               dropout_rate: float = 0.0,
               edges: int = 5,
               name: str = "",
               **kwargs):
    """
    Initialize this layer.

    Parameters
    ---------
    units: int
      Dimesion of dense layers used for convolution
    activation: function, optional (default=Tanh)
      activation function used across model, default is Tanh
    dropout_rate: float, optional (default=0.0)
     Dropout rate used by dropout layer
    edges: int, optional (default=5)
      How many dense layers to use in convolution.
      Typically equal to number of bond types used in the model.
    name: string, optional (default="")
      Name of the layer
    """

    super(MolGANConvolutionLayer, self).__init__(name=name, **kwargs)
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.units = units
    self.edges = edges

    self.dense1 = [Dense(units=self.units) for _ in range(edges - 1)]
    self.dense2 = Dense(units=self.units)
    self.dropout = Dropout(self.dropout_rate)
    self.activation_layer = Activation(self.activation)

  def call(self, inputs, training=False):
    """
    Invoke this layer

    Parameters
    ----------
    inputs: list
      List of two input matrices, adjacency tensor and node features tensors
      in one-hot encoding format.
    training: bool
      Should this layer be run in training mode.
      Typically decided by main model, influences things like dropout.

    Returns
    --------
    tuple(tf.Tensor,tf.Tensor,tf.Tensor)
      First and second are original input tensors
      Third is the result of convolution
    """

    ic = len(inputs)
    assert ic > 1, "MolGANConvolutionLayer requires at least two inputs: [adjacency_tensor, node_features_tensor]"

    adjacency_tensor = inputs[0]
    node_tensor = inputs[1]

    # means that this is second loop of convolution
    if ic > 2:
      hidden_tensor = inputs[2]
      annotations = tf.concat((hidden_tensor, node_tensor), -1)
    else:
      annotations = node_tensor

    output = tf.stack([dense(annotations) for dense in self.dense1], 1)

    adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))

    output = tf.matmul(adj, output)
    output = tf.reduce_sum(output, 1) + self.dense2(node_tensor)
    output = self.activation_layer(output)
    output = self.dropout(output)
    return adjacency_tensor, node_tensor, output

  def get_config(self) -> Dict:
    """
    Returns config dictionary for this layer.
    """

    config = super(MolGANConvolutionLayer, self).get_config()
    config["activation"] = self.activation
    config["dropout_rate"] = self.dropout_rate
    config["units"] = self.units
    config["edges"] = self.edges
    return config


class MolGANAggregationLayer(tf.keras.layers.Layer):
  """
  Graph Aggregation layer used in MolGAN model.
  MolGAN is a WGAN type model for generation of small molecules.
  Performs aggregation on tensor resulting from convolution layers.
  Given its simple nature it might be removed in future and moved to
  MolGANEncoderLayer.


  Example
  --------
  >>> from tensorflow.keras import Model
  >>> from tensorflow.keras.layers import Input
  >>> vertices = 9
  >>> nodes = 5
  >>> edges = 5
  >>> units = 128

  >>> layer_1 = MolGANConvolutionLayer(units=units,edges=edges)
  >>> layer_2 = MolGANConvolutionLayer(units=units,edges=edges)
  >>> layer_3 = MolGANAggregationLayer(units=128)
  >>> adjacency_tensor= Input(shape=(vertices, vertices, edges))
  >>> node_tensor = Input(shape=(vertices,nodes))
  >>> hidden_1 = layer_1([adjacency_tensor,node_tensor])
  >>> hidden_2 = layer_2(hidden_1)
  >>> output = layer_3(hidden_2[2])
  >>> model = Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])


  Example
  --------
  vertices = 9
  nodes = 5
  edges = 5
  units = 128

  layer_1 = MolGANConvolutionLayer(units=units,edges=edges)
  layer_2 = MolGANConvolutionLayer(units=units,edges=edges)
  layer_3 = MolGANAggregationLayer(units=128)
  adjacency_tensor= layers.Input(shape=(vertices, vertices, edges))
  node_tensor = layers.Input(shape=(vertices,nodes))
  hidden_1 = layer_1([adjacency_tensor,node_tensor])
  hidden_2 = layer_2(hidden_1)
  output = layer_3(hidden_2[2])
  model = keras.Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])

  References
  ----------
  .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
  for small molecular graphs", https://arxiv.org/abs/1805.11973
  """

  def __init__(self,
               units: int = 128,
               activation: Callable = activations.tanh,
               dropout_rate: float = 0.0,
               name: str = "",
               **kwargs):
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
    """

    super(MolGANAggregationLayer, self).__init__(name=name, **kwargs)
    self.units = units
    self.activation = activation
    self.dropout_rate = dropout_rate

    self.d1 = Dense(units=units, activation="sigmoid")
    self.d2 = Dense(units=units, activation=activation)
    self.dropout_layer = Dropout(dropout_rate)
    self.activation_layer = Activation(activation)

  def call(self, inputs, training=False):
    """
    Invoke this layer

    Parameters
    ----------
    inputs: List
      Single tensor resulting from graph convolution layer
    training: bool
      Should this layer be run in training mode.
      Typically decided by main model, influences things like dropout.

    Returns
    --------
    aggregation tensor: tf.Tensor
      Result of aggregation function on input convolution tensor.
    """

    i = self.d1(inputs)
    j = self.d2(inputs)
    output = tf.reduce_sum(i * j, 1)
    output = self.activation_layer(output)
    output = self.dropout_layer(output)
    return output

  def get_config(self) -> Dict:
    """
    Returns config dictionary for this layer.
    """

    config = super(MolGANAggregationLayer, self).get_config()
    config["units"] = self.units
    config["activation"] = self.activation
    config["dropout_rate"] = self.dropout_rate
    config["edges"] = self.edges
    return config


class MolGANMultiConvolutionLayer(tf.keras.layers.Layer):
  """
  Multiple pass convolution layer used in MolGAN model.
  MolGAN is a WGAN type model for generation of small molecules.
  It takes outputs of previous convolution layer and uses
  them as inputs for the next one.
  It simplifies the overall framework, but might be moved to
  MolGANEncoderLayer in the future in order to reduce number of layers.

  Example
  --------
  >>> from tensorflow.keras import Model
  >>> from tensorflow.keras.layers import Input
  >>> vertices = 9
  >>> nodes = 5
  >>> edges = 5
  >>> units = 128

  >>> layer_1 = MolGANMultiConvolutionLayer(units=(128,64))
  >>> layer_2 = MolGANAggregationLayer(units=128)
  >>> adjacency_tensor= Input(shape=(vertices, vertices, edges))
  >>> node_tensor = Input(shape=(vertices,nodes))
  >>> hidden = layer_1([adjacency_tensor,node_tensor])
  >>> output = layer_2(hidden)
  >>> model = Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])

  Example
  --------
  vertices = 9
  nodes = 5
  edges = 5
  units = 128

  layer_1 = MolGANMultiConvolutionLayer(units=(128,64))
  layer_2 = MolGANAggregationLayer(units=128)
  adjacency_tensor= layers.Input(shape=(vertices, vertices, edges))
  node_tensor = layers.Input(shape=(vertices,nodes))
  hidden = layer_1([adjacency_tensor,node_tensor])
  output = layer_2(hidden)
  model = keras.Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])

  References
  ----------
  .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
  for small molecular graphs", https://arxiv.org/abs/1805.11973
  """

  def __init__(self,
               units: Tuple = (128, 64),
               activation: Callable = activations.tanh,
               dropout_rate: float = 0.0,
               edges: int = 5,
               name: str = "",
               **kwargs):
    """
    Initialize the layer

    Parameters
    ---------
    units: Tuple, optional (default=(128,64)), min_length=2
      List of dimensions used by consecutive convolution layers.
      The more values the more convolution layers invoked.
    activation: function, optional (default=tanh)
      activation function used across model, default is Tanh
    dropout_rate: float, optional (default=0.0)
      Used by dropout layer
    edges: int, optional (default=0)
      Controls how many dense layers use for single convolution unit.
      Typically matches number of bond types used in the molecule.
    name: string, optional (default="")
      Name of the layer
    """

    super(MolGANMultiConvolutionLayer, self).__init__(name=name, **kwargs)
    assert len(units) > 1, "Layer requires at least two values"

    self.units = units
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.edges = edges

    self.first_convolution = MolGANConvolutionLayer(
        self.units[0], self.activation, self.dropout_rate, self.edges)
    self.gcl = [
        MolGANConvolutionLayer(u, self.activation, self.dropout_rate,
                               self.edges) for u in self.units[1:]
    ]

  def call(self, inputs, training=False):
    """
    Invoke this layer

    Parameters
    ----------
    inputs: list
      List of two input matrices, adjacency tensor and node features tensors
      in one-hot encoding format.
    training: bool
      Should this layer be run in training mode.
      Typically decided by main model, influences things like dropout.

    Returns
    --------
    convolution tensor: tf.Tensor
      Result of input tensors going through convolution a number of times.
    """

    adjacency_tensor = inputs[0]
    node_tensor = inputs[1]

    tensors = self.first_convolution([adjacency_tensor, node_tensor])

    for layer in self.gcl:
      tensors = layer(tensors)

    _, _, hidden_tensor = tensors

    return hidden_tensor

  def get_config(self) -> Dict:
    """
    Returns config dictionary for this layer.
    """

    config = super(MolGANMultiConvolutionLayer, self).get_config()
    config["units"] = self.units
    config["activation"] = self.activation
    config["dropout_rate"] = self.dropout_rate
    config["edges"] = self.edges
    return config


class MolGANEncoderLayer(tf.keras.layers.Layer):
  """
  Main learning layer used by MolGAN model.
  MolGAN is a WGAN type model for generation of small molecules.
  It role is to further simplify model.
  This layer can be manually built by stacking graph convolution layers
  followed by graph aggregation.

  Example
  --------
  >>> from tensorflow.keras import Model
  >>> from tensorflow.keras.layers import Input, Dropout,Dense
  >>> vertices = 9
  >>> edges = 5
  >>> nodes = 5
  >>> dropout_rate = .0
  >>> adjacency_tensor= Input(shape=(vertices, vertices, edges))
  >>> node_tensor = Input(shape=(vertices, nodes))

  >>> graph = MolGANEncoderLayer(units = [(128,64),128], dropout_rate= dropout_rate, edges=edges)([adjacency_tensor,node_tensor])
  >>> dense = Dense(units=128, activation='tanh')(graph)
  >>> dense = Dropout(dropout_rate)(dense)
  >>> dense = Dense(units=64, activation='tanh')(dense)
  >>> dense = Dropout(dropout_rate)(dense)
  >>> output = Dense(units=1)(dense)

  >>> model = Model(inputs=[adjacency_tensor,node_tensor], outputs=[output])

  References
  ----------
  .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
  for small molecular graphs", https://arxiv.org/abs/1805.11973
  """

  def __init__(self,
               units: List = [(128, 64), 128],
               activation: Callable = activations.tanh,
               dropout_rate: float = 0.0,
               edges: int = 5,
               name: str = "",
               **kwargs):
    """
    Initialize the layer.

    Parameters
    ---------
    units: List, optional (default=[(128, 64), 128])
      List of units for MolGANMultiConvolutionLayer and GraphAggregationLayer
      i.e. [(128,64),128] means two convolution layers dims = [128,64]
      followed by aggregation layer dims=128
    activation: function, optional (default=Tanh)
      activation function used across model, default is Tanh
    dropout_rate: float, optional (default=0.0)
      Used by dropout layer
    edges: int, optional (default=0)
      Controls how many dense layers use for single convolution unit.
      Typically matches number of bond types used in the molecule.
    name: string, optional (default="")
      Name of the layer
    """

    super(MolGANEncoderLayer, self).__init__(name=name, **kwargs)
    assert len(units) == 2
    self.graph_convolution_units, self.auxiliary_units = units
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.edges = edges

    self.multi_graph_convolution_layer = MolGANMultiConvolutionLayer(
        self.graph_convolution_units, self.activation, self.dropout_rate,
        self.edges)
    self.graph_aggregation_layer = MolGANAggregationLayer(
        self.auxiliary_units, self.activation, self.dropout_rate)

  def call(self, inputs, training=False):
    """
    Invoke this layer

    Parameters
    ----------
    inputs: list
      List of two input matrices, adjacency tensor and node features tensors
      in one-hot encoding format.
    training: bool
      Should this layer be run in training mode.
      Typically decided by main model, influences things like dropout.

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
      annotations = tf.concat((output, hidden_tensor, node_tensor), -1)
    else:
      _, node_tensor = inputs
      annotations = tf.concat((output, node_tensor), -1)

    output = self.graph_aggregation_layer(annotations)
    return output

  def get_config(self) -> Dict:
    """
    Returns config dictionary for this layer.
    """

    config = super(MolGANEncoderLayer, self).get_config()
    config["graph_convolution_units"] = self.graph_convolution_units
    config["auxiliary_units"] = self.auxiliary_units
    config["activation"] = self.activation
    config["dropout_rate"] = self.dropout_rate
    config["edges"] = self.edges
    return config


class LSTMStep(tf.keras.layers.Layer):
  """Layer that performs a single step LSTM update.

  This layer performs a single step LSTM update. Note that it is *not*
  a full LSTM recurrent network. The LSTMStep layer is useful as a
  primitive for designing layers such as the AttnLSTMEmbedding or the
  IterRefLSTMEmbedding below.
  """

  def __init__(self,
               output_dim,
               input_dim,
               init_fn='glorot_uniform',
               inner_init_fn='orthogonal',
               activation_fn='tanh',
               inner_activation_fn='hard_sigmoid',
               **kwargs):
    """
    Parameters
    ----------
    output_dim: int
      Dimensionality of output vectors.
    input_dim: int
      Dimensionality of input vectors.
    init_fn: str
      TensorFlow nitialization to use for W.
    inner_init_fn: str
      TensorFlow initialization to use for U.
    activation_fn: str
      TensorFlow activation to use for output.
    inner_activation_fn: str
      TensorFlow activation to use for inner steps.
    """

    super(LSTMStep, self).__init__(**kwargs)
    self.init = init_fn
    self.inner_init = inner_init_fn
    self.output_dim = output_dim
    # No other forget biases supported right now.
    self.activation = activation_fn
    self.inner_activation = inner_activation_fn
    self.activation_fn = activations.get(activation_fn)
    self.inner_activation_fn = activations.get(inner_activation_fn)
    self.input_dim = input_dim

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
    return [backend.zeros(input_shape), backend.zeros(input_shape)]

  def build(self, input_shape):
    """Constructs learnable weights for this layer."""
    init = initializers.get(self.init)
    inner_init = initializers.get(self.inner_init)
    self.W = init((self.input_dim, 4 * self.output_dim))
    self.U = inner_init((self.output_dim, 4 * self.output_dim))

    self.b = tf.Variable(
        np.hstack((np.zeros(self.output_dim), np.ones(self.output_dim),
                   np.zeros(self.output_dim), np.zeros(self.output_dim))),
        dtype=tf.float32)
    self.built = True

  def call(self, inputs):
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

    # Taken from Keras code [citation needed]
    z = backend.dot(x, self.W) + backend.dot(h_tm1, self.U) + self.b

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


def cosine_dist(x, y):
  """Computes the inner product (cosine similarity) between two tensors.

  This assumes that the two input tensors contain rows of vectors where
  each column represents a different feature. The output tensor will have
  elements that represent the inner product between pairs of normalized vectors
  in the rows of `x` and `y`. The two tensors need to have the same number of
  columns, because one cannot take the dot product between vectors of different
  lengths. For example, in sentence similarity and sentence classification tasks,
  the number of columns is the embedding size. In these tasks, the rows of the
  input tensors would be different test vectors or sentences. The input tensors
  themselves could be different batches. Using vectors or tensors of all 0s
  should be avoided.

  Methods
  -------
  The vectors in the input tensors are first l2-normalized such that each vector
  has length or magnitude of 1. The inner product (dot product) is then taken
  between corresponding pairs of row vectors in the input tensors and returned.

  Examples
  --------
  The cosine similarity between two equivalent vectors will be 1. The cosine
  similarity between two equivalent tensors (tensors where all the elements are
  the same) will be a tensor of 1s. In this scenario, if the input tensors `x` and
  `y` are each of shape `(n,p)`, where each element in `x` and `y` is the same, then
  the output tensor would be a tensor of shape `(n,n)` with 1 in every entry.

  >>> import tensorflow as tf
  >>> import deepchem.models.layers as layers
  >>> x = tf.ones((6, 4), dtype=tf.dtypes.float32, name=None)
  >>> y_same = tf.ones((6, 4), dtype=tf.dtypes.float32, name=None)
  >>> cos_sim_same = layers.cosine_dist(x,y_same)

  `x` and `y_same` are the same tensor (equivalent at every element, in this
  case 1). As such, the pairwise inner product of the rows in `x` and `y` will
  always be 1. The output tensor will be of shape (6,6).

  >>> diff = cos_sim_same - tf.ones((6, 6), dtype=tf.dtypes.float32, name=None)
  >>> tf.reduce_sum(diff) == 0 # True
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> cos_sim_same.shape
  TensorShape([6, 6])

  The cosine similarity between two orthogonal vectors will be 0 (by definition).
  If every row in `x` is orthogonal to every row in `y`, then the output will be a
  tensor of 0s. In the following example, each row in the tensor `x1` is orthogonal
  to each row in `x2` because they are halves of an identity matrix.

  >>> identity_tensor = tf.eye(512, dtype=tf.dtypes.float32)
  >>> x1 = identity_tensor[0:256,:]
  >>> x2 = identity_tensor[256:512,:]
  >>> cos_sim_orth = layers.cosine_dist(x1,x2)

  Each row in `x1` is orthogonal to each row in `x2`. As such, the pairwise inner
  product of the rows in `x1`and `x2` will always be 0. Furthermore, because the
  shape of the input tensors are both of shape `(256,512)`, the output tensor will
  be of shape `(256,256)`.

  >>> tf.reduce_sum(cos_sim_orth) == 0 # True
  <tf.Tensor: shape=(), dtype=bool, numpy=True>
  >>> cos_sim_orth.shape
  TensorShape([256, 256])

  Parameters
  ----------
  x: tf.Tensor
    Input Tensor of shape `(n, p)`.
    The shape of this input tensor should be `n` rows by `p` columns.
    Note that `n` need not equal `m` (the number of rows in `y`).
  y: tf.Tensor
    Input Tensor of shape `(m, p)`
    The shape of this input tensor should be `m` rows by `p` columns.
    Note that `m` need not equal `n` (the number of rows in `x`).

  Returns
  -------
  tf.Tensor
    Returns a tensor of shape `(n, m)`, that is, `n` rows by `m` columns.
    Each `i,j`-th entry of this output tensor is the inner product between
    the l2-normalized `i`-th row of the input tensor `x` and the
    the l2-normalized `j`-th row of the output tensor `y`.
  """
  x_norm = tf.math.l2_normalize(x, axis=1)
  y_norm = tf.math.l2_normalize(y, axis=1)
  return backend.dot(x_norm, tf.transpose(y_norm))


class AttnLSTMEmbedding(tf.keras.layers.Layer):
  """Implements AttnLSTM as in matching networks paper.

  The AttnLSTM embedding adjusts two sets of vectors, the "test" and
  "support" sets. The "support" consists of a set of evidence vectors.
  Think of these as the small training set for low-data machine
  learning.  The "test" consists of the queries we wish to answer with
  the small amounts of available data. The AttnLSTMEmbdding allows us to
  modify the embedding of the "test" set depending on the contents of
  the "support".  The AttnLSTMEmbedding is thus a type of learnable
  metric that allows a network to modify its internal notion of
  distance.

  See references [1]_ [2]_ for more details.

  References
  ----------
  .. [1] Vinyals, Oriol, et al. "Matching networks for one shot learning."
         Advances in neural information processing systems. 2016.
  .. [2] Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters:
         Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of features per atom
    max_depth: int
      Number of "processing steps" used by sequence-to-sequence for sets model.
    """
    super(AttnLSTMEmbedding, self).__init__(**kwargs)

    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def get_config(self):
    config = super(AttnLSTMEmbedding, self).get_config()
    config['n_test'] = self.n_test
    config['n_support'] = self.n_support
    config['n_feat'] = self.n_feat
    config['max_depth'] = self.max_depth
    return config

  def build(self, input_shape):
    n_feat = self.n_feat
    self.lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = backend.zeros([self.n_test, n_feat])
    self.states_init = self.lstm.get_initial_states([self.n_test, n_feat])
    self.built = True

  def call(self, inputs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    inputs: list
      List of two tensors (X, Xp). X should be of shape (n_test,
      n_feat) and Xp should be of shape (n_support, n_feat) where
      n_test is the size of the test set, n_support that of the support
      set, and n_feat is the number of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output
      shape will be [(n_test, n_feat), (n_support, n_feat)]
    """
    if len(inputs) != 2:
      raise ValueError("AttnLSTMEmbedding layer must have exactly two parents")
    # x is test set, xp is support set.
    x, xp = inputs

    # Get initializations
    q = self.q_init
    states = self.states_init

    for d in range(self.max_depth):
      # Process using attention
      # Eqn (4), appendix A.1 of Matching Networks paper
      e = cosine_dist(x + q, xp)
      a = tf.nn.softmax(e)
      r = backend.dot(a, xp)

      # Generate new attention states
      y = backend.concatenate([q, r], axis=1)
      q, states = self.lstm([y] + states)
    return [x + q, xp]


class IterRefLSTMEmbedding(tf.keras.layers.Layer):
  """Implements the Iterative Refinement LSTM.

  Much like AttnLSTMEmbedding, the IterRefLSTMEmbedding is another type
  of learnable metric which adjusts "test" and "support." Recall that
  "support" is the small amount of data available in a low data machine
  learning problem, and that "test" is the query. The AttnLSTMEmbedding
  only modifies the "test" based on the contents of the support.
  However, the IterRefLSTM modifies both the "support" and "test" based
  on each other. This allows the learnable metric to be more malleable
  than that from AttnLSTMEmbeding.
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Unlike the AttnLSTM model which only modifies the test vectors
    additively, this model allows for an additive update to be
    performed to both test and support using information from each
    other.

    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of input atom features
    max_depth: int
      Number of LSTM Embedding layers.
    """
    super(IterRefLSTMEmbedding, self).__init__(**kwargs)
    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def get_config(self):
    config = super(IterRefLSTMEmbedding, self).get_config()
    config['n_test'] = self.n_test
    config['n_support'] = self.n_support
    config['n_feat'] = self.n_feat
    config['max_depth'] = self.max_depth
    return config

  def build(self, input_shape):
    n_feat = self.n_feat

    # Support set lstm
    self.support_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = backend.zeros([self.n_support, n_feat])
    self.support_states_init = self.support_lstm.get_initial_states(
        [self.n_support, n_feat])

    # Test lstm
    self.test_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.p_init = backend.zeros([self.n_test, n_feat])
    self.test_states_init = self.test_lstm.get_initial_states(
        [self.n_test, n_feat])
    self.built = True

  def call(self, inputs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    inputs: list
      List of two tensors (X, Xp). X should be of shape (n_test,
      n_feat) and Xp should be of shape (n_support, n_feat) where
      n_test is the size of the test set, n_support that of the
      support set, and n_feat is the number of per-atom features.

    Returns
    -------
    Returns two tensors of same shape as input. Namely the output
    shape will be [(n_test, n_feat), (n_support, n_feat)]
    """
    if len(inputs) != 2:
      raise ValueError(
          "IterRefLSTMEmbedding layer must have exactly two parents")
    x, xp = inputs

    # Get initializations
    p = self.p_init
    q = self.q_init
    # Rename support
    z = xp
    states = self.support_states_init
    x_states = self.test_states_init

    for d in range(self.max_depth):
      # Process support xp using attention
      e = cosine_dist(z + q, xp)
      a = tf.nn.softmax(e)
      # Get linear combination of support set
      r = backend.dot(a, xp)

      # Process test x using attention
      x_e = cosine_dist(x + p, z)
      x_a = tf.nn.softmax(x_e)
      s = backend.dot(x_a, z)

      # Generate new support attention states
      qr = backend.concatenate([q, r], axis=1)
      q, states = self.support_lstm([qr] + states)

      # Generate new test attention states
      ps = backend.concatenate([p, s], axis=1)
      p, x_states = self.test_lstm([ps] + x_states)

      # Redefine
      z = r

    return [x + p, xp + q]


class SwitchedDropout(tf.keras.layers.Layer):
  """Apply dropout based on an input.

  This is required for uncertainty prediction.  The standard Keras
  Dropout layer only performs dropout during training, but we
  sometimes need to do it during prediction.  The second input to this
  layer should be a scalar equal to 0 or 1, indicating whether to
  perform dropout.
  """

  def __init__(self, rate, **kwargs):
    self.rate = rate
    super(SwitchedDropout, self).__init__(**kwargs)

  def get_config(self):
    config = super(SwitchedDropout, self).get_config()
    config['rate'] = self.rate
    return config

  def call(self, inputs):
    rate = self.rate * tf.squeeze(inputs[1])
    return tf.nn.dropout(inputs[0], rate=rate)


class WeightedLinearCombo(tf.keras.layers.Layer):
  """Computes a weighted linear combination of input layers, with the weights defined by trainable variables."""

  def __init__(self, std=0.3, **kwargs):
    """Initialize this layer.

    Parameters
    ----------
    std: float, optional (default 0.3)
      The standard deviation to use when randomly initializing weights.
    """
    super(WeightedLinearCombo, self).__init__(**kwargs)
    self.std = std

  def get_config(self):
    config = super(WeightedLinearCombo, self).get_config()
    config['std'] = self.std
    return config

  def build(self, input_shape):
    init = tf.keras.initializers.RandomNormal(stddev=self.std)
    self.input_weights = [
        self.add_weight(
            'weight_%d' % (i + 1), (1,), initializer=init, trainable=True)
        for i in range(len(input_shape))
    ]
    self.built = True

  def call(self, inputs):
    out_tensor = None
    for in_tensor, w in zip(inputs, self.input_weights):
      if out_tensor is None:
        out_tensor = w * in_tensor
      else:
        out_tensor += w * in_tensor
    return out_tensor


class CombineMeanStd(tf.keras.layers.Layer):
  """Generate Gaussian nose."""

  def __init__(self, training_only=False, noise_epsilon=1.0, **kwargs):
    """Create a CombineMeanStd layer.

    This layer should have two inputs with the same shape, and its
    output also has the same shape.  Each element of the output is a
    Gaussian distributed random number whose mean is the corresponding
    element of the first input, and whose standard deviation is the
    corresponding element of the second input.

    Parameters
    ----------
    training_only: bool
      if True, noise is only generated during training.  During
      prediction, the output is simply equal to the first input (that
      is, the mean of the distribution used during training).
    noise_epsilon: float
      The noise is scaled by this factor
    """
    super(CombineMeanStd, self).__init__(**kwargs)
    self.training_only = training_only
    self.noise_epsilon = noise_epsilon

  def get_config(self):
    config = super(CombineMeanStd, self).get_config()
    config['training_only'] = self.training_only
    config['noise_epsilon'] = self.noise_epsilon
    return config

  def call(self, inputs, training=True):
    if len(inputs) != 2:
      raise ValueError("Must have two in_layers")
    mean_parent, std_parent = inputs[0], inputs[1]
    noise_scale = tf.cast(training or not self.training_only, tf.float32)
    from tensorflow.python.ops import array_ops
    sample_noise = tf.random.normal(
        array_ops.shape(mean_parent), 0, self.noise_epsilon, dtype=tf.float32)
    return mean_parent + noise_scale * std_parent * sample_noise


class Stack(tf.keras.layers.Layer):
  """Stack the inputs along a new axis."""

  def __init__(self, axis=1, **kwargs):
    super(Stack, self).__init__(**kwargs)
    self.axis = axis

  def get_config(self):
    config = super(Stack, self).get_config()
    config['axis'] = self.axis
    return config

  def call(self, inputs):
    return tf.stack(inputs, axis=self.axis)


class Variable(tf.keras.layers.Layer):
  """Output a trainable value.

  Due to a quirk of Keras, you must pass an input value when invoking
  this layer.  It doesn't matter what value you pass.  Keras assumes
  every layer that is not an Input will have at least one parent, and
  violating this assumption causes errors during evaluation.
  """

  def __init__(self, initial_value, **kwargs):
    """Construct a variable layer.

    Parameters
    ----------
    initial_value: array or Tensor
      the initial value the layer should output
    """
    super(Variable, self).__init__(**kwargs)
    self.initial_value = initial_value

  def get_config(self):
    config = super(Variable, self).get_config()
    config['initial_value'] = self.initial_value
    return config

  def build(self, input_shape):
    self.var = tf.Variable(self.initial_value, dtype=self.dtype)
    self.built = True

  def call(self, inputs):
    return self.var


class VinaFreeEnergy(tf.keras.layers.Layer):
  """Computes free-energy as defined by Autodock Vina.

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self,
               N_atoms,
               M_nbrs,
               ndim,
               nbr_cutoff,
               start,
               stop,
               stddev=.3,
               Nrot=1,
               **kwargs):
    super(VinaFreeEnergy, self).__init__(**kwargs)
    self.stddev = stddev
    # Number of rotatable bonds
    # TODO(rbharath): Vina actually sets this per-molecule. See if makes
    # a difference.
    self.Nrot = Nrot
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop

  def get_config(self):
    config = super(VinaFreeEnergy, self).get_config()
    config['N_atoms'] = self.N_atoms
    config['M_nbrs'] = self.M_nbrs
    config['ndim'] = self.ndim
    config['nbr_cutoff'] = self.nbr_cutoff
    config['start'] = self.start
    config['stop'] = self.stop
    config['stddev'] = self.stddev
    config['Nrot'] = self.Nrot
    return config

  def build(self, input_shape):
    self.weighted_combo = WeightedLinearCombo()
    self.w = tf.Variable(tf.random.normal((1,), stddev=self.stddev))
    self.built = True

  def cutoff(self, d, x):
    out_tensor = tf.where(d < 8, x, tf.zeros_like(x))
    return out_tensor

  def nonlinearity(self, c, w):
    """Computes non-linearity used in Vina."""
    out_tensor = c / (1 + w * self.Nrot)
    return w, out_tensor

  def repulsion(self, d):
    """Computes Autodock Vina's repulsion interaction term."""
    out_tensor = tf.where(d < 0, d**2, tf.zeros_like(d))
    return out_tensor

  def hydrophobic(self, d):
    """Computes Autodock Vina's hydrophobic interaction term."""
    out_tensor = tf.where(d < 0.5, tf.ones_like(d),
                          tf.where(d < 1.5, 1.5 - d, tf.zeros_like(d)))
    return out_tensor

  def hydrogen_bond(self, d):
    """Computes Autodock Vina's hydrogen bond interaction term."""
    out_tensor = tf.where(
        d < -0.7, tf.ones_like(d),
        tf.where(d < 0, (1.0 / 0.7) * (0 - d), tf.zeros_like(d)))
    return out_tensor

  def gaussian_first(self, d):
    """Computes Autodock Vina's first Gaussian interaction term."""
    out_tensor = tf.exp(-(d / 0.5)**2)
    return out_tensor

  def gaussian_second(self, d):
    """Computes Autodock Vina's second Gaussian interaction term."""
    out_tensor = tf.exp(-((d - 3) / 2)**2)
    return out_tensor

  def call(self, inputs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (N, d)
      Coordinates/features.
    Z: tf.Tensor of shape (N)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B)
      The free energy of each complex in batch
    """
    X = inputs[0]
    Z = inputs[1]

    # TODO(rbharath): This layer shouldn't be neighbor-listing. Make
    # neighbors lists an argument instead of a part of this layer.
    nbr_list = NeighborList(self.N_atoms, self.M_nbrs, self.ndim,
                            self.nbr_cutoff, self.start, self.stop)(X)

    # Shape (N, M)
    dists = InteratomicL2Distances(self.N_atoms, self.M_nbrs,
                                   self.ndim)([X, nbr_list])

    repulsion = self.repulsion(dists)
    hydrophobic = self.hydrophobic(dists)
    hbond = self.hydrogen_bond(dists)
    gauss_1 = self.gaussian_first(dists)
    gauss_2 = self.gaussian_second(dists)

    # Shape (N, M)
    interactions = self.weighted_combo(
        [repulsion, hydrophobic, hbond, gauss_1, gauss_2])

    # Shape (N, M)
    thresholded = self.cutoff(dists, interactions)

    weight, free_energies = self.nonlinearity(thresholded, self.w)
    return tf.reduce_sum(free_energies)


class NeighborList(tf.keras.layers.Layer):
  """Computes a neighbor-list in Tensorflow.

  Neighbor-lists (also called Verlet Lists) are a tool for grouping
  atoms which are close to each other spatially. This layer computes a
  Neighbor List from a provided tensor of atomic coordinates. You can
  think of this as a general "k-means" layer, but optimized for the
  case `k==3`.

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self, N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop, **kwargs):
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
    nbr_cutoff: float
      Length in Angstroms (?) at which atom boxes are gridded.
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

  def get_config(self):
    config = super(NeighborList, self).get_config()
    config['N_atoms'] = self.N_atoms
    config['M_nbrs'] = self.M_nbrs
    config['ndim'] = self.ndim
    config['nbr_cutoff'] = self.nbr_cutoff
    config['start'] = self.start
    config['stop'] = self.stop
    return config

  def call(self, inputs):
    if isinstance(inputs, SequenceCollection):
      if len(inputs) != 1:
        raise ValueError("NeighborList can only have one input")
      inputs = inputs[0]
    if len(inputs.get_shape()) != 2:
      # TODO(rbharath): Support batching
      raise ValueError("Parent tensor must be (num_atoms, ndum)")
    return self.compute_nbr_list(inputs)

  def compute_nbr_list(self, coords):
    """Get closest neighbors for atoms.

    Needs to handle padding for atoms with no neighbors.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)

    Returns
    -------
    nbr_list: tf.Tensor
      Shape (N_atoms, M_nbrs) of atom indices
    """
    # Shape (n_cells, ndim)
    cells = self.get_cells()

    # List of length N_atoms, each element of different length uniques_i
    nbrs = self.get_atoms_in_nbrs(coords, cells)
    padding = tf.fill((self.M_nbrs,), -1)
    padded_nbrs = [tf.concat([unique_nbrs, padding], 0) for unique_nbrs in nbrs]

    # List of length N_atoms, each element of different length uniques_i
    # List of length N_atoms, each a tensor of shape
    # (uniques_i, ndim)
    nbr_coords = [tf.gather(coords, atom_nbrs) for atom_nbrs in nbrs]

    # Add phantom atoms that exist far outside the box
    coord_padding = tf.cast(
        tf.fill((self.M_nbrs, self.ndim), 2 * self.stop), tf.float32)
    padded_nbr_coords = [
        tf.concat([nbr_coord, coord_padding], 0) for nbr_coord in nbr_coords
    ]

    # List of length N_atoms, each of shape (1, ndim)
    atom_coords = tf.split(coords, self.N_atoms)
    # TODO(rbharath): How does distance need to be modified here to
    # account for periodic boundary conditions?
    # List of length N_atoms each of shape (M_nbrs)
    padded_dists = [
        tf.reduce_sum((atom_coord - padded_nbr_coord)**2, axis=1)
        for (atom_coord,
             padded_nbr_coord) in zip(atom_coords, padded_nbr_coords)
    ]

    padded_closest_nbrs = [
        tf.nn.top_k(-padded_dist, k=self.M_nbrs)[1]
        for padded_dist in padded_dists
    ]

    # N_atoms elts of size (M_nbrs,) each
    padded_neighbor_list = [
        tf.gather(padded_atom_nbrs, padded_closest_nbr)
        for (padded_atom_nbrs,
             padded_closest_nbr) in zip(padded_nbrs, padded_closest_nbrs)
    ]

    neighbor_list = tf.stack(padded_neighbor_list)

    return neighbor_list

  def get_atoms_in_nbrs(self, coords, cells):
    """Get the atoms in neighboring cells for each cells.

    Returns
    -------
    atoms_in_nbrs = (N_atoms, n_nbr_cells, M_nbrs)
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
    neighbor_cells = tf.squeeze(tf.gather(neighbor_cells, cells_for_atoms))

    # Shape (N_atoms, n_nbr_cells, M_nbrs)
    atoms_in_nbrs = tf.gather(closest_atoms, neighbor_cells)

    # Shape (N_atoms, n_nbr_cells*M_nbrs)
    atoms_in_nbrs = tf.reshape(atoms_in_nbrs, [self.N_atoms, -1])

    # List of length N_atoms, each element length uniques_i
    nbrs_per_atom = tf.split(atoms_in_nbrs, self.N_atoms)
    uniques = [
        tf.unique(tf.squeeze(atom_nbrs))[0] for atom_nbrs in nbrs_per_atom
    ]

    # TODO(rbharath): FRAGILE! Uses fact that identity seems to be the first
    # element removed to remove self from list of neighbors. Need to verify
    # this holds more broadly or come up with robust alternative.
    uniques = [unique[1:] for unique in uniques]

    return uniques

  def get_closest_atoms(self, coords, cells):
    """For each cell, find M_nbrs closest atoms.

    Let N_atoms be the number of atoms.

    Parameters
    ----------
    coords: tf.Tensor
      (N_atoms, ndim) shape.
    cells: tf.Tensor
      (n_cells, ndim) shape.

    Returns
    -------
    closest_inds: tf.Tensor
      Of shape (n_cells, M_nbrs)
    """
    N_atoms, n_cells, ndim, M_nbrs = (self.N_atoms, self.n_cells, self.ndim,
                                      self.M_nbrs)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.reshape(
        tf.tile(cells, (1, N_atoms)), (N_atoms * n_cells, ndim))

    # Shape (N_atoms*n_cells, ndim) after tile
    tiled_coords = tf.tile(coords, (n_cells, 1))

    # Shape (N_atoms*n_cells)
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    # Shape (n_cells, N_atoms)
    coords_norm = tf.reshape(coords_vec, (n_cells, N_atoms))

    # Find k atoms closest to this cell. Notice negative sign since
    # tf.nn.top_k returns *largest* not smallest.
    # Tensor of shape (n_cells, M_nbrs)
    closest_inds = tf.nn.top_k(-coords_norm, k=M_nbrs)[1]

    return closest_inds

  def get_cells_for_atoms(self, coords, cells):
    """Compute the cells each atom belongs to.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    cells_for_atoms: tf.Tensor
      Shape (N_atoms, 1)
    """
    N_atoms, n_cells, ndim = self.N_atoms, self.n_cells, self.ndim
    n_cells = int(n_cells)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.tile(cells, (N_atoms, 1))

    # Shape (N_atoms*n_cells, 1) after tile
    tiled_coords = tf.reshape(
        tf.tile(coords, (1, n_cells)), (n_cells * N_atoms, ndim))
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (N_atoms, n_cells))

    closest_inds = tf.nn.top_k(-coords_norm, k=1)[1]
    return closest_inds

  def _get_num_nbrs(self):
    """Get number of neighbors in current dimensionality space."""
    ndim = self.ndim
    if ndim == 1:
      n_nbr_cells = 3
    elif ndim == 2:
      # 9 neighbors in 2-space
      n_nbr_cells = 9
    # TODO(rbharath): Shoddy handling of higher dimensions...
    elif ndim >= 3:
      # Number of cells for cube in 3-space is
      n_nbr_cells = 27  # (26 faces on Rubik's cube for example)
    return n_nbr_cells

  def get_neighbor_cells(self, cells):
    """Compute neighbors of cells in grid.

    # TODO(rbharath): Do we need to handle periodic boundary conditions
    properly here?
    # TODO(rbharath): This doesn't handle boundaries well. We hard-code
    # looking for n_nbr_cells neighbors, which isn't right for boundary cells in
    # the cube.

    Parameters
    ----------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    nbr_cells: tf.Tensor
      (n_cells, n_nbr_cells)
    """
    ndim, n_cells = self.ndim, self.n_cells
    n_nbr_cells = self._get_num_nbrs()
    # Tile cells to form arrays of size (n_cells*n_cells, ndim)
    # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
    # Tile (a, a, a, b, b, b, etc.)
    tiled_centers = tf.reshape(
        tf.tile(cells, (1, n_cells)), (n_cells * n_cells, ndim))
    # Tile (a, b, c, a, b, c, ...)
    tiled_cells = tf.tile(cells, (n_cells, 1))

    coords_vec = tf.reduce_sum((tiled_centers - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (n_cells, n_cells))
    closest_inds = tf.nn.top_k(-coords_norm, k=n_nbr_cells)[1]

    return closest_inds

  def get_cells(self):
    """Returns the locations of all grid points in box.

    Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
    Then would return a list of length 20^3 whose entries would be
    [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

    Returns
    -------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    """
    start, stop, nbr_cutoff = self.start, self.stop, self.nbr_cutoff
    mesh_args = [tf.range(start, stop, nbr_cutoff) for _ in range(self.ndim)]
    return tf.cast(
        tf.reshape(
            tf.transpose(tf.stack(tf.meshgrid(*mesh_args))),
            (self.n_cells, self.ndim)), tf.float32)


class AtomicConvolution(tf.keras.layers.Layer):
  """Implements the atomic convolutional transform introduced in

  Gomes, Joseph, et al. "Atomic convolutional networks for predicting
  protein-ligand binding affinity." arXiv preprint arXiv:1703.10603
  (2017).

  At a high level, this transform performs a graph convolution
  on the nearest neighbors graph in 3D space.
  """

  def __init__(self,
               atom_types=None,
               radial_params=list(),
               boxsize=None,
               **kwargs):
    """Atomic convolution layer

    N = max_num_atoms, M = max_num_neighbors, B = batch_size, d = num_features
    l = num_radial_filters * num_atom_types

    Parameters
    ----------
    atom_types: list or None
      Of length a, where a is number of atom types for filtering.
    radial_params: list
      Of length l, where l is number of radial filters learned.
    boxsize: float or None
      Simulation box length [Angstrom].
    """
    super(AtomicConvolution, self).__init__(**kwargs)
    self.boxsize = boxsize
    self.radial_params = radial_params
    self.atom_types = atom_types

  def get_config(self):
    config = super(AtomicConvolution, self).get_config()
    config['atom_types'] = self.atom_types
    config['radial_params'] = self.radial_params
    config['boxsize'] = self.boxsize
    return config

  def build(self, input_shape):
    vars = []
    for i in range(3):
      val = np.array([p[i] for p in self.radial_params]).reshape((-1, 1, 1, 1))
      vars.append(tf.Variable(val, dtype=tf.float32))
    self.rc = vars[0]
    self.rs = vars[1]
    self.re = vars[2]
    self.built = True

  def call(self, inputs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list.
    Nbrs_Z: tf.Tensor of shape (B, N, M)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B, N, l)
      A new tensor representing the output of the atomic conv layer
    """
    X = inputs[0]
    Nbrs = tf.cast(inputs[1], tf.int32)
    Nbrs_Z = inputs[2]

    # N: Maximum number of atoms
    # M: Maximum number of neighbors
    # d: Number of coordinates/features/filters
    # B: Batch Size
    N = X.get_shape()[-2]
    d = X.get_shape()[-1]
    M = Nbrs.get_shape()[-1]
    B = X.get_shape()[0]

    # Compute the distances and radial symmetry functions.
    D = self.distance_tensor(X, Nbrs, self.boxsize, B, N, M, d)
    R = self.distance_matrix(D)
    R = tf.expand_dims(R, 0)
    rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)

    if not self.atom_types:
      cond = tf.cast(tf.not_equal(Nbrs_Z, 0), tf.float32)
      cond = tf.reshape(cond, (1, -1, N, M))
      layer = tf.reduce_sum(cond * rsf, 3)
    else:
      sym = []
      for j in range(len(self.atom_types)):
        cond = tf.cast(tf.equal(Nbrs_Z, self.atom_types[j]), tf.float32)
        cond = tf.reshape(cond, (1, -1, N, M))
        sym.append(tf.reduce_sum(cond * rsf, 3))
      layer = tf.concat(sym, 0)

    layer = tf.transpose(layer, [1, 2, 0])  # (l, B, N) -> (B, N, l)
    m, v = tf.nn.moments(layer, axes=[0])
    return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

  def radial_symmetry_function(self, R, rc, rs, e):
    """Calculates radial symmetry function.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_filters

    Parameters
    ----------
    R: tf.Tensor of shape (B, N, M)
      Distance matrix.
    rc: float
      Interaction cutoff [Angstrom].
    rs: float
      Gaussian distance matrix mean.
    e: float
      Gaussian distance matrix width.

    Returns
    -------
    retval: tf.Tensor of shape (B, N, M)
      Radial symmetry function (before summation)
    """
    K = self.gaussian_distance_matrix(R, rs, e)
    FC = self.radial_cutoff(R, rc)
    return tf.multiply(K, FC)

  def radial_cutoff(self, R, rc):
    """Calculates radial cutoff matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
      R [B, N, M]: tf.Tensor
        Distance matrix.
      rc: tf.Variable
        Interaction cutoff [Angstrom].

    Returns
    -------
    FC [B, N, M]: tf.Tensor
      Radial cutoff matrix.
    """
    T = 0.5 * (tf.cos(np.pi * R / (rc)) + 1)
    E = tf.zeros_like(T)
    cond = tf.less_equal(R, rc)
    FC = tf.where(cond, T, E)
    return FC

  def gaussian_distance_matrix(self, R, rs, e):
    """Calculates gaussian distance matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
    R [B, N, M]: tf.Tensor
      Distance matrix.
    rs: tf.Variable
      Gaussian distance matrix mean.
    e: tf.Variable
      Gaussian distance matrix width (e = .5/std**2).

    Returns
    -------
    retval [B, N, M]: tf.Tensor
      Gaussian distance matrix.
    """
    return tf.exp(-e * (R - rs)**2)

  def distance_tensor(self, X, Nbrs, boxsize, B, N, M, d):
    """Calculates distance tensor for batch of molecules.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features tensor.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list tensor.
    boxsize: float or None
      Simulation box length [Angstrom].

    Returns
    -------
    D: tf.Tensor of shape (B, N, M, d)
      Coordinates/features distance tensor.
    """
    flat_neighbors = tf.reshape(Nbrs, [-1, N * M])
    neighbor_coords = tf.gather(X, flat_neighbors, batch_dims=-1, axis=1)
    neighbor_coords = tf.reshape(neighbor_coords, [-1, N, M, d])
    D = neighbor_coords - tf.expand_dims(X, 2)
    if boxsize is not None:
      boxsize = tf.reshape(boxsize, [1, 1, 1, d])
      D -= tf.round(D / boxsize) * boxsize
    return D

  def distance_matrix(self, D):
    """Calcuates the distance matrix from the distance tensor

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    D: tf.Tensor of shape (B, N, M, d)
      Distance tensor.

    Returns
    -------
    R: tf.Tensor of shape (B, N, M)
       Distance matrix.
    """
    R = tf.reduce_sum(tf.multiply(D, D), 3)
    R = tf.sqrt(R)
    return R


class AlphaShareLayer(tf.keras.layers.Layer):
  """
  Part of a sluice network. Adds alpha parameters to control
  sharing between the main and auxillary tasks

  Factory method AlphaShare should be used for construction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  out_tensor: a tensor with shape [len(in_layers), x, y] where x, y were the original layer dimensions
  Distance matrix.
  """

  def __init__(self, **kwargs):
    super(AlphaShareLayer, self).__init__(**kwargs)

  def get_config(self):
    config = super(AlphaShareLayer, self).get_config()
    return config

  def build(self, input_shape):
    n_alphas = 2 * len(input_shape)
    self.alphas = tf.Variable(
        tf.random.normal([n_alphas, n_alphas]), name='alphas')
    self.built = True

  def call(self, inputs):
    # check that there isnt just one or zero inputs
    if len(inputs) <= 1:
      raise ValueError("AlphaShare must have more than one input")
    self.num_outputs = len(inputs)
    # create subspaces
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1])
    subspace_size = int(original_cols / 2)
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor[:, :subspace_size], [-1]))
      subspaces.append(tf.reshape(input_tensor[:, subspace_size:], [-1]))
    n_alphas = len(subspaces)
    subspaces = tf.reshape(tf.stack(subspaces), [n_alphas, -1])
    subspaces = tf.matmul(self.alphas, subspaces)

    # concatenate subspaces, reshape to size of original input, then stack
    # such that out_tensor has shape (2,?,original_cols)
    count = 0
    out_tensors = []
    tmp_tensor = []
    for row in range(n_alphas):
      tmp_tensor.append(tf.reshape(subspaces[row,], [-1, subspace_size]))
      count += 1
      if (count == 2):
        out_tensors.append(tf.concat(tmp_tensor, 1))
        tmp_tensor = []
        count = 0
    return out_tensors


class SluiceLoss(tf.keras.layers.Layer):
  """
  Calculates the loss in a Sluice Network
  Every input into an AlphaShare should be used in SluiceLoss
  """

  def __init__(self, **kwargs):
    super(SluiceLoss, self).__init__(**kwargs)

  def get_config(self):
    config = super(SluiceLoss, self).get_config()
    return config

  def call(self, inputs):
    temp = []
    subspaces = []
    # creates subspaces the same way it was done in AlphaShare
    for input_tensor in inputs:
      subspace_size = int(input_tensor.get_shape()[-1] / 2)
      subspaces.append(input_tensor[:, :subspace_size])
      subspaces.append(input_tensor[:, subspace_size:])
      product = tf.matmul(tf.transpose(subspaces[0]), subspaces[1])
      subspaces = []
      # calculate squared Frobenius norm
      temp.append(tf.reduce_sum(tf.pow(product, 2)))
    return tf.reduce_sum(temp)


class BetaShare(tf.keras.layers.Layer):
  """
  Part of a sluice network. Adds beta params to control which layer
  outputs are used for prediction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or
    more tensors

  Returns
  -------
  output_layers: list of Layers or tensors with same size as in_layers
    Distance matrix.
  """

  def __init__(self, **kwargs):
    super(BetaShare, self).__init__(**kwargs)

  def get_config(self):
    config = super(BetaShare, self).get_config()
    return config

  def build(self, input_shape):
    n_betas = len(input_shape)
    self.betas = tf.Variable(tf.random.normal([1, n_betas]), name='betas')
    self.built = True

  def call(self, inputs):
    """
    Size of input layers must all be the same
    """
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1])
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor, [-1]))
    n_betas = len(inputs)
    subspaces = tf.reshape(tf.stack(subspaces), [n_betas, -1])
    out_tensor = tf.matmul(self.betas, subspaces)
    return tf.reshape(out_tensor, [-1, original_cols])


class ANIFeat(tf.keras.layers.Layer):
  """Performs transform from 3D coordinates to ANI symmetry functions"""

  def __init__(self,
               max_atoms=23,
               radial_cutoff=4.6,
               angular_cutoff=3.1,
               radial_length=32,
               angular_length=8,
               atom_cases=[1, 6, 7, 8, 16],
               atomic_number_differentiated=True,
               coordinates_in_bohr=True,
               **kwargs):
    """
    Only X can be transformed
    """
    super(ANIFeat, self).__init__(**kwargs)
    self.max_atoms = max_atoms
    self.radial_cutoff = radial_cutoff
    self.angular_cutoff = angular_cutoff
    self.radial_length = radial_length
    self.angular_length = angular_length
    self.atom_cases = atom_cases
    self.atomic_number_differentiated = atomic_number_differentiated
    self.coordinates_in_bohr = coordinates_in_bohr

  def get_config(self):
    config = super(ANIFeat, self).get_config()
    config['max_atoms'] = self.max_atoms
    config['radial_cutoff'] = self.radial_cutoff
    config['angular_cutoff'] = self.angular_cutoff
    config['radial_length'] = self.radial_length
    config['angular_length'] = self.angular_length
    config['atom_cases'] = self.atom_cases
    config['atomic_number_differentiated'] = self.atomic_number_differentiated
    config['coordinates_in_bohr'] = self.coordinates_in_bohr
    return config

  def call(self, inputs):
    """In layers should be of shape dtype tf.float32, (None, self.max_atoms, 4)"""
    atom_numbers = tf.cast(inputs[:, :, 0], tf.int32)
    flags = tf.sign(atom_numbers)
    flags = tf.cast(
        tf.expand_dims(flags, 1) * tf.expand_dims(flags, 2), tf.float32)
    coordinates = inputs[:, :, 1:]
    if self.coordinates_in_bohr:
      coordinates = coordinates * 0.52917721092

    d = self.distance_matrix(coordinates, flags)

    d_radial_cutoff = self.distance_cutoff(d, self.radial_cutoff, flags)
    d_angular_cutoff = self.distance_cutoff(d, self.angular_cutoff, flags)

    radial_sym = self.radial_symmetry(d_radial_cutoff, d, atom_numbers)
    angular_sym = self.angular_symmetry(d_angular_cutoff, d, atom_numbers,
                                        coordinates)
    return tf.concat(
        [
            tf.cast(tf.expand_dims(atom_numbers, 2), tf.float32), radial_sym,
            angular_sym
        ],
        axis=2)

  def distance_matrix(self, coordinates, flags):
    """ Generate distance matrix """
    # (TODO YTZ:) faster, less memory intensive way
    # r = tf.reduce_sum(tf.square(coordinates), 2)
    # r = tf.expand_dims(r, -1)
    # inner = 2*tf.matmul(coordinates, tf.transpose(coordinates, perm=[0,2,1]))
    # # inner = 2*tf.matmul(coordinates, coordinates, transpose_b=True)

    # d = r - inner + tf.transpose(r, perm=[0,2,1])
    # d = tf.nn.relu(d) # fix numerical instabilities about diagonal
    # d = tf.sqrt(d) # does this have negative elements? may be unstable for diagonals

    max_atoms = self.max_atoms
    tensor1 = tf.stack([coordinates] * max_atoms, axis=1)
    tensor2 = tf.stack([coordinates] * max_atoms, axis=2)

    # Calculate pairwise distance
    d = tf.sqrt(
        tf.reduce_sum(tf.math.squared_difference(tensor1, tensor2), axis=3) +
        1e-7)

    d = d * flags
    return d

  def distance_cutoff(self, d, cutoff, flags):
    """ Generate distance matrix with trainable cutoff """
    # Cutoff with threshold Rc
    d_flag = flags * tf.sign(cutoff - d)
    d_flag = tf.nn.relu(d_flag)
    d_flag = d_flag * tf.expand_dims((1 - tf.eye(self.max_atoms)), 0)
    d = 0.5 * (tf.cos(np.pi * d / cutoff) + 1)
    return d * d_flag
    # return d

  def radial_symmetry(self, d_cutoff, d, atom_numbers):
    """ Radial Symmetry Function """
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.radial_cutoff, self.radial_length)
    ita = np.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
    Rs = tf.cast(np.reshape(Rs, (1, 1, 1, -1)), tf.float32)
    ita = tf.cast(np.reshape(ita, (1, 1, 1, -1)), tf.float32)
    length = ita.get_shape().as_list()[-1]

    d_cutoff = tf.stack([d_cutoff] * length, axis=3)
    d = tf.stack([d] * length, axis=3)

    out = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type in self.atom_cases:
        selected_atoms = tf.expand_dims(
            tf.expand_dims(atom_numbers_embedded[:, :, atom_type], axis=1),
            axis=3)
        out_tensors.append(tf.reduce_sum(out * selected_atoms, axis=2))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out, axis=2)

  def angular_symmetry(self, d_cutoff, d, atom_numbers, coordinates):
    """ Angular Symmetry Function """

    max_atoms = self.max_atoms
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.angular_cutoff, self.angular_length)
    ita = 3 / (Rs[1] - Rs[0])**2
    thetas = np.linspace(0., np.pi, self.angular_length)
    zeta = float(self.angular_length**2)

    ita, zeta, Rs, thetas = np.meshgrid(ita, zeta, Rs, thetas)
    zeta = tf.cast(np.reshape(zeta, (1, 1, 1, 1, -1)), tf.float32)
    ita = tf.cast(np.reshape(ita, (1, 1, 1, 1, -1)), tf.float32)
    Rs = tf.cast(np.reshape(Rs, (1, 1, 1, 1, -1)), tf.float32)
    thetas = tf.cast(np.reshape(thetas, (1, 1, 1, 1, -1)), tf.float32)
    length = zeta.get_shape().as_list()[-1]

    # tf.stack issues again...
    vector_distances = tf.stack([coordinates] * max_atoms, 1) - tf.stack(
        [coordinates] * max_atoms, 2)
    R_ij = tf.stack([d] * max_atoms, axis=3)
    R_ik = tf.stack([d] * max_atoms, axis=2)
    f_R_ij = tf.stack([d_cutoff] * max_atoms, axis=3)
    f_R_ik = tf.stack([d_cutoff] * max_atoms, axis=2)

    # Define angle theta = arccos(R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance))
    vector_mul = tf.reduce_sum(tf.stack([vector_distances] * max_atoms, axis=3) * \
                               tf.stack([vector_distances] * max_atoms, axis=2), axis=4)
    vector_mul = vector_mul * tf.sign(f_R_ij) * tf.sign(f_R_ik)
    theta = tf.acos(tf.math.divide(vector_mul, R_ij * R_ik + 1e-5))

    R_ij = tf.stack([R_ij] * length, axis=4)
    R_ik = tf.stack([R_ik] * length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * length, axis=4)
    theta = tf.stack([theta] * length, axis=4)

    out_tensor = tf.pow((1. + tf.cos(theta - thetas)) / 2., zeta) * \
                 tf.exp(-ita * tf.square((R_ij + R_ik) / 2. - Rs)) * f_R_ij * f_R_ik * 2

    if self.atomic_number_differentiated:
      out_tensors = []
      for id_j, atom_type_j in enumerate(self.atom_cases):
        for atom_type_k in self.atom_cases[id_j:]:
          selected_atoms = tf.stack([atom_numbers_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
                           tf.stack([atom_numbers_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
          selected_atoms = tf.expand_dims(
              tf.expand_dims(selected_atoms, axis=1), axis=4)
          out_tensors.append(
              tf.reduce_sum(out_tensor * selected_atoms, axis=(2, 3)))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out_tensor, axis=(2, 3))

  def get_num_feats(self):
    n_feat = self.outputs.get_shape().as_list()[-1]
    return n_feat


class GraphEmbedPoolLayer(tf.keras.layers.Layer):
  r"""
  GraphCNNPool Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  This is a learnable pool operation It constructs a new adjacency
  matrix for a graph of specified number of nodes.

  This differs from our other pool operations which set vertices to a
  function value without altering the adjacency matrix.

  ..math:: V_{emb} = SpatialGraphCNN({V_{in}})
  ..math:: V_{out} = \sigma(V_{emb})^{T} * V_{in}
  ..math:: A_{out} = V_{emb}^{T} * A_{in} * V_{emb}
  """

  def __init__(self, num_vertices, **kwargs):
    self.num_vertices = num_vertices
    super(GraphEmbedPoolLayer, self).__init__(**kwargs)

  def get_config(self):
    config = super(GraphEmbedPoolLayer, self).get_config()
    config['num_vertices'] = self.num_vertices
    return config

  def build(self, input_shape):
    no_features = int(input_shape[0][-1])
    self.W = tf.Variable(
        tf.random.truncated_normal(
            [no_features, self.num_vertices],
            stddev=1.0 / np.sqrt(no_features)),
        name='weights',
        dtype=tf.float32)
    self.b = tf.Variable(tf.constant(0.1), name='bias', dtype=tf.float32)
    self.built = True

  def call(self, inputs):
    """
    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output
    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns
    -------
    Returns a `tf.tensor` with a graph convolution applied
    The shape will be `(batch, vertex, self.num_filters)`.
    """
    if len(inputs) == 3:
      V, A, mask = inputs
    else:
      V, A = inputs
      mask = None
    factors = self.embedding_factors(V)

    if mask is not None:
      factors = tf.multiply(factors, mask)
    factors = self.softmax_factors(factors)

    result = tf.matmul(factors, V, transpose_a=True)

    result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
    result_A = tf.matmul(result_A, factors)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
    result_A = tf.matmul(factors, result_A, transpose_a=True)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], self.num_vertices,
                                     A.get_shape()[2], self.num_vertices))
    return result, result_A

  def embedding_factors(self, V):
    no_features = V.get_shape()[-1]
    V_reshape = tf.reshape(V, (-1, no_features))
    s = tf.slice(tf.shape(V), [0], [len(V.get_shape()) - 1])
    s = tf.concat([s, tf.stack([self.num_vertices])], 0)
    result = tf.reshape(tf.matmul(V_reshape, self.W) + self.b, s)
    return result

  def softmax_factors(self, V, axis=1):
    max_value = tf.reduce_max(V, axis=axis, keepdims=True)
    exp = tf.exp(tf.subtract(V, max_value))
    prob = tf.math.divide(exp, tf.reduce_sum(exp, axis=axis, keepdims=True))
    return prob


class GraphCNN(tf.keras.layers.Layer):
  r"""
  GraphCNN Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  Spatial-domain convolutions can be defined as
  H = h_0I + h_1A + h_2A^2 + ... + hkAk, H ∈ R**(N×N)

  We approximate it by
  H ≈ h_0I + h_1A

  We can define a convolution as applying multiple these linear filters
  over edges of different types (think up, down, left, right, diagonal in images)
  Where each edge type has its own adjacency matrix
  H ≈ h_0I + h_1A_1 + h_2A_2 + . . . h_(L−1)A_(L−1)

  V_out = \sum_{c=1}^{C} H^{c} V^{c} + b
  """

  def __init__(self, num_filters, **kwargs):
    """
    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output

    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns: tf.tensor
    Returns a tf.tensor with a graph convolution applied
    The shape will be (batch, vertex, self.num_filters)
    """
    super(GraphCNN, self).__init__(**kwargs)
    self.num_filters = num_filters

  def get_config(self):
    config = super(GraphCNN, self).get_config()
    config['num_filters'] = self.num_filters
    return config

  def build(self, input_shape):
    no_features = int(input_shape[0][2])
    no_A = int(input_shape[1][2])
    self.W = tf.Variable(
        tf.random.truncated_normal(
            [no_features * no_A, self.num_filters],
            stddev=np.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='weights',
        dtype=tf.float32)
    self.W_I = tf.Variable(
        tf.random.truncated_normal(
            [no_features, self.num_filters],
            stddev=np.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='weights_I',
        dtype=tf.float32)
    self.b = tf.Variable(tf.constant(0.1), name='bias', dtype=tf.float32)
    self.built = True

  def call(self, inputs):
    if len(inputs) == 3:
      V, A, mask = inputs
    else:
      V, A = inputs
    no_A = A.get_shape()[2]
    no_features = V.get_shape()[2]
    n = self.graphConvolution(V, A)
    A_shape = tf.shape(A)
    n = tf.reshape(n, [-1, A_shape[1], no_A * no_features])
    return self.batch_mat_mult(n, self.W) + self.batch_mat_mult(
        V, self.W_I) + self.b

  def graphConvolution(self, V, A):
    no_A = A.get_shape()[2]
    no_features = V.get_shape()[2]
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1] * no_A, A_shape[1]]))
    n = tf.matmul(A_reshape, V)
    return tf.reshape(n, [-1, A_shape[1], no_A, no_features])

  def batch_mat_mult(self, A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
      axis_2 = -1
    else:
      axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    return tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))


class Highway(tf.keras.layers.Layer):
  """ Create a highway layer. y = H(x) * T(x) + x * (1 - T(x))

  H(x) = activation_fn(matmul(W_H, x) + b_H) is the non-linear transformed output
  T(x) = sigmoid(matmul(W_T, x) + b_T) is the transform gate

  Implementation based on paper

  Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway networks." arXiv preprint arXiv:1505.00387 (2015).


  This layer expects its input to be a two dimensional tensor
  of shape (batch size, # input features).  Outputs will be in
  the same shape.
  """

  def __init__(self,
               activation_fn='relu',
               biases_initializer='zeros',
               weights_initializer=None,
               **kwargs):
    """
    Parameters
    ----------
    activation_fn: object
      the Tensorflow activation function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    super(Highway, self).__init__(**kwargs)
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer

  def get_config(self):
    config = super(Highway, self).get_config()
    config['activation_fn'] = self.activation_fn
    config['biases_initializer'] = self.biases_initializer
    config['weights_initializer'] = self.weights_initializer
    return config

  def build(self, input_shape):
    if isinstance(input_shape, SequenceCollection):
      input_shape = input_shape[0]
    out_channels = input_shape[1]

    if self.weights_initializer is None:
      weights_initializer = tf.keras.initializers.VarianceScaling
    else:
      weights_initializer = self.weights_initializer

    self.dense_H = tf.keras.layers.Dense(
        out_channels,
        activation=self.activation_fn,
        bias_initializer=self.biases_initializer,
        kernel_initializer=weights_initializer)
    self.dense_T = tf.keras.layers.Dense(
        out_channels,
        activation=tf.nn.sigmoid,
        bias_initializer=tf.constant_initializer(-1),
        kernel_initializer=weights_initializer)
    self.built = True

  def call(self, inputs):
    if isinstance(inputs, SequenceCollection):
      parent = inputs[0]
    else:
      parent = inputs
    dense_H = self.dense_H(parent)
    dense_T = self.dense_T(parent)
    return tf.multiply(dense_H, dense_T) + tf.multiply(parent, 1 - dense_T)


class WeaveLayer(tf.keras.layers.Layer):
  """This class implements the core Weave convolution from the
  Google graph convolution paper [1]_

  This model contains atom features and bond features
  separately.Here, bond features are also called pair features.
  There are 2 types of transformation, atom->atom, atom->pair,
  pair->atom, pair->pair that this model implements.

  Examples
  --------
  This layer expects 4 inputs in a list of the form `[atom_features,
  pair_features, pair_split, atom_to_pair]`. We'll walk through the structure
  of these inputs. Let's start with some basic definitions.

  >>> import deepchem as dc
  >>> import numpy as np

  Suppose you have a batch of molecules

  >>> smiles = ["CCC", "C"]

  Note that there are 4 atoms in total in this system. This layer expects its
  input molecules to be batched together.

  >>> total_n_atoms = 4

  Let's suppose that we have a featurizer that computes `n_atom_feat` features
  per atom.

  >>> n_atom_feat = 75

  Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
  n_atom_feat)` of atomic features. For simplicity, let's just go with a
  random such matrix.

  >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

  Let's suppose we have `n_pair_feat` pairwise features

  >>> n_pair_feat = 14

  For each molecule, we compute a matrix of shape `(n_atoms*n_atoms,
  n_pair_feat)` of pairwise features for each pair of atoms in the molecule.
  Let's construct this conceptually for our example.

  >>> pair_feat = [np.random.rand(3*3, n_pair_feat), np.random.rand(1*1, n_pair_feat)]
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

  >>> A = np.array(A)
  >>> A.shape
  (4, 50)
  >>> P = np.array(P)
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
               init: str = 'glorot_uniform',
               activation: str = 'relu',
               batch_normalize: bool = True,
               batch_normalize_kwargs: Dict = {"renorm": True},
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
    init: str, optional (default 'glorot_uniform')
      Weight initialization for filters.
    activation: str, optional (default 'relu')
      Activation function applied
    batch_normalize: bool, optional (default True)
      If this is turned on, apply batch normalization before applying
      activation functions on convolutional layers.
    batch_normalize_kwargs: Dict, optional (default `{renorm=True}`)
      Batch normalization is a complex layer which has many potential
      argumentswhich change behavior. This layer accepts user-defined
      parameters which are passed to all `BatchNormalization` layers in
      `WeaveModel`, `WeaveLayer`, and `WeaveGather`.
    """
    super(WeaveLayer, self).__init__(**kwargs)
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)
    self.update_pair = update_pair  # last weave layer does not need to update
    self.n_hidden_AA = n_hidden_AA
    self.n_hidden_PA = n_hidden_PA
    self.n_hidden_AP = n_hidden_AP
    self.n_hidden_PP = n_hidden_PP
    self.n_hidden_A = n_hidden_AA + n_hidden_PA
    self.n_hidden_P = n_hidden_AP + n_hidden_PP
    self.batch_normalize = batch_normalize
    self.batch_normalize_kwargs = batch_normalize_kwargs

    self.n_atom_input_feat = n_atom_input_feat
    self.n_pair_input_feat = n_pair_input_feat
    self.n_atom_output_feat = n_atom_output_feat
    self.n_pair_output_feat = n_pair_output_feat
    self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None

  def get_config(self) -> Dict:
    """Returns config dictionary for this layer."""
    config = super(WeaveLayer, self).get_config()
    config['n_atom_input_feat'] = self.n_atom_input_feat
    config['n_pair_input_feat'] = self.n_pair_input_feat
    config['n_atom_output_feat'] = self.n_atom_output_feat
    config['n_pair_output_feat'] = self.n_pair_output_feat
    config['n_hidden_AA'] = self.n_hidden_AA
    config['n_hidden_PA'] = self.n_hidden_PA
    config['n_hidden_AP'] = self.n_hidden_AP
    config['n_hidden_PP'] = self.n_hidden_PP
    config['batch_normalize'] = self.batch_normalize
    config['batch_normalize_kwargs'] = self.batch_normalize_kwargs
    config['update_pair'] = self.update_pair
    config['init'] = self.init
    config['activation'] = self.activation
    return config

  def build(self, input_shape):
    """ Construct internal trainable weights.

    Parameters
    ----------
    input_shape: tuple
      Ignored since we don't need the input shape to create internal weights.
    """

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    self.W_AA = init([self.n_atom_input_feat, self.n_hidden_AA])
    self.b_AA = backend.zeros(shape=[
        self.n_hidden_AA,
    ])
    self.AA_bn = BatchNormalization(**self.batch_normalize_kwargs)

    self.W_PA = init([self.n_pair_input_feat, self.n_hidden_PA])
    self.b_PA = backend.zeros(shape=[
        self.n_hidden_PA,
    ])
    self.PA_bn = BatchNormalization(**self.batch_normalize_kwargs)

    self.W_A = init([self.n_hidden_A, self.n_atom_output_feat])
    self.b_A = backend.zeros(shape=[
        self.n_atom_output_feat,
    ])
    self.A_bn = BatchNormalization(**self.batch_normalize_kwargs)

    if self.update_pair:
      self.W_AP = init([self.n_atom_input_feat * 2, self.n_hidden_AP])
      self.b_AP = backend.zeros(shape=[
          self.n_hidden_AP,
      ])
      self.AP_bn = BatchNormalization(**self.batch_normalize_kwargs)

      self.W_PP = init([self.n_pair_input_feat, self.n_hidden_PP])
      self.b_PP = backend.zeros(shape=[
          self.n_hidden_PP,
      ])
      self.PP_bn = BatchNormalization(**self.batch_normalize_kwargs)

      self.W_P = init([self.n_hidden_P, self.n_pair_output_feat])
      self.b_P = backend.zeros(shape=[
          self.n_pair_output_feat,
      ])
      self.P_bn = BatchNormalization(**self.batch_normalize_kwargs)
    self.built = True

  def call(self, inputs: List) -> List:
    """Creates weave tensors.

    Parameters
    ----------
    inputs: List
      Should contain 4 tensors [atom_features, pair_features, pair_split,
      atom_to_pair]
    """
    atom_features = inputs[0]
    pair_features = inputs[1]

    pair_split = inputs[2]
    atom_to_pair = inputs[3]

    activation = self.activation_fn

    AA = tf.matmul(atom_features, self.W_AA) + self.b_AA
    if self.batch_normalize:
      AA = self.AA_bn(AA)
    AA = activation(AA)
    PA = tf.matmul(pair_features, self.W_PA) + self.b_PA
    if self.batch_normalize:
      PA = self.PA_bn(PA)
    PA = activation(PA)
    PA = tf.math.segment_sum(PA, pair_split)

    A = tf.matmul(tf.concat([AA, PA], 1), self.W_A) + self.b_A
    if self.batch_normalize:
      A = self.A_bn(A)
    A = activation(A)

    if self.update_pair:
      # Note that AP_ij and AP_ji share the same self.AP_bn batch
      # normalization
      AP_ij = tf.matmul(
          tf.reshape(
              tf.gather(atom_features, atom_to_pair),
              [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
      if self.batch_normalize:
        AP_ij = self.AP_bn(AP_ij)
      AP_ij = activation(AP_ij)
      AP_ji = tf.matmul(
          tf.reshape(
              tf.gather(atom_features, tf.reverse(atom_to_pair, [1])),
              [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
      if self.batch_normalize:
        AP_ji = self.AP_bn(AP_ji)
      AP_ji = activation(AP_ji)

      PP = tf.matmul(pair_features, self.W_PP) + self.b_PP
      if self.batch_normalize:
        PP = self.PP_bn(PP)
      PP = activation(PP)
      P = tf.matmul(tf.concat([AP_ij + AP_ji, PP], 1), self.W_P) + self.b_P
      if self.batch_normalize:
        P = self.P_bn(P)
      P = activation(P)
    else:
      P = pair_features

    return [A, P]


class WeaveGather(tf.keras.layers.Layer):
  """Implements the weave-gathering section of weave convolutions.

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

  Note
  ----
  This class requires `tensorflow_probability` to be installed.
  """

  def __init__(self,
               batch_size: int,
               n_input: int = 128,
               gaussian_expand: bool = True,
               compress_post_gaussian_expansion: bool = False,
               init: str = 'glorot_uniform',
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
    init: str, optional (default 'glorot_uniform')
      Weight initialization for filters if `compress_post_gaussian_expansion`
      is True.
    activation: str, optional (default 'tanh')
      Activation function applied for filters if
      `compress_post_gaussian_expansion` is True. Should be recognizable by
      `tf.keras.activations`.
    """
    try:
      import tensorflow_probability as tfp
    except ModuleNotFoundError:
      raise ImportError(
          "This class requires tensorflow-probability to be installed.")
    super(WeaveGather, self).__init__(**kwargs)
    self.n_input = n_input
    self.batch_size = batch_size
    self.gaussian_expand = gaussian_expand
    self.compress_post_gaussian_expansion = compress_post_gaussian_expansion
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)

  def get_config(self):
    config = super(WeaveGather, self).get_config()
    config['batch_size'] = self.batch_size
    config['n_input'] = self.n_input
    config['gaussian_expand'] = self.gaussian_expand
    config['init'] = self.init
    config['activation'] = self.activation
    config[
        'compress_post_gaussian_expansion'] = self.compress_post_gaussian_expansion
    return config

  def build(self, input_shape):
    if self.compress_post_gaussian_expansion:

      def init(input_shape):
        return self.add_weight(
            name='kernel',
            shape=(input_shape[0], input_shape[1]),
            initializer=self.init,
            trainable=True)

      self.W = init([self.n_input * 11, self.n_input])
      self.b = backend.zeros(shape=[self.n_input])
    self.built = True

  def call(self, inputs: List) -> List:
    """Creates weave tensors.

    Parameters
    ----------
    inputs: List
      Should contain 2 tensors [atom_features, atom_split]

    Returns
    -------
    output_molecules: List
      Each entry in this list is of shape `(self.n_inputs,)`

    """
    outputs = inputs[0]
    atom_split = inputs[1]

    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)

    output_molecules = tf.math.segment_sum(outputs, atom_split)

    if self.compress_post_gaussian_expansion:
      output_molecules = tf.matmul(output_molecules, self.W) + self.b
      output_molecules = self.activation_fn(output_molecules)

    return output_molecules

  def gaussian_histogram(self, x):
    """Expands input into a set of gaussian histogram bins.

    Parameters
    ----------
    x: tf.Tensor
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
    outputs: tf.Tensor
      Of shape `(N, 11*n_feat)`
    """
    import tensorflow_probability as tfp
    gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                            (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                            (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                            (1.080, 0.170), (1.645, 0.283)]
    dist = [tfp.distributions.Normal(p[0], p[1]) for p in gaussian_memberships]
    dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
    outputs = [dist[i].prob(x) / dist_max[i] for i in range(11)]
    outputs = tf.stack(outputs, axis=2)
    outputs = outputs / tf.reduce_sum(outputs, axis=2, keepdims=True)
    outputs = tf.reshape(outputs, [-1, self.n_input * 11])
    return outputs


class DTNNEmbedding(tf.keras.layers.Layer):

  def __init__(self,
               n_embedding=30,
               periodic_table_length=30,
               init='glorot_uniform',
               **kwargs):
    """
    Parameters
    ----------
    n_embedding: int, optional
      Number of features for each atom
    periodic_table_length: int, optional
      Length of embedding, 83=Bi
    init: str, optional
      Weight initialization for filters.
    """
    super(DTNNEmbedding, self).__init__(**kwargs)
    self.n_embedding = n_embedding
    self.periodic_table_length = periodic_table_length
    self.init = init  # Set weight initialization

  def get_config(self):
    config = super(DTNNEmbedding, self).get_config()
    config['n_embedding'] = self.n_embedding
    config['periodic_table_length'] = self.periodic_table_length
    config['init'] = self.init
    return config

  def build(self, input_shape):

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    self.embedding_list = init([self.periodic_table_length, self.n_embedding])
    self.built = True

  def call(self, inputs):
    """
    parent layers: atom_number
    """
    atom_number = inputs
    return tf.nn.embedding_lookup(self.embedding_list, atom_number)


class DTNNStep(tf.keras.layers.Layer):

  def __init__(self,
               n_embedding=30,
               n_distance=100,
               n_hidden=60,
               init='glorot_uniform',
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
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    """
    super(DTNNStep, self).__init__(**kwargs)
    self.n_embedding = n_embedding
    self.n_distance = n_distance
    self.n_hidden = n_hidden
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)

  def get_config(self):
    config = super(DTNNStep, self).get_config()
    config['n_embedding'] = self.n_embedding
    config['n_distance'] = self.n_distance
    config['n_hidden'] = self.n_hidden
    config['activation'] = self.activation
    config['init'] = self.init
    return config

  def build(self, input_shape):

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    self.W_cf = init([self.n_embedding, self.n_hidden])
    self.W_df = init([self.n_distance, self.n_hidden])
    self.W_fc = init([self.n_hidden, self.n_embedding])
    self.b_cf = backend.zeros(shape=[
        self.n_hidden,
    ])
    self.b_df = backend.zeros(shape=[
        self.n_hidden,
    ])
    self.built = True

  def call(self, inputs):
    """
    parent layers: atom_features, distance, distance_membership_i, distance_membership_j
    """
    atom_features = inputs[0]
    distance = inputs[1]
    distance_membership_i = inputs[2]
    distance_membership_j = inputs[3]
    distance_hidden = tf.matmul(distance, self.W_df) + self.b_df
    atom_features_hidden = tf.matmul(atom_features, self.W_cf) + self.b_cf
    outputs = tf.multiply(
        distance_hidden, tf.gather(atom_features_hidden, distance_membership_j))

    # for atom i in a molecule m, this step multiplies together distance info of atom pair(i,j)
    # and embeddings of atom j(both gone through a hidden layer)
    outputs = tf.matmul(outputs, self.W_fc)
    outputs = self.activation_fn(outputs)

    output_ii = tf.multiply(self.b_df, atom_features_hidden)
    output_ii = tf.matmul(output_ii, self.W_fc)
    output_ii = self.activation_fn(output_ii)

    # for atom i, sum the influence from all other atom j in the molecule
    return tf.math.segment_sum(
        outputs, distance_membership_i) - output_ii + atom_features


class DTNNGather(tf.keras.layers.Layer):

  def __init__(self,
               n_embedding=30,
               n_outputs=100,
               layer_sizes=[100],
               output_activation=True,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
    Parameters
    ----------
    n_embedding: int, optional
      Number of features for each atom
    n_outputs: int, optional
      Number of features for each molecule(output)
    layer_sizes: list of int, optional(default=[1000])
      Structure of hidden layer(s)
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    """
    super(DTNNGather, self).__init__(**kwargs)
    self.n_embedding = n_embedding
    self.n_outputs = n_outputs
    self.layer_sizes = layer_sizes
    self.output_activation = output_activation
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)

  def get_config(self):
    config = super(DTNNGather, self).get_config()
    config['n_embedding'] = self.n_embedding
    config['n_outputs'] = self.n_outputs
    config['layer_sizes'] = self.layer_sizes
    config['output_activation'] = self.output_activation
    config['init'] = self.init
    config['activation'] = self.activation
    return config

  def build(self, input_shape):
    self.W_list = []
    self.b_list = []

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    prev_layer_size = self.n_embedding
    for i, layer_size in enumerate(self.layer_sizes):
      self.W_list.append(init([prev_layer_size, layer_size]))
      self.b_list.append(backend.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(init([prev_layer_size, self.n_outputs]))
    self.b_list.append(backend.zeros(shape=[
        self.n_outputs,
    ]))
    self.built = True

  def call(self, inputs):
    """
    parent layers: atom_features, atom_membership
    """
    output = inputs[0]
    atom_membership = inputs[1]

    for i, W in enumerate(self.W_list[:-1]):
      output = tf.matmul(output, W) + self.b_list[i]
      output = self.activation_fn(output)
    output = tf.matmul(output, self.W_list[-1]) + self.b_list[-1]
    if self.output_activation:
      output = self.activation_fn(output)
    return tf.math.segment_sum(output, atom_membership)


def _DAGgraph_step(batch_inputs, W_list, b_list, activation_fn, dropouts,
                   training):
  outputs = batch_inputs
  for idw, (dropout, W) in enumerate(zip(dropouts, W_list)):
    outputs = tf.nn.bias_add(tf.matmul(outputs, W), b_list[idw])
    outputs = activation_fn(outputs)
    if dropout is not None:
      outputs = dropout(outputs, training=training)
  return outputs


class DAGLayer(tf.keras.layers.Layer):
  """DAG computation layer.

  This layer generates a directed acyclic graph for each atom
  in a molecule. This layer is based on the algorithm from [1]_

  This layer performs a sort of inward sweep. Recall that for
  each atom, a DAG is generated that "points inward" to that
  atom from the undirected molecule graph. Picture this as
  "picking up" the atom as the vertex and using the natural
  tree structure that forms from gravity. The layer "sweeps
  inwards" from the leaf nodes of the DAG upwards to the
  atom. This is batched so the transformation is done for
  each atom.

  References
  ----------
  ..[1] Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. "Deep
  architectures and deep learning in chemoinformatics: the prediction of
  aqueous solubility for drug-like molecules." Journal of chemical information
  and modeling 53.7 (2013): 1563-1575.
  """

  def __init__(self,
               n_graph_feat=30,
               n_atom_feat=75,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               batch_size=64,
               **kwargs):
    """
    Parameters
    ----------
    n_graph_feat: int, optional
      Number of features for each node(and the whole grah).
    n_atom_feat: int, optional
      Number of features listed per atom.
    max_atoms: int, optional
      Maximum number of atoms in molecules.
    layer_sizes: list of int, optional(default=[100])
      List of hidden layer size(s):
      length of this list represents the number of hidden layers,
      and each element is the width of corresponding hidden layer.
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied.
    dropout: float, optional
      Dropout probability in hidden layer(s).
    batch_size: int, optional
      number of molecules in a batch.
    """
    super(DAGLayer, self).__init__(**kwargs)
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.batch_size = batch_size
    self.n_inputs = n_atom_feat + (self.max_atoms - 1) * n_graph_feat
    # number of inputs each step
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_graph_feat
    self.n_atom_feat = n_atom_feat

  def get_config(self):
    config = super(DAGLayer, self).get_config()
    config['n_graph_feat'] = self.n_graph_feat
    config['n_atom_feat'] = self.n_atom_feat
    config['max_atoms'] = self.max_atoms
    config['layer_sizes'] = self.layer_sizes
    config['init'] = self.init
    config['activation'] = self.activation
    config['dropout'] = self.dropout
    config['batch_size'] = self.batch_size
    return config

  def build(self, input_shape):
    """"Construct internal trainable weights."""
    self.W_list = []
    self.b_list = []
    self.dropouts = []
    prev_layer_size = self.n_inputs
    for layer_size in self.layer_sizes:
      self.W_list.append(
          self.add_weight(
              name='kernel',
              shape=(prev_layer_size, layer_size),
              initializer=self.init,
              trainable=True))
      self.b_list.append(
          self.add_weight(
              name='bias',
              shape=(layer_size,),
              initializer='zeros',
              trainable=True))
      if self.dropout is not None and self.dropout > 0.0:
        self.dropouts.append(Dropout(rate=self.dropout))
      else:
        self.dropouts.append(None)
      prev_layer_size = layer_size
    self.W_list.append(
        self.add_weight(
            name='kernel',
            shape=(prev_layer_size, self.n_outputs),
            initializer=self.init,
            trainable=True))
    self.b_list.append(
        self.add_weight(
            name='bias',
            shape=(self.n_outputs,),
            initializer='zeros',
            trainable=True))
    if self.dropout is not None and self.dropout > 0.0:
      self.dropouts.append(Dropout(rate=self.dropout))
    else:
      self.dropouts.append(None)
    self.built = True

  def call(self, inputs, training=True):
    """
    parent layers: atom_features, parents, calculation_orders, calculation_masks, n_atoms
    """
    atom_features = inputs[0]
    # each atom corresponds to a graph, which is represented by the `max_atoms*max_atoms` int32 matrix of index
    # each gragh include `max_atoms` of steps(corresponding to rows) of calculating graph features
    parents = tf.cast(inputs[1], dtype=tf.int32)
    # target atoms for each step: (batch_size*max_atoms) * max_atoms
    calculation_orders = inputs[2]
    calculation_masks = inputs[3]

    n_atoms = tf.squeeze(inputs[4])
    graph_features = tf.zeros((self.max_atoms * self.batch_size,
                               self.max_atoms + 1, self.n_graph_feat))

    for count in range(self.max_atoms):
      # `count`-th step
      # extracting atom features of target atoms: (batch_size*max_atoms) * n_atom_features
      mask = calculation_masks[:, count]
      current_round = tf.boolean_mask(calculation_orders[:, count], mask)
      batch_atom_features = tf.gather(atom_features, current_round)

      # generating index for graph features used in the inputs
      stack1 = tf.reshape(
          tf.stack(
              [tf.boolean_mask(tf.range(n_atoms), mask)] * (self.max_atoms - 1),
              axis=1), [-1])
      stack2 = tf.reshape(tf.boolean_mask(parents[:, count, 1:], mask), [-1])
      index = tf.stack([stack1, stack2], axis=1)
      # extracting graph features for parents of the target atoms, then flatten
      # shape: (batch_size*max_atoms) * [(max_atoms-1)*n_graph_features]
      batch_graph_features = tf.reshape(
          tf.gather_nd(graph_features, index),
          [-1, (self.max_atoms - 1) * self.n_graph_feat])

      # concat into the input tensor: (batch_size*max_atoms) * n_inputs
      batch_inputs = tf.concat(
          axis=1, values=[batch_atom_features, batch_graph_features])
      # DAGgraph_step maps from batch_inputs to a batch of graph_features
      # of shape: (batch_size*max_atoms) * n_graph_features
      # representing the graph features of target atoms in each graph
      batch_outputs = _DAGgraph_step(batch_inputs, self.W_list, self.b_list,
                                     self.activation_fn, self.dropouts,
                                     training)

      # index for targe atoms
      target_index = tf.stack([tf.range(n_atoms), parents[:, count, 0]], axis=1)
      target_index = tf.boolean_mask(target_index, mask)
      graph_features = tf.tensor_scatter_nd_update(graph_features, target_index,
                                                   batch_outputs)
    return batch_outputs


class DAGGather(tf.keras.layers.Layer):

  def __init__(self,
               n_graph_feat=30,
               n_outputs=30,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """DAG vector gathering layer

    Parameters
    ----------
    n_graph_feat: int, optional
      Number of features for each atom.
    n_outputs: int, optional
      Number of features for each molecule.
    max_atoms: int, optional
      Maximum number of atoms in molecules.
    layer_sizes: list of int, optional
      List of hidden layer size(s):
      length of this list represents the number of hidden layers,
      and each element is the width of corresponding hidden layer.
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied.
    dropout: float, optional
      Dropout probability in the hidden layer(s).
    """
    super(DAGGather, self).__init__(**kwargs)
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.activation_fn = activations.get(activation)
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs

  def get_config(self):
    config = super(DAGGather, self).get_config()
    config['n_graph_feat'] = self.n_graph_feat
    config['n_outputs'] = self.n_outputs
    config['max_atoms'] = self.max_atoms
    config['layer_sizes'] = self.layer_sizes
    config['init'] = self.init
    config['activation'] = self.activation
    config['dropout'] = self.dropout
    return config

  def build(self, input_shape):
    self.W_list = []
    self.b_list = []
    self.dropouts = []
    prev_layer_size = self.n_graph_feat
    for layer_size in self.layer_sizes:
      self.W_list.append(
          self.add_weight(
              name='kernel',
              shape=(prev_layer_size, layer_size),
              initializer=self.init,
              trainable=True))
      self.b_list.append(
          self.add_weight(
              name='bias',
              shape=(layer_size,),
              initializer='zeros',
              trainable=True))
      if self.dropout is not None and self.dropout > 0.0:
        self.dropouts.append(Dropout(rate=self.dropout))
      else:
        self.dropouts.append(None)
      prev_layer_size = layer_size
    self.W_list.append(
        self.add_weight(
            name='kernel',
            shape=(prev_layer_size, self.n_outputs),
            initializer=self.init,
            trainable=True))
    self.b_list.append(
        self.add_weight(
            name='bias',
            shape=(self.n_outputs,),
            initializer='zeros',
            trainable=True))
    if self.dropout is not None and self.dropout > 0.0:
      self.dropouts.append(Dropout(rate=self.dropout))
    else:
      self.dropouts.append(None)
    self.built = True

  def call(self, inputs, training=True):
    """
    parent layers: atom_features, membership
    """
    atom_features = inputs[0]
    membership = inputs[1]
    # Extract atom_features
    graph_features = tf.math.segment_sum(atom_features, membership)
    # sum all graph outputs
    return _DAGgraph_step(graph_features, self.W_list, self.b_list,
                          self.activation_fn, self.dropouts, training)


class MessagePassing(tf.keras.layers.Layer):
  """ General class for MPNN message passing.

  Default structures are built according to [1]_. This class performs `T`
  steps of message passing.

  Examples
  --------
  >>> import numpy as np
  >>> import deepchem as dc

  Suppose you have a batch of molecules

  >>> smiles = ["CCC", "C"]

  Note that there are 4 atoms in total in this system. This layer expects its
  input molecules to be batched together.

  >>> total_n_atoms = 4

  Let's suppose that we have a featurizer that computes `n_atom_feat` features
  per atom.

  >>> n_atom_feat = 75

  Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
  n_atom_feat)` of atomic features. For simplicity, let's just go with a
  random such matrix.

  >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

  Let's suppose we have `n_pair_feat` pairwise features

  >>> n_pair_feat = 14

  For each molecule, we compute a matrix of shape `(n_atoms*n_atoms,
  n_pair_feat)` of pairwise features for each pair of atoms in the molecule.
  Let's construct this conceptually for our example.

  >>> pair_feat = [np.random.rand(3*3, n_pair_feat), np.random.rand(1*1, n_pair_feat)]
  >>> pair_feat = np.concatenate(pair_feat, axis=0)
  >>> pair_feat.shape
  (10, 14)

  `atom_to_pair` tells us the precise pair each pair feature belongs to. In
  our case

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

  Let's now define the actual layer with `T=2` layers of message passing.

  >>> layer = MessagePassing(T=2)

  And invoke it

  >>> out = layer([atom_feat, pair_feat, atom_to_pair])


  References
  ----------
  .. [1] Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters:
  Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).
  """

  def __init__(self,
               T,
               message_fn='enn',
               update_fn='gru',
               n_hidden=100,
               **kwargs):
    """
    Parameters
    ----------
    T: int
      Number of message passing steps
    message_fn: str, optional
      message function in the model
    update_fn: str, optional
      update function in the model
    n_hidden: int, optional
      number of hidden units in the passing phase
    """
    super(MessagePassing, self).__init__(**kwargs)
    self.T = T
    self.message_fn = message_fn
    self.update_fn = update_fn
    self.n_hidden = n_hidden

  def get_config(self):
    config = super(MessagePassing, self).get_config()
    config['T'] = self.T
    config['message_fn'] = self.message_fn
    config['update_fn'] = self.update_fn
    config['n_hidden'] = self.n_hidden
    return config

  def build(self, input_shape):
    n_pair_features = int(input_shape[1][-1])
    if self.message_fn == 'enn':
      # Default message function: edge network, update function: GRU
      # more options to be implemented
      self.message_function = EdgeNetwork(n_pair_features, self.n_hidden)
    if self.update_fn == 'gru':
      self.update_function = GatedRecurrentUnit(self.n_hidden)
    self.built = True

  def call(self, inputs):
    """ Perform T steps of message passing 

    Parameters
    ----------
    inputs: List
      Should contain 3 tensors [atom_features, pair_features, atom_to_pair]

    Returns
    -------
    tf.Tensor
    """
    atom_features, pair_features, atom_to_pair = inputs
    n_atom_features = atom_features.get_shape().as_list()[-1]
    if n_atom_features < self.n_hidden:
      pad_length = self.n_hidden - n_atom_features
      out = tf.pad(atom_features, ((0, 0), (0, pad_length)), mode='CONSTANT')
    elif n_atom_features > self.n_hidden:
      raise ValueError("Too large initial feature vector")
    else:
      out = atom_features
    for i in range(self.T):
      message = self.message_function([pair_features, out, atom_to_pair])
      out = self.update_function([out, message])
    return out


class EdgeNetwork(tf.keras.layers.Layer):
  """ Submodule for Message Passing 

  Implements an edge network as described in [1]_

  References
  ----------
  .. [1] Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters:
  Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).
  """

  def __init__(self,
               n_pair_features=8,
               n_hidden=100,
               init='glorot_uniform',
               **kwargs):
    super(EdgeNetwork, self).__init__(**kwargs)
    self.n_pair_features = n_pair_features
    self.n_hidden = n_hidden
    self.init = init

  def get_config(self):
    config = super(EdgeNetwork, self).get_config()
    config['n_pair_features'] = self.n_pair_features
    config['n_hidden'] = self.n_hidden
    config['init'] = self.init
    return config

  def build(self, input_shape):

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    n_pair_features = self.n_pair_features
    n_hidden = self.n_hidden
    self.W = init([n_pair_features, n_hidden * n_hidden])
    self.b = backend.zeros(shape=(n_hidden * n_hidden,))
    self.built = True

  def call(self, inputs):
    pair_features, atom_features, atom_to_pair = inputs
    A = tf.nn.bias_add(tf.matmul(pair_features, self.W), self.b)
    A = tf.reshape(A, (-1, self.n_hidden, self.n_hidden))
    out = tf.expand_dims(tf.gather(atom_features, atom_to_pair[:, 1]), 2)
    out = tf.squeeze(tf.matmul(A, out), axis=2)
    return tf.math.segment_sum(out, atom_to_pair[:, 0])


class GatedRecurrentUnit(tf.keras.layers.Layer):
  """ Submodule for Message Passing

  Implements a GRU for message passing as described in [1]_

  References
  ----------
  .. [1] Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters:
  Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).

  """

  def __init__(self, n_hidden=100, init='glorot_uniform', **kwargs):
    super(GatedRecurrentUnit, self).__init__(**kwargs)
    self.n_hidden = n_hidden
    self.init = init

  def get_config(self):
    config = super(GatedRecurrentUnit, self).get_config()
    config['n_hidden'] = self.n_hidden
    config['init'] = self.init
    return config

  def build(self, input_shape):
    n_hidden = self.n_hidden

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    self.Wz = init([n_hidden, n_hidden])
    self.Wr = init([n_hidden, n_hidden])
    self.Wh = init([n_hidden, n_hidden])
    self.Uz = init([n_hidden, n_hidden])
    self.Ur = init([n_hidden, n_hidden])
    self.Uh = init([n_hidden, n_hidden])
    self.bz = backend.zeros(shape=(n_hidden,))
    self.br = backend.zeros(shape=(n_hidden,))
    self.bh = backend.zeros(shape=(n_hidden,))
    self.built = True

  def call(self, inputs):
    z = tf.nn.sigmoid(
        tf.matmul(inputs[1], self.Wz) + tf.matmul(inputs[0], self.Uz) + self.bz)
    r = tf.nn.sigmoid(
        tf.matmul(inputs[1], self.Wr) + tf.matmul(inputs[0], self.Ur) + self.br)
    h = (1 - z) * tf.nn.tanh(
        tf.matmul(inputs[1], self.Wh) + tf.matmul(inputs[0] * r, self.Uh) +
        self.bh) + z * inputs[0]
    return h


class SetGather(tf.keras.layers.Layer):
  """set2set gather layer for graph-based model

  Models using this layer must set `pad_batches=True`.
  """

  def __init__(self, M, batch_size, n_hidden=100, init='orthogonal', **kwargs):
    """
    Parameters
    ----------
    M: int
      Number of LSTM steps
    batch_size: int
      Number of samples in a batch(all batches must have same size)
    n_hidden: int, optional
      number of hidden units in the passing phase
    """
    super(SetGather, self).__init__(**kwargs)
    self.M = M
    self.batch_size = batch_size
    self.n_hidden = n_hidden
    self.init = init

  def get_config(self):
    config = super(SetGather, self).get_config()
    config['M'] = self.M
    config['batch_size'] = self.batch_size
    config['n_hidden'] = self.n_hidden
    config['init'] = self.init
    return config

  def build(self, input_shape):

    def init(input_shape):
      return self.add_weight(
          name='kernel',
          shape=(input_shape[0], input_shape[1]),
          initializer=self.init,
          trainable=True)

    self.U = init((2 * self.n_hidden, 4 * self.n_hidden))
    self.b = tf.Variable(
        np.concatenate((np.zeros(self.n_hidden), np.ones(self.n_hidden),
                        np.zeros(self.n_hidden), np.zeros(self.n_hidden))),
        dtype=tf.float32)
    self.built = True

  def call(self, inputs):
    """Perform M steps of set2set gather,

    Detailed descriptions in: https://arxiv.org/abs/1511.06391
    """
    atom_features, atom_split = inputs
    c = tf.zeros((self.batch_size, self.n_hidden))
    h = tf.zeros((self.batch_size, self.n_hidden))

    for i in range(self.M):
      q_expanded = tf.gather(h, atom_split)
      e = tf.reduce_sum(atom_features * q_expanded, 1)
      e_mols = tf.dynamic_partition(e, atom_split, self.batch_size)
      # Add another value(~-Inf) to prevent error in softmax
      e_mols = [
          tf.concat([e_mol, tf.constant([-1000.])], 0) for e_mol in e_mols
      ]
      a = tf.concat([tf.nn.softmax(e_mol)[:-1] for e_mol in e_mols], 0)
      r = tf.math.segment_sum(
          tf.reshape(a, [-1, 1]) * atom_features, atom_split)
      # Model using this layer must set pad_batches=True
      q_star = tf.concat([h, r], axis=1)
      h, c = self.LSTMStep(q_star, c)
    return q_star

  def LSTMStep(self, h, c, x=None):
    # Perform one step of LSTM
    z = tf.nn.bias_add(tf.matmul(h, self.U), self.b)
    i = tf.nn.sigmoid(z[:, :self.n_hidden])
    f = tf.nn.sigmoid(z[:, self.n_hidden:2 * self.n_hidden])
    o = tf.nn.sigmoid(z[:, 2 * self.n_hidden:3 * self.n_hidden])
    z3 = z[:, 3 * self.n_hidden:]
    c_out = f * c + i * tf.nn.tanh(z3)
    h_out = o * tf.nn.tanh(c_out)
    return h_out, c_out

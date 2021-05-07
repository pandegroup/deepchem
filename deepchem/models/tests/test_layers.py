import deepchem as dc
import numpy as np
import tensorflow as tf
import deepchem.models.layers as layers
from tensorflow.python.framework import test_util


def test_cosine_dist():
  """Test invoking cosine_dist."""
  x = tf.ones((5, 4), dtype=tf.dtypes.float32, name=None)
  y_same = tf.ones((5, 4), dtype=tf.dtypes.float32, name=None)
  # x and y are the same tensor (equivalent at every element)
  # the pairwise inner product of the rows in x and y will always be 1
  # the output tensor will be of shape (5,5)
  cos_sim_same = layers.cosine_dist(x, y_same)
  diff = cos_sim_same - tf.ones((5, 5), dtype=tf.dtypes.float32, name=None)
  assert tf.reduce_sum(diff) == 0  # True

  identity_tensor = tf.eye(
      512, dtype=tf.dtypes.float32)  # identity matrix of shape (512,512)
  x1 = identity_tensor[0:256, :]
  x2 = identity_tensor[256:512, :]
  # each row in x1 is orthogonal to each row in x2
  # the pairwise inner product of the rows in x and y will always be 0
  # the output tensor will be of shape (256,256)
  cos_sim_orth = layers.cosine_dist(x1, x2)
  assert tf.reduce_sum(cos_sim_orth) == 0  # True
  assert all([cos_sim_orth.shape[dim] == 256 for dim in range(2)])  # True


def test_highway():
  """Test invoking Highway."""
  width = 5
  batch_size = 10
  input = np.random.rand(batch_size, width).astype(np.float32)
  layer = layers.Highway()
  result = layer(input)
  assert result.shape == (batch_size, width)
  assert len(layer.trainable_variables) == 4

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.Highway()
  result2 = layer2(input)
  assert not np.allclose(result, result2)

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer(input)
  assert np.allclose(result, result3)


def test_combine_mean_std():
  """Test invoking CombineMeanStd."""
  mean = np.random.rand(5, 3).astype(np.float32)
  std = np.random.rand(5, 3).astype(np.float32)
  layer = layers.CombineMeanStd(training_only=True, noise_epsilon=0.01)
  result1 = layer([mean, std], training=False)
  assert np.array_equal(result1, mean)  # No noise in test mode
  result2 = layer([mean, std], training=True)
  assert not np.array_equal(result2, mean)
  assert np.allclose(result2, mean, atol=0.1)


def test_stack():
  """Test invoking Stack."""
  input1 = np.random.rand(5, 4).astype(np.float32)
  input2 = np.random.rand(5, 4).astype(np.float32)
  result = layers.Stack()([input1, input2])
  assert result.shape == (5, 2, 4)
  assert np.array_equal(input1, result[:, 0, :])
  assert np.array_equal(input2, result[:, 1, :])


def test_variable():
  """Test invoking Variable."""
  value = np.random.rand(5, 4).astype(np.float32)
  layer = layers.Variable(value)
  layer.build([])
  result = layer.call([]).numpy()
  assert np.allclose(result, value)
  assert len(layer.trainable_variables) == 1


def test_interatomic_l2_distances():
  """Test invoking InteratomicL2Distances."""
  atoms = 5
  neighbors = 2
  coords = np.random.rand(atoms, 3)
  neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
  layer = layers.InteratomicL2Distances(atoms, neighbors, 3)
  result = layer([coords, neighbor_list])
  assert result.shape == (atoms, neighbors)
  for atom in range(atoms):
    for neighbor in range(neighbors):
      delta = coords[atom] - coords[neighbor_list[atom, neighbor]]
      dist2 = np.dot(delta, delta)
      assert np.allclose(dist2, result[atom, neighbor])


def test_message_passing_layer():
  """Test invoking WeaveLayer."""
  out_channels = 2
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  mp = layers.MessagePassing(T=2)
  atom_feat = []
  pair_feat = []
  atom_to_pair = []
  start = 0
  n_pair_feat = 14
  for im, mol in enumerate(mols):
    n_atoms = mol.get_num_atoms()
    # index of pair features
    C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
    atom_to_pair.append(
        np.transpose(np.array([C1.flatten() + start,
                               C0.flatten() + start])))
    # number of pairs for each atom
    start = start + n_atoms

    # atom features
    atom_feat.append(mol.get_atom_features())
    # pair features
    pair_feat.append(
        np.reshape(mol.get_pair_features(), (n_atoms * n_atoms, n_pair_feat)))
  inputs = [
      np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
      np.concatenate(pair_feat, axis=0),
      np.concatenate(atom_to_pair, axis=0)
  ]
  # Outputs should be what?
  outputs = mp(inputs)
  assert len(outputs) == 4


def test_weave_layer():
  """Test invoking WeaveLayer."""
  out_channels = 2
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  weave = layers.WeaveLayer(
      init=tf.keras.initializers.TruncatedNormal(stddev=0.03))
  atom_feat = []
  pair_feat = []
  atom_to_pair = []
  pair_split = []
  start = 0
  n_pair_feat = 14
  for im, mol in enumerate(mols):
    n_atoms = mol.get_num_atoms()
    # index of pair features
    C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
    atom_to_pair.append(
        np.transpose(np.array([C1.flatten() + start,
                               C0.flatten() + start])))
    # number of pairs for each atom
    pair_split.extend(C1.flatten() + start)
    start = start + n_atoms

    # atom features
    atom_feat.append(mol.get_atom_features())
    # pair features
    pair_feat.append(
        np.reshape(mol.get_pair_features(), (n_atoms * n_atoms, n_pair_feat)))
  inputs = [
      np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
      np.concatenate(pair_feat, axis=0),
      np.array(pair_split),
      np.concatenate(atom_to_pair, axis=0)
  ]
  # Outputs should be [A, P]
  outputs = weave(inputs)
  assert len(outputs) == 2


def test_weave_gather():
  """Test invoking WeaveGather."""
  out_channels = 2
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  atom_feat = []
  atom_split = []
  for im, mol in enumerate(mols):
    n_atoms = mol.get_num_atoms()
    atom_split.extend([im] * n_atoms)

    # atom features
    atom_feat.append(mol.get_atom_features())
  inputs = [
      np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
      np.array(atom_split)
  ]
  # Try without compression
  gather = layers.WeaveGather(batch_size=2, n_input=75, gaussian_expand=True)
  # Outputs should be [mol1_vec, mol2_vec)
  outputs = gather(inputs)
  assert len(outputs) == 2
  assert np.array(outputs[0]).shape == (11 * 75,)
  assert np.array(outputs[1]).shape == (11 * 75,)

  # Try with compression
  gather = layers.WeaveGather(
      batch_size=2,
      n_input=75,
      gaussian_expand=True,
      compress_post_gaussian_expansion=True)
  # Outputs should be [mol1_vec, mol2_vec)
  outputs = gather(inputs)
  assert len(outputs) == 2
  assert np.array(outputs[0]).shape == (75,)
  assert np.array(outputs[1]).shape == (75,)


def test_weave_gather_gaussian_histogram():
  """Test Gaussian Histograms."""
  import tensorflow as tf
  from rdkit import Chem
  out_channels = 2
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.WeaveFeaturizer()
  mols = featurizer.featurize(mols)
  gather = layers.WeaveGather(batch_size=2, n_input=75)
  atom_feat = []
  atom_split = []
  for im, mol in enumerate(mols):
    n_atoms = mol.get_num_atoms()
    atom_split.extend([im] * n_atoms)

    # atom features
    atom_feat.append(mol.get_atom_features())
  inputs = [
      np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
      np.array(atom_split)
  ]
  #per_mol_features = tf.math.segment_sum(inputs[0], inputs[1])
  outputs = gather.gaussian_histogram(inputs[0])
  # Gaussian histograms expands into 11 Gaussian buckets.
  assert np.array(outputs).shape == (
      4,
      11 * 75,
  )
  #assert np.array(outputs[1]).shape == (11 * 75,)


def test_graph_conv():
  """Test invoking GraphConv."""
  out_channels = 2
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.graph_features.ConvMolFeaturizer()
  mols = featurizer.featurize(mols)
  multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
  atom_features = multi_mol.get_atom_features().astype(np.float32)
  degree_slice = multi_mol.deg_slice
  membership = multi_mol.membership
  deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
  args = [atom_features, degree_slice, membership] + deg_adjs
  layer = layers.GraphConv(out_channels)
  result = layer(args)
  assert result.shape == (n_atoms, out_channels)
  num_deg = 2 * layer.max_degree + (1 - layer.min_degree)
  assert len(layer.trainable_variables) == 2 * num_deg


def test_graph_pool():
  """Test invoking GraphPool."""
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.graph_features.ConvMolFeaturizer()
  mols = featurizer.featurize(mols)
  multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
  atom_features = multi_mol.get_atom_features().astype(np.float32)
  degree_slice = multi_mol.deg_slice
  membership = multi_mol.membership
  deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
  args = [atom_features, degree_slice, membership] + deg_adjs
  result = layers.GraphPool()(args)
  assert result.shape[0] == n_atoms
  # TODO What should shape[1] be?  It's not documented.


def test_graph_gather():
  """Test invoking GraphGather."""
  batch_size = 2
  n_features = 75
  n_atoms = 4  # In CCC and C, there are 4 atoms
  raw_smiles = ['CCC', 'C']
  from rdkit import Chem
  mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
  featurizer = dc.feat.graph_features.ConvMolFeaturizer()
  mols = featurizer.featurize(mols)
  multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
  atom_features = multi_mol.get_atom_features().astype(np.float32)
  degree_slice = multi_mol.deg_slice
  membership = multi_mol.membership
  deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
  args = [atom_features, degree_slice, membership] + deg_adjs
  result = layers.GraphGather(batch_size)(args)
  # TODO(rbharath): Why is it 2*n_features instead of n_features?
  assert result.shape == (batch_size, 2 * n_features)


def test_lstm_step():
  """Test invoking LSTMStep."""
  max_depth = 5
  n_test = 5
  n_feat = 10
  y = np.random.rand(n_test, 2 * n_feat).astype(np.float32)
  state_zero = np.random.rand(n_test, n_feat).astype(np.float32)
  state_one = np.random.rand(n_test, n_feat).astype(np.float32)
  layer = layers.LSTMStep(n_feat, 2 * n_feat)
  result = layer([y, state_zero, state_one])
  h_out, h_copy_out, c_out = (result[0], result[1][0], result[1][1])
  assert h_out.shape == (n_test, n_feat)
  assert h_copy_out.shape == (n_test, n_feat)
  assert c_out.shape == (n_test, n_feat)
  assert len(layer.trainable_variables) == 1


def test_attn_lstm_embedding():
  """Test invoking AttnLSTMEmbedding."""
  max_depth = 5
  n_test = 5
  n_support = 11
  n_feat = 10
  test = np.random.rand(n_test, n_feat).astype(np.float32)
  support = np.random.rand(n_support, n_feat).astype(np.float32)
  layer = layers.AttnLSTMEmbedding(n_test, n_support, n_feat, max_depth)
  test_out, support_out = layer([test, support])
  assert test_out.shape == (n_test, n_feat)
  assert support_out.shape == (n_support, n_feat)
  assert len(layer.trainable_variables) == 4


def test_iter_ref_lstm_embedding():
  """Test invoking IterRefLSTMEmbedding."""
  max_depth = 5
  n_test = 5
  n_support = 11
  n_feat = 10
  test = np.random.rand(n_test, n_feat).astype(np.float32)
  support = np.random.rand(n_support, n_feat).astype(np.float32)
  layer = layers.IterRefLSTMEmbedding(n_test, n_support, n_feat, max_depth)
  test_out, support_out = layer([test, support])
  assert test_out.shape == (n_test, n_feat)
  assert support_out.shape == (n_support, n_feat)
  assert len(layer.trainable_variables) == 8


def test_vina_free_energy():
  """Test invoking VinaFreeEnergy."""
  n_atoms = 5
  m_nbrs = 1
  ndim = 3
  nbr_cutoff = 1
  start = 0
  stop = 4
  X = np.random.rand(n_atoms, ndim).astype(np.float32)
  Z = np.random.randint(0, 2, (n_atoms)).astype(np.float32)
  layer = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start, stop)
  result = layer([X, Z])
  assert len(layer.trainable_variables) == 6
  assert result.shape == tuple()

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start, stop)
  result2 = layer2([X, Z])
  assert not np.allclose(result, result2)

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer([X, Z])
  assert np.allclose(result, result3)


def test_weighted_linear_combo():
  """Test invoking WeightedLinearCombo."""
  input1 = np.random.rand(5, 10).astype(np.float32)
  input2 = np.random.rand(5, 10).astype(np.float32)
  layer = layers.WeightedLinearCombo()
  result = layer([input1, input2])
  assert len(layer.trainable_variables) == 2
  expected = input1 * layer.trainable_variables[0] + input2 * layer.trainable_variables[1]
  assert np.allclose(result, expected)


def test_neighbor_list():
  """Test invoking NeighborList."""
  N_atoms = 5
  start = 0
  stop = 12
  nbr_cutoff = 3
  ndim = 3
  M_nbrs = 2
  coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
  coords = tf.cast(tf.stack(coords), tf.float32)
  layer = layers.NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop)
  result = layer(coords)
  assert result.shape == (N_atoms, M_nbrs)


def test_atomic_convolution():
  """Test invoking AtomicConvolution."""
  batch_size = 4
  max_atoms = 5
  max_neighbors = 2
  dimensions = 3
  params = [[5.0, 2.0, 0.5], [10.0, 2.0, 0.5]]
  input1 = np.random.rand(batch_size, max_atoms, dimensions).astype(np.float32)
  input2 = np.random.randint(
      max_atoms, size=(batch_size, max_atoms, max_neighbors))
  input3 = np.random.randint(1, 10, size=(batch_size, max_atoms, max_neighbors))
  layer = layers.AtomicConvolution(radial_params=params)
  result = layer([input1, input2, input3])
  assert result.shape == (batch_size, max_atoms, len(params))
  assert len(layer.trainable_variables) == 3


def test_alpha_share_layer():
  """Test invoking AlphaShareLayer."""
  batch_size = 10
  length = 6
  input1 = np.random.rand(batch_size, length).astype(np.float32)
  input2 = np.random.rand(batch_size, length).astype(np.float32)
  layer = layers.AlphaShareLayer()
  result = layer([input1, input2])
  assert input1.shape == result[0].shape
  assert input2.shape == result[1].shape

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.AlphaShareLayer()
  result2 = layer2([input1, input2])
  assert not np.allclose(result[0], result2[0])
  assert not np.allclose(result[1], result2[1])

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer([input1, input2])
  assert np.allclose(result[0], result3[0])
  assert np.allclose(result[1], result3[1])


def test_sluice_loss():
  """Test invoking SluiceLoss."""
  input1 = np.ones((3, 4)).astype(np.float32)
  input2 = np.ones((2, 2)).astype(np.float32)
  result = layers.SluiceLoss()([input1, input2])
  assert np.allclose(result, 40.0)


def test_beta_share():
  """Test invoking BetaShare."""
  batch_size = 10
  length = 6
  input1 = np.random.rand(batch_size, length).astype(np.float32)
  input2 = np.random.rand(batch_size, length).astype(np.float32)
  layer = layers.BetaShare()
  result = layer([input1, input2])
  assert input1.shape == result.shape
  assert input2.shape == result.shape

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.BetaShare()
  result2 = layer2([input1, input2])
  assert not np.allclose(result, result2)

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer([input1, input2])
  assert np.allclose(result, result3)


def test_ani_feat():
  """Test invoking ANIFeat."""
  batch_size = 10
  max_atoms = 5
  input = np.random.rand(batch_size, max_atoms, 4).astype(np.float32)
  layer = layers.ANIFeat(max_atoms=max_atoms)
  result = layer(input)
  # TODO What should the output shape be?  It's not documented, and there
  # are no other test cases for it.


def test_graph_embed_pool_layer():
  """Test invoking GraphEmbedPoolLayer."""
  V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
  adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
  layer = layers.GraphEmbedPoolLayer(num_vertices=6)
  result = layer([V, adjs])
  assert result[0].shape == (10, 6, 50)
  assert result[1].shape == (10, 6, 5, 6)

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.GraphEmbedPoolLayer(num_vertices=6)
  result2 = layer2([V, adjs])
  assert not np.allclose(result[0], result2[0])
  assert not np.allclose(result[1], result2[1])

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer([V, adjs])
  assert np.allclose(result[0], result3[0])
  assert np.allclose(result[1], result3[1])


def test_graph_cnn():
  """Test invoking GraphCNN."""
  V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
  adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
  layer = layers.GraphCNN(num_filters=6)
  result = layer([V, adjs])
  assert result.shape == (10, 100, 6)

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = layers.GraphCNN(num_filters=6)
  result2 = layer2([V, adjs])
  assert not np.allclose(result, result2)

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer([V, adjs])
  assert np.allclose(result, result3)


def test_DAG_layer():
  """Test invoking DAGLayer."""
  batch_size = 10
  n_graph_feat = 30
  n_atom_feat = 75
  max_atoms = 50
  layer_sizes = [100]
  atom_features = np.random.rand(batch_size, n_atom_feat)
  parents = np.random.randint(
      0, max_atoms, size=(batch_size, max_atoms, max_atoms))
  calculation_orders = np.random.randint(
      0, batch_size, size=(batch_size, max_atoms))
  calculation_masks = np.random.randint(0, 2, size=(batch_size, max_atoms))
  # Recall that the DAG layer expects a MultiConvMol as input,
  # so the "batch" is a pooled set of atoms from all the
  # molecules in the batch, just as it is for the graph conv.
  # This means that n_atoms is the batch-size
  n_atoms = batch_size
  #dropout_switch = False
  layer = layers.DAGLayer(
      n_graph_feat=n_graph_feat,
      n_atom_feat=n_atom_feat,
      max_atoms=max_atoms,
      layer_sizes=layer_sizes)
  outputs = layer([
      atom_features,
      parents,
      calculation_orders,
      calculation_masks,
      n_atoms,
      #dropout_switch
  ])
  ## TODO(rbharath): What is the shape of outputs supposed to be?
  ## I'm getting (7, 30) here. Where does 7 come from??


def test_DAG_gather():
  """Test invoking DAGGather."""
  # TODO(rbharath): We need more documentation about why
  # these numbers work.
  batch_size = 10
  n_graph_feat = 30
  n_atom_feat = 30
  n_outputs = 75
  max_atoms = 50
  layer_sizes = [100]
  layer = layers.DAGGather(
      n_graph_feat=n_graph_feat,
      n_outputs=n_outputs,
      max_atoms=max_atoms,
      layer_sizes=layer_sizes)
  atom_features = np.random.rand(batch_size, n_atom_feat)
  membership = np.sort(np.random.randint(0, batch_size, size=(batch_size)))
  outputs = layer([atom_features, membership])

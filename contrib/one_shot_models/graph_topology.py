"""Manages Placeholders for Graph convolution networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
# from deepchem.nn.copy import Input
from deepchem.feat.mol_graphs import ConvMol

def merge_two_dicts(x, y):
  z = x.copy()
  z.update(y)
  return z


def merge_dicts(l):
  """Convenience function to merge list of dictionaries."""
  merged = {}
  for dict in l:
    merged = merge_two_dicts(merged, dict)
  return merged


class GraphTopology(object):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self, n_feat, name='topology', max_deg=10, min_deg=0):
    """
    Note that batch size is not specified in a GraphTopology object. A batch
    of molecules must be combined into a disconnected graph and fed to topology
    directly to handle batches.

    Parameters
    ----------
    n_feat: int
      Number of features per atom.
    name: str, optional
      Name of this manager.
    max_deg: int, optional
      Maximum #bonds for atoms in molecules.
    min_deg: int, optional
      Minimum #bonds for atoms in molecules.
    """
    warnings.warn("GraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    #self.n_atoms = n_atoms
    self.n_feat = n_feat

    self.name = name
    self.max_deg = max_deg
    self.min_deg = min_deg

    self.atom_features_placeholder = tf.keras.Input(
      name=self.name + '_atom_features', 
      shape=(None, self.n_feat), 
      dtype=tf.dtypes.float32
    )

    self.deg_adj_lists_placeholders = [
      tf.keras.Input(
        dtype='int32',
        shape=(None, deg),
        name=self.name + '_deg_adj' + str(deg)
      )
      for deg in range (1, self.max_deg + 1)
    ]

    self.deg_slice_placeholder = tf.keras.Input(
        dtype='int32',
        shape=(self.max_deg - self.min_deg + 1, 2),
        name=self.name + '_deg_slice'
    )
    
    self.membership_placeholder = tf.keras.Input(
        dtype='int32', 
        shape=(None,), 
        name=self.name + '_membership'
    )
    
    self.topology = [self.deg_slice_placeholder, self.membership_placeholder]
    self.topology += self.deg_adj_lists_placeholders

    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_input_placeholders(self):
    """All placeholders.

    Contains atom_features placeholder and topology placeholders.
    """
    return self.inputs

  def get_topology_placeholders(self):
    """Returns topology placeholders

    Consists of deg_slice_placeholder, membership_placeholder, and the
    deg_adj_list_placeholders.
    """
    return self.topology

  def get_atom_features_placeholder(self):
    return self.atom_features_placeholder

  def get_deg_adjacency_lists_placeholders(self):
    return self.deg_adj_lists_placeholders

  def get_deg_slice_placeholder(self):
    return self.deg_slice_placeholder

  def get_membership_placeholder(self):
    return self.membership_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of mol_graphs into tensorflow feed_dict.

    Assigns the graph information in array of ConvMol objects to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of ConvMol objects

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Merge mol conv objects
    batch = ConvMol.agglomerate_mols(batch)
    atoms = batch.get_atom_features()
    deg_adj_lists = [
        batch.deg_adj_lists[deg] for deg in range(1, self.max_deg + 1)
    ]

    # Generate dicts
    deg_adj_dict = dict(
        list(zip(self.deg_adj_lists_placeholders, deg_adj_lists)))
    atoms_dict = {
        self.atom_features_placeholder: atoms,
        self.deg_slice_placeholder: batch.deg_slice,
        self.membership_placeholder: batch.membership
    }
    return merge_dicts([atoms_dict, deg_adj_dict])


class DTNNGraphTopology(GraphTopology):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self,
               n_distance=100,
               distance_min=-1.,
               distance_max=18.,
               name='DTNN_topology'):
    """
    Parameters
    ----------
    n_distance: int, optional
      granularity of distance matrix
      step size will be (distance_max-distance_min)/n_distance
    distance_min: float, optional
      minimum distance of atom pairs, default = -1 Angstorm
    distance_max: float, optional
      maximum distance of atom pairs, default = 18 Angstorm
    """
    warnings.warn("DTNNGraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    self.name = name
    self.n_distance = n_distance
    self.distance_min = distance_min
    self.distance_max = distance_max
    self.step_size = (distance_max - distance_min) / n_distance
    self.steps = np.array(
        [distance_min + i * self.step_size for i in range(n_distance)])
    self.steps = np.expand_dims(self.steps, 0)

    self.atom_number_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_atom_number')
    self.distance_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_distance),
        name=self.name + '_distance')
    self.atom_membership_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_atom_membership')
    self.distance_membership_i_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_distance_membership_i')
    self.distance_membership_j_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_distance_membership_j')

    # Define the list of tensors to be used as topology
    self.topology = [
        self.distance_placeholder,
        self.atom_membership_placeholder,
        self.distance_membership_i_placeholder,
        self.distance_membership_j_placeholder,
    ]
    self.inputs = [self.atom_number_placeholder]
    self.inputs += self.topology

  def get_atom_number_placeholder(self):
    return self.atom_number_placeholder

  def get_distance_placeholder(self):
    return self.distance_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of Coulomb Matrix into tensorflow feed_dict.

    Assigns the atom number and distance info to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of Coulomb Matrix

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Extract atom numbers
    num_atoms = list(map(sum, batch.astype(bool)[:, :, 0]))
    atom_number = [
        np.round(
            np.power(2 * np.diag(batch[i, :num_atoms[i], :num_atoms[i]]), 1 /
                     2.4)).astype(int) for i in range(len(num_atoms))
    ]

    distance = []
    atom_membership = []
    distance_membership_i = []
    distance_membership_j = []
    start = 0
    for im, molecule in enumerate(atom_number):
      distance_matrix = np.outer(
          molecule, molecule) / batch[im, :num_atoms[im], :num_atoms[im]]
      np.fill_diagonal(distance_matrix, -100)
      distance.append(np.expand_dims(distance_matrix.flatten(), 1))
      atom_membership.append([im] * num_atoms[im])
      membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
      membership_i = membership.flatten(order='F')
      membership_j = membership.flatten()
      distance_membership_i.append(membership_i + start)
      distance_membership_j.append(membership_j + start)
      start = start + num_atoms[im]
    atom_number = np.concatenate(atom_number)
    distance = np.concatenate(distance, 0)
    distance = np.exp(-np.square(distance - self.steps) /
                      (2 * self.step_size**2))
    distance_membership_i = np.concatenate(distance_membership_i)
    distance_membership_j = np.concatenate(distance_membership_j)
    atom_membership = np.concatenate(atom_membership)
    # Generate dicts
    dict_DTNN = {
        self.atom_number_placeholder: atom_number,
        self.distance_placeholder: distance,
        self.atom_membership_placeholder: atom_membership,
        self.distance_membership_i_placeholder: distance_membership_i,
        self.distance_membership_j_placeholder: distance_membership_j
    }
    return dict_DTNN


class DAGGraphTopology(GraphTopology):
  """GraphTopology for DAG models
  """

  def __init__(self, n_atom_feat=75, max_atoms=50, name='topology'):
    """
    Parameters
    ----------
    n_atom_feat: int, optional
      Number of features per atom.
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    """
    warnings.warn("DAGGraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.n_atom_feat = n_atom_feat
    self.max_atoms = max_atoms
    self.name = name
    self.atom_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_atom_feat),
        name=self.name + '_atom_features')

    self.parents_placeholder = tf.placeholder(
        dtype='int32',
        shape=(None, self.max_atoms, self.max_atoms),
        # molecule * atom(graph) => step => features
        name=self.name + '_parents')

    self.calculation_orders_placeholder = tf.placeholder(
        dtype='int32',
        shape=(None, self.max_atoms),
        # molecule * atom(graph) => step
        name=self.name + '_orders')

    self.calculation_masks_placeholder = tf.placeholder(
        dtype='bool',
        shape=(None, self.max_atoms),
        # molecule * atom(graph) => step
        name=self.name + '_masks')

    self.membership_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_membership')

    self.n_atoms_placeholder = tf.placeholder(
        dtype='int32', shape=(), name=self.name + '_n_atoms')

    # Define the list of tensors to be used as topology
    self.topology = [
        self.parents_placeholder, self.calculation_orders_placeholder,
        self.calculation_masks_placeholder, self.membership_placeholder,
        self.n_atoms_placeholder
    ]

    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_parents_placeholder(self):
    return self.parents_placeholder

  def get_calculation_orders_placeholder(self):
    return self.calculation_orders_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of mol_graphs into tensorflow feed_dict.

    Assigns the graph information in array of ConvMol objects to the
    placeholders tensors for DAG models

    params
    ------
    batch : np.ndarray
      Array of ConvMol objects

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """

    atoms_per_mol = [mol.get_num_atoms() for mol in batch]
    n_atoms = sum(atoms_per_mol)
    start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

    atoms_all = []
    # calculation orders for a batch of molecules
    parents_all = []
    calculation_orders = []
    calculation_masks = []
    membership = []
    for idm, mol in enumerate(batch):
      # padding atom features vector of each molecule with 0
      atoms_all.append(mol.get_atom_features())
      parents = mol.parents
      parents_all.extend(parents)
      calculation_index = np.array(parents)[:, :, 0]
      mask = np.array(calculation_index - self.max_atoms, dtype=bool)
      calculation_orders.append(calculation_index + start_index[idm])
      calculation_masks.append(mask)
      membership.extend([idm] * atoms_per_mol[idm])

    atoms_all = np.concatenate(atoms_all, axis=0)
    parents_all = np.stack(parents_all, axis=0)
    calculation_orders = np.concatenate(calculation_orders, axis=0)
    calculation_masks = np.concatenate(calculation_masks, axis=0)
    membership = np.array(membership)

    atoms_dict = {
        self.atom_features_placeholder: atoms_all,
        self.parents_placeholder: parents_all,
        self.calculation_orders_placeholder: calculation_orders,
        self.calculation_masks_placeholder: calculation_masks,
        self.membership_placeholder: membership,
        self.n_atoms_placeholder: n_atoms
    }

    return atoms_dict


class WeaveGraphTopology(GraphTopology):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self,
               max_atoms=50,
               n_atom_feat=75,
               n_pair_feat=14,
               name='Weave_topology'):
    """
    Parameters
    ----------
    max_atoms: int, optional
      maximum number of atoms in a molecule
    n_atom_feat: int, optional
      number of basic features of each atom
    n_pair_feat: int, optional
      number of basic features of each pair
    """
    warnings.warn("WeaveGraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    #self.n_atoms = n_atoms
    self.name = name
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat

    self.atom_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_atoms, self.n_atom_feat),
        name=self.name + '_atom_features')
    self.atom_mask_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_atoms),
        name=self.name + '_atom_mask')
    self.pair_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_atoms, self.max_atoms, self.n_pair_feat),
        name=self.name + '_pair_features')
    self.pair_mask_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_atoms, self.max_atoms),
        name=self.name + '_pair_mask')
    self.membership_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_membership')
    # Define the list of tensors to be used as topology
    self.topology = [self.atom_mask_placeholder, self.pair_mask_placeholder]
    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_pair_features_placeholder(self):
    return self.pair_features_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of WeaveMol into tensorflow feed_dict.

    Assigns the atom features and pair features to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of WeaveMol

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Extract atom numbers
    atom_feat = []
    pair_feat = []
    atom_mask = []
    pair_mask = []
    membership = []
    max_atoms = self.max_atoms
    for im, mol in enumerate(batch):
      n_atoms = mol.get_num_atoms()
      atom_feat.append(
          np.pad(mol.get_atom_features(), ((0, max_atoms - n_atoms), (0, 0)),
                 'constant'))
      atom_mask.append(
          np.array([1] * n_atoms + [0] * (max_atoms - n_atoms), dtype=float))
      pair_feat.append(
          np.pad(mol.get_pair_features(), ((0, max_atoms - n_atoms), (
              0, max_atoms - n_atoms), (0, 0)), 'constant'))
      pair_mask.append(np.array([[1]*n_atoms + [0]*(max_atoms-n_atoms)]*n_atoms + \
                       [[0]*max_atoms]*(max_atoms-n_atoms), dtype=float))
      membership.extend([im] * n_atoms)
    atom_feat = np.stack(atom_feat)
    pair_feat = np.stack(pair_feat)
    atom_mask = np.stack(atom_mask)
    pair_mask = np.stack(pair_mask)
    membership = np.array(membership)
    # Generate dicts
    dict_DTNN = {
        self.atom_features_placeholder: atom_feat,
        self.pair_features_placeholder: pair_feat,
        self.atom_mask_placeholder: atom_mask,
        self.pair_mask_placeholder: pair_mask,
        self.membership_placeholder: membership
    }
    return dict_DTNN


class AlternateWeaveGraphTopology(GraphTopology):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self,
               batch_size,
               max_atoms=50,
               n_atom_feat=75,
               n_pair_feat=14,
               name='Weave_topology'):
    """
    Parameters
    ----------
    batch_size: int
      number of molecules in a batch
    max_atoms: int, optional
      maximum number of atoms in a molecule
    n_atom_feat: int, optional
      number of basic features of each atom
    n_pair_feat: int, optional
      number of basic features of each pair
    """
    warnings.warn("AlternateWeaveGraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    #self.n_atoms = n_atoms
    self.name = name
    self.batch_size = batch_size
    self.max_atoms = max_atoms * batch_size
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat

    self.atom_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_atom_feat),
        name=self.name + '_atom_features')
    self.pair_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_pair_feat),
        name=self.name + '_pair_features')
    self.pair_split_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_pair_split')
    self.atom_split_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_atom_split')
    self.atom_to_pair_placeholder = tf.placeholder(
        dtype='int32', shape=(None, 2), name=self.name + '_atom_to_pair')

    # Define the list of tensors to be used as topology
    self.topology = [
        self.pair_split_placeholder, self.atom_split_placeholder,
        self.atom_to_pair_placeholder
    ]
    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_pair_features_placeholder(self):
    return self.pair_features_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of WeaveMol into tensorflow feed_dict.

    Assigns the atom features and pair features to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of WeaveMol

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Extract atom numbers
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    max_atoms = self.max_atoms
    start = 0
    for im, mol in enumerate(batch):
      n_atoms = mol.get_num_atoms()
      # number of atoms in each molecule
      atom_split.extend([im] * n_atoms)
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
          np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                               self.n_pair_feat)))

    atom_feat = np.concatenate(atom_feat, axis=0)
    pair_feat = np.concatenate(pair_feat, axis=0)
    atom_to_pair = np.concatenate(atom_to_pair, axis=0)
    atom_split = np.array(atom_split)
    # Generate dicts
    dict_DTNN = {
        self.atom_features_placeholder: atom_feat,
        self.pair_features_placeholder: pair_feat,
        self.pair_split_placeholder: pair_split,
        self.atom_split_placeholder: atom_split,
        self.atom_to_pair_placeholder: atom_to_pair
    }
    return dict_DTNN

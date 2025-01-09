from collections import deque

import sys
import tensorflow as tf
import pickle

import os
import fnmatch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from rdkit import Chem
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.graph_features import atom_features
from scipy.sparse import csr_matrix


def get_atom_type(atom):
  elem = atom.GetAtomicNum()
  hyb = str(atom.GetHybridization).lower()
  if elem == 1:
    return (0)
  if elem == 4:
    return (1)
  if elem == 5:
    return (2)
  if elem == 6:
    if "sp2" in hyb:
      return (3)
    elif "sp3" in hyb:
      return (4)
    else:
      return (5)
  if elem == 7:
    if "sp2" in hyb:
      return (6)
    elif "sp3" in hyb:
      return (7)
    else:
      return (8)
  if elem == 8:
    if "sp2" in hyb:
      return (9)
    elif "sp3" in hyb:
      return (10)
    else:
      return (11)
  if elem == 9:
    return (12)
  if elem == 15:
    if "sp2" in hyb:
      return (13)
    elif "sp3" in hyb:
      return (14)
    else:
      return (15)
  if elem == 16:
    if "sp2" in hyb:
      return (16)
    elif "sp3" in hyb:
      return (17)
    else:
      return (18)
  if elem == 17:
    return (19)
  if elem == 35:
    return (20)
  if elem == 53:
    return (21)
  return (22)


def get_atom_adj_matrices(mol,
                          n_atom_types,
                          max_n_atoms=200,
                          max_valence=4,
                          graph_conv_features=True,
                          nxn=True):
  if not graph_conv_features:
    bond_matrix = np.zeros((max_n_atoms, 4 * max_valence)).astype(np.uint8)

  if nxn:
    adj_matrix = np.zeros((max_n_atoms, max_n_atoms)).astype(np.uint8)
  else:
    adj_matrix = np.zeros((max_n_atoms, max_valence)).astype(np.uint8)
    adj_matrix += (adj_matrix.shape[0] - 1)

  if not graph_conv_features:
    atom_matrix = np.zeros((max_n_atoms, n_atom_types + 3)).astype(np.uint8)
    atom_matrix[:, atom_matrix.shape[1] - 1] = 1

  atom_arrays = []
  for a_idx in range(0, mol.GetNumAtoms()):
    atom = mol.GetAtomWithIdx(a_idx)
    if graph_conv_features:
      atom_arrays.append(atom_features(atom))
    else:

      atom_type = get_atom_type(atom)
      atom_matrix[a_idx][-1] = 0
      atom_matrix[a_idx][atom_type] = 1

    for n_idx, neighbor in enumerate(atom.GetNeighbors()):
      if nxn:
        adj_matrix[a_idx][neighbor.GetIdx()] = 1
        adj_matrix[a_idx][a_idx] = 1
      else:
        adj_matrix[a_idx][n_idx] = neighbor.GetIdx()

      if not graph_conv_features:
        bond = mol.GetBondBetweenAtoms(a_idx, neighbor.GetIdx())
        bond_type = str(bond.GetBondType()).lower()
        if "single" in bond_type:
          bond_order = 0
        elif "double" in bond_type:
          bond_order = 1
        elif "triple" in bond_type:
          bond_order = 2
        elif "aromatic" in bond_type:
          bond_order = 3
        bond_matrix[a_idx][(4 * n_idx) + bond_order] = 1

  if graph_conv_features:
    n_feat = len(atom_arrays[0])
    atom_matrix = np.zeros((max_n_atoms, n_feat)).astype(np.uint8)
    for idx, atom_array in enumerate(atom_arrays):
      atom_matrix[idx, :] = atom_array
  else:
    atom_matrix = np.concatenate(
        [atom_matrix, bond_matrix], axis=1).astype(np.uint8)

  return (adj_matrix.astype(np.uint8), atom_matrix.astype(np.uint8))


def featurize_mol(mol, n_atom_types, max_n_atoms, max_valence,
                  num_atoms_feature):
  adj_matrix, atom_matrix = get_atom_adj_matrices(mol, n_atom_types,
                                                  max_n_atoms, max_valence)
  if num_atoms_feature:
    return ((adj_matrix, atom_matrix, mol.GetNumAtoms()))
  return ((adj_matrix, atom_matrix))


class AdjacencyFingerprint(Featurizer):

  def __init__(self,
               n_atom_types=23,
               max_n_atoms=200,
               add_hydrogens=False,
               max_valence=4,
               num_atoms_feature=False):
    self.n_atom_types = n_atom_types
    self.max_n_atoms = max_n_atoms
    self.add_hydrogens = add_hydrogens
    self.max_valence = max_valence
    self.num_atoms_feature = num_atoms_feature

  def featurize(self, rdkit_mols: list) -> np.ndarray:
        featurized_mols = np.empty((len(rdkit_mols)), dtype=object)

        for idx, mol in enumerate(rdkit_mols):
            try:
                if self.add_hydrogens:
                    mol = Chem.AddHs(mol)
                featurized_mols[idx] = featurize_mol(mol, self.n_atom_types,
                                                     self.max_n_atoms, self.max_valence,
                                                     self.num_atoms_feature)
            except Exception as e:
                print(f"Error featurizing molecule {idx}: {e}")
                featurized_mols[idx] = None
        return featurized_mols

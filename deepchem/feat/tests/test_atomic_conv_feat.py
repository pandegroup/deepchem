"""
Test atomic conv featurizer.
"""

import os
import logging
import numpy as np
import unittest
from deepchem.feat import AtomicConvFeaturizer

logger = logging.getLogger(__name__)


def test_atomic_conv_featurization():
  """Unit test for AtomicConvFeaturizer."""
  dir_path = os.path.dirname(os.path.realpath(__file__))
  ligand_file = os.path.join(dir_path, "data/3zso_ligand_hyd.pdb")
  protein_file = os.path.join(dir_path, "data/3zso_protein.pdb")
  # Pulled from PDB files. For larger datasets with more PDBs, would use
  # max num atoms instead of exact.
  frag1_num_atoms = 44  # for ligand atoms
  frag2_num_atoms = 2336  # for protein atoms
  complex_num_atoms = 2380  # in total
  max_num_neighbors = 4
  # Cutoff in angstroms
  neighbor_cutoff = 4
  complex_featurizer = AtomicConvFeaturizer(frag1_num_atoms, frag2_num_atoms,
                                            complex_num_atoms,
                                            max_num_neighbors, neighbor_cutoff)
  (frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords,
   frag2_neighbor_list, frag2_z, complex_coords, complex_neighbor_list,
   complex_z) = complex_featurizer._featurize((ligand_file, protein_file))

  assert frag1_coords.shape == (frag1_num_atoms, 3)
  assert (sorted(list(frag1_neighbor_list.keys())) == list(
      range(frag1_num_atoms)))
  assert frag1_z.shape == (frag1_num_atoms,)

  assert frag2_coords.shape == (frag2_num_atoms, 3)
  ##################
  print("len(sorted(list(frag2_neighbor_list.keys())))")
  print(len(sorted(list(frag2_neighbor_list.keys()))))
  print("len(list(range(frag2_num_atoms)))")
  print(len(list(range(frag2_num_atoms))))
  ##################
  assert (sorted(list(frag2_neighbor_list.keys())) == list(
      range(frag2_num_atoms)))
  assert frag2_z.shape == (frag2_num_atoms,)

  assert complex_coords.shape == (complex_num_atoms, 3)
  assert (sorted(list(complex_neighbor_list.keys())) == list(
      range(complex_num_atoms)))
  assert (complex_z.shape == (complex_num_atoms,))

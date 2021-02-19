"""
Test atomic coordinates and neighbor lists.
"""
import os
import logging
import numpy as np
import unittest
from deepchem.utils import conformers
from deepchem.feat import AtomicCoordinates
from deepchem.feat import NeighborListAtomicCoordinates
from deepchem.feat import NeighborListComplexAtomicCoordinates

logger = logging.getLogger(__name__)


class TestAtomicCoordinates(unittest.TestCase):
  """
  Test AtomicCoordinates.
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    engine = conformers.ConformerGenerator(max_conformers=1)
    self.mol = engine.generate_conformers(mol)
    self.get_angstrom_coords = AtomicCoordinates()._featurize
    assert self.mol.GetNumConformers() > 0

  def test_atomic_coordinates(self):
    """
    Simple test that atomic coordinates returns ndarray of right shape.
    """
    N = self.mol.GetNumAtoms()
    atomic_coords_featurizer = AtomicCoordinates()
    coords = atomic_coords_featurizer._featurize(self.mol)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (N, 3)

  def test_neighbor_list_shape(self):
    """
    Simple test that Neighbor Lists have right shape.
    """
    nblist_featurizer = NeighborListAtomicCoordinates()
    N = self.mol.GetNumAtoms()
    coords = self.get_angstrom_coords(self.mol)

    nblist_featurizer = NeighborListAtomicCoordinates()
    nblist = nblist_featurizer._featurize(self.mol)[1]
    assert isinstance(nblist, dict)
    assert len(nblist.keys()) == N
    for (atom, neighbors) in nblist.items():
      assert isinstance(atom, int)
      assert isinstance(neighbors, list)
      assert len(neighbors) <= N

    # Do a manual distance computation and make
    for i in range(N):
      for j in range(N):
        dist = np.linalg.norm(coords[i] - coords[j])
        logger.info("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < nblist_featurizer.neighbor_cutoff and i != j:
          assert j in nblist[i]
        else:
          assert j not in nblist[i]

  def test_neighbor_list_extremes(self):
    """
    Test Neighbor Lists with large/small boxes.
    """
    N = self.mol.GetNumAtoms()

    # Test with cutoff 0 angstroms. There should be no neighbors in this case.
    nblist_featurizer = NeighborListAtomicCoordinates(neighbor_cutoff=.1)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == 0

    # Test with cutoff 100 angstroms. Everything should be neighbors now.
    nblist_featurizer = NeighborListAtomicCoordinates(neighbor_cutoff=100)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == N - 1

  def test_neighbor_list_max_num_neighbors(self):
    """
    Test that neighbor lists return only max_num_neighbors.
    """
    N = self.mol.GetNumAtoms()

    max_num_neighbors = 1
    nblist_featurizer = NeighborListAtomicCoordinates(max_num_neighbors)
    nblist = nblist_featurizer._featurize(self.mol)[1]

    for atom in range(N):
      assert len(nblist[atom]) <= max_num_neighbors

    # Do a manual distance computation and ensure that selected neighbor is
    # closest since we set max_num_neighbors = 1
    coords = self.get_angstrom_coords(self.mol)
    for i in range(N):
      closest_dist = np.inf
      closest_nbr = None
      for j in range(N):
        if i == j:
          continue
        dist = np.linalg.norm(coords[i] - coords[j])
        logger.info("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < closest_dist:
          closest_dist = dist
          closest_nbr = j
      logger.info("Closest neighbor to %d is %d" % (i, closest_nbr))
      logger.info("Distance: %f" % closest_dist)
      if closest_dist < nblist_featurizer.neighbor_cutoff:
        assert nblist[i] == [closest_nbr]
      else:
        assert nblist[i] == []

  def test_neighbor_list_periodic(self):
    """Test building a neighbor list with periodic boundary conditions."""
    cutoff = 4.0
    box_size = np.array([10.0, 8.0, 9.0])
    N = self.mol.GetNumAtoms()
    coords = self.get_angstrom_coords(self.mol)
    featurizer = NeighborListAtomicCoordinates(
        neighbor_cutoff=cutoff, periodic_box_size=box_size)
    neighborlist = featurizer._featurize(self.mol)[1]
    expected_neighbors = [set() for i in range(N)]
    for i in range(N):
      for j in range(i):
        delta = coords[i] - coords[j]
        delta -= np.round(delta / box_size) * box_size
        if np.linalg.norm(delta) < cutoff:
          expected_neighbors[i].add(j)
          expected_neighbors[j].add(i)
    for i in range(N):
      assert (set(neighborlist[i]) == expected_neighbors[i])

  def test_complex_featurization_simple(self):
    """Test Neighbor List computation on protein-ligand complex."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(dir_path, "data/3zso_ligand_hyd.pdb")
    protein_file = os.path.join(dir_path, "data/3zso_protein.pdb")
    max_num_neighbors = 4
    complex_featurizer = NeighborListComplexAtomicCoordinates(max_num_neighbors)

    system_coords, system_neighbor_list = complex_featurizer._featurize(
        (ligand_file, protein_file))

    N = system_coords.shape[0]
    assert len(system_neighbor_list.keys()) == N
    for atom in range(N):
      assert len(system_neighbor_list[atom]) <= max_num_neighbors

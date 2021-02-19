"""
Atomic coordinate featurizer.
"""
import logging

import numpy as np

from deepchem.feat.base_classes import Featurizer, ComplexFeaturizer
from deepchem.feat.molecule_featurizers import AtomicCoordinates
from deepchem.utils.data_utils import pad_array
from deepchem.utils.rdkit_utils import MoleculeLoadException, get_xyz_from_mol, \
  load_molecule, merge_molecules_xyz, merge_molecules

from typing import Tuple


def compute_neighbor_list(coords, neighbor_cutoff, max_num_neighbors,
                          periodic_box_size):
  """Computes a neighbor list from atom coordinates."""
  N = coords.shape[0]
  import mdtraj
  traj = mdtraj.Trajectory(coords.reshape((1, N, 3)), None)
  box_size = None
  if periodic_box_size is not None:
    box_size = np.array(periodic_box_size)
    traj.unitcell_vectors = np.array(
        [[[box_size[0], 0, 0], [0, box_size[1], 0], [0, 0, box_size[2]]]],
        dtype=np.float32)
  neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
  neighbor_list = {}
  for i in range(N):
    if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
      delta = coords[i] - coords.take(neighbors[i], axis=0)
      if box_size is not None:
        delta -= np.round(delta / box_size) * box_size
      dist = np.linalg.norm(delta, axis=1)
      sorted_neighbors = list(zip(dist, neighbors[i]))
      sorted_neighbors.sort()
      neighbor_list[i] = [
          sorted_neighbors[j][1] for j in range(max_num_neighbors)
      ]
    else:
      neighbor_list[i] = list(neighbors[i])
  return neighbor_list


class NeighborListAtomicCoordinates(Featurizer):
  """
  Adjacency List of neighbors in 3-space

  Neighbors determined by user-defined distance cutoff [in Angstrom].

  https://en.wikipedia.org/wiki/Cell_list
  Ref: http://www.cs.cornell.edu/ron/references/1989/Calculations%20of%20a%20List%20of%20Neighbors%20in%20Molecular%20Dynamics%20Si.pdf

  Parameters
  ----------
  neighbor_cutoff: float
    Threshold distance [Angstroms] for counting neighbors.
  periodic_box_size: 3 element array
    Dimensions of the periodic box in Angstroms, or None to not use periodic boundary conditions
  """

  def __init__(self,
               max_num_neighbors=None,
               neighbor_cutoff=4,
               periodic_box_size=None):
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.periodic_box_size = periodic_box_size
    # Type of data created by this featurizer
    self.dtype = object
    self.bohr_coords_featurizer = AtomicCoordinates(use_bohr=True)
    self.coords_featurizer = AtomicCoordinates(use_bohr=False)

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
      mol: rdkit Mol
        To be featurized.
    """
    # TODO(rbharath): Should this return a list?
    bohr_coords = self.bohr_coords_featurizer._featurize(mol)
    coords = self.coords_featurizer._featurize(mol)
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                          self.max_num_neighbors,
                                          self.periodic_box_size)
    return (bohr_coords, neighbor_list)


class NeighborListComplexAtomicCoordinates(ComplexFeaturizer):
  """
  Adjacency list of neighbors for protein-ligand complexes in 3-space.

  Neighbors determined by user-defined distance cutoff.
  """

  def __init__(self, max_num_neighbors=None, neighbor_cutoff=4):
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    # Type of data created by this featurizer
    self.dtype = object

  def _featurize(self, complex: Tuple[str, str]):
    """
    Compute neighbor list for complex.

    Parameters
    ----------
    complex: Tuple[str, str]
      Filenames for molecule and protein.
    """
    mol_pdb_file, protein_pdb_file = complex
    mol_coords, ob_mol = load_molecule(mol_pdb_file)
    protein_coords, protein_mol = load_molecule(protein_pdb_file)
    system_coords = merge_molecules_xyz([mol_coords, protein_coords])

    system_neighbor_list = compute_neighbor_list(
        system_coords, self.neighbor_cutoff, self.max_num_neighbors, None)

    return (system_coords, system_neighbor_list)


class AtomicConvFeaturizer(ComplexFeaturizer):
  """This class computes the featurization that corresponds to AtomicConvModel.

  This class computes featurizations needed for AtomicConvModel.
  Given a two molecular structures, it computes a number of useful
  geometric features. In particular, for each molecule and the global
  complex, it computes a coordinates matrix of size (N_atoms, 3)
  where N_atoms is the number of atoms. It also computes a
  neighbor-list, a dictionary with N_atoms elements where
  neighbor-list[i] is a list of the atoms the i-th atom has as
  neighbors. In addition, it computes a z-matrix for the molecule
  which is an array of shape (N_atoms,) that contains the atomic
  number of that atom.

  Since the featurization computes these three quantities for each of
  the two molecules and the complex, a total of 9 quantities are
  returned for each complex. Note that for efficiency, fragments of
  the molecules can be provided rather than the full molecules
  themselves.
  """

  def __init__(self,
               frag1_num_atoms,
               frag2_num_atoms,
               complex_num_atoms,
               max_num_neighbors,
               neighbor_cutoff,
               strip_hydrogens=True):
    """

    Parameters
    ----------
    frag1_num_atoms: int
      Maximum number of atoms in fragment 1.
    frag2_num_atoms: int
      Maximum number of atoms in fragment 2.
    complex_num_atoms: int
      Maximum number of atoms in complex of frag1/frag2 together.
    max_num_neighbors: int
      Maximum number of atoms considered as neighbors.
    neighbor_cutoff: float
      Maximum distance (angstroms) for two atoms to be considered as
      neighbors. If more than `max_num_neighbors` atoms fall within
      this cutoff, the closest `max_num_neighbors` will be used.
    strip_hydrogens: bool (default True)
      Remove hydrogens before computing featurization.
    """
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.complex_num_atoms = complex_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.strip_hydrogens = strip_hydrogens
    self.neighborlist_featurizer = NeighborListComplexAtomicCoordinates(
        self.max_num_neighbors, self.neighbor_cutoff)

  def _featurize(self, complex):
    mol_pdb_file, protein_pdb_file = complex
    try:
      frag1_coords, frag1_mol = load_molecule(
          mol_pdb_file, is_protein=False, sanitize=True, add_hydrogens=False)
      frag2_coords, frag2_mol = load_molecule(
          protein_pdb_file, is_protein=True, sanitize=True, add_hydrogens=False)
    except MoleculeLoadException:
      # Currently handles loading failures by returning None
      # TODO: Is there a better handling procedure?
      logging.warning("Some molecules cannot be loaded by Rdkit. Skipping")
      return None
    system_mol = merge_molecules([frag1_mol, frag2_mol])
    system_coords = get_xyz_from_mol(system_mol)

    frag1_coords, frag1_mol = self._strip_hydrogens(frag1_coords, frag1_mol)
    frag2_coords, frag2_mol = self._strip_hydrogens(frag2_coords, frag2_mol)
    system_coords, system_mol = self._strip_hydrogens(system_coords, system_mol)

    try:
      frag1_coords, frag1_neighbor_list, frag1_z = self.featurize_mol(
          frag1_coords, frag1_mol, self.frag1_num_atoms)

      frag2_coords, frag2_neighbor_list, frag2_z = self.featurize_mol(
          frag2_coords, frag2_mol, self.frag2_num_atoms)

      system_coords, system_neighbor_list, system_z = self.featurize_mol(
          system_coords, system_mol, self.complex_num_atoms)
    except ValueError:
      logging.warning(
          "max_atoms was set too low. Some complexes too large and skipped")
      return None

    return frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords, frag2_neighbor_list, frag2_z, \
           system_coords, system_neighbor_list, system_z

  def get_Z_matrix(self, mol, max_atoms):
    if len(mol.GetAtoms()) > max_atoms:
      raise ValueError("A molecule is larger than permitted by max_atoms. "
                       "Increase max_atoms and try again.")
    return pad_array(
        np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_atoms)

  def featurize_mol(self, coords, mol, max_num_atoms):
    logging.info("Featurizing molecule of size: %d", len(mol.GetAtoms()))
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                          self.max_num_neighbors, None)
    z = self.get_Z_matrix(mol, max_num_atoms)
    z = pad_array(z, max_num_atoms)
    coords = pad_array(coords, (max_num_atoms, 3))
    return coords, neighbor_list, z

  def _strip_hydrogens(self, coords, mol):

    class MoleculeShim(object):
      """
      Shim of a Molecule which supports #GetAtoms()
      """

      def __init__(self, atoms):
        self.atoms = [AtomShim(x) for x in atoms]

      def GetAtoms(self):
        return self.atoms

    class AtomShim(object):

      def __init__(self, atomic_num):
        self.atomic_num = atomic_num

      def GetAtomicNum(self):
        return self.atomic_num

    if not self.strip_hydrogens:
      return coords, mol
    indexes_to_keep = []
    atomic_numbers = []
    for index, atom in enumerate(mol.GetAtoms()):
      if atom.GetAtomicNum() != 1:
        indexes_to_keep.append(index)
        atomic_numbers.append(atom.GetAtomicNum())
    mol = MoleculeShim(atomic_numbers)
    coords = coords[indexes_to_keep]
    return coords, mol


#################### Deprecation warnings for old atomic conv featurizer name ####################

ATOMICCONV_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class ComplexNeighborListFragmentAtomicCoordinates(AtomicConvFeaturizer):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        ATOMICCONV_DEPRECATION.format(
            "ComplexNeighorListFragmentAtomicCoordinates",
            "AtomicConvFeaturizer"), FutureWarning)

    super(ComplexNeighborListFragmentAtomicCoordinates, self).__init__(
        *args, **kwargs)

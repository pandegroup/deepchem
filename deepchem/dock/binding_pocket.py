"""
Computes putative binding pockets on protein.
"""
import logging
import numpy as np
from typing import Any, List, Optional, Tuple

from deepchem.models import Model
from deepchem.utils.rdkit_utils import load_molecule
from deepchem.utils.coordinate_box_utils \
  import CoordinateBox, get_face_boxes, merge_overlapping_boxes
from deepchem.utils.fragment_utils import get_contact_atom_indices

logger = logging.getLogger(__name__)


def extract_active_site(protein_file: str,
                        ligand_file: str,
                        cutoff: float = 4.0
                      ) -> Tuple[CoordinateBox, np.ndarray]:
  """Extracts a box for the active site.

  Parameters
  ----------
  protein_file: str
    Location of protein PDB
  ligand_file: str
    Location of ligand input file
  cutoff: float, optional (default 4.0)
    The distance in angstroms from the protein pocket to
    consider for featurization.

  Returns
  -------
  Tuple[CoordinateBox, np.ndarray]
    A tuple of `(CoordinateBox, np.ndarray)` where the second entry is
    of shape `(N, 3)` with `N` the number of atoms in the active site.
  """
  protein = load_molecule(protein_file, add_hydrogens=False)
  ligand = load_molecule(ligand_file, add_hydrogens=True, calc_charges=True)
  protein_contacts, ligand_contacts = get_contact_atom_indices(
      [protein, ligand], cutoff=cutoff)
  protein_coords = protein[0]
  pocket_coords = protein_coords[protein_contacts]

  x_min = int(np.floor(np.amin(pocket_coords[:, 0])))
  x_max = int(np.ceil(np.amax(pocket_coords[:, 0])))
  y_min = int(np.floor(np.amin(pocket_coords[:, 1])))
  y_max = int(np.ceil(np.amax(pocket_coords[:, 1])))
  z_min = int(np.floor(np.amin(pocket_coords[:, 2])))
  z_max = int(np.ceil(np.amax(pocket_coords[:, 2])))
  box = CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))
  return box, pocket_coords


class BindingPocketFinder(object):
  """Abstract superclass for binding pocket detectors

  Many times when working with a new protein or other macromolecule,
  it's not clear what zones of the macromolecule may be good targets
  for potential ligands or other molecules to interact with. This
  abstract class provides a template for child classes that
  algorithmically locate potential binding pockets that are good
  potential interaction sites.

  Note that potential interactions sites can be found by many
  different methods, and that this abstract class doesn't specify the
  technique to be used.
  """

  def find_pockets(self, molecule: Any):
    """Finds potential binding pockets in proteins.

    Parameters
    ----------
    molecule: object
      Some representation of a molecule.
    """
    raise NotImplementedError


class ConvexHullPocketFinder(BindingPocketFinder):
  """Implementation that uses convex hull of protein to find pockets.

  Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112621/pdf/1472-6807-14-18.pdf
  """

  def __init__(self, scoring_model: Optional[Model] = None, pad: float = 5.0):
    """Initialize the pocket finder.

    Parameters
    ----------
    scoring_model: Model, optional (default None)
      If specified, use this model to prune pockets.
    pad: float, optional (default 5.0)
      The number of angstroms to pad around a binding pocket's atoms
      to get a binding pocket box.
    """
    self.scoring_model = scoring_model
    self.pad = pad

  def find_all_pockets(self, protein_file: str) -> List[CoordinateBox]:
    """Find list of binding pockets on protein.

    Parameters
    ----------
    protein_file: str
      Protein to load in.

    Returns
    -------
    List[CoordinateBox]
      List of binding pockets on protein. Each pocket is a `CoordinateBox`
    """
    coords, _ = load_molecule(protein_file)
    return get_face_boxes(coords, self.pad)

  def find_pockets(self, macromolecule_file: str) -> List[CoordinateBox]:
    """Find list of suitable binding pockets on protein.

    This function computes putative binding pockets on this protein.
    This class uses the `ConvexHull` to compute binding pockets. Each
    face of the hull is converted into a coordinate box used for
    binding.

    Parameters
    ----------
    macromolecule_file: str
      Location of the macromolecule file to load

    Returns
    -------
    List[CoordinateBox]
      List of pockets. Each pocket is a `CoordinateBox`
    """
    coords, _ = load_molecule(
        macromolecule_file, add_hydrogens=False, calc_charges=False)
    boxes = get_face_boxes(coords, self.pad)
    boxes = merge_overlapping_boxes(boxes)
    return boxes

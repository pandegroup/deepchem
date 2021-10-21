"""
This file contains utilities for molecular docking.
"""
from typing import List, Optional, Tuple

import os
import numpy as np
from deepchem.utils.typing import RDKitMol
from deepchem.utils.pdbqt_utils import pdbqt_to_pdb


def write_vina_conf(protein_filename: str,
                    ligand_filename: str,
                    centroid: np.ndarray,
                    box_dims: np.ndarray,
                    conf_filename: str,
                    num_modes: int = 9,
                    exhaustiveness: int = None) -> None:
  """Writes Vina configuration file to disk.

  Autodock Vina accepts a configuration file which provides options
  under which Vina is invoked. This utility function writes a vina
  configuration file which directs Autodock vina to perform docking
  under the provided options.

  Parameters
  ----------
  protein_filename: str
    Filename for protein
  ligand_filename: str
    Filename for the ligand
  centroid: np.ndarray
    A numpy array with shape `(3,)` holding centroid of system
  box_dims: np.ndarray
    A numpy array of shape `(3,)` holding the size of the box to dock
  conf_filename: str
    Filename to write Autodock Vina configuration to.
  num_modes: int, optional (default 9)
    The number of binding modes Autodock Vina should find
  exhaustiveness: int, optional
    The exhaustiveness of the search to be performed by Vina
  """
  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % protein_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("center_x = %f\n" % centroid[0])
    f.write("center_y = %f\n" % centroid[1])
    f.write("center_z = %f\n\n" % centroid[2])

    f.write("size_x = %f\n" % box_dims[0])
    f.write("size_y = %f\n" % box_dims[1])
    f.write("size_z = %f\n\n" % box_dims[2])

    f.write("num_modes = %d\n\n" % num_modes)
    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)


def write_gnina_conf(protein_filename: str,
                     ligand_filename: str,
                     conf_filename: str,
                     num_modes: int = 9,
                     exhaustiveness: int = None,
                     **kwargs) -> None:
  """Writes GNINA configuration file to disk.

  GNINA accepts a configuration file which provides options
  under which GNINA is invoked. This utility function writes a
  configuration file which directs GNINA to perform docking
  under the provided options.

  Parameters
  ----------
  protein_filename: str
    Filename for protein
  ligand_filename: str
    Filename for the ligand
  conf_filename: str
    Filename to write Autodock Vina configuration to.
  num_modes: int, optional (default 9)
    The number of binding modes GNINA should find
  exhaustiveness: int, optional
    The exhaustiveness of the search to be performed by GNINA
  kwargs:
    Args supported by GNINA documented here
    https://github.com/gnina/gnina#usage

  """

  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % protein_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("autobox_ligand = %s\n\n" % protein_filename)

    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)
    f.write("num_modes = %d\n\n" % num_modes)

    for k, v in kwargs.items():
      f.write("%s = %s\n" % (str(k), str(v)))


def read_gnina_log(log_file: str) -> np.ndarray:
  """Read GNINA logfile and get docking scores.

  GNINA writes computed binding affinities to a logfile.

  Parameters
  ----------
  log_file: str
    Filename of logfile generated by GNINA.

  Returns
  -------
  scores: np.array, dimension (num_modes, 3)
    Array of binding affinity (kcal/mol), CNN pose score,
    and CNN affinity for each binding mode.

  """

  scores = []
  lines = open(log_file).readlines()
  mode_start = np.inf
  for idx, line in enumerate(lines):
    if line[:6] == '-----+':
      mode_start = idx
    if idx > mode_start:
      mode = line.split()
      score = [float(x) for x in mode[1:]]
      scores.append(score)

  scores = np.array(scores)
  return scores


def load_docked_ligands(
    pdbqt_output: str) -> Tuple[List[RDKitMol], List[float]]:
  """This function loads ligands docked by autodock vina.

  Autodock vina writes outputs to disk in a PDBQT file format. This
  PDBQT file can contain multiple docked "poses". Recall that a pose
  is an energetically favorable 3D conformation of a molecule. This
  utility function reads and loads the structures for multiple poses
  from vina's output file.

  Parameters
  ----------
  pdbqt_output: str
    Should be the filename of a file generated by autodock vina's
    docking software.

  Returns
  -------
  Tuple[List[rdkit.Chem.rdchem.Mol], List[float]]
    Tuple of `molecules, scores`. `molecules` is a list of rdkit
    molecules with 3D information. `scores` is the associated vina
    score.

  Notes
  -----
  This function requires RDKit to be installed.
  """
  try:
    from rdkit import Chem
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  lines = open(pdbqt_output).readlines()
  molecule_pdbqts = []
  scores = []
  current_pdbqt: Optional[List[str]] = None
  for line in lines:
    if line[:5] == "MODEL":
      current_pdbqt = []
    elif line[:19] == "REMARK VINA RESULT:":
      words = line.split()
      # the line has format
      # REMARK VINA RESULT: score ...
      # There is only 1 such line per model so we can append it
      scores.append(float(words[3]))
    elif line[:6] == "ENDMDL":
      molecule_pdbqts.append(current_pdbqt)
      current_pdbqt = None
    else:
      # FIXME: Item "None" of "Optional[List[str]]" has no attribute "append"
      current_pdbqt.append(line)  # type: ignore

  molecules = []
  for pdbqt_data in molecule_pdbqts:
    pdb_block = pdbqt_to_pdb(pdbqt_data=pdbqt_data)
    mol = Chem.MolFromPDBBlock(str(pdb_block), sanitize=False, removeHs=False)
    molecules.append(mol)
  return molecules, scores


def prepare_inputs(protein: str,
                   ligand: str,
                   replace_nonstandard_residues: bool = True,
                   remove_heterogens: bool = True,
                   remove_water: bool = True,
                   add_hydrogens: bool = True,
                   pH: float = 7.0,
                   optimize_ligand: bool = True,
                   pdb_name: Optional[str] = None) -> Tuple[RDKitMol, RDKitMol]:
  """This prepares protein-ligand complexes for docking.

  Autodock Vina requires PDB files for proteins and ligands with
  sensible inputs. This function uses PDBFixer and RDKit to ensure
  that inputs are reasonable and ready for docking. Default values
  are given for convenience, but fixing PDB files is complicated and
  human judgement is required to produce protein structures suitable
  for docking. Always inspect the results carefully before trying to
  perform docking.

  Parameters
  ----------
  protein: str
    Filename for protein PDB file or a PDBID.
  ligand: str
    Either a filename for a ligand PDB file or a SMILES string.
  replace_nonstandard_residues: bool (default True)
    Replace nonstandard residues with standard residues.
  remove_heterogens: bool (default True)
    Removes residues that are not standard amino acids or nucleotides.
  remove_water: bool (default True)
    Remove water molecules.
  add_hydrogens: bool (default True)
    Add missing hydrogens at the protonation state given by `pH`.
  pH: float (default 7.0)
    Most common form of each residue at given `pH` value is used.
  optimize_ligand: bool (default True)
    If True, optimize ligand with RDKit. Required for SMILES inputs.
  pdb_name: Optional[str]
    If given, write sanitized protein and ligand to files called
    "pdb_name.pdb" and "ligand_pdb_name.pdb"

  Returns
  -------
  Tuple[RDKitMol, RDKitMol]
    Tuple of `protein_molecule, ligand_molecule` with 3D information.

  Note
  ----
  This function requires RDKit and OpenMM to be installed.
  Read more about PDBFixer here: https://github.com/openmm/pdbfixer.

  Examples
  --------
  >>> from deepchem.utils import prepare_inputs
  >>> p, m = prepare_inputs('3cyx', 'CCC')
  >>> p.GetNumAtoms()
  1415
  >>> m.GetNumAtoms()
  11

  >>> p, m = prepare_inputs('3cyx', 'CCC', remove_heterogens=False)
  >>> p.GetNumAtoms()
  1720

  """

  try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
  except ModuleNotFoundError:
    raise ImportError(
        "This function requires RDKit and OpenMM to be installed.")

  if protein.endswith('.pdb'):
    fixer = PDBFixer(protein)
  else:
    fixer = PDBFixer(url='https://files.rcsb.org/download/%s.pdb' % (protein))

  if ligand.endswith('.pdb'):
    m = Chem.MolFromPDBFile(ligand)
  else:
    m = Chem.MolFromSmiles(ligand, sanitize=True)

  # Apply common fixes to PDB files
  if replace_nonstandard_residues:
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
  if remove_heterogens and not remove_water:
    fixer.removeHeterogens(True)
  if remove_heterogens and remove_water:
    fixer.removeHeterogens(False)
  if add_hydrogens:
    fixer.addMissingHydrogens(pH)

  PDBFile.writeFile(fixer.topology, fixer.positions, open('tmp.pdb', 'w'))
  p = Chem.MolFromPDBFile('tmp.pdb', sanitize=True)
  os.remove('tmp.pdb')

  # Optimize ligand
  if optimize_ligand:
    m = Chem.AddHs(m)  # need hydrogens for optimization
    AllChem.EmbedMolecule(m)
    AllChem.MMFFOptimizeMolecule(m)

  if pdb_name:
    Chem.rdmolfiles.MolToPDBFile(p, '%s.pdb' % (pdb_name))
    Chem.rdmolfiles.MolToPDBFile(m, 'ligand_%s.pdb' % (pdb_name))

  return (p, m)

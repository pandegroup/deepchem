from typing import Union

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class RawFeaturizer(MolecularFeaturizer):
  """Encodes a molecule as a SMILES string or RDKit mol.

  This featurizer can be useful when you're trying to transform a large
  collection of RDKit mol objects as Smiles strings, or alternatively as a
  "no-op" featurizer in your molecular pipeline.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, smiles: bool = False):
    """Initialize this featurizer.

    Parameters
    ----------
    smiles: bool, optional (default False)
      If True, encode this molecule as a SMILES string. Else as a RDKit mol.
    """
    self.smiles = smiles

  def _featurize(self, mol: RDKitMol) -> Union[str, RDKitMol]:
    """Calculate either smiles string or pass through raw molecule.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    str or rdkit.Chem.rdchem.Mol
      SMILES string or RDKit Mol object.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    if self.smiles:
      return [Chem.MolToSmiles(mol)]
    else:
      return [mol]

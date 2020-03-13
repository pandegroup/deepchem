"""
Topological fingerprints.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "MIT"

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from deepchem.feat import Featurizer


class CircularFingerprint(Featurizer):
  """
  Circular (Morgan) fingerprints.

  Parameters
  ----------
  radius : int, optional (default 2)
      Fingerprint radius.
  size : int, optional (default 2048)
      Length of generated bit vector.
  chiral : bool, optional (default False)
      Whether to consider chirality in fingerprint generation.
  counts : bool, optional (default False)
      Whether to calculated fingerprint counts (0..n) or bits (0,1).
  bonds : bool, optional (default True)
      Whether to consider bond order in fingerprint generation.
  features : bool, optional (default False)
      Whether to use feature information instead of atom information; see
      RDKit docs for more info.
  sparse : bool, optional (default False)
      Whether to return a dict for each molecule containing the sparse
      fingerprint.
  smiles : bool, optional (default False)
      Whether to calculate SMILES strings for fragment IDs (only applicable
      when calculating sparse fingerprints).
  """
  name = 'circular'

    def __init__(self,
                 radius=2,
                 size=2048,
                 chiral=False,
                 bonds=True,
                 features=False,
                 sparse=False,
                 smiles=False,
                 counts=False):
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.counts = counts
        self.bonds = bonds
        self.features = features
        self.sparse = sparse
        self.smiles = smiles

  def _featurize(self, mol):
    """
    Calculate circular fingerprint.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
        if self.sparse:
            info = {}
            fp = rdMolDescriptors.GetMorganFingerprint(
                mol,
                self.radius,
                useChirality=self.chiral,
                useBondTypes=self.bonds,
                useFeatures=self.features,
                bitInfo=info)
            fp = fp.GetNonzeroElements()  # convert to a dict

            # generate SMILES for fragments
            if self.smiles:
                fp_smiles = {}
                for fragment_id, count in fp.items():
                    root, radius = info[fragment_id][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root)
                    frag = Chem.PathToSubmol(mol, env)
                    smiles = Chem.MolToSmiles(frag)
                    fp_smiles[fragment_id] = {'smiles': smiles, 'count': count}
                fp = fp_smiles
        elif self.counts:
            info = {}
            fpbv = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.size,
                useChirality=self.chiral,
                useBondTypes=self.bonds,
                useFeatures=self.features,
                bitInfo=info)
            fp = np.zeros((1, ), dtype=int)
            DataStructs.ConvertToNumpyArray(fpbv, fp)
            for b, c in info.items():
                fp[b] = len(c)
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.size,
                useChirality=self.chiral,
                useBondTypes=self.bonds,
                useFeatures=self.features)
        return fp

  def __hash__(self):
    return hash((self.radius, self.size, self.chiral, self.bonds,
                    self.features, self.sparse, self.smiles))

  def __eq__(self, other):
    if not isinstance(self, other.__class__):
      return False
    return self.radius == other.radius and \
              self.size == other.size and \
              self.chiral == other.chiral and \
              self.bonds == other.bonds and \
              self.features == other.features and \
              self.sparse == other.sparse and \
              self.smiles == other.smiles
               

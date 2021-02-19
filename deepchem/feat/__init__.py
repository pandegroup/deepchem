"""
Making it easy to import in classes.
"""
# flake8: noqa

# base classes for featurizers
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.base_classes import MaterialStructureFeaturizer
from deepchem.feat.base_classes import MaterialCompositionFeaturizer
from deepchem.feat.base_classes import ComplexFeaturizer
from deepchem.feat.base_classes import UserDefinedFeaturizer

from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.graph_features import WeaveFeaturizer
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer

# molecule featurizers
from deepchem.feat.molecule_featurizers import AtomicCoordinates
from deepchem.feat.molecule_featurizers import BPSymmetryFunctionInput
from deepchem.feat.molecule_featurizers import CircularFingerprint
from deepchem.feat.molecule_featurizers import CoulombMatrix
from deepchem.feat.molecule_featurizers import CoulombMatrixEig
from deepchem.feat.molecule_featurizers import MACCSKeysFingerprint
from deepchem.feat.molecule_featurizers import MordredDescriptors
from deepchem.feat.molecule_featurizers import Mol2VecFingerprint
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.feat.molecule_featurizers import MolGanFeaturizer
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.feat.molecule_featurizers import PubChemFingerprint
from deepchem.feat.molecule_featurizers import RawFeaturizer
from deepchem.feat.molecule_featurizers import RDKitDescriptors
from deepchem.feat.molecule_featurizers import SmilesToImage
from deepchem.feat.molecule_featurizers import SmilesToSeq, create_char_to_idx

# complex featurizers
from deepchem.feat.complex_featurizers import RdkitGridFeaturizer
from deepchem.feat.complex_featurizers import NeighborListAtomicCoordinates
from deepchem.feat.complex_featurizers import NeighborListComplexAtomicCoordinates
from deepchem.feat.complex_featurizers import AtomicConvFeaturizer
from deepchem.feat.complex_featurizers import (
    ComplexNeighborListFragmentAtomicCoordinates,)
from deepchem.feat.complex_featurizers import ContactCircularFingerprint
from deepchem.feat.complex_featurizers import ContactCircularVoxelizer
from deepchem.feat.complex_featurizers import SplifFingerprint
from deepchem.feat.complex_featurizers import SplifVoxelizer
from deepchem.feat.complex_featurizers import ChargeVoxelizer
from deepchem.feat.complex_featurizers import SaltBridgeVoxelizer
from deepchem.feat.complex_featurizers import CationPiVoxelizer
from deepchem.feat.complex_featurizers import PiStackVoxelizer
from deepchem.feat.complex_featurizers import HydrogenBondVoxelizer
from deepchem.feat.complex_featurizers import HydrogenBondCounter

# material featurizers
from deepchem.feat.material_featurizers import ElementPropertyFingerprint
from deepchem.feat.material_featurizers import SineCoulombMatrix
from deepchem.feat.material_featurizers import CGCNNFeaturizer
from deepchem.feat.material_featurizers import ElemNetFeaturizer
from deepchem.feat.material_featurizers import LCNNFeaturizer

from deepchem.feat.atomic_conformation import AtomicConformation
from deepchem.feat.atomic_conformation import AtomicConformationFeaturizer

try:
  import transformers
  from transformers import BertTokenizer

  from deepchem.feat.smiles_tokenizer import SmilesTokenizer
  from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
except ModuleNotFoundError:
  pass

# support classes
from deepchem.feat.molecule_featurizers import GraphMatrix

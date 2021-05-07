import deepchem as dc
from deepchem.feat.molecule_featurizers import RawFeaturizer


def test_multi_featurizer_hppb():
  """
    Expect this to exit with status 0.
    """
  feat = RawFeaturizer()
  tasks, datasets, transformers = dc.molnet.load_hppb(
      featurizer=feat, reload=False)

  tasks, datasets, transformers = dc.molnet.load_hppb(
      featurizer="ECFP", reload=False)

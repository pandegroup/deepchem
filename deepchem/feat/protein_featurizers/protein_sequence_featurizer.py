import os
import numpy as np
import tempfile
from amino_acids import AminoAcid
from deepchem.utils.protein_sequence_data_utils import read_fasta
from deepchem.utils.protein_sequence_data_utils import read_hhm
from deepchem.utils.protein_sequence_data_utils import run_hhblits_local
from deepchem.utils.protein_sequence_feature_utils import read_a3m_as_mat
from deepchem.utils.protein_sequence_feature_utils import sequence_one_hot_encoding
from deepchem.utils.protein_sequence_feature_utils import sequence_deletion_probability
from deepchem.utils.protein_sequence_feature_utils import sequence_weights
from deepchem.utils.protein_sequence_feature_utils import sequence_profile
from deepchem.utils.protein_sequence_feature_utils import sequence_profile_no_gap
from deepchem.utils.protein_sequence_feature_utils import sequence_profile_with_prior
from deepchem.utils.protein_sequence_feature_utils import sequence_identity
from deepchem.utils.protein_sequence_feature_utils import sequence_static_prop
from deepchem.utils.protein_sequence_feature_utils import sequence_gap_matrix
from deepchem.utils.protein_sequence_feature_utils import profile_combinatorial
from deepchem.utils.protein_sequence_feature_utils import mutual_information
from deepchem.utils.protein_sequence_feature_utils import mean_contact_potential

from deepchem.feat.base_classes import Featurizer



class ProteinSequenceFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for a
  protein sequence (used for structure prediction).

  The defining feature of a `ProteinSequenceFeaturizer` is that it
  reads protein sequences (and pre-saved multiple sequence alignments)
  to generate features for structure prediction.

  Child classes need to implement the `_featurize` method for
  calculating features for a single protein sequence. Note that `_featurize`
  methods should take two arguments: protein sequence and a directory path 
  for saving multiple sequence alignments
  """

  def prepare_msa(self, sequence, path):
    msa_file = os.path.join(path, 'results.a3m')
    if os.path.exists(msa_file):
      return True
    else:
      run_hhblits_local(sequence, path)
      if not os.path.exists(msa_file):
        return False
      return True


  def featurize(self, 
                protein_seqs: Iterable[str], 
                log_every_n : int=1000):
    """Calculate features for molecules.
    Parameters
    ----------
    protein_seqs: str
      protein sequence or path to the folder of pre-saved multiple sequence alignments
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: 
    """

    if isinstance(protein_seqs, str):
      # Handle single sequence/path
      protein_seqs = [protein_seqs]
    else:
      # Convert iterables to list
      protein_seqs = list(protein_seqs)

    features = []
    for i, s in enumerate(protein_seqs):
      if os.path.isdir(s):
        # `s` is a folder, pre-saved MSAs are provided.
        # Raw sequence should be saved under this folder as "input.seq"
        _, sequence = read_fasta(os.path.join(s, "input.seq"))
        assert len(sequence) == 1
        sequence = sequence[0]
        path = s
      else:
        # `s` is a protein sequence, a temp folder 
        # will be used for featurization
        sequence = s
        path = tempfile.mkdtemp()

      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      features.append(self._featurize(sequence, path))

    features = np.asarray(features)
    return features


class ContactMapProteinSequenceFeaturizer(ProteinSequenceFeaturizer):
  
  def _featurize(sequence, path):
    # sec_structure
    # solv_surf
    assert self.prepare_msa(sequence, path), "MSA not found under %s" % path
    hhm_profile = read_hhm(sequence, os.path.join(path, 'results.hhm'))
      
    a3m_seq_IDs, a3m_seqs = read_fasta(os.path.join(path, 'results.a3m'))
    a3m_seq_mat = read_a3m_as_mat(sequence, a3m_seq_IDs, a3m_seqs)
    
    seq_one_hot = sequence_one_hot_encoding(sequence)
    seq_del_prob = sequence_deletion_probability(sequence, a3m_seqs)
    prof = sequence_profile(a3m_seq_mat)
    prof_no_gap = sequence_profile_no_gap(a3m_seq_mat)
    
    weights = sequence_weights(a3m_seq_mat)
    w_prof = sequence_profile(a3m_seq_mat, weights=weights)
    w_prof_no_gap = sequence_profile_no_gap(a3m_seq_mat, weights=weights)

    prior_prof = sequence_profile_with_prior(w_prof)
    prior_prof_no_gap = sequence_profile_with_prior(w_prof_no_gap)    

    static_prop = sequence_static_prop(a3m_seq_mat, weights)

    feats_1D = np.concatenate([seq_one_hot, seq_del_prob, prof, prof_no_gap,
                               w_prof, w_prof_no_gap, prior_prof, 
                               prior_prof_no_gap, hhm_profile, static_prop], 1)


    gap_matrix = sequence_gap_matrix(a3m_seq_mat)
    w_prof_2D = profile_combinatorial(a3m_seq_mat, weights, w_prof)
    MI = mutual_information(w_prof, w_prof_2D)
    MCP = mean_contact_potential(w_prof_2D)

    #pseudo_bias
    #pseudo_frob
    #pseudolikelihood
    feats_2D = np.concatenate([gap_matrix, MI, MCP], 2)
    
    
    return feats_1D, feats_2D

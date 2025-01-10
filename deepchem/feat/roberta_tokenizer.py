import pandas as pd
from deepchem.feat import Featurizer
from typing import List, Union
try:
    from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for RobertaFeaturizer to be used!')
    pass


class RobertaFeaturizer(RobertaTokenizerFast, Featurizer):
    """Roberta Featurizer.

    The Roberta Featurizer is a wrapper class of the Roberta Tokenizer,
    which is used by Huggingface's transformers library for tokenizing large corpuses for Roberta Models.
    Please confirm the details in [1]_.

    Please see https://github.com/huggingface/transformers
    and https://github.com/seyonechithrananda/bert-loves-chemistry for more details.

    Examples
    --------
    >>> from deepchem.feat import RobertaFeaturizer
    >>> smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    >>> featurizer = RobertaFeaturizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
    >>> out = featurizer(smiles, add_special_tokens=True, truncation=True)

    References
    ----------
    .. [1] Chithrananda, Seyone, Grand, Gabriel, and Ramsundar, Bharath (2020): "Chemberta: Large-scale self-supervised
        pretraining for molecular property prediction." arXiv. preprint. arXiv:2010.09885.


    Note
    -----
    This class requires transformers to be installed.
    RobertaFeaturizer uses dual inheritance with RobertaTokenizerFast in Huggingface for rapid tokenization,
    as well as DeepChem's MolecularFeaturizer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """Calculate encoding using HuggingFace's RobertaTokenizerFast.

        Parameters
        ----------
        datapoint: str
            Arbitrary string sequence to be tokenized.

        Returns
        -------
        encoding: List[List[int]]
            List containing two lists; the `input_ids` and the `attention_mask`
        """

        # the encoding is natively a dictionary with keys 'input_ids' and 'attention_mask'
        encoding = list(super().__call__(datapoint, **kwargs).values())
        return encoding

    def __call__(self,
                 datapoints: Union[str, List[str], List[List[str]], pd.Series],
                 padding: bool = True,
                 **kwargs) -> List[List[int]]:
        """Convert RobertaFeaturizer into a callable.

        The embeddings of the datapoints are all padded to the
        same length so that they can be featurized all together. Only
        the input_ids are returned as they represent the features that are
        used by _featurize_shard in downstream DataLoader object.

        Parameters
        ----------
        datapoints: str | List[str] | List[List[str]] | pd.Series
            Arbitrary string sequence to be featurized.
        padding: bool, default True
            Pad all the embeddings to the same length.

        Returns
        -------
        results['input_ids]: List[List[int]]
            A List of Lists where each sublist represents the input ids of the embedding.
            All the sublists are padded to the same length.
        """

        if isinstance(datapoints, pd.Series):
            datapoints = datapoints.to_list()
        # results is natively a dictionary with keys 'input_ids' and 'attention_mask'
        results = super().__call__(datapoints, padding=padding, **kwargs)
        return results['input_ids']

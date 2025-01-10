import pytest


@pytest.mark.torch
def test_smiles_call():
    """Test __call__ method for the featurizer, which is inherited from HuggingFace's RobertaTokenizerFast"""
    from deepchem.feat.roberta_tokenizer import RobertaFeaturizer
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    long_molecule_smiles = [
        "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
    ]
    featurizer = RobertaFeaturizer.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k")
    embedding = featurizer(smiles, add_special_tokens=True, truncation=True)
    embedding_long = featurizer(long_molecule_smiles * 2,
                                add_special_tokens=True,
                                truncation=True)
    for emb in [embedding, embedding_long]:
        assert len(emb[0]) == len(emb[1])


@pytest.mark.torch
def test_smiles_featurize():
    """Test the .featurize method, which will convert the dictionary output to an array

    Checks that all SMILES are featurized and that each featurization
    contains input_ids and attention_mask
    """
    from deepchem.feat.roberta_tokenizer import RobertaFeaturizer
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    long_molecule_smiles = [
        "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
    ]
    featurizer = RobertaFeaturizer.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k")
    max_length = 100
    feat_kwargs = {
        'add_special_tokens': True,
        'truncation': True,
        'padding': 'max_length',
        'max_length': max_length
    }
    feats = featurizer.featurize(smiles, **feat_kwargs)
    assert len(feats) == 2
    assert all([len(f) == 2 for f in feats])
    assert all([len(f[0]) == max_length for f in feats])

    long_feat = featurizer.featurize(long_molecule_smiles, **feat_kwargs)
    assert len(long_feat) == 1
    assert len(long_feat[0]) == 2  # the tokens and attention mask
    assert len(
        long_feat[0][0]) == 100  # number of tokens for each smiles string

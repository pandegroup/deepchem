import os
import deepchem as dc
import numpy as np
import pytest
import torch

from deepchem.models.torch_models.MolPropModel import ChemBERTaGNN  # Adjusted to import the new model


@pytest.mark.hf
def test_chemberta_gnn_pretraining(smiles_regression_dataset, smiles_multitask_regression_dataset):
    # Pretraining in MLM mode
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression')
    loss = model.fit(smiles_regression_dataset, nb_epoch=1)
    assert loss

    # Pretraining in Multitask Regression Mode
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', n_tasks=2)
    loss = model.fit(smiles_multitask_regression_dataset, nb_epoch=1)
    assert loss


@pytest.mark.hf
def test_chemberta_gnn_finetuning(smiles_regression_dataset, smiles_multitask_regression_dataset):
    # test regression
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression')
    loss = model.fit(smiles_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(smiles_regression_dataset, metrics=dc.metrics.Metric(dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(smiles_regression_dataset)
    assert prediction.shape == smiles_regression_dataset.y.shape

    # test multitask regression
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', n_tasks=2)
    loss = model.fit(smiles_multitask_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(smiles_multitask_regression_dataset, metrics=dc.metrics.Metric(dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(smiles_multitask_regression_dataset)
    assert prediction.shape == smiles_multitask_regression_dataset.y.shape

    # test classification
    y = np.random.choice([0, 1], size=smiles_regression_dataset.y.shape)
    dataset = dc.data.NumpyDataset(X=smiles_regression_dataset.X, y=y, w=smiles_regression_dataset.w, ids=smiles_regression_dataset.ids)
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='classification')
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


@pytest.mark.hf
def test_chemberta_gnn_load_from_pretrained(tmpdir, smiles_regression_dataset):
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', model_dir=pretrain_model_dir)
    model.save_checkpoint()

    finetune_model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # Check if weights match
    pretrain_model_state_dict = model.state_dict()
    finetune_model_state_dict = finetune_model.state_dict()
    matches = [
        torch.allclose(pretrain_model_state_dict[key], finetune_model_state_dict[key])
        for key in pretrain_model_state_dict.keys()
    ]
    assert all(matches)


@pytest.mark.hf
def test_chemberta_gnn_save_reload(tmpdir):
    model = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = ChemBERTaGNN(bert_model_name='seyonec/ChemBERTa-2', gnn_type='GCN', mode='regression', model_dir=tmpdir)
    model_new.restore()

    old_state = model.state_dict()
    new_state = model_new.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]
    # All keys should match
    assert all(matches)


@pytest.mark.hf
def test_chemberta_gnn_load_weights_from_hf_hub():
    pretrained_model_path = 'DeepChem/ChemBERTa-77M-MLM'
    model = ChemBERTaGNN(bert_model_name='DeepChem/ChemBERTa-77M-MLM', gnn_type='GCN', mode='regression')
    old_model_id = id(model)
    model.load_from_pretrained(pretrained_model_path, from_hf_checkpoint=True)
    new_model_id = id(model)
    # The model ID should change after loading new weights
    assert old_model_id != new_model_id

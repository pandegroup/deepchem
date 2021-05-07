import unittest
import tempfile

import numpy as np

import deepchem as dc
from deepchem.feat import PagtnMolGraphFeaturizer
from deepchem.models import PagtnModel
from deepchem.models.tests.test_graph_models import get_dataset

try:
  import dgl
  import dgllife
  import torch
  has_torch_and_dgl = True
except:
  has_torch_and_dgl = False


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_pagtn_regression():
  # load datasets
  featurizer = PagtnMolGraphFeaturizer(max_length=5)
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = PagtnModel(mode='regression', n_tasks=n_tasks, batch_size=16)

  # overfit test
  model.fit(dataset, nb_epoch=20)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.5

  # test on a small MoleculeNet dataset
  from deepchem.molnet import load_delaney

  tasks, all_dataset, transformers = load_delaney(featurizer=featurizer)
  train_set, _, _ = all_dataset
  model = PagtnModel(mode='regression', n_tasks=n_tasks, batch_size=16)
  model.fit(train_set, nb_epoch=1)


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_attentivefp_classification():
  # load datasets
  featurizer = PagtnMolGraphFeaturizer(max_length=5)
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = PagtnModel(mode='classification', n_tasks=n_tasks, batch_size=16)

  # overfit test
  model.fit(dataset, nb_epoch=100)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.85

  # test on a small MoleculeNet dataset
  from deepchem.molnet import load_bace_classification

  tasks, all_dataset, transformers = load_bace_classification(
      featurizer=featurizer)
  train_set, _, _ = all_dataset
  model = PagtnModel(mode='classification', n_tasks=n_tasks, batch_size=16)
  model.fit(train_set, nb_epoch=1)


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_attentivefp_reload():
  # load datasets
  featurizer = PagtnMolGraphFeaturizer(max_length=5)
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model_dir = tempfile.mkdtemp()
  model = PagtnModel(
      mode='classification',
      n_tasks=n_tasks,
      model_dir=model_dir,
      batch_size=16)

  model.fit(dataset, nb_epoch=100)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.85

  reloaded_model = PagtnModel(
      mode='classification',
      n_tasks=n_tasks,
      model_dir=model_dir,
      batch_size=16)
  reloaded_model.restore()

  pred_mols = ["CCCC", "CCCCCO", "CCCCC"]
  X_pred = featurizer(pred_mols)
  random_dataset = dc.data.NumpyDataset(X_pred)
  original_pred = model.predict(random_dataset)
  reload_pred = reloaded_model.predict(random_dataset)
  assert np.all(original_pred == reload_pred)

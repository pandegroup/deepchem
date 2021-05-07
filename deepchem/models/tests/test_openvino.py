import pytest
import numpy as np
import tensorflow as tf
import torch
import os

import deepchem as dc
from deepchem.molnet import load_tox21, load_perovskite
from deepchem.data import NumpyDataset
from deepchem.models.torch_models import TorchModel


def test_cgcnn_regression():
  np.random.seed(123)
  torch.manual_seed(124)

  # load datasets
  current_dir = os.path.dirname(os.path.abspath(__file__))
  config = {
      "reload": False,
      "featurizer": dc.feat.CGCNNFeaturizer(),
      # disable transformer
      "transformers": [],
      "data_dir": current_dir
  }

  tasks, datasets, transformers = load_perovskite(**config)
  train, valid, _ = datasets
  n_tasks = len(tasks)

  def init(**kwargs):
    torch.manual_seed(124)
    return dc.models.CGCNNModel(
        n_tasks=n_tasks,
        mode='regression',
        batch_size=1,
        learning_rate=0.001,
        **kwargs)

  ref_model = init()
  model = init(use_openvino=True)

  # check overfit
  metric = dc.metrics.Metric(dc.metrics.mae_score, n_tasks=n_tasks)
  ref_scores = ref_model.evaluate(train, [metric], transformers)
  scores = model.evaluate(train, [metric], transformers)

  if os.path.exists(os.path.join(current_dir, 'perovskite.json')):
    os.remove(os.path.join(current_dir, 'perovskite.json'))

  assert abs(scores['mae_score'] - ref_scores['mae_score']) < 0.2


def test_tox21_tf_progressive():
  np.random.seed(123)
  tf.random.set_seed(124)

  n_features = 1024
  tox21_tasks, tox21_datasets, transformers = load_tox21()
  _, valid_dataset, _ = tox21_datasets

  metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

  def init(**kwargs):
    tf.random.set_seed(572)
    return dc.models.ProgressiveMultitaskClassifier(
        len(tox21_tasks),
        n_features,
        layer_sizes=[1000],
        dropouts=[.25],
        learning_rate=0.001,
        batch_size=50,
        **kwargs)

  ref_model = init()
  model = init(use_openvino=True)

  ref_scores = ref_model.evaluate(valid_dataset, [metric], transformers)
  scores = model.evaluate(valid_dataset, [metric], transformers)

  assert scores['mean-roc_auc_score'] == pytest.approx(
      ref_scores['mean-roc_auc_score'], 1e-5)
  assert model._openvino_model.is_available()


def test_torch_model():
  np.random.seed(123)

  def init(**kwargs):
    torch.manual_seed(1412)
    pytorch_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1000), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(1000, 1))
    return TorchModel(pytorch_model, dc.models.losses.L2Loss(), **kwargs)

  ref_model = init(batch_size=8)
  model = init(batch_size=8, use_openvino=True)

  dataset = NumpyDataset(np.random.standard_normal((100, 1024)))
  ref = ref_model.predict(dataset)
  out = model.predict(dataset)
  diff = np.abs(np.max(ref - out))

  assert diff < 1e-6
  assert model._openvino_model.is_available()

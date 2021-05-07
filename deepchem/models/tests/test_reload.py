"""
Test reload for trained models.
"""
import os
import pytest
import unittest
import tempfile
import numpy as np
import deepchem as dc
import tensorflow as tf
import scipy
from flaky import flaky
from sklearn.ensemble import RandomForestClassifier
from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS
from deepchem.feat import create_char_to_idx


def test_sklearn_classifier_reload():
  """Test that trained model can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  sklearn_model = RandomForestClassifier()
  model_dir = tempfile.mkdtemp()
  model = dc.models.SklearnModel(sklearn_model, model_dir)

  # Fit trained model
  model.fit(dataset)
  model.save()

  # Load trained model
  reloaded_model = dc.models.SklearnModel(None, model_dir)
  reloaded_model.reload()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_multitaskregressor_reload():
  """Test that MultitaskRegressor can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      learning_rate=0.003,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.MultitaskRegressor(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      learning_rate=0.003,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_multitaskclassification_reload():
  """Test that MultitaskClassifier can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.0003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Reload trained model
  reloaded_model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      dropouts=[0.],
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.0003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_residual_classification_reload():
  """Test that a residual network can reload correctly."""
  n_samples = 10
  n_features = 5
  n_tasks = 1
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=500)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload trained model
  reloaded_model = dc.models.MultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[20] * 10,
      dropouts=0.0,
      batch_size=n_samples,
      residual=True,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_robust_multitask_classification_reload():
  """Test robust multitask overfits tiny data."""
  n_tasks = 10
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=25)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reloaded Trained Model
  reloaded_model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_atomic_conv_model_reload():
  from deepchem.models.atomic_conv import AtomicConvModel
  from deepchem.data import NumpyDataset
  model_dir = tempfile.mkdtemp()
  batch_size = 1
  N_atoms = 5

  acm = AtomicConvModel(
      n_tasks=1,
      batch_size=batch_size,
      layer_sizes=[
          1,
      ],
      frag1_num_atoms=5,
      frag2_num_atoms=5,
      complex_num_atoms=10,
      model_dir=model_dir)

  features = []
  frag1_coords = np.random.rand(N_atoms, 3)
  frag1_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
  frag1_z = np.random.randint(10, size=(N_atoms))
  frag2_coords = np.random.rand(N_atoms, 3)
  frag2_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
  frag2_z = np.random.randint(10, size=(N_atoms))
  system_coords = np.random.rand(2 * N_atoms, 3)
  system_nbr_list = {
      0: [],
      1: [],
      2: [],
      3: [],
      4: [],
      5: [],
      6: [],
      7: [],
      8: [],
      9: []
  }
  system_z = np.random.randint(10, size=(2 * N_atoms))

  features.append(
      (frag1_coords, frag1_nbr_list, frag1_z, frag2_coords, frag2_nbr_list,
       frag2_z, system_coords, system_nbr_list, system_z))
  features = np.asarray(features)
  labels = np.random.rand(batch_size)
  dataset = NumpyDataset(features, labels)

  acm.fit(dataset, nb_epoch=1)

  reloaded_model = AtomicConvModel(
      n_tasks=1,
      batch_size=batch_size,
      layer_sizes=[
          1,
      ],
      frag1_num_atoms=5,
      frag2_num_atoms=5,
      complex_num_atoms=10,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  origpred = acm.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)


def test_normalizing_flow_model_reload():
  """Test that NormalizingFlowModel can be reloaded correctly."""
  from deepchem.models.normalizing_flows import NormalizingFlow, NormalizingFlowModel
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors
  tfk = tf.keras

  model_dir = tempfile.mkdtemp()

  Made = tfb.AutoregressiveNetwork(
      params=2, hidden_units=[512, 512], activation='relu', dtype='float64')

  flow_layers = [tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=Made)]
  # 3D Multivariate Gaussian base distribution
  nf = NormalizingFlow(
      base_distribution=tfd.MultivariateNormalDiag(
          loc=np.zeros(2), scale_diag=np.ones(2)),
      flow_layers=flow_layers)

  nfm = NormalizingFlowModel(nf, model_dir=model_dir)

  target_distribution = tfd.MultivariateNormalDiag(loc=np.array([1., 0.]))
  dataset = dc.data.NumpyDataset(X=target_distribution.sample(96))
  final = nfm.fit(dataset, nb_epoch=1)

  x = np.zeros(2)
  lp1 = nfm.flow.log_prob(x).numpy()

  assert nfm.flow.sample().numpy().shape == (2,)

  reloaded_model = NormalizingFlowModel(nf, model_dir=model_dir)
  reloaded_model.restore()

  # Check that reloaded model can sample from the distribution
  assert reloaded_model.flow.sample().numpy().shape == (2,)

  lp2 = reloaded_model.flow.log_prob(x).numpy()

  # Check that density estimation is same for reloaded model
  assert np.all(lp1 == lp2)


def test_robust_multitask_regressor_reload():
  """Test that RobustMultitaskRegressor can be reloaded correctly."""
  n_tasks = 10
  n_samples = 10
  n_features = 3

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_IRV_multitask_classification_reload():
  """Test IRV classifier can be reloaded."""
  n_tasks = 5
  n_samples = 10
  n_features = 128
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.randint(2, size=(n_samples, n_features))
  y = np.ones((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)
  IRV_transformer = dc.trans.IRVTransformer(5, n_tasks, dataset)
  dataset_trans = IRV_transformer.transform(dataset)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskIRVClassifier(
      n_tasks,
      K=5,
      learning_rate=0.01,
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset_trans)

  # Eval model on train
  scores = model.evaluate(dataset_trans, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload Trained Model
  reloaded_model = dc.models.MultitaskIRVClassifier(
      n_tasks,
      K=5,
      learning_rate=0.01,
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.random(dataset_trans.X.shape)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset_trans, [classification_metric])
  assert scores[classification_metric.name] > .9


@flaky
def test_progressive_classification_reload():
  """Test progressive multitask can reload."""
  np.random.seed(123)
  n_tasks = 5
  n_samples = 10
  n_features = 6

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model_dir = tempfile.mkdtemp()
  model = dc.models.ProgressiveMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=400)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9

  # Reload Trained Model
  reloaded_model = dc.models.ProgressiveMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_progressivemultitaskregressor_reload():
  """Test that ProgressiveMultitaskRegressor can be reloaded correctly."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.random.rand(n_samples, n_tasks)
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

  model_dir = tempfile.mkdtemp()
  model = dc.models.ProgressiveMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  # Reload trained model
  reloaded_model = dc.models.ProgressiveMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.001,
      weight_init_stddevs=[.1],
      alpha_init_stddevs=[.02],
      batch_size=n_samples,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_DAG_regression_reload():
  """Test DAG regressor reloads."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.ConvMolFeaturizer()
  tasks = ["outcome"]
  mols = ["CC", "CCO", "CC", "CCC", "CCCCO", "CO", "CC", "CCCCC", "CCC", "CCCO"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.rand(n_samples, n_tasks)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_feat = 75
  batch_size = 10
  transformer = dc.trans.DAGTransformer(max_atoms=50)
  dataset = transformer.transform(dataset)

  model_dir = tempfile.mkdtemp()
  model = dc.models.DAGModel(
      n_tasks,
      max_atoms=50,
      n_atom_feat=n_feat,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .1

  reloaded_model = dc.models.DAGModel(
      n_tasks,
      max_atoms=50,
      n_atom_feat=n_feat,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  predset = transformer.transform(predset)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)

  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .1


def test_weave_classification_reload():
  """Test weave model can be reloaded."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  tasks = ["outcome"]
  mols = ["CC", "CCCCC", "CCCCC", "CCC", "COOO", "COO", "OO"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = [1, 1, 1, 1, 0, 0, 0]
  dataset = dc.data.NumpyDataset(X, y)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  batch_size = 5

  model_dir = tempfile.mkdtemp()
  model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.01,
      mode="classification",
      dropouts=0.0,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)

  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)

  reloaded_model = dc.models.WeaveModel(
      n_tasks,
      batch_size=batch_size,
      learning_rate=0.003,
      mode="classification",
      dropouts=0.0,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  #Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6


def test_MPNN_regression_reload():
  """Test MPNN can reload datasets."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  # Load mini log-solubility dataset.
  featurizer = dc.feat.WeaveFeaturizer()
  tasks = ["outcome"]
  mols = ["C", "CO", "CC"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.rand(n_samples, n_tasks)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  n_atom_feat = 75
  n_pair_feat = 14
  batch_size = 10
  model_dir = tempfile.mkdtemp()
  model = dc.models.MPNNModel(
      n_tasks,
      n_atom_feat=n_atom_feat,
      n_pair_feat=n_pair_feat,
      T=2,
      M=3,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=50)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .8

  # Reload trained model
  reloaded_model = dc.models.MPNNModel(
      n_tasks,
      n_atom_feat=n_atom_feat,
      n_pair_feat=n_pair_feat,
      T=2,
      M=3,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)
  reloaded_model.restore()

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] > .8

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)


def test_textCNN_classification_reload():
  """Test textCNN model reloadinng."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  featurizer = dc.feat.RawFeaturizer()
  tasks = ["outcome"]
  mols = ["C", "CO", "CC"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.random.randint(2, size=(n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, ids=mols)

  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

  char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)
  batch_size = 3

  model_dir = tempfile.mkdtemp()
  model = dc.models.TextCNNModel(
      n_tasks,
      char_dict,
      seq_length=length,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="classification",
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .8

  # Reload trained model
  reloaded_model = dc.models.TextCNNModel(
      n_tasks,
      char_dict,
      seq_length=length,
      batch_size=batch_size,
      learning_rate=0.001,
      use_queue=False,
      mode="classification",
      model_dir=model_dir)
  reloaded_model.restore()

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .8

  assert len(reloaded_model.model.get_weights()) == len(
      model.model.get_weights())
  for (reloaded, orig) in zip(reloaded_model.model.get_weights(),
                              model.model.get_weights()):
    assert np.all(reloaded == orig)

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred, ids=predmols)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  assert len(model.model.layers) == len(reloaded_model.model.layers)


def test_1d_cnn_regression_reload():
  """Test that a 1D CNN can reload."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  np.random.seed(123)
  X = np.random.rand(n_samples, 10, n_features)
  y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
  dataset = dc.data.NumpyDataset(X, y)

  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  model_dir = tempfile.mkdtemp()

  model = dc.models.CNN(
      n_tasks,
      n_features,
      dims=1,
      dropouts=0,
      kernel_size=3,
      mode='regression',
      learning_rate=0.003,
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=200)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1

  # Reload trained model
  reloaded_model = dc.models.CNN(
      n_tasks,
      n_features,
      dims=1,
      dropouts=0,
      kernel_size=3,
      mode='regression',
      learning_rate=0.003,
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  Xpred = np.random.rand(n_samples, 10, n_features)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < 0.1


def test_graphconvmodel_reload():
  featurizer = dc.feat.ConvMolFeaturizer()
  tasks = ["outcome"]
  n_tasks = len(tasks)
  mols = ["C", "CO", "CC"]
  n_samples = len(mols)
  X = featurizer(mols)
  y = np.array([0, 1, 0])
  dataset = dc.data.NumpyDataset(X, y)

  classification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  batch_size = 10
  model_dir = tempfile.mkdtemp()
  model = dc.models.GraphConvModel(
      len(tasks),
      batch_size=batch_size,
      batch_normalize=False,
      mode='classification',
      model_dir=model_dir)

  model.fit(dataset, nb_epoch=10)
  scores = model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] >= 0.6

  # Reload trained Model
  reloaded_model = dc.models.GraphConvModel(
      len(tasks),
      batch_size=batch_size,
      batch_normalize=False,
      mode='classification',
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)

  # Eval model on train
  scores = reloaded_model.evaluate(dataset, [classification_metric])
  assert scores[classification_metric.name] > .6


def test_chemception_reload():
  """Test that chemception models can be saved and reloaded."""
  img_size = 80
  img_spec = "engd"
  res = 0.5
  n_tasks = 1
  featurizer = dc.feat.SmilesToImage(
      img_size=img_size, img_spec=img_spec, res=res)

  data_points = 10
  mols = ["CCCCCCCC"] * data_points
  X = featurizer(mols)

  y = np.random.randint(0, 2, size=(data_points, n_tasks))
  w = np.ones(shape=(data_points, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, mols)
  classsification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  model_dir = tempfile.mkdtemp()
  model = dc.models.ChemCeption(
      n_tasks=n_tasks,
      img_spec="engd",
      model_dir=model_dir,
      mode="classification")
  model.fit(dataset, nb_epoch=3)

  # Reload Trained Model
  reloaded_model = dc.models.ChemCeption(
      n_tasks=n_tasks,
      img_spec="engd",
      model_dir=model_dir,
      mode="classification")
  reloaded_model.restore()

  # Check predictions match on random sample
  predmols = ["CCCC", "CCCCCO", "CCCCC"]
  Xpred = featurizer(predmols)
  predset = dc.data.NumpyDataset(Xpred)
  origpred = model.predict(predset)
  reloadpred = reloaded_model.predict(predset)
  assert np.all(origpred == reloadpred)


# TODO: This test is a little awkward. The Smiles2Vec model awkwardly depends on a dataset_file being available on disk. This needs to be cleaned up to match the standard model handling API.
def test_smiles2vec_reload():
  """Test that smiles2vec models can be saved and reloaded."""
  dataset_file = os.path.join(os.path.dirname(__file__), "chembl_25_small.csv")
  max_len = 250
  pad_len = 10
  max_seq_len = 20
  char_to_idx = create_char_to_idx(
      dataset_file, max_len=max_len, smiles_field="smiles")
  feat = dc.feat.SmilesToSeq(
      char_to_idx=char_to_idx, max_len=max_len, pad_len=pad_len)

  n_tasks = 5
  data_points = 10

  loader = dc.data.CSVLoader(
      tasks=CHEMBL25_TASKS, smiles_field='smiles', featurizer=feat)
  dataset = loader.create_dataset(
      inputs=[dataset_file], shard_size=10000, data_dir=tempfile.mkdtemp())
  y = np.random.randint(0, 2, size=(data_points, n_tasks))
  w = np.ones(shape=(data_points, n_tasks))
  dataset = dc.data.NumpyDataset(dataset.X[:data_points, :max_seq_len], y, w,
                                 dataset.ids[:data_points])

  classsification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  model_dir = tempfile.mkdtemp()
  model = dc.models.Smiles2Vec(
      char_to_idx=char_to_idx,
      max_seq_len=max_seq_len,
      use_conv=True,
      n_tasks=n_tasks,
      model_dir=model_dir,
      mode="classification")
  model.fit(dataset, nb_epoch=3)

  # Reload Trained Model
  reloaded_model = dc.models.Smiles2Vec(
      char_to_idx=char_to_idx,
      max_seq_len=max_seq_len,
      use_conv=True,
      n_tasks=n_tasks,
      model_dir=model_dir,
      mode="classification")
  reloaded_model.restore()

  # Check predictions match on original dataset
  origpred = model.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)


# TODO: We need a cleaner usage example for this
def test_DTNN_regression_reload():
  """Test DTNN can reload datasets."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1

  current_dir = os.path.dirname(os.path.abspath(__file__))
  input_file = os.path.join(current_dir, "example_DTNN.mat")
  dataset = scipy.io.loadmat(input_file)
  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.NumpyDataset(X, y, w, ids=None)
  n_tasks = y.shape[1]

  regression_metric = dc.metrics.Metric(
      dc.metrics.pearson_r2_score, task_averager=np.mean)

  model_dir = tempfile.mkdtemp()
  model = dc.models.DTNNModel(
      n_tasks,
      n_embedding=20,
      n_distance=100,
      learning_rate=1.0,
      model_dir=model_dir,
      mode="regression")

  # Fit trained model
  model.fit(dataset, nb_epoch=250)

  # Eval model on train
  pred = model.predict(dataset)
  mean_rel_error = np.mean(np.abs(1 - pred / y))
  assert mean_rel_error < 0.2

  reloaded_model = dc.models.DTNNModel(
      n_tasks,
      n_embedding=20,
      n_distance=100,
      learning_rate=1.0,
      model_dir=model_dir,
      mode="regression")
  reloaded_model.restore()

  # Check predictions match on random sample
  origpred = model.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)


def generate_sequences(sequence_length, num_sequences):
  for i in range(num_sequences):
    seq = [
        np.random.randint(10)
        for x in range(np.random.randint(1, sequence_length + 1))
    ]
    yield (seq, seq)


def test_seq2seq_reload():
  """Test reloading for seq2seq models."""

  sequence_length = 8
  tokens = list(range(10))
  model_dir = tempfile.mkdtemp()
  s = dc.models.SeqToSeq(
      tokens,
      tokens,
      sequence_length,
      encoder_layers=2,
      decoder_layers=2,
      embedding_dimension=150,
      learning_rate=0.01,
      dropout=0.1,
      model_dir=model_dir)

  # Train the model on random sequences.  We aren't training long enough to
  # really make it reliable, but I want to keep this test fast, and it should
  # still be able to reproduce a reasonable fraction of input sequences.

  s.fit_sequences(generate_sequences(sequence_length, 25000))

  # Test it out.

  tests = [seq for seq, target in generate_sequences(sequence_length, 50)]
  pred1 = s.predict_from_sequences(tests, beam_width=1)
  pred4 = s.predict_from_sequences(tests, beam_width=4)

  reloaded_s = dc.models.SeqToSeq(
      tokens,
      tokens,
      sequence_length,
      encoder_layers=2,
      decoder_layers=2,
      embedding_dimension=150,
      learning_rate=0.01,
      dropout=0.1,
      model_dir=model_dir)
  reloaded_s.restore()

  reloaded_pred1 = reloaded_s.predict_from_sequences(tests, beam_width=1)
  assert len(pred1) == len(reloaded_pred1)
  for (p1, r1) in zip(pred1, reloaded_pred1):
    assert p1 == r1
  reloaded_pred4 = reloaded_s.predict_from_sequences(tests, beam_width=4)
  assert len(pred4) == len(reloaded_pred4)
  for (p4, r4) in zip(pred4, reloaded_pred4):
    assert p4 == r4
  embeddings = s.predict_embeddings(tests)
  pred1e = s.predict_from_embeddings(embeddings, beam_width=1)
  pred4e = s.predict_from_embeddings(embeddings, beam_width=4)

  reloaded_embeddings = reloaded_s.predict_embeddings(tests)
  reloaded_pred1e = reloaded_s.predict_from_embeddings(
      reloaded_embeddings, beam_width=1)
  reloaded_pred4e = reloaded_s.predict_from_embeddings(
      reloaded_embeddings, beam_width=4)

  assert np.all(embeddings == reloaded_embeddings)

  assert len(pred1e) == len(reloaded_pred1e)
  for (p1e, r1e) in zip(pred1e, reloaded_pred1e):
    assert p1e == r1e

  assert len(pred4e) == len(reloaded_pred4e)
  for (p4e, r4e) in zip(pred4e, reloaded_pred4e):
    assert p4e == r4e


def test_fittransform_regression_reload():
  """Test that MultitaskFitTransformRegressor can reload simple regression datasets."""
  n_samples = 10
  n_features = 3
  n_tasks = 1

  # Generate dummy dataset
  np.random.seed(123)
  tf.random.set_seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
  regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  model_dir = tempfile.mkdtemp()
  model = dc.models.MultitaskFitTransformRegressor(
      n_tasks, [n_features, n_features],
      dropouts=[0.01],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      fit_transformers=fit_transformers,
      n_evals=1,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)

  # Fit trained model
  model.fit(dataset, nb_epoch=100)

  # Eval model on train
  scores = model.evaluate(dataset, [regression_metric])
  assert scores[regression_metric.name] < .1

  reloaded_model = dc.models.MultitaskFitTransformRegressor(
      n_tasks, [n_features, n_features],
      dropouts=[0.01],
      weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
      batch_size=n_samples,
      fit_transformers=fit_transformers,
      n_evals=1,
      optimizer=dc.models.optimizers.Adam(
          learning_rate=0.003, beta1=0.9, beta2=0.999),
      model_dir=model_dir)
  reloaded_model.restore()

  # Check predictions match on random sample
  origpred = model.predict(dataset)
  reloadpred = reloaded_model.predict(dataset)
  assert np.all(origpred == reloadpred)

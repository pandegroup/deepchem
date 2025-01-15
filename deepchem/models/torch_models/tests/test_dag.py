import pytest
import torch
import numpy as np
import deepchem as dc
import tempfile
from deepchem.models.torch_models import DAGModel
from deepchem.trans import DAGTransformer
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.data import NumpyDataset


def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer, reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,
                                                        reload=False)

    train, valid, test = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@pytest.mark.torch
def test_dag_model_classification():
    tasks, dataset, _, _ = get_dataset('classification', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        mode='classification',
        batch_size=10,
    )
    model.fit(dataset, nb_epoch=1)
    _ = model.predict(dataset)


@pytest.mark.torch
def test_dag_regression_model():
    np.random.seed(1234)
    torch.manual_seed(1234)
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        mode='regression',
    )
    model.fit(dataset, nb_epoch=1)
    _ = model.predict(dataset)


@pytest.mark.torch
def test_dag_regression_model_uncertainty():
    np.random.seed(1234)
    torch.manual_seed(1234)
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    transformer = DAGTransformer(max_atoms=max_atoms)
    dataset = transformer.transform(dataset)

    model = DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        mode='regression',
        uncertainty=True,
        dropout=0.05,
    )
    model.fit(dataset, nb_epoch=1)
    _ = model.predict(dataset)


@pytest.mark.torch
def test_DAG_regression_reload():
    """Test DAG regressor reloads."""
    np.random.seed(123)
    n_tasks = 1

    # Load mini log-solubility dataset.
    featurizer = dc.feat.ConvMolFeaturizer()
    mols = [
        "CC", "CCO", "CC", "CCC", "CCCCO", "CO", "CC", "CCCCC", "CCC", "CCCO"
    ]
    n_samples = len(mols)
    X = featurizer(mols)
    y = np.random.rand(n_samples, n_tasks)
    dataset = dc.data.NumpyDataset(X, y)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    n_feat = 75
    batch_size = 10
    transformer = dc.trans.DAGTransformer(max_atoms=50)
    dataset = transformer.transform(dataset)

    model_dir = tempfile.mkdtemp()
    model = DAGModel(n_tasks,
                     max_atoms=50,
                     n_atom_feat=n_feat,
                     batch_size=batch_size,
                     learning_rate=0.001,
                     use_queue=False,
                     mode="regression",
                     model_dir=model_dir)

    # Fit trained model
    model.fit(dataset, nb_epoch=1)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])
    # assert scores[regression_metric.name] > .1

    reloaded_model = DAGModel(n_tasks,
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

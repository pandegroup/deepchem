"""
Tests to make sure deepchem models can fit models on easy datasets.
"""
import unittest

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)

import deepchem as dc


def test_sklearn_regression():
    """Test that sklearn models can learn on simple regression datasets."""
    np.random.seed(123)

    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.expand_dims(y, 1)
    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)

    sklearn_model = LinearRegression()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] > .5


def test_sklearn_transformed_regression():
    """Test that sklearn models can learn on simple transformed regression datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.expand_dims(y, 1)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    # Eval model on train
    transformers = [
        dc.trans.NormalizationTransformer(transform_X=True,
                                          dataset=train_dataset),
        dc.trans.ClippingTransformer(transform_X=True, dataset=train_dataset),
        dc.trans.NormalizationTransformer(transform_y=True,
                                          dataset=train_dataset)
    ]
    for data in [train_dataset, test_dataset]:
        for transformer in transformers:
            data = transformer.transform(data)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)
    sklearn_model = LinearRegression()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    train_scores = model.evaluate(train_dataset, [regression_metric],
                                  transformers)
    assert train_scores[regression_metric.name] > .5

    # Eval model on test
    test_scores = model.evaluate(test_dataset, [regression_metric],
                                 transformers)
    assert test_scores[regression_metric.name] > .5


def test_sklearn_multitask_regression():
    """Test that sklearn models can learn on simple multitask regression."""
    np.random.seed(123)
    n_tasks = 4
    tasks = range(n_tasks)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
    test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)

    def model_builder(model_dir):
        sklearn_model = LinearRegression()
        return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    score = scores[regression_metric.name]
    assert score > .5


def test_sklearn_classification():
    """Test that sklearn models can learn on simple classification datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_digits(n_class=2)
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    sklearn_model = LogisticRegression()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [classification_metric])
    assert scores[classification_metric.name] > .5


def test_sklearn_multitask_classification():
    """Test that sklearn models can learn on simple multitask classification."""
    np.random.seed(123)
    n_tasks = 4
    tasks = range(n_tasks)
    dataset = sklearn.datasets.load_digits(n_class=2)
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
    test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    def model_builder(model_dir):
        sklearn_model = LogisticRegression()
        return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(train_dataset)
    model.save()
    # Eval model on test
    scores = model.evaluate(test_dataset, [classification_metric])
    assert scores[classification_metric.name] > .5


def test_sklearn_fit_on_batch_regression():
    """Test that sklearn models can learn on simple regression datasets using fit_on_batch."""
    np.random.seed(123)
    X, y = sklearn.datasets.make_regression(n_samples=100,
                                            n_features=2,
                                            noise=1,
                                            random_state=123)
    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)

    sklearn_model = SGDRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit model using fit_on_batch
    batch_size = 64
    for x, y, w, _ in train_dataset.iterbatches(batch_size=batch_size,
                                                epochs=100):
        model.fit_on_batch(x, y, w)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] > .5


def test_sklearn_fit_on_batch_classification():
    """Test that sklearn models can learn on simple classification datasets using fit_on_batch."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    sklearn_model = SGDClassifier()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit model using fit_on_batch
    classes = np.unique(y)
    batch_size = 64
    for x, y, w, _ in train_dataset.iterbatches(batch_size=batch_size):
        model.fit_on_batch(x, y, w, classes=classes)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [classification_metric])
    assert scores[classification_metric.name] > .5


class TestSklearnErrors(unittest.TestCase):

    def test_sklearn_fit_on_batch_attribute_error(self):
        """Test that the appropriate error is raised when fit_on_batch() is called when using a model that does not implement partial_fit()"""
        with self.assertRaises(AttributeError):
            np.random.seed(123)
            # dummy values
            X = np.random.randn(10)
            y = np.ones(10)
            # linear regression has no "partial_fit"
            sklearn_model = LinearRegression()
            model = dc.models.SklearnModel(sklearn_model)
            model.fit_on_batch(X, y)

    def test_sklearn_fit_on_batch_value_error(self):
        """Test that the appropriate error is raised when fit_on_batch() is called without classes for classification model"""
        with self.assertRaises(ValueError):
            np.random.seed(123)
            # dummy values
            X = np.random.randn(10)
            y = np.ones(10)
            sklearn_model = SGDClassifier()
            model = dc.models.SklearnModel(sklearn_model)
            model.fit_on_batch(X, y)

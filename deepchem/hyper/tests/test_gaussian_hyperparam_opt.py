"""
Tests for Gaussian Process Hyperparameter Optimization.
These tests fails every so often. I think it's when the Gaussian
process optimizer doesn't find an optimal point. This is still a
valuable test suite so leaving it in despite the flakiness.
"""
import numpy as np
import sklearn
import sklearn.ensemble
import deepchem as dc
import unittest
import pytest
import tempfile
from flaky import flaky


class TestGaussianHyperparamOpt(unittest.TestCase):
    """
    Test Gaussian Hyperparameter Optimization.
    """

    def setUp(self):
        """Set up common resources."""

        def rf_model_builder(**model_params):
            rf_params = {
                k: v for (k, v) in model_params.items() if k != 'model_dir'
            }
            model_dir = model_params['model_dir']
            sklearn_model = sklearn.ensemble.RandomForestRegressor(**rf_params)
            return dc.models.SklearnModel(sklearn_model, model_dir)

        self.rf_model_builder = rf_model_builder
        self.train_dataset = dc.data.NumpyDataset(X=np.random.rand(50, 5),
                                                  y=np.random.rand(50, 1))
        self.valid_dataset = dc.data.NumpyDataset(X=np.random.rand(20, 5),
                                                  y=np.random.rand(20, 1))

    def test_rf_example(self):
        """Test a simple example of optimizing a RF model with a gaussian process."""

        optimizer = dc.hyper.GaussianProcessHyperparamOpt(self.rf_model_builder,
                                                          max_iter=2)
        params_dict = {"n_estimators": 10}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict, self.train_dataset, self.valid_dataset, metric)

        valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                          transformers)
        assert valid_score["pearson_r2_score"] == max(all_results.values())
        assert valid_score["pearson_r2_score"] > 0

    def test_rf_example_min(self):
        """Test a simple example of optimizing a RF model with a gaussian process looking for minimum score."""

        optimizer = dc.hyper.GaussianProcessHyperparamOpt(self.rf_model_builder,
                                                          max_iter=2)
        params_dict = {"n_estimators": 10}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict,
            self.train_dataset,
            self.valid_dataset,
            metric,
            transformers,
            use_max=False)

        valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                          transformers)
        assert valid_score["pearson_r2_score"] == min(all_results.values())
        assert valid_score["pearson_r2_score"] > 0

    def test_rf_with_logdir(self):
        """Test that using a logdir can work correctly."""
        optimizer = dc.hyper.GaussianProcessHyperparamOpt(self.rf_model_builder,
                                                          max_iter=2)
        params_dict = {"n_estimators": 10}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
        with tempfile.TemporaryDirectory() as tmpdirname:
            best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
                params_dict,
                self.train_dataset,
                self.valid_dataset,
                metric,
                transformers,
                logdir=tmpdirname)
        valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                          transformers)
        assert valid_score["pearson_r2_score"] == max(all_results.values())
        assert valid_score["pearson_r2_score"] > 0

    @flaky
    @pytest.mark.torch
    def test_multitask_example(self):
        """Test a simple example of optimizing a multitask model with a gaussian process search."""
        # Generate dummy dataset
        np.random.seed(123)
        train_dataset = dc.data.NumpyDataset(np.random.rand(10, 3),
                                             np.zeros((10, 2)), np.ones(
                                                 (10, 2)), np.arange(10))
        valid_dataset = dc.data.NumpyDataset(np.random.rand(5, 3),
                                             np.zeros((5, 2)), np.ones((5, 2)),
                                             np.arange(5))
        transformers = []

        optimizer = dc.hyper.GaussianProcessHyperparamOpt(
            lambda **params: dc.models.MultitaskRegressor(
                n_tasks=2,
                n_features=3,
                dropouts=[0.],
                weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
                learning_rate=0.003,
                **params),
            max_iter=1)

        params_dict = {"batch_size": 10}
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                   task_averager=np.mean)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict,
            train_dataset,
            valid_dataset,
            metric,
            transformers,
            use_max=False)

        valid_score = best_model.evaluate(valid_dataset, [metric], transformers)
        assert valid_score["mean-mean_squared_error"] == min(
            all_results.values())
        assert valid_score["mean-mean_squared_error"] > 0

    @flaky
    @pytest.mark.torch
    def test_multitask_example_different_search_range(self):
        """Test a simple example of optimizing a multitask model with a gaussian process search with per-parameter search range."""
        # Generate dummy dataset
        np.random.seed(123)
        train_dataset = dc.data.NumpyDataset(np.random.rand(10, 3),
                                             np.zeros((10, 2)), np.ones(
                                                 (10, 2)), np.arange(10))
        valid_dataset = dc.data.NumpyDataset(np.random.rand(5, 3),
                                             np.zeros((5, 2)), np.ones((5, 2)),
                                             np.arange(5))
        transformers = []

        # These are per-example multiplier
        search_range = {"learning_rate": 10, "batch_size": 4}
        optimizer = dc.hyper.GaussianProcessHyperparamOpt(
            lambda **params: dc.models.MultitaskRegressor(
                n_tasks=2,
                n_features=3,
                dropouts=[0.],
                weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
                **params),
            search_range=search_range,
            max_iter=2)

        params_dict = {"learning_rate": 0.003, "batch_size": 10}
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                   task_averager=np.mean)

        with tempfile.TemporaryDirectory() as tmpdirname:
            best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
                params_dict,
                train_dataset,
                valid_dataset,
                metric,
                transformers,
                logdir=tmpdirname,
                use_max=False)
            valid_score = best_model.evaluate(valid_dataset, [metric],
                                              transformers)
        # Test that 2 parameters were optimized
        for hp_str in all_results.keys():
            # Recall that the key is a string of the form _batch_size_39_learning_rate_0.01 for example
            assert "batch_size" in hp_str
            assert "learning_rate" in hp_str
        assert valid_score["mean-mean_squared_error"] == min(
            all_results.values())
        assert valid_score["mean-mean_squared_error"] > 0

    @flaky
    @pytest.mark.torch
    def test_multitask_example_nb_epoch(self):
        """Test a simple example of optimizing a multitask model with a gaussian process search with a different number of training epochs."""
        # Generate dummy dataset
        np.random.seed(123)
        train_dataset = dc.data.NumpyDataset(np.random.rand(10, 3),
                                             np.zeros((10, 2)), np.ones(
                                                 (10, 2)), np.arange(10))
        valid_dataset = dc.data.NumpyDataset(np.random.rand(5, 3),
                                             np.zeros((5, 2)), np.ones((5, 2)),
                                             np.arange(5))
        transformers = []

        optimizer = dc.hyper.GaussianProcessHyperparamOpt(
            lambda **params: dc.models.MultitaskRegressor(
                n_tasks=2,
                n_features=3,
                dropouts=[0.],
                weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
                learning_rate=0.003,
                **params),
            max_iter=1)

        params_dict = {"batch_size": 10}
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                   task_averager=np.mean)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict,
            train_dataset,
            valid_dataset,
            metric,
            transformers,
            nb_epoch=3,
            use_max=False)

        valid_score = best_model.evaluate(valid_dataset, [metric], transformers)
        assert valid_score["mean-mean_squared_error"] == min(
            all_results.values())
        assert valid_score["mean-mean_squared_error"] > 0

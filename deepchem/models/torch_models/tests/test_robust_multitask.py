import numpy as np
import pytest
import tempfile
import os
try:
    import torch
    import torch.nn as nn
    import deepchem as dc
    from deepchem.models.torch_models import RobustMultitask, RobustMultitaskClassifier, RobustMultitaskRegressor
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass

# Input parameters used in the tensorflow models prior to extracting the weights.
n_tasks_tf = 3
n_features_tf = 100
layer_sizes_tf = [512, 1024]


@pytest.mark.torch
def test_robust_multitask_construction():
    """Test that RobustMultiTask Model can be constructed without crash.
    """

    model = RobustMultitask(
        n_tasks=1,
        n_features=100,
        mode="regression",
        layer_sizes=[128, 256],
    )

    assert model is not None


@pytest.mark.torch
def test_robust_multitask_forward():
    n_tasks = n_tasks_tf
    n_features = n_features_tf
    layer_sizes = layer_sizes_tf

    torch_model = RobustMultitask(n_tasks=n_tasks,
                                  n_features=n_features,
                                  layer_sizes=layer_sizes,
                                  mode='classification')

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_classifier_weights.npz"))

    move_weights(torch_model, weights)
    input_x = weights['input']
    output = weights['output']

    torch_model.eval()  # Disable dropout for deterministic output
    torch_out = torch_model(torch.tensor(input_x).float())[0]
    assert np.allclose(output, torch_out.detach().numpy(),
                       atol=1e-4), "Predictions are not close"


@pytest.mark.torch
def test_robust_multitask_classifier_construction():
    """Test that RobustMultiTaskClassifier Model can be constructed without crash.
    """

    model = RobustMultitaskClassifier(
        n_tasks=1,
        n_features=100,
        layer_sizes=[128, 256],
        n_classes=2,
    )

    assert model is not None


@pytest.mark.torch
def test_robust_multitask_classification_forward():
    """Test that the forward pass of RobustMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = n_tasks_tf
    n_features = n_features_tf
    layer_sizes = layer_sizes_tf

    torch_model = RobustMultitaskClassifier(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=layer_sizes,
    )

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_classifier_weights.npz"))

    move_weights(torch_model, weights)

    input_x = weights["input"]
    output = weights["output"]

    # Inference using TorchModel's predict() method works with NumpyDataset only. Hence we need to convert our numpy arrays to NumpyDataset.
    y = np.random.randint(0, 2, size=(input_x.shape[0], n_tasks))
    w = np.ones((input_x.shape[0], n_tasks))
    ids = np.arange(input_x.shape[0])
    input_x = dc.data.NumpyDataset(input_x, y, w, ids)
    torch_out = torch_model.predict(input_x)
    assert np.allclose(output, torch_out,
                       atol=1e-1), "Predictions are not close"


def move_weights(torch_model, weights):
    """Porting weights from Tensorflow to PyTorch"""

    def to_torch_param(weights):
        """Convert numpy weights to torch parameters to be used as model weights"""
        weights = weights.T
        return nn.Parameter(torch.from_numpy(weights))

    torch_weights = {
        k: to_torch_param(v) for k, v in weights.items() if k != "output"
    }

    # Shared layers
    torch_model.shared_layers[0].weight = torch_weights["shared-layers-dense-w"]
    torch_model.shared_layers[0].bias = torch_weights["shared-layers-dense-b"]

    torch_model.shared_layers[3].weight = torch_weights[
        "shared-layers-dense_1-w"]
    torch_model.shared_layers[3].bias = torch_weights["shared-layers-dense_1-b"]

    # Bypass layers for each tasks
    for i in range(n_tasks_tf):
        torch_model.bypass_layers[i][0].weight = torch_weights[
            f"bypass-layers-dense_{2 + i * 2}-w"]
        torch_model.bypass_layers[i][0].bias = torch_weights[
            f"bypass-layers-dense_{2 + i * 2}-b"]

        torch_model.output_layers[i].weight = torch_weights[
            f"bypass-layers-dense_{3 + i * 2}-w"]
        torch_model.output_layers[i].bias = torch_weights[
            f"bypass-layers-dense_{3 + i * 2}-b"]


@pytest.mark.torch
def test_robust_multitask_classifier_overfit():
    """Test that the model can overfit simple classification datasets."""
    n_samples = 20
    n_features = 5
    n_tasks = 3

    # Generate dummy dataset
    np.random.seed(123)
    torch.manual_seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    model = RobustMultitaskClassifier(
        n_tasks,
        n_features,
        layer_sizes=[128, 256],
        dropouts=0.2,
        weight_init_stddevs=0.02,
        bias_init_consts=0.0,
    )

    model.fit(dataset, nb_epoch=300)

    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > 0.9, "Failed to overfit"


@pytest.mark.torch
def test_robust_multitask_classifier_reload():
    """Test that the model can be reloaded from disk."""

    n_samples = 60
    n_features = 5
    n_tasks = 3

    # Generate dummy dataset
    np.random.seed(123)
    torch.manual_seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=(X.shape[0], n_tasks))
    w = np.ones((X.shape[0], n_tasks))
    ids = np.arange(X.shape[0])
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    model_dir = tempfile.mkdtemp()

    orig_model = RobustMultitaskClassifier(n_tasks,
                                           n_features,
                                           layer_sizes=[128, 256],
                                           dropouts=0.2,
                                           weight_init_stddevs=0.02,
                                           bias_init_consts=0.0,
                                           model_dir=model_dir)

    orig_model.fit(dataset, nb_epoch=200)

    reloaded_model = RobustMultitaskClassifier(n_tasks,
                                               n_features,
                                               layer_sizes=[128, 256],
                                               dropouts=0.2,
                                               weight_init_stddevs=0.02,
                                               bias_init_consts=0.0,
                                               model_dir=model_dir)

    reloaded_model.restore()

    X_new = np.random.rand(n_samples, n_features)
    orig_preds = orig_model.predict_on_batch(X_new)
    reloaded_preds = reloaded_model.predict_on_batch(X_new)

    assert np.all(orig_preds == reloaded_preds), "Predictions are not the same"


def test_robust_multitask_regressor_construction():
    """Test that RobustMultiTaskRegressor Model can be constructed without crash.
    """

    model = RobustMultitaskRegressor(
        n_tasks=1,
        n_features=100,
        layer_sizes=[128, 256],
    )

    assert model is not None


@pytest.mark.torch
def test_robust_multitask_regression_forward():
    """Test that the forward pass of RobustMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = n_tasks_tf
    n_features = n_features_tf
    layer_sizes = layer_sizes_tf

    torch_model = RobustMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=layer_sizes,
    )

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_regressor_weights.npz"))

    move_weights(torch_model, weights)

    input_x = weights["input"]
    output = weights["output"]

    # Inference using TorchModel's predict() method works with NumpyDataset only. Hence we need to convert our numpy arrays to NumpyDataset.
    y = np.random.rand(input_x.shape[0], 1)
    w = np.ones((input_x.shape[0], 1))
    ids = np.arange(input_x.shape[0])
    input_x = dc.data.NumpyDataset(input_x, y, w, ids)

    torch_out = torch_model.predict(input_x)
    assert np.allclose(output, torch_out,
                       atol=1e-4), "Predictions are not close"


@pytest.mark.torch
def test_robust_multitask_regressor_overfit():
    """Test that the model can overfit simple regression datasets."""
    n_samples = 10
    n_features = 5
    n_tasks = 3

    np.random.seed(123)
    torch.manual_seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          task_averager=np.mean,
                                          mode="regression")

    model = RobustMultitaskRegressor(
        n_tasks,
        n_features,
        layer_sizes=[128, 256],
        dropouts=0.2,
        weight_init_stddevs=0.02,
        bias_init_consts=0.0,
    )

    model.fit(dataset, nb_epoch=300)

    scores = model.evaluate(dataset, [regression_metric])
    assert scores[regression_metric.name] < 0.05, "Failed to overfit"


@pytest.mark.torch
def test_robust_multitask_regressor_reload():
    """Test that the model can be reloaded from disk."""

    n_samples = 20
    n_features = 5
    n_tasks = 3

    # Generate dummy dataset
    np.random.seed(123)
    torch.manual_seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks).astype(np.float32)
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    model_dir = tempfile.mkdtemp()

    orig_model = RobustMultitaskRegressor(n_tasks,
                                          n_features,
                                          layer_sizes=[128, 256],
                                          dropouts=0.2,
                                          alpha_init_stddevs=0.02,
                                          weight_init_stddevs=0.02,
                                          bias_init_consts=0.0,
                                          model_dir=model_dir)
    orig_model.fit(dataset, nb_epoch=200)

    reloaded_model = RobustMultitaskRegressor(n_tasks,
                                              n_features,
                                              layer_sizes=[128, 256],
                                              dropouts=0.2,
                                              alpha_init_stddevs=0.02,
                                              weight_init_stddevs=0.02,
                                              bias_init_consts=0.0,
                                              model_dir=model_dir)

    reloaded_model.restore()

    X_new = np.random.rand(n_samples, n_features)
    orig_preds = orig_model.predict_on_batch(X_new)
    reloaded_preds = reloaded_model.predict_on_batch(X_new)

    assert np.all(orig_preds == reloaded_preds), "Predictions are not the same"

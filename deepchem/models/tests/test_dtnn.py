import os
import numpy as np
import pytest

from deepchem.data import SDFLoader
from deepchem.feat import CoulombMatrix

try:
    from deepchem.models.torch_models import DTNNModel
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def test_dtnn():
    """Tests DTNN for Shape and trainable parameter count.

    - Used dataset files: qm9_mini.sdf, qm9_mini.sdf.csv (A subset of qm9 dataset.)
    - Tasks selected are only of regression type."""
    import os
    import torch
    from deepchem.models.torch_models import dtnn
    from deepchem.data import SDFLoader
    from deepchem.feat import CoulombMatrix
    from deepchem.models.torch_models import dtnn
    # Get Data
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = os.path.join(model_dir, 'tests/assets/qm9_mini.sdf')
    TASKS = ["alpha", "homo"]
    loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    data = loader.create_dataset(dataset_file, shard_size=100)
    inputs = dtnn._compute_features_on_batch(data.X)
    atom_number, distance, atom_membership, distance_membership_i, distance_membership_j = inputs
    inputs = [
        torch.tensor(atom_number).to(torch.int64),
        torch.tensor(distance).to(torch.float32),
        torch.tensor(atom_membership).to(torch.int64),
        torch.tensor(distance_membership_i).to(torch.int64),
        torch.tensor(distance_membership_j).to(torch.int64)
    ]
    n_tasks = data.y.shape[0]
    model = dtnn.DTNN(n_tasks)

    pred = model(inputs)

    # Check Shape
    assert pred.shape == (21, 21)

    # Check number of parameters
    assert len(list(model.parameters())) == 17


@pytest.mark.torch
def test_dtnn_model():
    """Tests DTNN Model for Shape and prediction.

    - Used dataset files: qm9_mini.sdf, qm9_mini.sdf.csv (A subset of qm9 dataset.)
    - Tasks selected are only of regression type.

    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_dir, "assets/qm9_mini.sdf")
    TASKS = ["alpha", "homo"]
    loader = SDFLoader(tasks=TASKS, featurizer=CoulombMatrix(29), sanitize=True)
    data = loader.create_dataset(dataset_file, shard_size=100)

    model = DTNNModel(data.y.shape[1],
                      n_embedding=40,
                      n_distance=100,
                      learning_rate=0.8,
                      mode="regression")
    model.fit(data, nb_epoch=1000)

    # Eval model on train
    pred = model.predict(data)

    mean_rel_error = np.mean(np.abs(1 - pred / (data.y)))

    assert mean_rel_error < 0.5
    assert pred.shape == data.y.shape

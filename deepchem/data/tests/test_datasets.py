"""
Tests for dataset creation
"""
import random
import math
import unittest
import os
import numpy as np
import pytest
import deepchem as dc
import pandas as pd
import tempfile

try:
    import torch  # noqa
    PYTORCH_IMPORT_FAILED = False
except ImportError:
    PYTORCH_IMPORT_FAILED = True


def load_solubility_data():
    """Loads solubility dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    return loader.create_dataset(input_file)


def load_multitask_data():
    """Load example multitask data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = [
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ]
    input_file = os.path.join(
        current_dir, "../../models/tests/assets/multitask_example.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    return loader.create_dataset(input_file)


class TestTransformer(dc.trans.Transformer):

    def transform_array(self, X, y, w, ids):
        return (2 * X, 1.5 * y, w, ids)


def test_transform_disk():
    """Test that the transform() method works for DiskDatasets."""
    dataset = load_solubility_data()
    X = dataset.X
    y = dataset.y
    w = dataset.w
    ids = dataset.ids

    # Transform it

    transformer = TestTransformer(transform_X=True, transform_y=True)
    for parallel in (True, False):
        transformed = dataset.transform(transformer, parallel=parallel)
        np.testing.assert_array_equal(X, dataset.X)
        np.testing.assert_array_equal(y, dataset.y)
        np.testing.assert_array_equal(w, dataset.w)
        np.testing.assert_array_equal(ids, dataset.ids)
        np.testing.assert_array_equal(2 * X, transformed.X)
        np.testing.assert_array_equal(1.5 * y, transformed.y)
        np.testing.assert_array_equal(w, transformed.w)
        np.testing.assert_array_equal(ids, transformed.ids)


def test_sparsify_and_densify():
    """Test that sparsify and densify work as inverses."""
    # Test on identity matrix
    num_samples = 10
    num_features = num_samples
    X = np.eye(num_samples)
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Generate random sparse features dataset
    np.random.seed(123)
    p = .05
    X = np.random.binomial(1, p, size=(num_samples, num_features))
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)

    # Test edge case with array of all zeros
    X = np.zeros((num_samples, num_features))
    X_sparse = dc.data.sparsify_features(X)
    X_reconstructed = dc.data.densify_features(X_sparse, num_features)
    np.testing.assert_array_equal(X, X_reconstructed)


def test_pad_features():
    """Test that pad_features pads features correctly."""
    batch_size = 100
    num_features = 10

    # Test cases where n_samples < 2*n_samples < batch_size
    n_samples = 29
    X_b = np.zeros((n_samples, num_features))

    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100
    X_b = np.zeros((n_samples, num_features))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    X_out = dc.data.pad_features(batch_size, X_b)
    assert len(X_out) == batch_size


def test_pad_batches():
    """Test that pad_batch pads batches correctly."""
    batch_size = 100
    num_features = 10
    num_tasks = 5

    # Test cases where n_samples < 2*n_samples < batch_size
    n_samples = 29
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test cases where n_samples < batch_size
    n_samples = 79
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case where n_samples == batch_size
    n_samples = 100
    X_b = np.zeros((n_samples, num_features))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for object featurization.
    n_samples = 2
    X_b = np.array([{"a": 1}, {"b": 2}])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case for more complicated object featurization
    n_samples = 2
    X_b = np.array([(1, {"a": 1}), (2, {"b": 2})])
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))
    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size

    # Test case with multidimensional data
    n_samples = 50
    num_atoms = 15
    d = 3
    X_b = np.zeros((n_samples, num_atoms, d))
    y_b = np.zeros((n_samples, num_tasks))
    w_b = np.zeros((n_samples, num_tasks))
    ids_b = np.zeros((n_samples,))

    X_out, y_out, w_out, ids_out = dc.data.pad_batch(batch_size, X_b, y_b, w_b,
                                                     ids_b)
    assert len(X_out) == len(y_out) == len(w_out) == len(ids_out) == batch_size


def test_get_task_names():
    """Test that get_task_names returns correct task_names"""
    solubility_dataset = load_solubility_data()
    assert solubility_dataset.get_task_names() == ["log-solubility"]

    multitask_dataset = load_multitask_data()
    assert sorted(multitask_dataset.get_task_names()) == sorted([
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ])


def test_get_data_shape():
    """Test that get_data_shape returns currect data shape"""
    solubility_dataset = load_solubility_data()
    assert solubility_dataset.get_data_shape() == (1024,)

    multitask_dataset = load_multitask_data()
    assert multitask_dataset.get_data_shape() == (1024,)


def test_len():
    """Test that len(dataset) works."""
    solubility_dataset = load_solubility_data()
    assert len(solubility_dataset) == 10


def test_reshard():
    """Test that resharding the dataset works."""
    solubility_dataset = load_solubility_data()
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    assert solubility_dataset.get_number_shards() == 1
    solubility_dataset.reshard(shard_size=1)
    assert solubility_dataset.get_shard_size() == 1
    X_r, y_r, w_r, ids_r = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    assert solubility_dataset.get_number_shards() == 10
    solubility_dataset.reshard(shard_size=10)
    assert solubility_dataset.get_shard_size() == 10
    X_rr, y_rr, w_rr, ids_rr = (solubility_dataset.X, solubility_dataset.y,
                                solubility_dataset.w, solubility_dataset.ids)

    # Test first resharding worked
    np.testing.assert_array_equal(X, X_r)
    np.testing.assert_array_equal(y, y_r)
    np.testing.assert_array_equal(w, w_r)
    np.testing.assert_array_equal(ids, ids_r)

    # Test second resharding worked
    np.testing.assert_array_equal(X, X_rr)
    np.testing.assert_array_equal(y, y_rr)
    np.testing.assert_array_equal(w, w_rr)
    np.testing.assert_array_equal(ids, ids_rr)


def test_complete_shuffle():
    shard_sizes = [1, 2, 3, 4, 5]

    all_Xs, all_ys, all_ws, all_ids = [], [], [], []

    def shard_generator():
        for sz in shard_sizes:
            X_b = np.random.rand(sz, 1)
            y_b = np.random.rand(sz, 1)
            w_b = np.random.rand(sz, 1)
            ids_b = np.random.rand(sz)

            all_Xs.append(X_b)
            all_ys.append(y_b)
            all_ws.append(w_b)
            all_ids.append(ids_b)

            yield X_b, y_b, w_b, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    res = dataset.complete_shuffle()

    # approx 1/15! chance of equality
    np.testing.assert_equal(np.any(np.not_equal(dataset.X, res.X)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.y, res.w)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.w, res.y)), True)
    np.testing.assert_equal(np.any(np.not_equal(dataset.ids, res.ids)), True)

    np.testing.assert_array_equal(np.sort(dataset.X, axis=0),
                                  np.sort(res.X, axis=0))
    np.testing.assert_array_equal(np.sort(dataset.y, axis=0),
                                  np.sort(res.y, axis=0))
    np.testing.assert_array_equal(np.sort(dataset.w, axis=0),
                                  np.sort(res.w, axis=0))
    np.testing.assert_array_equal(np.sort(dataset.ids), np.sort(res.ids))


def test_iterbatches():
    """Test that iterating over batches of data works."""
    solubility_dataset = load_solubility_data()
    batch_size = 2
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    for (X_b, y_b, w_b, ids_b) in solubility_dataset.iterbatches(batch_size):
        assert X_b.shape == (batch_size,) + data_shape
        assert y_b.shape == (batch_size,) + (len(tasks),)
        assert w_b.shape == (batch_size,) + (len(tasks),)
        assert ids_b.shape == (batch_size,)


def test_itersamples_numpy():
    """Test that iterating over samples in a NumpyDataset works."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10
    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    for i, (sx, sy, sw, sid) in enumerate(dataset.itersamples()):
        np.testing.assert_array_equal(sx, X[i])
        np.testing.assert_array_equal(sy, y[i])
        np.testing.assert_array_equal(sw, w[i])
        np.testing.assert_array_equal(sid, ids[i])


def test_itersamples_disk():
    """Test that iterating over samples in a DiskDataset works."""
    solubility_dataset = load_solubility_data()
    X = solubility_dataset.X
    y = solubility_dataset.y
    w = solubility_dataset.w
    ids = solubility_dataset.ids
    for i, (sx, sy, sw, sid) in enumerate(solubility_dataset.itersamples()):
        np.testing.assert_array_equal(sx, X[i])
        np.testing.assert_array_equal(sy, y[i])
        np.testing.assert_array_equal(sw, w[i])
        np.testing.assert_array_equal(sid, ids[i])


def test_transform_numpy():
    """Test that the transform() method works for NumpyDatasets."""
    num_datapoints = 100
    num_features = 10
    num_tasks = 10

    # Generate data
    X = np.random.rand(num_datapoints, num_features)
    y = np.random.randint(2, size=(num_datapoints, num_tasks))
    w = np.random.randint(2, size=(num_datapoints, num_tasks))
    ids = np.array(["id"] * num_datapoints)
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    # Transform it

    transformer = TestTransformer(transform_X=True, transform_y=True)
    transformed = dataset.transform(transformer)
    np.testing.assert_array_equal(X, dataset.X)
    np.testing.assert_array_equal(y, dataset.y)
    np.testing.assert_array_equal(w, dataset.w)
    np.testing.assert_array_equal(ids, dataset.ids)
    np.testing.assert_array_equal(2 * X, transformed.X)
    np.testing.assert_array_equal(1.5 * y, transformed.y)
    np.testing.assert_array_equal(w, transformed.w)
    np.testing.assert_array_equal(ids, transformed.ids)


def test_to_numpy():
    """Test that transformation to numpy arrays is sensible."""
    solubility_dataset = load_solubility_data()
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    N_samples = len(solubility_dataset)
    N_tasks = len(tasks)

    assert X.shape == (N_samples,) + data_shape
    assert y.shape == (N_samples, N_tasks)
    assert w.shape == (N_samples, N_tasks)
    assert ids.shape == (N_samples,)


def test_consistent_ordering():
    """Test that ordering of labels is consistent over time."""
    solubility_dataset = load_solubility_data()

    ids1 = solubility_dataset.ids
    ids2 = solubility_dataset.ids

    assert np.array_equal(ids1, ids2)


def test_get_statistics():
    """Test statistics computation of this dataset."""
    solubility_dataset = load_solubility_data()
    X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                  solubility_dataset.w, solubility_dataset.ids)
    X_means, y_means = np.mean(X, axis=0), np.mean(y, axis=0)
    X_stds, y_stds = np.std(X, axis=0), np.std(y, axis=0)
    comp_X_means, comp_X_stds, comp_y_means, comp_y_stds = \
      solubility_dataset.get_statistics()
    np.testing.assert_allclose(comp_X_means, X_means)
    np.testing.assert_allclose(comp_y_means, y_means)
    np.testing.assert_allclose(comp_X_stds, X_stds)
    np.testing.assert_allclose(comp_y_stds, y_stds)


def test_disk_iterate_batch_size():
    solubility_dataset = load_solubility_data()
    X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                  solubility_dataset.w, solubility_dataset.ids)
    batch_sizes = []
    for X, y, _, _ in solubility_dataset.iterbatches(3,
                                                     epochs=2,
                                                     pad_batches=False,
                                                     deterministic=True):
        batch_sizes.append(len(X))
    assert [3, 3, 3, 1, 3, 3, 3, 1] == batch_sizes


def test_disk_pad_batches():
    shard_sizes = [21, 11, 41, 21, 51]
    batch_size = 10

    all_Xs, all_ys, all_ws, all_ids = [], [], [], []

    def shard_generator():
        for sz in shard_sizes:
            X_b = np.random.rand(sz, 1)
            y_b = np.random.rand(sz, 1)
            w_b = np.random.rand(sz, 1)
            ids_b = np.random.rand(sz)

            all_Xs.append(X_b)
            all_ys.append(y_b)
            all_ws.append(w_b)
            all_ids.append(ids_b)

            yield X_b, y_b, w_b, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_ys = np.concatenate(all_ys, axis=0)
    all_ws = np.concatenate(all_ws, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    test_Xs, test_ys, test_ws, test_ids = [], [], [], []
    for bidx, (a, b, c, d) in enumerate(
            dataset.iterbatches(batch_size=batch_size,
                                pad_batches=True,
                                deterministic=True)):

        test_Xs.append(a)
        test_ys.append(b)
        test_ws.append(c)
        test_ids.append(d)

    test_Xs = np.concatenate(test_Xs, axis=0)
    test_ys = np.concatenate(test_ys, axis=0)
    test_ws = np.concatenate(test_ws, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    total_size = sum(shard_sizes)

    assert bidx == math.ceil(total_size / batch_size) - 1

    expected_batches = math.ceil(total_size / batch_size) * batch_size

    assert len(test_Xs) == expected_batches
    assert len(test_ys) == expected_batches
    assert len(test_ws) == expected_batches
    assert len(test_ids) == expected_batches

    np.testing.assert_array_equal(all_Xs, test_Xs[:total_size, :])
    np.testing.assert_array_equal(all_ys, test_ys[:total_size, :])
    np.testing.assert_array_equal(all_ws, test_ws[:total_size, :])
    np.testing.assert_array_equal(all_ids, test_ids[:total_size])


def test_disk_iterate_y_w_None():
    shard_sizes = [21, 11, 41, 21, 51]
    batch_size = 10

    all_Xs, all_ids = [], []

    def shard_generator():
        for sz in shard_sizes:
            X_b = np.random.rand(sz, 1)
            ids_b = np.random.rand(sz)

            all_Xs.append(X_b)
            all_ids.append(ids_b)

            yield X_b, None, None, ids_b

    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    test_Xs, test_ids = [], []
    for bidx, (a, _, _, d) in enumerate(
            dataset.iterbatches(batch_size=batch_size,
                                pad_batches=True,
                                deterministic=True)):

        test_Xs.append(a)
        test_ids.append(d)

    test_Xs = np.concatenate(test_Xs, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    total_size = sum(shard_sizes)

    assert bidx == math.ceil(total_size / batch_size) - 1

    expected_batches = math.ceil(total_size / batch_size) * batch_size

    assert len(test_Xs) == expected_batches
    assert len(test_ids) == expected_batches

    np.testing.assert_array_equal(all_Xs, test_Xs[:total_size, :])
    np.testing.assert_array_equal(all_ids, test_ids[:total_size])


def test_disk_iterate_batch():

    all_batch_sizes = [None, 32, 17, 11]
    all_shard_sizes = [[7, 3, 12, 4, 5], [1, 1, 1, 1, 1], [31, 31, 31, 31, 31],
                       [21, 11, 41, 21, 51]]

    for idx in range(25):
        shard_length = random.randint(1, 32)
        shard_sizes = []
        for _ in range(shard_length):
            shard_sizes.append(random.randint(1, 128))
        all_shard_sizes.append(shard_sizes)
        if idx == 0:
            # special case to test
            all_batch_sizes.append(None)
        else:
            all_batch_sizes.append(random.randint(1, 256))

    for shard_sizes, batch_size in zip(all_shard_sizes, all_batch_sizes):

        all_Xs, all_ys, all_ws, all_ids = [], [], [], []

        def shard_generator():
            for sz in shard_sizes:
                X_b = np.random.rand(sz, 1)
                y_b = np.random.rand(sz, 1)
                w_b = np.random.rand(sz, 1)
                ids_b = np.random.rand(sz)

                all_Xs.append(X_b)
                all_ys.append(y_b)
                all_ws.append(w_b)
                all_ids.append(ids_b)

                yield X_b, y_b, w_b, ids_b

        dataset = dc.data.DiskDataset.create_dataset(shard_generator())

        all_Xs = np.concatenate(all_Xs, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        all_ws = np.concatenate(all_ws, axis=0)
        all_ids = np.concatenate(all_ids, axis=0)

        total_size = sum(shard_sizes)

        assert dataset.X.shape[0] == total_size

        # deterministic
        test_Xs, test_ys, test_ws, test_ids = [], [], [], []
        for bidx, (a, b, c, d) in enumerate(
                dataset.iterbatches(batch_size=batch_size,
                                    pad_batches=False,
                                    deterministic=True)):

            test_Xs.append(a)
            test_ys.append(b)
            test_ws.append(c)
            test_ids.append(d)

        if batch_size is None:
            for idx, (tx, ty, tw, tids) in enumerate(
                    zip(test_Xs, test_ys, test_ws, test_ids)):
                assert len(tx) == shard_sizes[idx]
                assert len(ty) == shard_sizes[idx]
                assert len(tw) == shard_sizes[idx]
                assert len(tids) == shard_sizes[idx]

        test_Xs = np.concatenate(test_Xs, axis=0)
        test_ys = np.concatenate(test_ys, axis=0)
        test_ws = np.concatenate(test_ws, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)

        if batch_size is None:
            assert bidx == len(shard_sizes) - 1
        else:
            assert bidx == math.ceil(total_size / batch_size) - 1

        np.testing.assert_array_equal(all_Xs, test_Xs)
        np.testing.assert_array_equal(all_ys, test_ys)
        np.testing.assert_array_equal(all_ws, test_ws)
        np.testing.assert_array_equal(all_ids, test_ids)

        # non-deterministic
        test_Xs, test_ys, test_ws, test_ids = [], [], [], []

        for bidx, (a, b, c, d) in enumerate(
                dataset.iterbatches(batch_size=batch_size,
                                    pad_batches=False,
                                    deterministic=False)):

            test_Xs.append(a)
            test_ys.append(b)
            test_ws.append(c)
            test_ids.append(d)

        # we don't know the order in which the shards are iterated in.
        test_Xs = np.concatenate(test_Xs, axis=0)
        test_ys = np.concatenate(test_ys, axis=0)
        test_ws = np.concatenate(test_ws, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)

        if batch_size is None:
            assert bidx == len(shard_sizes) - 1
        else:
            assert bidx == math.ceil(total_size / batch_size) - 1

        np.testing.assert_array_equal(np.sort(all_Xs, axis=0),
                                      np.sort(test_Xs, axis=0))
        np.testing.assert_array_equal(np.sort(all_ys, axis=0),
                                      np.sort(test_ys, axis=0))
        np.testing.assert_array_equal(np.sort(all_ws, axis=0),
                                      np.sort(test_ws, axis=0))
        np.testing.assert_array_equal(np.sort(all_ids, axis=0),
                                      np.sort(test_ids, axis=0))


def test_merge():
    """Test that dataset merge works."""
    num_datapoints = 10
    num_features = 10
    num_tasks = 1
    num_datasets = 4
    datasets = []
    for i in range(num_datasets):
        Xi = np.random.rand(num_datapoints, num_features)
        yi = np.random.randint(2, size=(num_datapoints, num_tasks))
        wi = np.ones((num_datapoints, num_tasks))
        idsi = np.array(["id"] * num_datapoints)
        dataseti = dc.data.DiskDataset.from_numpy(Xi, yi, wi, idsi)
        datasets.append(dataseti)

    new_data = dc.data.datasets.DiskDataset.merge(datasets)
    x = [1, 2, 3]
    X = np.array(x)
    y = np.array(x)
    a = dc.data.NumpyDataset(X, y)
    b = dc.data.NumpyDataset(X, y)
    c = dc.data.NumpyDataset.merge([a, b])
    assert c.y.shape == (
        6, 1)  # test to check if merge works when y is one dimensional
    # Check that we have all the data in
    assert new_data.X.shape == (num_datapoints * num_datasets, num_features)
    assert new_data.y.shape == (num_datapoints * num_datasets, num_tasks)
    assert len(new_data.tasks) == len(datasets[0].tasks)


def test_disk_cache():
    """Test that disk cache works."""
    shard_sizes = [10, 20, 30, 40, 50, 60]  # 6 shards total
    num_features = 10
    num_tasks = 4

    def shard_generator():
        for sz in shard_sizes:
            X_b = np.random.rand(sz, num_features)
            y_b = np.random.rand(sz, num_tasks)
            w_b = np.random.rand(sz, num_tasks)
            ids_b = np.random.rand(sz)

            yield X_b, y_b, w_b, ids_b

    # Tests with cache_data = True
    # By default the cache_data is True, so the dataset will cache the data
    dataset = dc.data.DiskDataset.create_dataset(shard_generator())

    # Test that the cache is initially empty
    assert dataset._cached_shards is None

    shards_to_test = 3

    # Test that the cache is populated
    for i, (X, y, w, ids) in enumerate(dataset.itershards()):
        assert np.array_equal(X, dataset._cached_shards[i].X)
        assert np.array_equal(y, dataset._cached_shards[i].y)
        assert np.array_equal(w, dataset._cached_shards[i].w)
        assert np.array_equal(ids, dataset._cached_shards[i].ids)

        # we are only testing for the first 3 shards
        # this will work only if the first three shards are small enough to fit in the cache
        if i + 1 == shards_to_test:
            break

    # Test the memory cache size
    memory_used = 0
    for i, (X, y, w, ids) in enumerate(dataset.itershards()):
        memory_used += X.nbytes + y.nbytes + w.nbytes + ids.nbytes

        if i + 1 == shards_to_test:
            break

    assert dataset._cache_used == memory_used

    # Test that the cache is cleared
    dataset.cache_data = False
    assert dataset._cached_shards is None

    # Tests with cache_data = False
    dataset2 = dc.data.DiskDataset(dataset.data_dir, cache_data=False)

    # Test that the cache is initially empty
    assert dataset2._cached_shards is None

    # Test that the cache is not populated
    for i, (X, y, w, ids) in enumerate(dataset2.itershards()):
        assert dataset2._cached_shards[i] is None


@pytest.mark.tensorflow
def test_make_tf_dataset():
    """Test creating a Tensorflow Iterator from a Dataset."""
    X = np.random.random((100, 5))
    y = np.random.random((100, 1))
    dataset = dc.data.NumpyDataset(X, y)
    iterator = dataset.make_tf_dataset(batch_size=10,
                                       epochs=2,
                                       deterministic=True)
    for i, (batch_X, batch_y, batch_w) in enumerate(iterator):
        offset = (i % 10) * 10
        np.testing.assert_array_equal(X[offset:offset + 10, :], batch_X)
        np.testing.assert_array_equal(y[offset:offset + 10, :], batch_y)
        np.testing.assert_array_equal(np.ones((10, 1)), batch_w)
    assert i == 19


@pytest.mark.torch
def _validate_pytorch_dataset(dataset):
    X = dataset.X
    y = dataset.y
    w = dataset.w
    ids = dataset.ids
    n_samples = X.shape[0]

    # Test iterating in order.

    ds = dataset.make_pytorch_dataset(epochs=2, deterministic=True)
    for i, (iter_X, iter_y, iter_w, iter_id) in enumerate(ds):
        j = i % n_samples
        np.testing.assert_array_equal(X[j, :], iter_X)
        np.testing.assert_array_equal(y[j, :], iter_y)
        np.testing.assert_array_equal(w[j, :], iter_w)
        assert ids[j] == iter_id
    assert i == 2 * n_samples - 1

    # Test iterating out of order.

    ds = dataset.make_pytorch_dataset(epochs=2, deterministic=False)
    id_to_index = dict((id, i) for i, id in enumerate(ids))
    id_count = dict((id, 0) for id in ids)
    for iter_X, iter_y, iter_w, iter_id in ds:
        j = id_to_index[iter_id]
        np.testing.assert_array_equal(X[j, :], iter_X)
        np.testing.assert_array_equal(y[j, :], iter_y)
        np.testing.assert_array_equal(w[j, :], iter_w)
        id_count[iter_id] += 1
    assert all(id_count[id] == 2 for id in ids)

    # Test iterating in batches.

    ds = dataset.make_pytorch_dataset(epochs=2,
                                      deterministic=False,
                                      batch_size=7)
    id_to_index = dict((id, i) for i, id in enumerate(ids))
    id_count = dict((id, 0) for id in ids)
    for iter_X, iter_y, iter_w, iter_id in ds:
        size = len(iter_id)
        assert size <= 7
        for i in range(size):
            j = id_to_index[iter_id[i]]
            np.testing.assert_array_equal(X[j, :], iter_X[i])
            np.testing.assert_array_equal(y[j, :], iter_y[i])
            np.testing.assert_array_equal(w[j, :], iter_w[i])
            id_count[iter_id[i]] += 1
    assert all(id_count[id] == 2 for id in ids)

    # Test iterating with multiple workers.

    import torch  # noqa
    ds = dataset.make_pytorch_dataset(epochs=2, deterministic=False)
    loader = torch.utils.data.DataLoader(ds, num_workers=3)
    id_count = dict((id, 0) for id in ids)
    for iter_X, iter_y, iter_w, iter_id in loader:
        j = id_to_index[iter_id[0]]
        np.testing.assert_array_equal(X[j, :], iter_X[0])
        np.testing.assert_array_equal(y[j, :], iter_y[0])
        np.testing.assert_array_equal(w[j, :], iter_w[0])
        id_count[iter_id[0]] += 1
    assert all(id_count[id] == 2 for id in ids)


def test_dataframe():
    """Test converting between Datasets and DataFrames."""
    dataset = load_solubility_data()

    # A round trip from Dataset to DataFrame to Dataset should produce identical arrays.

    df = dataset.to_dataframe()
    dataset2 = dc.data.Dataset.from_dataframe(df)
    np.testing.assert_array_equal(dataset.X, dataset2.X)
    np.testing.assert_array_equal(dataset.y, dataset2.y)
    np.testing.assert_array_equal(dataset.w, dataset2.w)
    np.testing.assert_array_equal(dataset.ids, dataset2.ids)

    # Try specifying particular columns.

    dataset3 = dc.data.Dataset.from_dataframe(df,
                                              X=['X2', 'X4'],
                                              y='w',
                                              w=['y', 'X1'])
    np.testing.assert_array_equal(dataset.X[:, (1, 3)], dataset3.X)
    np.testing.assert_array_equal(dataset.w, dataset3.y)
    np.testing.assert_array_equal(
        np.stack([dataset.y[:, 0], dataset.X[:, 0]], axis=1), dataset3.w)


def test_to_csv():
    """Test converting between Dataset and CSV"""
    tmpdir = tempfile.TemporaryDirectory()
    csv_path = os.path.join(tmpdir.name, 'tmp.csv')

    dataset = load_solubility_data()
    dataset.reshard(2)
    dataset.to_csv(csv_path)

    df = pd.read_csv(csv_path)
    dataset2 = dc.data.Dataset.from_dataframe(df)
    np.testing.assert_array_equal(dataset.X, dataset2.X)
    np.testing.assert_array_equal(dataset.y, dataset2.y)
    np.testing.assert_array_equal(dataset.w, dataset2.w)
    np.testing.assert_array_equal(dataset.ids, dataset2.ids)

    # Try specifying particular columns
    dataset3 = dc.data.Dataset.from_dataframe(df,
                                              X=['X2', 'X4'],
                                              y='w',
                                              w=['y', 'X1'])
    np.testing.assert_array_equal(dataset.X[:, (1, 3)], dataset3.X)
    np.testing.assert_array_equal(dataset.w, dataset3.y)
    np.testing.assert_array_equal(
        np.stack([dataset.y[:, 0], dataset.X[:, 0]], axis=1), dataset3.w)

    x = np.array([np.random.randn(2, 3), np.random.randn(2, 3)])
    dataset = dc.data.NumpyDataset(X=x, y=np.array([[1], [2]]))
    with pytest.raises(AssertionError):
        dataset.to_csv(csv_path)


def test_to_str():
    """Tests to string representation of Dataset."""
    dataset = dc.data.NumpyDataset(X=np.random.rand(5, 3),
                                   y=np.random.rand(5,),
                                   ids=np.arange(5))
    ref_str = '<NumpyDataset X.shape: (5, 3), y.shape: (5,), w.shape: (5,), ids: [0 1 2 3 4], task_names: [0]>'
    assert str(dataset) == ref_str

    # Test id shrinkage
    dc.utils.set_print_threshold(10)
    dataset = dc.data.NumpyDataset(X=np.random.rand(50, 3),
                                   y=np.random.rand(50,),
                                   ids=np.arange(50))
    ref_str = '<NumpyDataset X.shape: (50, 3), y.shape: (50,), w.shape: (50,), ids: [0 1 2 ... 47 48 49], task_names: [0]>'
    assert str(dataset) == ref_str

    # Test task shrinkage
    dataset = dc.data.NumpyDataset(X=np.random.rand(50, 3),
                                   y=np.random.rand(50, 20),
                                   ids=np.arange(50))
    ref_str = '<NumpyDataset X.shape: (50, 3), y.shape: (50, 20), w.shape: (50, 1), ids: [0 1 2 ... 47 48 49], task_names: [ 0  1  2 ... 17 18 19]>'
    assert str(dataset) == ref_str

    # Test max print size
    dc.utils.set_max_print_size(25)
    dataset = dc.data.NumpyDataset(X=np.random.rand(50, 3),
                                   y=np.random.rand(50,),
                                   ids=np.arange(50))
    ref_str = '<NumpyDataset X.shape: (50, 3), y.shape: (50,), w.shape: (50,), task_names: [0]>'
    assert str(dataset) == ref_str


class TestDatasets(unittest.TestCase):
    """
    Test basic top-level API for dataset objects.
    """

    def test_numpy_iterate_batch_size(self):
        solubility_dataset = load_solubility_data()
        X, y, _, _ = (solubility_dataset.X, solubility_dataset.y,
                      solubility_dataset.w, solubility_dataset.ids)
        solubility_dataset = dc.data.NumpyDataset.from_DiskDataset(
            solubility_dataset)
        batch_sizes = []
        for X, y, _, _ in solubility_dataset.iterbatches(3,
                                                         epochs=2,
                                                         pad_batches=False,
                                                         deterministic=True):
            batch_sizes.append(len(X))
        self.assertEqual([3, 3, 3, 1, 3, 3, 3, 1], batch_sizes)

    @pytest.mark.torch
    def test_make_pytorch_dataset_from_numpy(self):
        """Test creating a PyTorch Dataset from a NumpyDataset."""
        X = np.random.random((100, 5))
        y = np.random.random((100, 1))
        ids = [str(i) for i in range(100)]
        dataset = dc.data.NumpyDataset(X, y, ids=ids)
        _validate_pytorch_dataset(dataset)

    @pytest.mark.torch
    def test_make_pytorch_dataset_from_images(self):
        """Test creating a PyTorch Dataset from an ImageDataset."""
        path = os.path.join(os.path.dirname(__file__), 'images')
        files = [os.path.join(path, f) for f in os.listdir(path)]
        y = np.random.random((10, 1))
        ids = [str(i) for i in range(len(files))]
        dataset = dc.data.ImageDataset(files, y, ids=ids)
        _validate_pytorch_dataset(dataset)

    @pytest.mark.torch
    def test_make_pytorch_dataset_from_disk(self):
        """Test creating a PyTorch Dataset from a DiskDataset."""
        dataset = load_solubility_data()
        _validate_pytorch_dataset(dataset)

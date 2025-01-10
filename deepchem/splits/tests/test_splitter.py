"""
Tests for splitter objects.
"""
import os
import unittest
import numpy as np
import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.splits import IndexSplitter


def load_sparse_multitask_dataset():
    """Load sparse tox multitask data, sample dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = [
        "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8",
        "task9"
    ]
    input_file = os.path.join(current_dir,
                              "assets/sparse_multitask_example.csv")
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


def load_butina_data():
    """Loads solubility dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["task"]
    # task_type = "regression"
    input_file = os.path.join(current_dir, "assets/butina_example.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    return loader.create_dataset(input_file)


class TestSplitter(unittest.TestCase):
    """
    Test some basic splitters.
    """

    def test_random_group_split(self):
        solubility_dataset = load_solubility_data()

        groups = [0, 4, 1, 2, 3, 7, 0, 3, 1, 0]
        # 0 1 2 3 4 5 6 7 8 9

        group_splitter = dc.splits.RandomGroupSplitter(groups)

        train_idxs, valid_idxs, test_idxs = group_splitter.split(
            solubility_dataset, frac_train=0.5, frac_valid=0.25, frac_test=0.25)

        class_ind = [-1] * 10

        all_idxs = []
        for s in train_idxs + valid_idxs + test_idxs:
            all_idxs.append(s)

        assert sorted(all_idxs) == list(range(10))

        for split_idx, split in enumerate([train_idxs, valid_idxs, test_idxs]):
            for s in split:
                if class_ind[s] == -1:
                    class_ind[s] = split_idx
                else:
                    assert class_ind[s] == split_idx

    def test_singletask_random_split(self):
        """
        Test singletask RandomSplitter class.
        """
        solubility_dataset = load_solubility_data()
        random_splitter = dc.splits.RandomSplitter()
        train_data, valid_data, test_data = \
          random_splitter.train_valid_test_split(
            solubility_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

        merged_dataset = dc.data.DiskDataset.merge(
            [train_data, valid_data, test_data])
        assert sorted(merged_dataset.ids) == (sorted(solubility_dataset.ids))

    def test_singletask_index_split(self):
        """
        Test singletask IndexSplitter class.
        """
        solubility_dataset = load_solubility_data()
        random_splitter = dc.splits.IndexSplitter()
        train_data, valid_data, test_data = \
          random_splitter.train_valid_test_split(
            solubility_dataset)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

        merged_dataset = dc.data.DiskDataset.merge(
            [train_data, valid_data, test_data])
        assert sorted(merged_dataset.ids) == (sorted(solubility_dataset.ids))

    # TODO(rbharath): The IndexSplitter() had a bug with splitting sharded
    # data. Make a test for properly splitting of sharded data. Perhaps using
    # reshard() to handle this?

    def test_singletask_scaffold_split(self):
        """
        Test singletask ScaffoldSplitter class.
        """
        solubility_dataset = load_solubility_data()
        scaffold_splitter = dc.splits.ScaffoldSplitter()
        train_data, valid_data, test_data = \
          scaffold_splitter.train_valid_test_split(
            solubility_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_singletask_fingerprint_split(self):
        """
        Test singletask Fingerprint class.
        """
        solubility_dataset = load_solubility_data()
        assert (len(solubility_dataset.X) == 10)
        scaffold_splitter = dc.splits.FingerprintSplitter()
        train_data, valid_data, test_data = \
          scaffold_splitter.train_valid_test_split(
            solubility_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1
        s1 = set(train_data.ids)
        assert valid_data.ids[0] not in s1
        assert test_data.ids[0] not in s1

    def test_singletask_stratified_split(self):
        """
        Test singletask SingletaskStratifiedSplitter class.
        """
        solubility_dataset = load_solubility_data()
        stratified_splitter = dc.splits.SingletaskStratifiedSplitter()
        train_data, valid_data, test_data = \
          stratified_splitter.train_valid_test_split(
            solubility_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

        merged_dataset = dc.data.NumpyDataset.merge(
            [train_data, valid_data, test_data])
        assert sorted(merged_dataset.ids) == (sorted(solubility_dataset.ids))

    def test_singletask_maxmin_split(self):
        """
        Test singletask MaxMinSplitter class.
        """
        solubility_dataset = load_butina_data()
        maxmin_splitter = dc.splits.MaxMinSplitter()
        train_data, valid_data, test_data = \
          maxmin_splitter.train_valid_test_split(
            solubility_dataset)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_singletask_butina_split(self):
        """
        Test singletask ButinaSplitter class.
        """
        solubility_dataset = load_butina_data()
        butina_splitter = dc.splits.ButinaSplitter()
        train_data, valid_data, test_data = \
          butina_splitter.train_valid_test_split(
            solubility_dataset)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_k_fold_splitter(self):
        """
        Test that we can 5 fold index wise over 5 points
        """
        ds = NumpyDataset(np.array(range(5)), np.array(range(5)))
        index_splitter = IndexSplitter()

        K = 5
        fold_datasets = index_splitter.k_fold_split(ds, K)

        for fold in range(K):
            train, cv = fold_datasets[fold][0], fold_datasets[fold][1]
            self.assertTrue(cv.X[0] == fold)
            train_data = set(list(train.X))
            self.assertFalse(fold in train_data)
            self.assertEqual(K - 1, len(train))
            self.assertEqual(1, len(cv))

    def test_singletask_random_k_fold_split(self):
        """
        Test singletask RandomSplitter class.
        """
        solubility_dataset = load_solubility_data()
        random_splitter = dc.splits.RandomSplitter()
        ids_set = set(solubility_dataset.ids)

        K = 5
        fold_datasets = random_splitter.k_fold_split(solubility_dataset, K)
        for fold in range(K):
            fold_dataset = fold_datasets[fold][1]
            # Verify lengths is 10/k == 2
            assert len(fold_dataset) == 2
            # Verify that compounds in this fold are subset of original compounds
            fold_ids_set = set(fold_dataset.ids)
            assert fold_ids_set.issubset(ids_set)
            # Verify that no two folds have overlapping compounds.
            for other_fold in range(K):
                if fold == other_fold:
                    continue
                other_fold_dataset = fold_datasets[other_fold][1]
                other_fold_ids_set = set(other_fold_dataset.ids)
                assert fold_ids_set.isdisjoint(other_fold_ids_set)

    def test_singletask_index_k_fold_split(self):
        """
        Test singletask IndexSplitter class.
        """
        solubility_dataset = load_solubility_data()
        index_splitter = dc.splits.IndexSplitter()
        ids_set = set(solubility_dataset.ids)

        K = 5
        fold_datasets = index_splitter.k_fold_split(solubility_dataset, K)

        for fold in range(K):
            fold_dataset = fold_datasets[fold][1]
            # Verify lengths is 10/k == 2
            assert len(fold_dataset) == 2
            # Verify that compounds in this fold are subset of original compounds
            fold_ids_set = set(fold_dataset.ids)
            assert fold_ids_set.issubset(ids_set)
            # Verify that no two folds have overlapping compounds.
            for other_fold in range(K):
                if fold == other_fold:
                    continue
                other_fold_dataset = fold_datasets[other_fold][1]
                other_fold_ids_set = set(other_fold_dataset.ids)
                assert fold_ids_set.isdisjoint(other_fold_ids_set)

        merged_dataset = dc.data.DiskDataset.merge(
            [x[1] for x in fold_datasets])
        assert len(merged_dataset) == len(solubility_dataset)
        assert sorted(merged_dataset.ids) == (sorted(solubility_dataset.ids))

    def test_singletask_scaffold_k_fold_split(self):
        """
        Test singletask ScaffoldSplitter class.
        """
        solubility_dataset = load_solubility_data()
        scaffold_splitter = dc.splits.ScaffoldSplitter()
        ids_set = set(solubility_dataset.ids)

        K = 5
        fold_datasets = scaffold_splitter.k_fold_split(solubility_dataset, K)

        for fold in range(K):
            fold_dataset = fold_datasets[fold][1]
            # Verify lengths is 10/k == 2
            assert len(fold_dataset) == 2
            # Verify that compounds in this fold are subset of original compounds
            fold_ids_set = set(fold_dataset.ids)
            assert fold_ids_set.issubset(ids_set)
            # Verify that no two folds have overlapping compounds.
            for other_fold in range(K):
                if fold == other_fold:
                    continue
                other_fold_dataset = fold_datasets[other_fold][1]
                other_fold_ids_set = set(other_fold_dataset.ids)
                assert fold_ids_set.isdisjoint(other_fold_ids_set)

        merged_dataset = dc.data.DiskDataset.merge(
            [x[1] for x in fold_datasets])
        assert len(merged_dataset) == len(solubility_dataset)
        assert sorted(merged_dataset.ids) == (sorted(solubility_dataset.ids))

    def test_singletask_stratified_column_indices(self):
        """
        Test RandomStratifiedSplitter's split method on simple singletas.
        """
        # Test singletask case.
        n_samples = 100
        n_positives = 20
        n_tasks = 1

        X = np.ones(n_samples)
        y = np.zeros((n_samples, n_tasks))
        y[:n_positives] = 1
        w = np.ones((n_samples, n_tasks))
        dataset = dc.data.NumpyDataset(X, y, w)
        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        train, valid, test = stratified_splitter.split(dataset, 0.5, 0, 0.5)

        # The split index should partition dataset in half.
        assert len(train) == 50
        assert len(valid) == 0
        assert len(test) == 50
        assert np.count_nonzero(y[train]) == 10
        assert np.count_nonzero(y[test]) == 10

    def test_singletask_stratified_column_indices_mask(self):
        """
        Test RandomStratifiedSplitter's split method on dataset with mask.
        """
        # Test singletask case.
        n_samples = 100
        n_positives = 20
        n_tasks = 1

        # Test case where some weights are zero (i.e. masked)
        X = np.ones(n_samples)
        y = np.zeros((n_samples, n_tasks))
        y[:n_positives] = 1
        w = np.ones((n_samples, n_tasks))
        # Set half the positives to have zero weight
        w[:n_positives // 2] = 0
        dataset = dc.data.NumpyDataset(X, y, w)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        train, valid, test = stratified_splitter.split(dataset, 0.5, 0, 0.5)

        # There are 10 nonzero actives.
        # The split index should partition this into half, so expect 5
        w_present = (w != 0)
        y_present = y * w_present
        assert np.count_nonzero(y_present[train]) == 5

    def test_multitask_stratified_column_indices(self):
        """
        Test RandomStratifiedSplitter split on multitask dataset.
        """
        n_samples = 100
        n_tasks = 10
        p = .05  # proportion actives
        X = np.ones(n_samples)
        y = np.random.binomial(1, p, size=(n_samples, n_tasks))
        w = np.ones((n_samples, n_tasks))
        dataset = dc.data.NumpyDataset(X, y, w)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        train, valid, test = stratified_splitter.split(dataset, 0.5, 0, 0.5)

        for task in range(n_tasks):
            task_actives = np.count_nonzero(y[:, task])
            # The split index should partition the positives for each task roughly in half.
            target = task_actives / 2
            assert target - 2 <= np.count_nonzero(y[train, task]) <= target + 2

    def test_multitask_stratified_column_indices_masked(self):
        """
        Test RandomStratifiedSplitter split on multitask dataset.
        """
        n_samples = 200
        n_tasks = 10
        p = .05  # proportion actives
        X = np.ones(n_samples)
        y = np.random.binomial(1, p, size=(n_samples, n_tasks))
        w = np.ones((n_samples, n_tasks))
        # Mask half the examples
        w[:n_samples // 2] = 0
        dataset = dc.data.NumpyDataset(X, y, w)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        train, valid, test = stratified_splitter.split(dataset, 0.5, 0, 0.5)

        w_present = (w != 0)
        y_present = y * w_present
        for task in range(n_tasks):
            task_actives = np.count_nonzero(y_present[:, task])
            target = task_actives / 2
            # The split index should partition dataset in half.
            assert target - 1 <= np.count_nonzero(y_present[train,
                                                            task]) <= target + 1

    def test_random_stratified_split(self):
        """
        Test RandomStratifiedSplitter on a singletask split.
        """
        np.random.seed(2314)
        # Test singletask case.
        n_samples = 20
        n_positives = 10
        n_features = 10
        n_tasks = 1

        X = np.random.rand(n_samples, n_features)
        y = np.zeros((n_samples, n_tasks))
        y[:n_positives] = 1
        w = np.ones((n_samples, n_tasks))
        ids = np.arange(n_samples)
        dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        dataset_1, dataset_2 = stratified_splitter.train_test_split(
            dataset, frac_train=.5)
        print(dataset_1.get_shape())
        print(dataset_2.get_shape())

        # Should have split cleanly in half (picked random seed to ensure this)
        assert len(dataset_1) == 10
        assert len(dataset_2) == 10

        # Check positives are correctly distributed
        y_1 = dataset_1.y
        assert np.count_nonzero(y_1) == n_positives / 2

        y_2 = dataset_2.y
        assert np.count_nonzero(y_2) == n_positives / 2

    def test_singletask_stratified_train_valid_test_split(self):
        """
        Test RandomStratifiedSplitter on a singletask train/valid/test split.
        """
        np.random.seed(2314)
        # Test singletask case.
        n_samples = 100
        n_positives = 10
        n_features = 10
        n_tasks = 1

        X = np.random.rand(n_samples, n_features)
        y = np.zeros((n_samples, n_tasks))
        y[:n_positives] = 1
        w = np.ones((n_samples, n_tasks))
        ids = np.arange(n_samples)
        dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        train, valid, test = stratified_splitter.train_valid_test_split(
            dataset, frac_train=.8, frac_valid=.1, frac_test=.1)

        # Should have made an 80/10/10 train/valid/test split of actives.
        self.assertEqual(np.count_nonzero(train.y), 8)
        self.assertEqual(np.count_nonzero(valid.y), 1)
        self.assertEqual(np.count_nonzero(test.y), 1)

    def test_singletask_stratified_k_fold_split(self):
        """
        Test RandomStratifiedSplitter k-fold class.
        """
        n_samples = 100
        n_positives = 20
        n_features = 10

        X = np.random.rand(n_samples, n_features)
        y = np.zeros(n_samples)
        y[:n_positives] = 1
        w = np.ones(n_samples)
        ids = np.arange(n_samples)

        dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

        stratified_splitter = dc.splits.RandomStratifiedSplitter()
        ids_set = set(dataset.ids)

        K = 5
        fold_datasets = stratified_splitter.k_fold_split(dataset, K)
        fold_datasets = [f[1] for f in fold_datasets]

        for fold in range(K):
            fold_dataset = fold_datasets[fold]
            # Verify lengths is 100/k == 20
            # Note: This wouldn't work for multitask str
            # assert len(fold_dataset) == n_samples/K
            fold_labels = fold_dataset.y
            # Verify that each fold has n_positives/K = 4 positive examples.
            assert np.count_nonzero(fold_labels == 1) == n_positives / K
            # Verify that compounds in this fold are subset of original compounds
            fold_ids_set = set(fold_dataset.ids)
            assert fold_ids_set.issubset(ids_set)
            # Verify that no two folds have overlapping compounds.
            for other_fold in range(K):
                if fold == other_fold:
                    continue
                other_fold_dataset = fold_datasets[other_fold]
                other_fold_ids_set = set(other_fold_dataset.ids)
                assert fold_ids_set.isdisjoint(other_fold_ids_set)

        merged_dataset = dc.data.DiskDataset.merge(fold_datasets)
        assert len(merged_dataset) == len(dataset)
        assert sorted(merged_dataset.ids) == (sorted(dataset.ids))

    def test_multitask_random_split(self):
        """
        Test multitask RandomSplitter class.
        """
        multitask_dataset = load_multitask_data()
        random_splitter = dc.splits.RandomSplitter()
        train_data, valid_data, test_data = \
          random_splitter.train_valid_test_split(
            multitask_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_multitask_index_split(self):
        """
        Test multitask IndexSplitter class.
        """
        multitask_dataset = load_multitask_data()
        index_splitter = dc.splits.IndexSplitter()
        train_data, valid_data, test_data = \
          index_splitter.train_valid_test_split(
            multitask_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_multitask_scaffold_split(self):
        """
        Test multitask ScaffoldSplitter class.
        """
        multitask_dataset = load_multitask_data()
        scaffold_splitter = dc.splits.ScaffoldSplitter()
        train_data, valid_data, test_data = \
          scaffold_splitter.train_valid_test_split(
            multitask_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_specified_split(self):

        solubility_dataset = load_solubility_data()
        random_splitter = dc.splits.SpecifiedSplitter(valid_indices=[7],
                                                      test_indices=[8])
        train_data, valid_data, test_data = \
          random_splitter.split(
            solubility_dataset)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_random_seed(self):
        """Test that splitters use the random seed correctly."""
        dataset = load_solubility_data()
        splitter = dc.splits.RandomSplitter()
        train1, valid1, test1 = splitter.train_valid_test_split(dataset, seed=1)
        train2, valid2, test2 = splitter.train_valid_test_split(dataset, seed=2)
        train3, valid3, test3 = splitter.train_valid_test_split(dataset, seed=1)
        assert np.array_equal(train1.X, train3.X)
        assert np.array_equal(valid1.X, valid3.X)
        assert np.array_equal(test1.X, test3.X)
        assert not np.array_equal(train1.X, train2.X)
        assert not np.array_equal(valid1.X, valid2.X)
        assert not np.array_equal(test1.X, test2.X)

    def test_fingerprint_split(self):
        """
        Test FingerprintSplitter.
        """
        multitask_dataset = load_multitask_data()
        splitter = dc.splits.FingerprintSplitter()
        train_data, valid_data, test_data = \
          splitter.train_valid_test_split(
            multitask_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        assert len(train_data) == 8
        assert len(valid_data) == 1
        assert len(test_data) == 1

    def test_fingerprint_k_fold_split(self):
        """
        Test FingerprintSplitter.k_fold_split.
        """
        multitask_dataset = load_multitask_data()
        splitter = dc.splits.FingerprintSplitter()
        cv_folds = splitter.k_fold_split(multitask_dataset, k=3)
        assert len(multitask_dataset) == len(cv_folds[0][0]) + len(
            cv_folds[0][1])
        assert len(multitask_dataset) == len(cv_folds[1][0]) + len(
            cv_folds[1][1])
        assert len(multitask_dataset) == len(cv_folds[2][0]) + len(
            cv_folds[2][1])

    def test_SingletaskStratified_k_fold_split(self):
        """
        Test SingletaskStratifiedSplitter.k_fold_split
        """

        n_samples = 20
        n_tasks = 10
        n_features = 10

        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples, n_tasks)

        K = 7
        # When n_samples is not divisible by K, the function will return n_samples % K sub-arrays of size n_samples//K + 1 and the rest of size n_samples//K
        # Thus in our example, we will have 20%7 = 6 sub-arrays of size 3 and 1 sub-array of size 2

        dataset = dc.data.DiskDataset.from_numpy(X, y)
        splitter = dc.splits.SingletaskStratifiedSplitter(task_number=5)
        folds = splitter.k_fold_split(dataset, k=K)
        assert len(folds) == 7
        assert folds[0][0].y.shape[0] == 17
        assert folds[0][1].y.shape[0] == 3

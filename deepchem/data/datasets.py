"""
Contains wrapper class for datasets.
"""
import json
import os
import csv
import math
import random
import logging
import tempfile
import time
import shutil
import multiprocessing
from multiprocessing.dummy import Pool
from ast import literal_eval as make_tuple
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

import deepchem as dc
from deepchem.utils.typing import OneOrMany, Shape
from deepchem.utils.data_utils import save_to_disk, load_from_disk, load_image_files

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

logger = logging.getLogger(__name__)


def sparsify_features(X: np.ndarray) -> np.ndarray:
    """Extracts a sparse feature representation from dense feature array.

    Parameters
    ----------
    X: np.ndarray
        A numpy array of shape `(n_samples, ...)`.

    Returns
    -------
    X_sparse: np.ndarray
        A numpy array with `dtype=object` where `X_sparse[i]` is a
        typle of `(nonzero_inds, nonzero_vals)` with nonzero indices and
        values in the i-th sample of `X`.
    """
    n_samples = len(X)
    X_sparse = []
    for i in range(n_samples):
        nonzero_inds = np.nonzero(X[i])[0]
        nonzero_vals = X[i][nonzero_inds]
        X_sparse.append((nonzero_inds, nonzero_vals))
    return np.array(X_sparse, dtype=object)


def densify_features(X_sparse: np.ndarray, num_features: int) -> np.ndarray:
    """Expands sparse feature representation to dense feature array.

    Assumes that the sparse representation was constructed from an array
    which had original shape `(n_samples, num_features)` so doesn't
    support reconstructing multidimensional dense arrays.

    Parameters
    ----------
    X_sparse: np.ndarray
        Must have `dtype=object`. `X_sparse[i]` must be a tuple of nonzero
        indices and values.
    num_features: int
        Number of features in dense array.

    Returns
    -------
    X: np.ndarray
        A numpy array of shape `(n_samples, num_features)`.
    """
    n_samples = len(X_sparse)
    X = np.zeros((n_samples, num_features))
    for i in range(n_samples):
        nonzero_inds, nonzero_vals = X_sparse[i]
        X[i][nonzero_inds.astype(int)] = nonzero_vals
    return X


def pad_features(batch_size: int, X_b: np.ndarray) -> np.ndarray:
    """Pads a batch of features to have precisely batch_size elements.

    Given an array of features with length less than or equal to
    batch-size, pads it to `batch_size` length. It does this by
    repeating the original features in tiled fashion. For illustration,
    suppose that `len(X_b) == 3` and `batch_size == 10`.

    >>> X_b = np.arange(3)
    >>> X_b
    array([0, 1, 2])
    >>> batch_size = 10
    >>> X_manual = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> X_out = pad_features(batch_size, X_b)
    >>> assert (X_manual == X_out).all()

    This function is similar to `pad_batch` but doesn't handle labels
    `y` or weights `w` and is intended to be used for inference-time
    query processing.

    Parameters
    ----------
    batch_size: int
        The number of datapoints in a batch
    X_b: np.ndarray
        Must be such that `len(X_b) <= batch_size`

    Returns
    -------
    X_out: np.ndarray
        A numpy array with `len(X_out) == batch_size`.
    """
    num_samples = len(X_b)
    if num_samples > batch_size:
        raise ValueError("Cannot pad an array longer than `batch_size`")
    elif num_samples == batch_size:
        return X_b
    else:
        # By invariant of when this is called, can assume num_samples > 0
        # and num_samples < batch_size
        if len(X_b.shape) > 1:
            feature_shape = X_b.shape[1:]
            X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
        else:
            X_out = np.zeros((batch_size,), dtype=X_b.dtype)

        # Fill in batch arrays
        start = 0
        while start < batch_size:
            num_left = batch_size - start
            if num_left < num_samples:
                increment = num_left
            else:
                increment = num_samples
            X_out[start:start + increment] = X_b[:increment]
            start += increment
        return X_out


def pad_batch(batch_size: int, X_b: np.ndarray, y_b: np.ndarray,
              w_b: np.ndarray, ids_b: np.ndarray) -> Batch:
    """Pads batch to have size precisely batch_size elements.

    Given arrays of features `X_b`, labels `y_b`, weights `w_b`, and
    identifiers `ids_b` all with length less than or equal to
    batch-size, pads them to `batch_size` length. It does this by
    repeating the original entries in tiled fashion. Note that `X_b,
    y_b, w_b, ids_b` must all have the same length.

    Parameters
    ----------
    batch_size: int
        The number of datapoints in a batch
    X_b: np.ndarray
        Must be such that `len(X_b) <= batch_size`
    y_b: np.ndarray
        Must be such that `len(y_b) <= batch_size`
    w_b: np.ndarray
        Must be such that `len(w_b) <= batch_size`
    ids_b: np.ndarray
        Must be such that `len(ids_b) <= batch_size`

    Returns
    -------
    Batch
        The batch is a tuple of `(X_out, y_out, w_out, ids_out)`,
        all numpy arrays with length `batch_size`.
    """
    num_samples = len(X_b)
    if num_samples == batch_size:
        return (X_b, y_b, w_b, ids_b)
    # By invariant of when this is called, can assume num_samples > 0
    # and num_samples < batch_size
    if len(X_b.shape) > 1:
        feature_shape = X_b.shape[1:]
        X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
    else:
        X_out = np.zeros((batch_size,), dtype=X_b.dtype)

    if y_b is None:
        y_out = None
    elif len(y_b.shape) < 2:
        y_out = np.zeros(batch_size, dtype=y_b.dtype)
    else:
        y_out = np.zeros((batch_size,) + y_b.shape[1:], dtype=y_b.dtype)

    if w_b is None:
        w_out = None
    elif len(w_b.shape) < 2:
        w_out = np.zeros(batch_size, dtype=w_b.dtype)
    else:
        w_out = np.zeros((batch_size,) + w_b.shape[1:], dtype=w_b.dtype)

    ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

    # Fill in batch arrays
    start = 0
    # Only the first set of copy will be counted in training loss
    if w_out is not None:
        w_out[start:start + num_samples] = w_b[:]

    while start < batch_size:
        num_left = batch_size - start
        if num_left < num_samples:
            increment = num_left
        else:
            increment = num_samples
        X_out[start:start + increment] = X_b[:increment]

        if y_out is not None:
            y_out[start:start + increment] = y_b[:increment]

        ids_out[start:start + increment] = ids_b[:increment]
        start += increment

    return (X_out, y_out, w_out, ids_out)


class Dataset(object):
    """Abstract base class for datasets defined by X, y, w elements.

    `Dataset` objects are used to store representations of a dataset as
    used in a machine learning task. Datasets contain features `X`,
    labels `y`, weights `w` and identifiers `ids`. Different subclasses
    of `Dataset` may choose to hold `X, y, w, ids` in memory or on disk.

    The `Dataset` class attempts to provide for strong interoperability
    with other machine learning representations for datasets.
    Interconversion methods allow for `Dataset` objects to be converted
    to and from numpy arrays, pandas dataframes, tensorflow datasets,
    and pytorch datasets (only to and not from for pytorch at present).

    Note that you can never instantiate a `Dataset` object directly.
    Instead you will need to instantiate one of the concrete subclasses.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get the number of elements in the dataset.

        Returns
        -------
        int
            The number of elements in the dataset.
        """
        raise NotImplementedError()

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids
        arrays.

        Returns
        -------
        Tuple
            The tuple contains four elements, which are the shapes of
            the X, y, w, and ids arrays.
        """
        raise NotImplementedError()

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        raise NotImplementedError()

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `X`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `y`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `ids`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of weights `w`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Convert self to REPL print representation."""
        threshold = dc.utils.get_print_threshold()
        task_str = np.array2string(np.array(self.get_task_names()),
                                   threshold=threshold)
        X_shape, y_shape, w_shape, _ = self.get_shape()
        if self.__len__() < dc.utils.get_max_print_size():
            id_str = np.array2string(self.ids, threshold=threshold)
            return "<%s X.shape: %s, y.shape: %s, w.shape: %s, ids: %s, task_names: %s>" % (
                self.__class__.__name__, str(X_shape), str(y_shape),
                str(w_shape), id_str, task_str)
        else:
            return "<%s X.shape: %s, y.shape: %s, w.shape: %s, task_names: %s>" % (
                self.__class__.__name__, str(X_shape), str(y_shape),
                str(w_shape), task_str)

    def __str__(self) -> str:
        """Convert self to str representation."""
        return self.__repr__()

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays:
        `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default None)
            Number of elements in each batch.
        epochs: int, optional (default 1)
            Number of epochs to walk over dataset.
        deterministic: bool, optional (default False)
            If True, follow deterministic order.
        pad_batches: bool, optional (default False)
            If True, pad each batch to `batch_size`.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """
        raise NotImplementedError()

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Examples
        --------
        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x.tolist(), y.tolist(), w.tolist(), id)
        [1.0, 1.0] [0.0] [0.0] 0
        [1.0, 1.0] [0.0] [0.0] 1
        """
        raise NotImplementedError()

    def transform(self, transformer: "dc.trans.Transformer",
                  **args) -> "Dataset":
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:
        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple
        times with different subsets of the data.  Each time it is called,
        it should transform the samples and return the transformed data.

        Parameters
        ----------
        transformer: dc.trans.Transformer
            The transformation to apply to each sample in the dataset.

        Returns
        -------
        Dataset
            A newly constructed Dataset object.
        """
        raise NotImplementedError()

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None) -> "Dataset":
        """Creates a new dataset from a selection of indices from self.

        Parameters
        ----------
        indices: Sequence
            List of indices to select.
        select_dir: str, optional (default None)
            Path to new directory that the selected indices will be copied to.
        """
        raise NotImplementedError()

    def get_statistics(self,
                       X_stats: bool = True,
                       y_stats: bool = True) -> Tuple[np.ndarray, ...]:
        """Compute and return statistics of this dataset.

        Uses `self.itersamples()` to compute means and standard deviations
        of the dataset. Can compute on large datasets that don't fit in
        memory.

        Parameters
        ----------
        X_stats: bool, optional (default True)
            If True, compute feature-level mean and standard deviations.
        y_stats: bool, optional (default True)
            If True, compute label-level mean and standard deviations.

        Returns
        -------
        Tuple
            - If `X_stats == True`, returns `(X_means, X_stds)`.
            - If `y_stats == True`, returns `(y_means, y_stds)`.
            - If both are true, returns `(X_means, X_stds, y_means, y_stds)`.
        """
        x_shape, y_shape, w_shape, ids_shape = self.get_shape()
        X_means = np.zeros(x_shape[1:])
        X_m2 = np.zeros(x_shape[1:])
        y_means = np.zeros(y_shape[1:])
        y_m2 = np.zeros(y_shape[1:])
        n = 0
        for X, y, _, _ in self.itersamples():
            n += 1
            if X_stats:
                dx = X - X_means
                X_means += dx / n
                X_m2 += dx * (X - X_means)
            if y_stats:
                dy = y - y_means
                y_means += dy / n
                y_m2 += dy * (y - y_means)
        if n < 2:
            X_stds = np.zeros(x_shape[1:])
            y_stds = np.zeros(y_shape[1:])
        else:
            X_stds = np.sqrt(X_m2 / n)
            y_stds = np.sqrt(y_m2 / n)
        if X_stats and not y_stats:
            return X_means, X_stds
        elif y_stats and not X_stats:
            return y_means, y_stds
        elif X_stats and y_stats:
            return X_means, X_stds, y_means, y_stds
        else:
            return tuple()

    def make_tf_dataset(self,
                        batch_size: int = 100,
                        epochs: int = 1,
                        deterministic: bool = False,
                        pad_batches: bool = False):
        """Create a tf.data.Dataset that iterates over the data in this Dataset.

        Each value returned by the Dataset's iterator is a tuple of (X, y,
        w) for one batch.

        Parameters
        ----------
        batch_size: int, default 100
            The number of samples to include in each batch.
        epochs: int, default 1
            The number of times to iterate over the Dataset.
        deterministic: bool, default False
            If True, the data is produced in order.  If False, a different
            random permutation of the data is used for each epoch.
        pad_batches: bool, default False
            If True, batches are padded as necessary to make the size of
            each batch exactly equal batch_size.

        Returns
        -------
        tf.data.Dataset
            TensorFlow Dataset that iterates over the same data.

        Note
        ----
        This class requires TensorFlow to be installed.
        """
        try:
            import tensorflow as tf
        except:
            raise ImportError(
                "This method requires TensorFlow to be installed.")

        # Retrieve the first sample so we can determine the dtypes.
        X, y, w, ids = next(self.itersamples())
        dtypes = (tf.as_dtype(X.dtype), tf.as_dtype(y.dtype),
                  tf.as_dtype(w.dtype))
        shapes = (
            tf.TensorShape([None] + list(X.shape)),  # type: ignore
            tf.TensorShape([None] + list(y.shape)),  # type: ignore
            tf.TensorShape([None] + list(w.shape)))  # type: ignore

        # Create a Tensorflow Dataset.
        def gen_data():
            for X, y, w, ids in self.iterbatches(batch_size, epochs,
                                                 deterministic, pad_batches):
                yield (X, y, w)

        return tf.data.Dataset.from_generator(gen_data, dtypes, shapes)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

        Each value returned by the Dataset's iterator is a tuple of (X, y, w, id)
        containing the data for one batch, or for a single sample if batch_size is None.

        Parameters
        ----------
        epochs: int, default 1
            The number of times to iterate over the Dataset.
        deterministic: bool, default False
            If True, the data is produced in order. If False, a different
            random permutation of the data is used for each epoch.
        batch_size: int, optional (default None)
            The number of samples to return in each batch. If None, each returned
            value is a single sample.

        Returns
        -------
        torch.utils.data.IterableDataset
            `torch.utils.data.IterableDataset` that iterates over the data in
            this dataset.

        Note
        ----
        This class requires PyTorch to be installed.
        """
        raise NotImplementedError()

    def to_dataframe(self) -> pd.DataFrame:
        """Construct a pandas DataFrame containing the data from this Dataset.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe. If there is only a single feature per datapoint,
            will have column "X" else will have columns "X1,X2,..." for
            features.  If there is only a single label per datapoint, will
            have column "y" else will have columns "y1,y2,..." for labels. If
            there is only a single weight per datapoint will have column "w"
            else will have columns "w1,w2,...". Will have column "ids" for
            identifiers.
        """
        X = self.X
        y = self.y
        w = self.w
        ids = self.ids
        if len(X.shape) == 1 or X.shape[1] == 1:
            columns = ['X']
        else:
            columns = [f'X{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)
        if len(y.shape) == 1 or y.shape[1] == 1:
            columns = ['y']
        else:
            columns = [f'y{i+1}' for i in range(y.shape[1])]
        y_df = pd.DataFrame(y, columns=columns)
        if len(w.shape) == 1 or w.shape[1] == 1:
            columns = ['w']
        else:
            columns = [f'w{i+1}' for i in range(w.shape[1])]
        w_df = pd.DataFrame(w, columns=columns)
        ids_df = pd.DataFrame(ids, columns=['ids'])
        return pd.concat([X_df, y_df, w_df, ids_df], axis=1, sort=False)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       X: Optional[OneOrMany[str]] = None,
                       y: Optional[OneOrMany[str]] = None,
                       w: Optional[OneOrMany[str]] = None,
                       ids: Optional[str] = None):
        """Construct a Dataset from the contents of a pandas DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            The pandas DataFrame
        X: str or List[str], optional (default None)
            The name of the column or columns containing the X array.  If
            this is None, it will look for default column names that match
            those produced by to_dataframe().
        y: str or List[str], optional (default None)
            The name of the column or columns containing the y array.  If
            this is None, it will look for default column names that match
            those produced by to_dataframe().
        w: str or List[str], optional (default None)
            The name of the column or columns containing the w array.  If
            this is None, it will look for default column names that match
            those produced by to_dataframe().
        ids: str, optional (default None)
            The name of the column containing the ids.  If this is None, it
            will look for default column names that match those produced by
            to_dataframe().
        """
        # Find the X values.
        if X is not None:
            X_val = df[X]
        elif 'X' in df.columns:
            X_val = df['X']
        else:
            columns = []
            i = 1
            while f'X{i}' in df.columns:
                columns.append(f'X{i}')
                i += 1
            X_val = df[columns]
        if len(X_val.shape) == 1:
            X_val = np.expand_dims(X_val, 1)

        # Find the y values.
        if y is not None:
            y_val = df[y]
        elif 'y' in df.columns:
            y_val = df['y']
        else:
            columns = []
            i = 1
            while f'y{i}' in df.columns:
                columns.append(f'y{i}')
                i += 1
            y_val = df[columns]
        if len(y_val.shape) == 1:
            y_val = np.expand_dims(y_val, 1)

        # Find the w values.
        if w is not None:
            w_val = df[w]
        elif 'w' in df.columns:
            w_val = df['w']
        else:
            columns = []
            i = 1
            while f'w{i}' in df.columns:
                columns.append(f'w{i}')
                i += 1
            w_val = df[columns]
        if len(w_val.shape) == 1:
            w_val = np.expand_dims(w_val, 1)

        # Find the ids.
        if ids is not None:
            ids_val = df[ids]
        elif 'ids' in df.columns:
            ids_val = df['ids']
        else:
            ids_val = None
        return NumpyDataset(X_val, y_val, w_val, ids_val)

    def to_csv(self, path: str) -> None:
        """Write object to a comma-seperated values (CSV) file

        Example
        -------
        >>> import numpy as np
        >>> X = np.random.rand(10, 10)
        >>> dataset = dc.data.DiskDataset.from_numpy(X)
        >>> dataset.to_csv('out.csv')  # doctest: +SKIP

        Parameters
        ----------
        path: str
            File path or object

        Returns
        -------
        None
        """
        columns = []
        X_shape, y_shape, w_shape, id_shape = self.get_shape()
        assert len(
            X_shape) == 2, "dataset's X values should be scalar or 1-D arrays"
        assert len(
            y_shape) == 2, "dataset's y values should be scalar or 1-D arrays"
        if X_shape[1] == 1:
            columns.append('X')
        else:
            columns.extend([f'X{i+1}' for i in range(X_shape[1])])
        if y_shape[1] == 1:
            columns.append('y')
        else:
            columns.extend([f'y{i+1}' for i in range(y_shape[1])])
        if w_shape[1] == 1:
            columns.append('w')
        else:
            columns.extend([f'w{i+1}' for i in range(w_shape[1])])
        columns.append('ids')
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for (x, y, w, ids) in self.itersamples():
                writer.writerow(list(x) + list(y) + list(w) + [ids])
        return None


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays.

    This subclass of `Dataset` stores arrays `X,y,w,ids` in memory as
    numpy arrays. This makes it very easy to construct `NumpyDataset`
    objects.

    Examples
    --------
    >>> import numpy as np
    >>> dataset = NumpyDataset(X=np.random.rand(5, 3), y=np.random.rand(5,), ids=np.arange(5))
    """

    def __init__(self,
                 X: ArrayLike,
                 y: Optional[ArrayLike] = None,
                 w: Optional[ArrayLike] = None,
                 ids: Optional[ArrayLike] = None,
                 n_tasks: int = 1) -> None:
        """Initialize this object.

        Parameters
        ----------
        X: np.ndarray
            Input features. A numpy array of shape `(n_samples,...)`.
        y: np.ndarray, optional (default None)
            Labels. A numpy array of shape `(n_samples, ...)`. Note that each label can
            have an arbitrary shape.
        w: np.ndarray, optional (default None)
            Weights. Should either be 1D array of shape `(n_samples,)` or if
            there's more than one task, of shape `(n_samples, n_tasks)`.
        ids: np.ndarray, optional (default None)
            Identifiers. A numpy array of shape `(n_samples,)`
        n_tasks: int, default 1
            Number of learning tasks.
        """
        n_samples = np.shape(X)[0]
        if n_samples > 0:
            if y is None:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, n_tasks), np.float32)
                w = np.zeros((n_samples, 1), np.float32)
        if ids is None:
            ids = np.arange(n_samples)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if w is None:
            if len(y.shape) == 1:
                w = np.ones(y.shape[0], np.float32)
            else:
                w = np.ones((y.shape[0], 1), np.float32)
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        self._X = X
        self._y = y
        self._w = w
        self._ids = np.array(ids, dtype=object)

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return len(self._y)

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        if len(self._y.shape) < 2:
            return np.array([0])
        return np.arange(self._y.shape[1])

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        return self._y

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays:
        `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default None)
            Number of elements in each batch.
        epochs: int, default 1
            Number of epochs to walk over dataset.
        deterministic: bool, optional (default False)
            If True, follow deterministic order.
        pad_batches: bool, optional (default False)
            If True, pad each batch to `batch_size`.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """

        def iterate(dataset: NumpyDataset, batch_size: Optional[int],
                    epochs: int, deterministic: bool, pad_batches: bool):
            n_samples = dataset._X.shape[0]
            if deterministic:
                sample_perm = np.arange(n_samples)
            if batch_size is None:
                batch_size = n_samples
            for epoch in range(epochs):
                if not deterministic:
                    sample_perm = np.random.permutation(n_samples)
                batch_idx = 0
                num_batches = math.ceil(n_samples / batch_size)
                while batch_idx < num_batches:
                    start = batch_idx * batch_size
                    end = min(n_samples, (batch_idx + 1) * batch_size)
                    indices = range(start, end)
                    perm_indices = sample_perm[indices]
                    X_batch = dataset._X[perm_indices]
                    y_batch = dataset._y[perm_indices]
                    w_batch = dataset._w[perm_indices]
                    ids_batch = dataset._ids[perm_indices]
                    if pad_batches:
                        (X_batch, y_batch, w_batch,
                         ids_batch) = pad_batch(batch_size, X_batch, y_batch,
                                                w_batch, ids_batch)
                    batch_idx += 1
                    yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self, batch_size, epochs, deterministic, pad_batches)

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Returns
        -------
        Iterator[Batch]
            Iterator which yields tuples of four numpy arrays `(X, y, w, ids)`.

        Examples
        --------
        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x.tolist(), y.tolist(), w.tolist(), id)
        [1.0, 1.0] [0.0] [0.0] 0
        [1.0, 1.0] [0.0] [0.0] 1
        """
        n_samples = self._X.shape[0]
        return ((self._X[i], self._y[i], self._w[i], self._ids[i])
                for i in range(n_samples))

    def transform(self, transformer: "dc.trans.Transformer",
                  **args) -> "NumpyDataset":
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:
        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple
        times with different subsets of the data.  Each time it is called,
        it should transform the samples and return the transformed data.

        Parameters
        ----------
        transformer: dc.trans.Transformer
            The transformation to apply to each sample in the dataset

        Returns
        -------
        NumpyDataset
            A newly constructed NumpyDataset object
        """
        newx, newy, neww, newids = transformer.transform_array(
            self._X, self._y, self._w, self._ids)
        return NumpyDataset(newx, newy, neww, newids)

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None) -> "NumpyDataset":
        """Creates a new dataset from a selection of indices from self.

        Parameters
        ----------
        indices: List[int]
            List of indices to select.
        select_dir: str, optional (default None)
            Used to provide same API as `DiskDataset`. Ignored since
            `NumpyDataset` is purely in-memory.

        Returns
        -------
        NumpyDataset
            A selected NumpyDataset object
        """
        X = self.X[indices]
        y = self.y[indices]
        w = self.w[indices]
        ids = self.ids[indices]
        return NumpyDataset(X, y, w, ids)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

        Each value returned by the Dataset's iterator is a tuple of (X, y, w, id)
        containing the data for one batch, or for a single sample if batch_size is None.

        Parameters
        ----------
        epochs: int, default 1
            The number of times to iterate over the Dataset
        deterministic: bool, default False
            If True, the data is produced in order. If False, a different
            random permutation of the data is used for each epoch.
        batch_size: int, optional (default None)
            The number of samples to return in each batch. If None, each returned
            value is a single sample.

        Returns
        -------
        torch.utils.data.IterableDataset
            `torch.utils.data.IterableDataset` that iterates over the data in
            this dataset.

        Note
        ----
        This method requires PyTorch to be installed.
        """
        try:
            from deepchem.data.pytorch_datasets import _TorchNumpyDataset
        except:
            raise ImportError("This method requires PyTorch to be installed.")

        pytorch_ds = _TorchNumpyDataset(numpy_dataset=self,
                                        epochs=epochs,
                                        deterministic=deterministic,
                                        batch_size=batch_size)
        return pytorch_ds

    @staticmethod
    def from_DiskDataset(ds: "DiskDataset") -> "NumpyDataset":
        """Convert DiskDataset to NumpyDataset.

        Parameters
        ----------
        ds: DiskDataset
            DiskDataset to transform to NumpyDataset.

        Returns
        -------
        NumpyDataset
            A new NumpyDataset created from DiskDataset.
        """
        return NumpyDataset(ds.X, ds.y, ds.w, ds.ids)

    def to_json(self, fname: str) -> None:
        """Dump NumpyDataset to the json file .

        Parameters
        ----------
        fname: str
            The name of the json file.
        """
        d = {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'w': self.w.tolist(),
            'ids': self.ids.tolist()
        }
        with open(fname, 'w') as fout:
            json.dump(d, fout)

    @staticmethod
    def from_json(fname: str) -> "NumpyDataset":
        """Create NumpyDataset from the json file.

        Parameters
        ----------
        fname: str
            The name of the json file.

        Returns
        -------
        NumpyDataset
            A new NumpyDataset created from the json file.
        """
        with open(fname) as fin:
            d = json.load(fin)
            return NumpyDataset(d['X'], d['y'], d['w'], d['ids'])

    @staticmethod
    def merge(datasets: Sequence[Dataset]) -> "NumpyDataset":
        """Merge multiple NumpyDatasets.

        Parameters
        ----------
        datasets: List[Dataset]
            List of datasets to merge.

        Returns
        -------
        NumpyDataset
            A single NumpyDataset containing all the samples from all datasets.

        Example
        -------
        >>> X1, y1 = np.random.rand(5, 3), np.random.randn(5, 1)
        >>> first_dataset = dc.data.NumpyDataset(X1, y1)
        >>> X2, y2 = np.random.rand(5, 3), np.random.randn(5, 1)
        >>> second_dataset = dc.data.NumpyDataset(X2, y2)
        >>> merged_dataset = dc.data.NumpyDataset.merge([first_dataset, second_dataset])
        >>> print(len(merged_dataset) == len(first_dataset) + len(second_dataset))
        True

        """
        X, y, w, ids = datasets[0].X, datasets[0].y, datasets[0].w, datasets[
            0].ids
        for dataset in datasets[1:]:
            X = np.concatenate([X, dataset.X], axis=0)
            y = np.concatenate([y, dataset.y], axis=0)
            w = np.concatenate([w, dataset.w], axis=0)
            ids = np.concatenate(
                [ids, dataset.ids],
                axis=0,
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return NumpyDataset(X, y, w, ids, n_tasks=y.shape[1])


class _Shard(object):

    def __init__(self, X, y, w, ids):
        self.X = X
        self.y = y
        self.w = w
        self.ids = ids


class DiskDataset(Dataset):
    """
    A Dataset that is stored as a set of files on disk.

    The DiskDataset is the workhorse class of DeepChem that facilitates analyses
    on large datasets. Use this class whenever you're working with a large
    dataset that can't be easily manipulated in RAM.

    On disk, a `DiskDataset` has a simple structure. All files for a given
    `DiskDataset` are stored in a `data_dir`. The contents of `data_dir` should
    be laid out as follows:

    | data_dir/
    |   |
    |   ---> metadata.csv.gzip
    |   |
    |   ---> tasks.json
    |   |
    |   ---> shard-0-X.npy
    |   |
    |   ---> shard-0-y.npy
    |   |
    |   ---> shard-0-w.npy
    |   |
    |   ---> shard-0-ids.npy
    |   |
    |   ---> shard-1-X.npy
    |   .
    |   .
    |   .

    The metadata is constructed by static method
    `DiskDataset._construct_metadata` and saved to disk by
    `DiskDataset._save_metadata`. The metadata itself consists of a csv file
    which has columns `('ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape',
    'w_shape')`. `tasks.json` consists of a list of task names for this dataset.

    The actual data is stored in `.npy` files (numpy array files) of the form
    'shard-0-X.npy', 'shard-0-y.npy', etc.

    The basic structure of `DiskDataset` is quite robust and will likely serve
    you well for datasets up to about 100 GB or larger. However note that
    `DiskDataset` has not been tested for very large datasets at the terabyte
    range and beyond. You may be better served by implementing a custom
    `Dataset` class for those use cases.

    Examples
    --------
    Let's walk through a simple example of constructing a new `DiskDataset`.

    >>> import deepchem as dc
    >>> import numpy as np
    >>> X = np.random.rand(10, 10)
    >>> dataset = dc.data.DiskDataset.from_numpy(X)

    If you have already saved a `DiskDataset` to `data_dir`, you can reinitialize it with

    >> data_dir = "/path/to/my/data"
    >> dataset = dc.data.DiskDataset(data_dir)

    Once you have a dataset you can access its attributes as follows

    >>> X = np.random.rand(10, 10)
    >>> y = np.random.rand(10,)
    >>> w = np.ones_like(y)
    >>> dataset = dc.data.DiskDataset.from_numpy(X)
    >>> X, y, w = dataset.X, dataset.y, dataset.w

    One thing to beware of is that `dataset.X`, `dataset.y`, `dataset.w` are
    loading data from disk! If you have a large dataset, these operations can be
    extremely slow. Instead try iterating through the dataset instead.

    >>> for (xi, yi, wi, idi) in dataset.itersamples():
    ...   pass

    Attributes
    ----------
    data_dir: str
        Location of directory where this `DiskDataset` is stored to disk
    metadata_df: pd.DataFrame
        Pandas Dataframe holding metadata for this `DiskDataset`
    legacy_metadata: bool
        Whether this `DiskDataset` uses legacy format.

    Note
    ----
    `DiskDataset` originally had a simpler metadata format without shape
    information. Older `DiskDataset` objects had metadata files with columns
    `('ids', 'X', 'y', 'w')` and not additional shape columns. `DiskDataset`
    maintains backwards compatibility with this older metadata format, but we
    recommend for performance reasons not using legacy metadata for new
    projects.
    """

    def __init__(self, data_dir: str) -> None:
        """Load a constructed DiskDataset from disk

        Note that this method cannot construct a new disk dataset. Instead use
        static methods `DiskDataset.create_dataset` or `DiskDataset.from_numpy`
        for that purpose. Use this constructor instead to load a `DiskDataset`
        that has already been created on disk.

        Parameters
        ----------
        data_dir: str
            Location on disk of an existing `DiskDataset`.
        """
        self.data_dir = data_dir

        logger.info("Loading dataset from disk.")
        tasks, self.metadata_df = self.load_metadata()
        self.tasks = np.array(tasks)
        if len(self.metadata_df.columns) == 4 and list(
                self.metadata_df.columns) == ['ids', 'X', 'y', 'w']:
            logger.info(
                "Detected legacy metatadata on disk. You can upgrade from legacy metadata "
                "to the more efficient current metadata by resharding this dataset "
                "by calling the reshard() method of this object.")
            self.legacy_metadata = True
        elif len(self.metadata_df.columns) == 8 and list(
                self.metadata_df.columns) == [
                    'ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape',
                    'w_shape'
                ]:  # noqa
            self.legacy_metadata = False
        else:
            raise ValueError(
                "Malformed metadata on disk. Metadata must have columns 'ids', 'X', 'y', 'w', "
                "'ids_shape', 'X_shape', 'y_shape', 'w_shape' (or if in legacy metadata format,"
                "columns 'ids', 'X', 'y', 'w')")
        self._cached_shards: Optional[List] = None
        self._memory_cache_size = 20 * (1 << 20)  # 20 MB
        self._cache_used = 0

    @staticmethod
    def create_dataset(shard_generator: Iterable[Batch],
                       data_dir: Optional[str] = None,
                       tasks: Optional[ArrayLike] = None) -> "DiskDataset":
        """Creates a new DiskDataset

        Parameters
        ----------
        shard_generator: Iterable[Batch]
            An iterable (either a list or generator) that provides tuples of data
            (X, y, w, ids). Each tuple will be written to a separate shard on disk.
        data_dir: str, optional (default None)
            Filename for data directory. Creates a temp directory if none specified.
        tasks: Sequence, optional (default [])
            List of tasks for this dataset.

        Returns
        -------
        DiskDataset
            A new `DiskDataset` constructed from the given data
        """
        if data_dir is None:
            data_dir = tempfile.mkdtemp()
        elif not os.path.exists(data_dir):
            os.makedirs(data_dir)

        metadata_rows = []
        time1 = time.time()
        for shard_num, (X, y, w, ids) in enumerate(shard_generator):
            if shard_num == 0:
                if tasks is None and y is not None:
                    # The line here assumes that y generated by shard_generator is a numpy array
                    tasks = np.array([0]) if y.ndim < 2 else np.arange(
                        y.shape[1])
            basename = "shard-%d" % shard_num
            metadata_rows.append(
                DiskDataset.write_data_to_disk(data_dir, basename, X, y, w,
                                               ids))
        metadata_df = DiskDataset._construct_metadata(metadata_rows)
        DiskDataset._save_metadata(metadata_df, data_dir, tasks)
        time2 = time.time()
        logger.info("TIMING: dataset construction took %0.3f s" %
                    (time2 - time1))
        return DiskDataset(data_dir)

    def load_metadata(self) -> Tuple[List[str], pd.DataFrame]:
        """Helper method that loads metadata from disk."""
        try:
            tasks_filename, metadata_filename = self._get_metadata_filename()
            with open(tasks_filename) as fin:
                tasks = json.load(fin)
            metadata_df = pd.read_csv(metadata_filename,
                                      compression='gzip',
                                      dtype=object)
            metadata_df = metadata_df.where((pd.notnull(metadata_df)), None)
            return tasks, metadata_df
        except Exception:
            pass

        # Load obsolete format -> save in new format
        metadata_filename = os.path.join(self.data_dir, "metadata.joblib")
        if os.path.exists(metadata_filename):
            tasks, metadata_df = load_from_disk(metadata_filename)
            del metadata_df['task_names']
            del metadata_df['basename']
            DiskDataset._save_metadata(metadata_df, self.data_dir, tasks)
            return tasks, metadata_df
        raise ValueError(f"No Metadata found in the path {self.data_dir}")

    @staticmethod
    def _save_metadata(metadata_df: pd.DataFrame, data_dir: str,
                       tasks: Optional[ArrayLike]) -> None:
        """Saves the metadata for a DiskDataset

        Parameters
        ----------
        metadata_df: pd.DataFrame
            The dataframe which will be written to disk.
        data_dir: str
            Directory to store metadata.
        tasks: Sequence, optional
            Tasks of DiskDataset. If `None`, an empty list of tasks is written to
            disk.
        """
        if tasks is None:
            tasks = []
        elif isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
        tasks_filename = os.path.join(data_dir, "tasks.json")
        with open(tasks_filename, 'w') as fout:
            json.dump(tasks, fout)
        metadata_df.to_csv(metadata_filename, index=False, compression='gzip')

    @staticmethod
    def _construct_metadata(metadata_entries: List) -> pd.DataFrame:
        """Construct a dataframe containing metadata.

        Parameters
        ----------
        metadata_entries: List
            `metadata_entries` should have elements returned by write_data_to_disk
            above.

        Returns
        -------
        pd.DataFrame
            A Pandas Dataframe object contains metadata.
        """
        columns = ('ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape',
                   'w_shape')
        metadata_df = pd.DataFrame(metadata_entries, columns=columns)
        return metadata_df

    @staticmethod
    def write_data_to_disk(data_dir: str,
                           basename: str,
                           X: Optional[np.ndarray] = None,
                           y: Optional[np.ndarray] = None,
                           w: Optional[np.ndarray] = None,
                           ids: Optional[np.ndarray] = None) -> List[Any]:
        """Static helper method to write data to disk.

        This helper method is used to write a shard of data to disk.

        Parameters
        ----------
        data_dir: str
            Data directory to write shard to.
        basename: str
            Basename for the shard in question.
        X: np.ndarray, optional (default None)
            The features array.
        y: np.ndarray, optional (default None)
            The labels array.
        w: np.ndarray, optional (default None)
            The weights array.
        ids: np.ndarray, optional (default None)
            The identifiers array.

        Returns
        -------
        List[Optional[str]]
            List with values `[out_ids, out_X, out_y, out_w, out_ids_shape,
            out_X_shape, out_y_shape, out_w_shape]` with filenames of locations to
            disk which these respective arrays were written.
        """
        if X is not None:
            out_X: Optional[str] = "%s-X.npy" % basename
            save_to_disk(X, os.path.join(data_dir, out_X))  # type: ignore
            out_X_shape: Optional[Tuple[int, ...]] = X.shape
        else:
            out_X = None
            out_X_shape = None

        if y is not None:
            out_y: Optional[str] = "%s-y.npy" % basename
            save_to_disk(y, os.path.join(data_dir, out_y))  # type: ignore
            out_y_shape: Optional[Tuple[int, ...]] = y.shape
        else:
            out_y = None
            out_y_shape = None

        if w is not None:
            out_w: Optional[str] = "%s-w.npy" % basename
            save_to_disk(w, os.path.join(data_dir, out_w))  # type: ignore
            out_w_shape: Optional[Tuple[int, ...]] = w.shape
        else:
            out_w = None
            out_w_shape = None

        if ids is not None:
            out_ids: Optional[str] = "%s-ids.npy" % basename
            save_to_disk(ids, os.path.join(data_dir, out_ids))  # type: ignore
            out_ids_shape: Optional[Tuple[int, ...]] = ids.shape
        else:
            out_ids = None
            out_ids_shape = None

        # note that this corresponds to the _construct_metadata column order
        return [
            out_ids, out_X, out_y, out_w, out_ids_shape, out_X_shape,
            out_y_shape, out_w_shape
        ]

    def save_to_disk(self) -> None:
        """Save dataset to disk."""
        DiskDataset._save_metadata(self.metadata_df, self.data_dir, self.tasks)
        self._cached_shards = None

    def move(self,
             new_data_dir: str,
             delete_if_exists: Optional[bool] = True) -> None:
        """Moves dataset to new directory.

        Parameters
        ----------
        new_data_dir: str
            The new directory name to move this to dataset to.
        delete_if_exists: bool, optional (default True)
            If this option is set, delete the destination directory if it exists
            before moving. This is set to True by default to be backwards compatible
            with behavior in earlier versions of DeepChem.

        Note
        ----
        This is a stateful operation! `self.data_dir` will be moved into
        `new_data_dir`. If `delete_if_exists` is set to `True` (by default this is
        set `True`), then `new_data_dir` is deleted if it's a pre-existing
        directory.
        """
        if delete_if_exists and os.path.isdir(new_data_dir):
            shutil.rmtree(new_data_dir)
        shutil.move(self.data_dir, new_data_dir)
        if delete_if_exists:
            self.data_dir = new_data_dir
        else:
            self.data_dir = os.path.join(new_data_dir,
                                         os.path.basename(self.data_dir))

    def copy(self, new_data_dir: str) -> "DiskDataset":
        """Copies dataset to new directory.

        Parameters
        ----------
        new_data_dir: str
            The new directory name to copy this to dataset to.

        Returns
        -------
        DiskDataset
            A copied DiskDataset object.

        Note
        ----
        This is a stateful operation! Any data at `new_data_dir` will be deleted
        and `self.data_dir` will be deep copied into `new_data_dir`.
        """
        if os.path.isdir(new_data_dir):
            shutil.rmtree(new_data_dir)
        shutil.copytree(self.data_dir, new_data_dir)
        return DiskDataset(new_data_dir)

    def get_task_names(self) -> np.ndarray:
        """Gets learning tasks associated with this dataset."""
        return self.tasks

    def reshard(self, shard_size: int) -> None:
        """Reshards data to have specified shard size.

        Parameters
        ----------
        shard_size: int
            The size of shard.

        Examples
        --------
        >>> import deepchem as dc
        >>> import numpy as np
        >>> X = np.random.rand(100, 10)
        >>> d = dc.data.DiskDataset.from_numpy(X)
        >>> d.reshard(shard_size=10)
        >>> d.get_number_shards()
        10

        Note
        ----
        If this `DiskDataset` is in `legacy_metadata` format, reshard will
        convert this dataset to have non-legacy metadata.
        """
        # Create temp directory to store resharded version
        reshard_dir = tempfile.mkdtemp()
        n_shards = self.get_number_shards()

        # Get correct shapes for y/w
        tasks = self.get_task_names()
        _, y_shape, w_shape, _ = self.get_shape()
        if len(y_shape) == 1:
            y_shape = (len(y_shape), len(tasks))
        if len(w_shape) == 1:
            w_shape = (len(w_shape), len(tasks))

        # Write data in new shards
        def generator():
            X_next = np.zeros((0,) + self.get_data_shape())
            y_next = np.zeros((0,) + y_shape[1:])
            w_next = np.zeros((0,) + w_shape[1:])
            ids_next = np.zeros((0,), dtype=object)
            for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
                logger.info("Resharding shard %d/%d" %
                            (shard_num + 1, n_shards))
                # Handle shapes
                X = np.reshape(X, (len(X),) + self.get_data_shape())
                # Note that this means that DiskDataset resharding currently doesn't
                # work for datasets that aren't regression/classification.
                if y is None:  # datasets without label
                    y = y_next
                    w = w_next
                else:
                    y = np.reshape(y, (len(y),) + y_shape[1:])
                    w = np.reshape(w, (len(w),) + w_shape[1:])
                X_next = np.concatenate([X_next, X], axis=0)
                y_next = np.concatenate([y_next, y], axis=0)
                w_next = np.concatenate([w_next, w], axis=0)
                ids_next = np.concatenate([ids_next, ids])
                while len(X_next) > shard_size:
                    X_batch, X_next = X_next[:shard_size], X_next[shard_size:]
                    y_batch, y_next = y_next[:shard_size], y_next[shard_size:]
                    w_batch, w_next = w_next[:shard_size], w_next[shard_size:]
                    ids_batch, ids_next = ids_next[:shard_size], ids_next[
                        shard_size:]
                    yield (X_batch, y_batch, w_batch, ids_batch)
            # Handle spillover from last shard
            yield (X_next, y_next, w_next, ids_next)

        resharded_dataset = DiskDataset.create_dataset(generator(),
                                                       data_dir=reshard_dir,
                                                       tasks=self.tasks)
        shutil.rmtree(self.data_dir)
        shutil.move(reshard_dir, self.data_dir)
        # Should have updated to non-legacy metadata
        self.legacy_metadata = False
        self.metadata_df = resharded_dataset.metadata_df
        # Note that this resets the cache internally
        self.save_to_disk()

    def get_data_shape(self) -> Shape:
        """Gets array shape of datapoints in this dataset."""
        if not len(self.metadata_df):
            raise ValueError("No data in dataset.")
        if self.legacy_metadata:
            sample_X = load_from_disk(
                os.path.join(self.data_dir,
                             next(self.metadata_df.iterrows())[1]['X']))
            return np.shape(sample_X)[1:]
        else:
            X_shape, _, _, _ = self.get_shape()
            return X_shape[1:]

    def get_shard_size(self) -> int:
        """Gets size of shards on disk."""
        if not len(self.metadata_df):
            raise ValueError("No data in dataset.")
        sample_ids = load_from_disk(
            os.path.join(self.data_dir,
                         next(self.metadata_df.iterrows())[1]['ids']))
        return len(sample_ids)

    def _get_metadata_filename(self) -> Tuple[str, str]:
        """Get standard location for metadata file."""
        metadata_filename = os.path.join(self.data_dir, "metadata.csv.gzip")
        tasks_filename = os.path.join(self.data_dir, "tasks.json")
        return tasks_filename, metadata_filename

    def get_number_shards(self) -> int:
        """Returns the number of shards for this dataset."""
        return self.metadata_df.shape[0]

    def itershards(self) -> Iterator[Batch]:
        """Return an object that iterates over all shards in dataset.

        Datasets are stored in sharded fashion on disk. Each call to next() for the
        generator defined by this function returns the data from a particular shard.
        The order of shards returned is guaranteed to remain fixed.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """
        return (self.get_shard(i) for i in range(self.get_number_shards()))

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """ Get an object that iterates over minibatches from the dataset.

        It is guaranteed that the number of batches returned is
        `math.ceil(len(dataset)/batch_size)`. Each minibatch is returned as
        a tuple of four numpy arrays: `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default None)
            Number of elements in a batch. If None, then it yields batches
            with size equal to the size of each individual shard.
        epoch: int, default 1
            Number of epochs to walk over dataset
        deterministic: bool, default False
            Whether or not we should should shuffle each shard before
            generating the batches.  Note that this is only local in the
            sense that it does not ever mix between different shards.
        pad_batches: bool, default False
            Whether or not we should pad the last batch, globally, such that
            it has exactly batch_size elements.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """
        shard_indices = list(range(self.get_number_shards()))
        return self._iterbatches_from_shards(shard_indices, batch_size, epochs,
                                             deterministic, pad_batches)

    def _iterbatches_from_shards(self,
                                 shard_indices: Sequence[int],
                                 batch_size: Optional[int] = None,
                                 epochs: int = 1,
                                 deterministic: bool = False,
                                 pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over batches from a restricted set of shards."""

        def iterate(dataset: DiskDataset, batch_size: Optional[int],
                    epochs: int):
            num_shards = len(shard_indices)
            if deterministic:
                shard_perm = np.arange(num_shards)

            # (ytz): Depending on the application, thread-based pools may be faster
            # than process based pools, since process based pools need to pickle/serialize
            # objects as an extra overhead. Also, as hideously as un-thread safe this looks,
            # we're actually protected by the GIL.
            # mp.dummy aliases ThreadPool to Pool
            pool = Pool(1)

            if batch_size is None:
                num_global_batches = num_shards
            else:
                num_global_batches = math.ceil(dataset.get_shape()[0][0] /
                                               batch_size)

            for epoch in range(epochs):
                if not deterministic:
                    shard_perm = np.random.permutation(num_shards)
                next_shard = pool.apply_async(dataset.get_shard,
                                              (shard_indices[shard_perm[0]],))
                cur_global_batch = 0
                cur_shard = 0
                carry = None

                while cur_global_batch < num_global_batches:

                    X, y, w, ids = next_shard.get()
                    if cur_shard < num_shards - 1:
                        next_shard = pool.apply_async(
                            dataset.get_shard,
                            (shard_indices[shard_perm[cur_shard + 1]],))
                    elif epoch == epochs - 1:
                        pool.close()

                    if carry is not None:
                        X = np.concatenate([carry[0], X], axis=0)
                        if y is not None:
                            y = np.concatenate([carry[1], y], axis=0)
                        if w is not None:
                            w = np.concatenate([carry[2], w], axis=0)
                        ids = np.concatenate([carry[3], ids], axis=0)
                        carry = None

                    n_shard_samples = X.shape[0]
                    cur_local_batch = 0
                    if batch_size is None:
                        shard_batch_size = n_shard_samples
                    else:
                        shard_batch_size = batch_size

                    if n_shard_samples == 0:
                        cur_shard += 1
                        if batch_size is None:
                            cur_global_batch += 1
                        continue

                    num_local_batches = math.ceil(n_shard_samples /
                                                  shard_batch_size)
                    if not deterministic:
                        sample_perm = np.random.permutation(n_shard_samples)
                    else:
                        sample_perm = np.arange(n_shard_samples)

                    while cur_local_batch < num_local_batches:
                        start = cur_local_batch * shard_batch_size
                        end = min(n_shard_samples,
                                  (cur_local_batch + 1) * shard_batch_size)

                        indices = range(start, end)
                        perm_indices = sample_perm[indices]
                        X_b = X[perm_indices]

                        if y is not None:
                            y_b = y[perm_indices]
                        else:
                            y_b = None

                        if w is not None:
                            w_b = w[perm_indices]
                        else:
                            w_b = None

                        ids_b = ids[perm_indices]

                        assert len(X_b) <= shard_batch_size
                        if len(
                                X_b
                        ) < shard_batch_size and cur_shard != num_shards - 1:
                            assert carry is None
                            carry = [X_b, y_b, w_b, ids_b]
                        else:

                            # (ytz): this skips everything except possibly the last shard
                            if pad_batches:
                                (X_b, y_b, w_b,
                                 ids_b) = pad_batch(shard_batch_size, X_b, y_b,
                                                    w_b, ids_b)

                            yield X_b, y_b, w_b, ids_b
                            cur_global_batch += 1
                        cur_local_batch += 1
                    cur_shard += 1

        return iterate(self, batch_size, epochs)

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.

        Examples
        --------
        >>> dataset = DiskDataset.from_numpy(np.ones((2,2)), np.ones((2,1)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x.tolist(), y.tolist(), w.tolist(), id)
        [1.0, 1.0] [1.0] [1.0] 0
        [1.0, 1.0] [1.0] [1.0] 1
        """

        def iterate(dataset):
            for (X_shard, y_shard, w_shard, ids_shard) in dataset.itershards():
                n_samples = X_shard.shape[0]
                for i in range(n_samples):

                    def sanitize(elem):
                        if elem is None:
                            return None
                        else:
                            return elem[i]

                    yield map(sanitize, [X_shard, y_shard, w_shard, ids_shard])

        return iterate(self)

    def transform(self,
                  transformer: "dc.trans.Transformer",
                  parallel: bool = False,
                  out_dir: Optional[str] = None,
                  **args) -> "DiskDataset":
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:
        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple times
        with different subsets of the data.  Each time it is called, it should
        transform the samples and return the transformed data.

        Parameters
        ----------
        transformer: dc.trans.Transformer
            The transformation to apply to each sample in the dataset.
        parallel: bool, default False
            If True, use multiple processes to transform the dataset in parallel.
        out_dir: str, optional (default None)
            The directory to save the new dataset in. If this is omitted, a
            temporary directory is created automaticall.

        Returns
        -------
        DiskDataset
            A newly constructed Dataset object
        """
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        tasks = self.get_task_names()
        n_shards = self.get_number_shards()

        time1 = time.time()
        if parallel:
            results = []
            pool = multiprocessing.Pool()
            for i in range(self.get_number_shards()):
                row = self.metadata_df.iloc[i]
                X_file = os.path.join(self.data_dir, row['X'])
                if row['y'] is not None:
                    y_file: Optional[str] = os.path.join(
                        self.data_dir, row['y'])
                else:
                    y_file = None
                if row['w'] is not None:
                    w_file: Optional[str] = os.path.join(
                        self.data_dir, row['w'])
                else:
                    w_file = None
                ids_file = os.path.join(self.data_dir, row['ids'])
                results.append(
                    pool.apply_async(DiskDataset._transform_shard,
                                     (transformer, i, X_file, y_file, w_file,
                                      ids_file, out_dir, tasks)))
            pool.close()
            metadata_rows = [r.get() for r in results]
            metadata_df = DiskDataset._construct_metadata(metadata_rows)
            DiskDataset._save_metadata(metadata_df, out_dir, tasks)
            dataset = DiskDataset(out_dir)
        else:

            def generator():
                for shard_num, row in self.metadata_df.iterrows():
                    logger.info("Transforming shard %d/%d" %
                                (shard_num, n_shards))
                    X, y, w, ids = self.get_shard(shard_num)
                    newx, newy, neww, newids = transformer.transform_array(
                        X, y, w, ids)
                    yield (newx, newy, neww, newids)

            dataset = DiskDataset.create_dataset(generator(),
                                                 data_dir=out_dir,
                                                 tasks=tasks)
        time2 = time.time()
        logger.info("TIMING: transforming took %0.3f s" % (time2 - time1))
        return dataset

    @staticmethod
    def _transform_shard(transformer: "dc.trans.Transformer", shard_num: int,
                         X_file: str, y_file: str, w_file: str, ids_file: str,
                         out_dir: str,
                         tasks: np.ndarray) -> List[Optional[str]]:
        """This is called by transform() to transform a single shard."""
        X = None if X_file is None else np.array(load_from_disk(X_file))
        y = None if y_file is None else np.array(load_from_disk(y_file))
        w = None if w_file is None else np.array(load_from_disk(w_file))
        ids = np.array(load_from_disk(ids_file))
        X, y, w, ids = transformer.transform_array(X, y, w, ids)
        basename = "shard-%d" % shard_num
        return DiskDataset.write_data_to_disk(out_dir, basename, X, y, w, ids)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

        Each value returned by the Dataset's iterator is a tuple of (X, y, w, id)
        containing the data for one batch, or for a single sample if batch_size is None.

        Parameters
        ----------
        epochs: int, default 1
            The number of times to iterate over the Dataset
        deterministic: bool, default False
            If True, the data is produced in order. If False, a different
            random permutation of the data is used for each epoch.
        batch_size: int, optional (default None)
            The number of samples to return in each batch. If None, each returned
            value is a single sample.

        Returns
        -------
        torch.utils.data.IterableDataset
            `torch.utils.data.IterableDataset` that iterates over the data in
            this dataset.

        Note
        ----
        This method requires PyTorch to be installed.
        """
        try:
            from deepchem.data.pytorch_datasets import _TorchDiskDataset
        except:
            raise ImportError("This method requires PyTorch to be installed.")

        pytorch_ds = _TorchDiskDataset(disk_dataset=self,
                                       epochs=epochs,
                                       deterministic=deterministic,
                                       batch_size=batch_size)
        return pytorch_ds

    @staticmethod
    def from_numpy(X: ArrayLike,
                   y: Optional[ArrayLike] = None,
                   w: Optional[ArrayLike] = None,
                   ids: Optional[ArrayLike] = None,
                   tasks: Optional[ArrayLike] = None,
                   data_dir: Optional[str] = None) -> "DiskDataset":
        """Creates a DiskDataset object from specified Numpy arrays.

        Parameters
        ----------
        X: np.ndarray
            Feature array.
        y: np.ndarray, optional (default None)
            Labels array.
        w: np.ndarray, optional (default None)
            Weights array.
        ids: np.ndarray, optional (default None)
            Identifiers array.
        tasks: Sequence, optional (default None)
            Tasks in this dataset
        data_dir: str, optional (default None)
            The directory to write this dataset to. If none is specified, will use
            a temporary directory instead.

        Returns
        -------
        DiskDataset
            A new `DiskDataset` constructed from the provided information.
        """
        # To unify shape handling so from_numpy behaves like NumpyDataset, we just
        # make a NumpyDataset under the hood
        dataset = NumpyDataset(X, y, w, ids)
        if tasks is None:
            tasks = dataset.get_task_names()

        # raw_data = (X, y, w, ids)
        return DiskDataset.create_dataset(
            [(dataset.X, dataset.y, dataset.w, dataset.ids)],
            data_dir=data_dir,
            tasks=tasks)

    @staticmethod
    def merge(datasets: Iterable["Dataset"],
              merge_dir: Optional[str] = None) -> "DiskDataset":
        """Merges provided datasets into a merged dataset.

        Parameters
        ----------
        datasets: Iterable[Dataset]
            List of datasets to merge.
        merge_dir: str, optional (default None)
            The new directory path to store the merged DiskDataset.

        Returns
        -------
        DiskDataset
            A merged DiskDataset.
        """
        if merge_dir is not None:
            if not os.path.exists(merge_dir):
                os.makedirs(merge_dir)
        else:
            merge_dir = tempfile.mkdtemp()

        # Protect against generator exhaustion
        datasets = list(datasets)

        # This ensures tasks are consistent for all datasets
        tasks = []
        for dataset in datasets:
            try:
                tasks.append(dataset.tasks)  # type: ignore
            except AttributeError:
                pass
        if tasks:
            task_tuples = [tuple(task_list) for task_list in tasks]
            if len(tasks) < len(datasets) or len(set(task_tuples)) > 1:
                raise ValueError(
                    'Cannot merge datasets with different task specifications')
            merge_tasks = tasks[0]
        else:
            merge_tasks = []

        # determine the shard sizes of the datasets to merge
        shard_sizes = []
        for dataset in datasets:
            if hasattr(dataset, 'get_shard_size'):
                shard_sizes.append(dataset.get_shard_size())  # type: ignore
            # otherwise the entire dataset is the "shard size"
            else:
                shard_sizes.append(len(dataset))

        def generator():
            for ind, dataset in enumerate(datasets):
                logger.info("Merging in dataset %d/%d" % (ind, len(datasets)))
                if hasattr(dataset, 'itershards'):
                    for (X, y, w, ids) in dataset.itershards():
                        yield (X, y, w, ids)
                else:
                    yield (dataset.X, dataset.y, dataset.w, dataset.ids)

        merged_dataset = DiskDataset.create_dataset(generator(),
                                                    data_dir=merge_dir,
                                                    tasks=merge_tasks)

        # we must reshard the dataset to have a uniform size
        # choose the smallest shard size
        if len(set(shard_sizes)) > 1:
            merged_dataset.reshard(min(shard_sizes))

        return merged_dataset

    def subset(self,
               shard_nums: Sequence[int],
               subset_dir: Optional[str] = None) -> "DiskDataset":
        """Creates a subset of the original dataset on disk.

        Parameters
        ----------
        shard_nums: Sequence[int]
            The indices of shard to extract from the original DiskDataset.
        subset_dir: str, optional (default None)
            The new directory path to store the subset DiskDataset.

        Returns
        -------
        DiskDataset
            A subset DiskDataset.
        """
        if subset_dir is not None:
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)
        else:
            subset_dir = tempfile.mkdtemp()
        tasks = self.get_task_names()

        def generator():
            for shard_num, row in self.metadata_df.iterrows():
                if shard_num not in shard_nums:
                    continue
                X, y, w, ids = self.get_shard(shard_num)
                yield (X, y, w, ids)

        return DiskDataset.create_dataset(generator(),
                                          data_dir=subset_dir,
                                          tasks=tasks)

    def sparse_shuffle(self) -> None:
        """Shuffling that exploits data sparsity to shuffle large datasets.

        If feature vectors are sparse, say circular fingerprints or any other
        representation that contains few nonzero values, it can be possible to
        exploit the sparsity of the vector to simplify shuffles. This method
        implements a sparse shuffle by compressing sparse feature vectors down
        into a compressed representation, then shuffles this compressed dataset in
        memory and writes the results to disk.

        Note
        ----
        This method only works for 1-dimensional feature vectors (does not work
        for tensorial featurizations). Note that this shuffle is performed in
        place.
        """
        time1 = time.time()
        shard_size = self.get_shard_size()
        num_shards = self.get_number_shards()
        X_sparse_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        ids_list: List[np.ndarray] = []
        num_features = -1
        for i in range(num_shards):
            logger.info("Sparsifying shard %d/%d" % (i, num_shards))
            (X_s, y_s, w_s, ids_s) = self.get_shard(i)
            if num_features == -1:
                num_features = X_s.shape[1]
            X_sparse = sparsify_features(X_s)
            X_sparse_list, y_list, w_list, ids_list = (
                X_sparse_list + [X_sparse], y_list + [y_s], w_list + [w_s],
                ids_list + [np.atleast_1d(np.squeeze(ids_s))])
        # Get full dataset in memory
        (X_sparse, y, w, ids) = (np.vstack(X_sparse_list), np.vstack(y_list),
                                 np.vstack(w_list), np.concatenate(ids_list))
        # Shuffle in memory
        num_samples = len(X_sparse)
        permutation = np.random.permutation(num_samples)
        X_sparse, y, w, ids = (X_sparse[permutation], y[permutation],
                               w[permutation], ids[permutation])
        # Write shuffled shards out to disk
        for i in range(num_shards):
            logger.info("Sparse shuffling shard %d/%d" % (i, num_shards))
            start, stop = i * shard_size, (i + 1) * shard_size
            (X_sparse_s, y_s, w_s,
             ids_s) = (X_sparse[start:stop], y[start:stop], w[start:stop],
                       ids[start:stop])
            X_s = densify_features(X_sparse_s, num_features)
            self.set_shard(i, X_s, y_s, w_s, ids_s)
        time2 = time.time()
        logger.info("TIMING: sparse_shuffle took %0.3f s" % (time2 - time1))

    def complete_shuffle(self, data_dir: Optional[str] = None) -> Dataset:
        """Completely shuffle across all data, across all shards.

        Note
        ----
        The algorithm used for this complete shuffle is O(N^2) where N is the
        number of shards. It simply constructs each shard of the output dataset
        one at a time. Since the complete shuffle can take a long time, it's
        useful to watch the logging output. Each shuffled shard is constructed
        using select() which logs as it selects from each original shard. This
        will results in O(N^2) logging statements, one for each extraction of
        shuffled shard i's contributions from original shard j.

        Parameters
        ----------
        data_dir: Optional[str], (default None)
            Directory to write the shuffled dataset to. If none is specified a
            temporary directory will be used.

        Returns
        -------
        DiskDataset
            A DiskDataset whose data is a randomly shuffled version of this dataset.
        """
        N = len(self)
        perm = np.random.permutation(N).tolist()
        shard_size = self.get_shard_size()
        return self.select(perm, data_dir, shard_size)

    def shuffle_each_shard(self,
                           shard_basenames: Optional[List[str]] = None) -> None:
        """Shuffles elements within each shard of the dataset.

        Parameters
        ----------
        shard_basenames: List[str], optional (default None)
            The basenames for each shard. If this isn't specified, will assume the
            basenames of form "shard-i" used by `create_dataset` and `reshard`.
        """
        # Shuffle the arrays corresponding to each row in metadata_df
        n_rows = len(self.metadata_df.index)
        if shard_basenames is not None:
            if len(shard_basenames) != n_rows:
                raise ValueError(
                    "shard_basenames must provide a basename for each shard in this DiskDataset."
                )
        else:
            shard_basenames = [
                "shard-%d" % shard_num for shard_num in range(n_rows)
            ]
        for i, basename in zip(range(n_rows), shard_basenames):
            logger.info("Shuffling shard %d/%d" % (i, n_rows))
            X, y, w, ids = self.get_shard(i)
            n = X.shape[0]
            permutation = np.random.permutation(n)
            X, y, w, ids = (X[permutation], y[permutation], w[permutation],
                            ids[permutation])
            DiskDataset.write_data_to_disk(self.data_dir, basename, X, y, w,
                                           ids)
        # Reset cache
        self._cached_shards = None

    def shuffle_shards(self) -> None:
        """Shuffles the order of the shards for this dataset."""
        metadata_rows = self.metadata_df.values.tolist()
        random.shuffle(metadata_rows)
        self.metadata_df = DiskDataset._construct_metadata(metadata_rows)
        self.save_to_disk()

    def get_shard(self, i: int) -> Batch:
        """Retrieves data for the i-th shard from disk.

        Parameters
        ----------
        i: int
            Shard index for shard to retrieve batch from.

        Returns
        -------
        Batch
            A batch data for i-th shard.
        """

        # See if we have a cached copy of this shard.
        if self._cached_shards is None:
            self._cached_shards = [None] * self.get_number_shards()
            self._cache_used = 0
        if self._cached_shards[i] is not None:
            shard = self._cached_shards[i]
            return (shard.X, shard.y, shard.w, shard.ids)

        # We don't, so load it from disk.
        row = self.metadata_df.iloc[i]
        X = np.array(load_from_disk(os.path.join(self.data_dir, row['X'])))

        if row['y'] is not None:
            y: Optional[np.ndarray] = np.array(
                load_from_disk(os.path.join(self.data_dir, row['y'])))
        else:
            y = None

        if row['w'] is not None:
            # TODO (ytz): Under what condition does this exist but the file itself doesn't?
            w_filename = os.path.join(self.data_dir, row['w'])
            if os.path.exists(w_filename):
                w: Optional[np.ndarray] = np.array(load_from_disk(w_filename))
            elif y is not None:
                if len(y.shape) == 1:
                    w = np.ones(y.shape[0], np.float32)
                else:
                    w = np.ones((y.shape[0], 1), np.float32)
            else:
                w = None
        else:
            w = None

        ids = np.array(load_from_disk(os.path.join(self.data_dir, row['ids'])),
                       dtype=object)

        # Try to cache this shard for later use.  Since the normal usage pattern is
        # a series of passes through the whole dataset, there's no point doing
        # anything fancy.  It never makes sense to evict another shard from the
        # cache to make room for this one, because we'll probably want that other
        # shard again before the next time we want this one.  So just cache as many
        # as we can and then stop.

        shard = _Shard(X, y, w, ids)
        shard_size = X.nbytes + ids.nbytes
        if y is not None:
            shard_size += y.nbytes
        if w is not None:
            shard_size += w.nbytes
        if self._cache_used + shard_size < self._memory_cache_size:
            self._cached_shards[i] = shard
            self._cache_used += shard_size
        return (shard.X, shard.y, shard.w, shard.ids)

    def get_shard_ids(self, i: int) -> np.ndarray:
        """Retrieves the list of IDs for the i-th shard from disk.

        Parameters
        ----------
        i: int
            Shard index for shard to retrieve weights from.

        Returns
        -------
        np.ndarray
            A numpy array of ids for i-th shard.
        """

        if self._cached_shards is not None and self._cached_shards[
                i] is not None:
            return self._cached_shards[i].ids
        row = self.metadata_df.iloc[i]
        return np.array(load_from_disk(os.path.join(self.data_dir, row['ids'])),
                        dtype=object)

    def get_shard_y(self, i: int) -> np.ndarray:
        """Retrieves the labels for the i-th shard from disk.

        Parameters
        ----------
        i: int
            Shard index for shard to retrieve labels from.

        Returns
        -------
        np.ndarray
            A numpy array of labels for i-th shard.
        """

        if self._cached_shards is not None and self._cached_shards[
                i] is not None:
            return self._cached_shards[i].y
        row = self.metadata_df.iloc[i]
        return np.array(load_from_disk(os.path.join(self.data_dir, row['y'])))

    def get_shard_w(self, i: int) -> np.ndarray:
        """Retrieves the weights for the i-th shard from disk.

        Parameters
        ----------
        i: int
            Shard index for shard to retrieve weights from.

        Returns
        -------
        np.ndarray
            A numpy array of weights for i-th shard.
        """

        if self._cached_shards is not None and self._cached_shards[
                i] is not None:
            return self._cached_shards[i].w
        row = self.metadata_df.iloc[i]
        return np.array(load_from_disk(os.path.join(self.data_dir, row['w'])))

    def add_shard(self,
                  X: np.ndarray,
                  y: Optional[np.ndarray] = None,
                  w: Optional[np.ndarray] = None,
                  ids: Optional[np.ndarray] = None) -> None:
        """Adds a data shard.

        Parameters
        ----------
        X: np.ndarray
            Feature array.
        y: np.ndarray, optioanl (default None)
            Labels array.
        w: np.ndarray, optioanl (default None)
            Weights array.
        ids: np.ndarray, optioanl (default None)
            Identifiers array.
        """
        metadata_rows = self.metadata_df.values.tolist()
        shard_num = len(metadata_rows)
        basename = "shard-%d" % shard_num
        metadata_rows.append(
            DiskDataset.write_data_to_disk(self.data_dir, basename, X, y, w,
                                           ids))
        self.metadata_df = DiskDataset._construct_metadata(metadata_rows)
        self.save_to_disk()

    def set_shard(self,
                  shard_num: int,
                  X: np.ndarray,
                  y: Optional[np.ndarray] = None,
                  w: Optional[np.ndarray] = None,
                  ids: Optional[np.ndarray] = None) -> None:
        """Writes data shard to disk.

        Parameters
        ----------
        shard_num: int
            Shard index for shard to set new data.
        X: np.ndarray
            Feature array.
        y: np.ndarray, optioanl (default None)
            Labels array.
        w: np.ndarray, optioanl (default None)
            Weights array.
        ids: np.ndarray, optioanl (default None)
            Identifiers array.
        """
        basename = "shard-%d" % shard_num
        DiskDataset.write_data_to_disk(self.data_dir, basename, X, y, w, ids)
        self._cached_shards = None
        self.legacy_metadata = True

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None,
               select_shard_size: Optional[int] = None,
               output_numpy_dataset: Optional[bool] = False) -> Dataset:
        """Creates a new dataset from a selection of indices from self.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.random.rand(10, 10)
        >>> dataset = dc.data.DiskDataset.from_numpy(X)
        >>> selected = dataset.select([1, 3, 4])
        >>> len(selected)
        3

        Parameters
        ----------
        indices: Sequence
            List of indices to select.
        select_dir: str, optional (default None)
            Path to new directory that the selected indices will be copied to.
        select_shard_size: Optional[int], (default None)
            If specified, the shard-size to use for output selected `DiskDataset`.
            If not output_numpy_dataset, then this is set to this current dataset's
            shard size if not manually specified.
        output_numpy_dataset: Optional[bool], (default False)
            If True, output an in-memory `NumpyDataset` instead of a `DiskDataset`.
            Note that `select_dir` and `select_shard_size` must be `None` if this
            is `True`

        Returns
        -------
        Dataset
            A dataset containing the selected samples. The default dataset is `DiskDataset`.
            If `output_numpy_dataset` is True, the dataset is `NumpyDataset`.
        """
        if output_numpy_dataset and (select_dir is not None or
                                     select_shard_size is not None):
            raise ValueError(
                "If output_numpy_dataset is set, then select_dir and select_shard_size must both be None"
            )
        if output_numpy_dataset:
            # When outputting a NumpyDataset, we have 1 in-memory shard
            select_shard_size = len(indices)
        else:
            if select_dir is not None:
                if not os.path.exists(select_dir):
                    os.makedirs(select_dir)
            else:
                select_dir = tempfile.mkdtemp()
            if select_shard_size is None:
                select_shard_size = self.get_shard_size()

        tasks = self.get_task_names()
        N = len(indices)
        n_shards = self.get_number_shards()
        # Handle edge case with empty indices
        if not N:
            if not output_numpy_dataset:
                return DiskDataset.create_dataset([],
                                                  data_dir=select_dir,
                                                  tasks=tasks)
            else:
                return NumpyDataset(np.array([]), np.array([]), np.array([]),
                                    np.array([]))

        # We use two loops here. The outer while loop walks over selection shards
        # (the chunks of the indices to select that should go into separate
        # output shards), while the inner for loop walks over the shards in the
        # source datasets to select out the shard indices from that  source shard
        def generator():
            start = 0
            select_shard_num = 0
            while start < N:
                logger.info("Constructing selection output shard %d" %
                            (select_shard_num + 1))
                end = min(start + select_shard_size, N)
                select_shard_indices = indices[start:end]
                sorted_indices = np.array(
                    sorted(select_shard_indices)).astype(int)

                Xs, ys, ws, ids_s = [], [], [], []
                count, indices_count = 0, 0
                for shard_num in range(self.get_number_shards()):
                    logger.info(
                        "Selecting from input shard %d/%d for selection output shard %d"
                        % (shard_num + 1, n_shards, select_shard_num + 1))
                    if self.legacy_metadata:
                        ids = self.get_shard_ids(shard_num)
                        shard_len = len(ids)
                    else:
                        shard_X_shape, _, _, _ = self._get_shard_shape(
                            shard_num)
                        if len(shard_X_shape) > 0:
                            shard_len = shard_X_shape[0]
                        else:
                            shard_len = 0
                    # Find indices which rest in this shard
                    num_shard_elts = 0
                    while sorted_indices[indices_count +
                                         num_shard_elts] < count + shard_len:
                        num_shard_elts += 1
                        if (indices_count +
                                num_shard_elts) >= len(sorted_indices):
                            break
                    if num_shard_elts == 0:
                        count += shard_len
                        continue
                    else:
                        X, y, w, ids = self.get_shard(shard_num)
                    # Need to offset indices to fit within shard_size
                    shard_inds = sorted_indices[indices_count:indices_count +
                                                num_shard_elts] - count
                    # Handle empty case where no data from this shard needed
                    X_sel = X[shard_inds]
                    # Handle the case of datasets with y/w missing
                    if y is not None:
                        y_sel = y[shard_inds]
                    else:
                        y_sel = np.array([])
                    if w is not None:
                        w_sel = w[shard_inds]
                    else:
                        w_sel = np.array([])
                    ids_sel = ids[shard_inds]
                    Xs.append(X_sel)
                    ys.append(y_sel)
                    ws.append(w_sel)
                    ids_s.append(ids_sel)
                    indices_count += num_shard_elts
                    count += shard_len
                    # Break if all indices have been used up already
                    if indices_count >= len(sorted_indices):
                        break
                # Note these will be in the sorted order
                X = np.concatenate(Xs, axis=0)
                y = np.concatenate(ys, axis=0)
                w = np.concatenate(ws, axis=0)
                ids = np.concatenate(ids_s, axis=0)
                # We need to recover the original ordering. We can do this by using
                # np.where to find the locatios of the original indices in the sorted
                # indices.
                reverted_indices = np.array(
                    # We know there's only one match for np.where since this is a
                    # permutation, so the [0][0] pulls out the exact match location.
                    [
                        np.where(sorted_indices == orig_index)[0][0]
                        for orig_index in select_shard_indices
                    ])
                if y.size == 0:
                    tup_y = y
                else:
                    tup_y = y[reverted_indices]
                if w.size == 0:
                    tup_w = w
                else:
                    tup_w = w[reverted_indices]
                X, ids = X[reverted_indices], ids[reverted_indices]
                yield (X, tup_y, tup_w, ids)
                start = end
                select_shard_num += 1

        if not output_numpy_dataset:
            return DiskDataset.create_dataset(generator(),
                                              data_dir=select_dir,
                                              tasks=tasks)
        else:
            X, y, w, ids = next(generator())
            return NumpyDataset(X, y, w, ids)

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        if len(self) == 0:
            return np.array([])
        ids = []
        for i in range(self.get_number_shards()):
            ids.append(np.atleast_1d(np.squeeze(self.get_shard_ids(i))))
        return np.concatenate(ids)

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        Xs = []
        one_dimensional = False
        for (X_b, _, _, _) in self.itershards():
            Xs.append(X_b)
            if len(X_b.shape) == 1:
                one_dimensional = True
        if not one_dimensional:
            return np.vstack(Xs)
        else:
            return np.concatenate(Xs)

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        if len(self) == 0 or len(self.get_task_names()) == 0:
            return np.array([])
        ys = []
        one_dimensional = False
        for i in range(self.get_number_shards()):
            y_b = self.get_shard_y(i)
            ys.append(y_b)
            if len(y_b.shape) == 1:
                one_dimensional = True
        if not one_dimensional:
            return np.vstack(ys)
        else:
            return np.concatenate(ys)

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array."""
        ws = []
        one_dimensional = False
        for i in range(self.get_number_shards()):
            w_b = self.get_shard_w(i)
            ws.append(w_b)
            if len(w_b.shape) == 1:
                one_dimensional = True
        if not one_dimensional:
            return np.vstack(ws)
        else:
            return np.concatenate(ws)

    @property
    def memory_cache_size(self) -> int:
        """Get the size of the memory cache for this dataset, measured in bytes."""
        return self._memory_cache_size

    @memory_cache_size.setter
    def memory_cache_size(self, size: int) -> None:
        """Get the size of the memory cache for this dataset, measured in bytes."""
        self._memory_cache_size = size
        if self._cache_used > size:
            self._cached_shards = None

    def __len__(self) -> int:
        """Finds number of elements in dataset."""
        total = 0
        for _, row in self.metadata_df.iterrows():
            y = load_from_disk(os.path.join(self.data_dir, row['ids']))
            total += len(y)
        return total

    def _get_shard_shape(self,
                         shard_num: int) -> Tuple[Shape, Shape, Shape, Shape]:
        """Finds the shape of the specified shard."""
        if self.legacy_metadata:
            raise ValueError(
                "This function requires the new metadata format to be called. Please reshard this dataset by calling the reshard() method."
            )
        n_tasks = len(self.get_task_names())
        row = self.metadata_df.iloc[shard_num]
        if row['X_shape'] is not None:
            shard_X_shape = make_tuple(str(row['X_shape']))
        else:
            shard_X_shape = tuple()
        if n_tasks > 0:
            if row['y_shape'] is not None:
                shard_y_shape = make_tuple(str(row['y_shape']))
            else:
                shard_y_shape = tuple()
            if row['w_shape'] is not None:
                shard_w_shape = make_tuple(str(row['w_shape']))
            else:
                shard_w_shape = tuple()
        else:
            shard_y_shape = tuple()
            shard_w_shape = tuple()
        if row['ids_shape'] is not None:
            shard_ids_shape = make_tuple(str(row['ids_shape']))
        else:
            shard_ids_shape = tuple()
        X_shape, y_shape, w_shape, ids_shape = tuple(
            np.array(shard_X_shape)), tuple(np.array(shard_y_shape)), tuple(
                np.array(shard_w_shape)), tuple(np.array(shard_ids_shape))
        return X_shape, y_shape, w_shape, ids_shape

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        """Finds shape of dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        n_tasks = len(self.get_task_names())
        n_rows = len(self.metadata_df.index)
        # If shape metadata is available use it to directly compute shape from
        # metadata
        if not self.legacy_metadata:
            for shard_num in range(n_rows):
                shard_X_shape, shard_y_shape, shard_w_shape, shard_ids_shape = self._get_shard_shape(
                    shard_num)
                if shard_num == 0:
                    X_shape, y_shape, w_shape, ids_shape = np.array(
                        shard_X_shape), np.array(shard_y_shape), np.array(
                            shard_w_shape), np.array(shard_ids_shape)
                else:
                    X_shape[0] += shard_X_shape[0]
                    if n_tasks > 0:
                        y_shape[0] += shard_y_shape[0]
                        w_shape[0] += shard_w_shape[0]
                    ids_shape[0] += shard_ids_shape[0]
            return tuple(X_shape), tuple(y_shape), tuple(w_shape), tuple(
                ids_shape)
        # In absense of shape metadata, fall back to loading data from disk to
        # find shape.
        else:
            for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
                if shard_num == 0:
                    X_shape = np.array(X.shape)
                    if n_tasks > 0:
                        y_shape = np.array(y.shape)
                        w_shape = np.array(w.shape)
                    else:
                        y_shape = np.array([])
                        w_shape = np.array([])
                    ids_shape = np.array(ids.shape)
                else:
                    X_shape[0] += np.array(X.shape)[0]
                    if n_tasks > 0:
                        y_shape[0] += np.array(y.shape)[0]
                        w_shape[0] += np.array(w.shape)[0]
                    ids_shape[0] += np.array(ids.shape)[0]
            return tuple(X_shape), tuple(y_shape), tuple(w_shape), tuple(
                ids_shape)

    def get_label_means(self) -> pd.DataFrame:
        """Return pandas series of label means."""
        return self.metadata_df["y_means"]

    def get_label_stds(self) -> pd.DataFrame:
        """Return pandas series of label stds."""
        return self.metadata_df["y_stds"]


class ImageDataset(Dataset):
    """A Dataset that loads data from image files on disk."""

    def __init__(self,
                 X: Union[np.ndarray, List[str]],
                 y: Optional[Union[np.ndarray, List[str]]],
                 w: Optional[ArrayLike] = None,
                 ids: Optional[ArrayLike] = None) -> None:
        """Create a dataset whose X and/or y array is defined by image files on disk.

        Parameters
        ----------
        X: np.ndarray or List[str]
            The dataset's input data.  This may be either a single NumPy
            array directly containing the data, or a list containing the
            paths to the image files
        y: np.ndarray or List[str]
            The dataset's labels.  This may be either a single NumPy array
            directly containing the data, or a list containing the paths to
            the image files
        w: np.ndarray, optional (default None)
            a 1D or 2D array containing the weights for each sample or
            sample/task pair
        ids: np.ndarray, optional (default None)
            the sample IDs
        """
        n_samples = len(X)
        if y is None:
            y = np.zeros((n_samples,))
        self._X_shape = self._find_array_shape(X)
        self._y_shape = self._find_array_shape(y)
        if w is None:
            if len(self._y_shape) == 0:
                # Case n_samples should be 1
                if n_samples != 1:
                    raise ValueError("y can only be a scalar if n_samples == 1")
                w = np.ones_like(y)
            elif len(self._y_shape) == 1:
                w = np.ones(self._y_shape[0], np.float32)
            else:
                w = np.ones((self._y_shape[0], 1), np.float32)
        if ids is None:
            if not isinstance(X, np.ndarray):
                ids = X
            elif not isinstance(y, np.ndarray):
                ids = y
            else:
                ids = np.arange(n_samples)
        self._X = X
        self._y = y
        self._w = np.asarray(w)
        self._ids = np.array(ids, dtype=object)

    def _find_array_shape(self, array: Union[np.ndarray, List[str]]) -> Shape:
        if isinstance(array, np.ndarray):
            return array.shape
        image_shape = load_image_files([array[0]]).shape[1:]
        return tuple(np.concatenate([[len(array)], image_shape]))

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return self._X_shape[0]

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X_shape, self._y_shape, self._w.shape, self._ids.shape

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        if len(self._y_shape) < 2:
            return np.array([0])
        return np.arange(self._y_shape[1])

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        if isinstance(self._X, np.ndarray):
            return self._X
        return load_image_files(self._X)

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        if isinstance(self._y, np.ndarray):
            return self._y
        return load_image_files(self._y)

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays:
        `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default None)
            Number of elements in each batch.
        epochs: int, default 1
            Number of epochs to walk over dataset.
        deterministic: bool, default False
            If True, follow deterministic order.
        pad_batches: bool, default False
            If True, pad each batch to `batch_size`.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """

        def iterate(dataset, batch_size, epochs, deterministic, pad_batches):
            n_samples = dataset._X_shape[0]
            if deterministic:
                sample_perm = np.arange(n_samples)
            if batch_size is None:
                batch_size = n_samples
            for epoch in range(epochs):
                if not deterministic:
                    sample_perm = np.random.permutation(n_samples)
                batch_idx = 0
                num_batches = np.math.ceil(n_samples / batch_size)
                while batch_idx < num_batches:
                    start = batch_idx * batch_size
                    end = min(n_samples, (batch_idx + 1) * batch_size)
                    indices = range(start, end)
                    perm_indices = sample_perm[indices]
                    if isinstance(dataset._X, np.ndarray):
                        X_batch = dataset._X[perm_indices]
                    else:
                        X_batch = load_image_files(
                            [dataset._X[i] for i in perm_indices])
                    if isinstance(dataset._y, np.ndarray):
                        y_batch = dataset._y[perm_indices]
                    else:
                        y_batch = load_image_files(
                            [dataset._y[i] for i in perm_indices])
                    w_batch = dataset._w[perm_indices]
                    ids_batch = dataset._ids[perm_indices]
                    if pad_batches:
                        (X_batch, y_batch, w_batch,
                         ids_batch) = pad_batch(batch_size, X_batch, y_batch,
                                                w_batch, ids_batch)
                    batch_idx += 1
                    yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self, batch_size, epochs, deterministic, pad_batches)

    def _get_image(self, array: Union[np.ndarray, List[str]],
                   indices: Union[int, np.ndarray]) -> np.ndarray:
        """Method for loading an image

        Parameters
        ----------
        array: Union[np.ndarray, List[str]]
            A numpy array which contains images or List of image filenames
        indices: Union[int, np.ndarray]
            Index you want to get the images

        Returns
        -------
        np.ndarray
            Loaded images
        """
        if isinstance(array, np.ndarray):
            return array[indices]
        if isinstance(indices, np.ndarray):
            return load_image_files([array[i] for i in indices])
        return load_image_files([array[indices]])[0]

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Returns
        -------
        Iterator[Batch]
            Iterator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """
        n_samples = self._X_shape[0]
        return ((self._get_image(self._X, i), self._get_image(self._y, i),
                 self._w[i], self._ids[i]) for i in range(n_samples))

    def transform(
        self,
        transformer: "dc.trans.Transformer",
        **args,
    ) -> "NumpyDataset":
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:

        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple times with
        different subsets of the data.  Each time it is called, it should transform
        the samples and return the transformed data.

        Parameters
        ----------
        transformer: dc.trans.Transformer
            The transformation to apply to each sample in the dataset

        Returns
        -------
        NumpyDataset
            A newly constructed NumpyDataset object
        """
        newx, newy, neww, newids = transformer.transform_array(
            self.X, self.y, self.w, self.ids)
        return NumpyDataset(newx, newy, neww, newids)

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None) -> "ImageDataset":
        """Creates a new dataset from a selection of indices from self.

        Parameters
        ----------
        indices: Sequence
            List of indices to select.
        select_dir: str, optional (default None)
            Used to provide same API as `DiskDataset`. Ignored since
            `ImageDataset` is purely in-memory.

        Returns
        -------
        ImageDataset
            A selected ImageDataset object
        """
        X: Union[List[str], np.ndarray]
        y: Union[List[str], np.ndarray]
        if isinstance(self._X, np.ndarray):
            X = self._X[indices]
        else:
            X = [self._X[i] for i in indices]
        if isinstance(self._y, np.ndarray):
            y = self._y[indices]
        else:
            y = [self._y[i] for i in indices]
        w = self._w[indices]
        ids = self._ids[indices]
        return ImageDataset(X, y, w, ids)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

        Each value returned by the Dataset's iterator is a tuple of (X, y, w, id)
        containing the data for one batch, or for a single sample if batch_size is None.

        Parameters
        ----------
        epochs: int, default 1
            The number of times to iterate over the Dataset.
        deterministic: bool, default False
            If True, the data is produced in order. If False, a different
            random permutation of the data is used for each epoch.
        batch_size: int, optional (default None)
            The number of samples to return in each batch. If None, each returned
            value is a single sample.

        Returns
        -------
        torch.utils.data.IterableDataset
            `torch.utils.data.IterableDataset` that iterates over the data in
            this dataset.

        Note
        ----
        This method requires PyTorch to be installed.
        """
        try:
            from deepchem.data.pytorch_datasets import _TorchImageDataset
        except:
            raise ValueError("This method requires PyTorch to be installed.")

        pytorch_ds = _TorchImageDataset(image_dataset=self,
                                        epochs=epochs,
                                        deterministic=deterministic,
                                        batch_size=batch_size)
        return pytorch_ds


class Databag(object):
    """A utility class to iterate through multiple datasets together.


    A `Databag` is useful when you have multiple datasets that you want
    to iterate in locksteps. This might be easiest to grasp with a
    simple code example.

    >>> ones_dataset = NumpyDataset(X=np.ones((5, 3)))
    >>> zeros_dataset = NumpyDataset(X=np.zeros((5, 3)))
    >>> databag = Databag({"ones": ones_dataset, "zeros": zeros_dataset})
    >>> for sample_dict in databag.iterbatches(batch_size=1):
    ...   print(sample_dict)
    {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
    {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
    {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
    {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
    {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}

    Note how we get a batch at a time from each of the datasets in the
    `Databag`. This can be useful for training models that combine data
    from multiple `Dataset` objects at a time.
    """

    def __init__(self, datasets: Optional[Dict[Any, Dataset]] = None) -> None:
        """Initialize this `Databag`.

        Parameters
        ----------
        datasets: dict, optional (default None)
          A dictionary mapping keys to `Dataset` objects.
        """
        if datasets is None:
            self.datasets = dict()
        else:
            self.datasets = datasets

    def add_dataset(self, key: Any, dataset: Dataset) -> None:
        """Adds a dataset to this databag.

        Parameters
        ----------
        key: Any, hashable value
            Key to be added
        dataset: Dataset
            The dataset that `key` should point to.
        """
        self.datasets[key] = dataset

    def iterbatches(self, **kwargs) -> Iterator[Dict[str, np.ndarray]]:
        """Loop through all internal datasets in the same order.

        Parameters
        ----------
        batch_size: int
            Number of samples from each dataset to return
        epochs: int
            Number of times to loop through the datasets
        pad_batches: bool
            Should all batches==batch_size

        Returns
        -------
        Iterator[Dict[str, np.ndarray]]
            Generator which yields a dictionary {key: dataset.X[batch]}
        """
        key_order = [x for x in self.datasets.keys()]
        if "epochs" in kwargs:
            epochs = kwargs['epochs']
            del kwargs['epochs']
        else:
            epochs = 1
        kwargs['deterministic'] = True
        for epoch in range(epochs):
            iterators = [
                self.datasets[x].iterbatches(**kwargs) for x in key_order
            ]
            for tup in zip(*iterators):
                m_d = {key_order[i]: tup[i][0] for i in range(len(key_order))}
                yield m_d

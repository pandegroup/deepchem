"""
Process an input dataset into a format suitable for machine learning.
"""
import os
import tempfile
import zipfile
import time
import logging
import warnings
from typing import List, Optional, Tuple, Any, Sequence, Union, Iterator

import pandas as pd
import numpy as np
from deepchem.utils.typing import OneOrMany
from deepchem.utils.data_utils import load_image_files, load_csv_files, load_json_files, load_sdf_files, unzip_file
from deepchem.feat import UserDefinedFeaturizer, Featurizer
from deepchem.data import Dataset, DiskDataset, NumpyDataset, ImageDataset
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.utils.genomics_utils import encode_bio_sequence

try:
    import yaml
    from yaml.loader import SafeLoader
except ModuleNotFoundError:
    pass

try:
    import pysam
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _convert_df_to_numpy(df: pd.DataFrame,
                         tasks: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms a dataframe containing deepchem input into numpy arrays.

    This is a private helper method intended to help parse labels and
    weights arrays from a pandas dataframe. Here `df` is a dataframe
    which has columns for each task in `tasks`. These labels are
    extracted into a labels array `y`. Weights `w` are initialized to
    all ones, but weights for any missing labels are set to 0.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe with columns for all tasks.
    tasks: List[str]
        List of tasks

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The tuple is `(w, y)`.
    """
    n_samples = df.shape[0]
    n_tasks = len(tasks)

    y = np.hstack([
        np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks
    ])
    w = np.ones((n_samples, n_tasks))
    if y.dtype.kind in ['O', 'U']:
        missing = (y == '')
        y[missing] = 0
        w[missing] = 0

    return y.astype(float), w.astype(float)


class DataLoader(object):
    """Handles loading/featurizing of data from disk.

    The main use of `DataLoader` and its child classes is to make it
    easier to load large datasets into `Dataset` objects.`

    `DataLoader` is an abstract superclass that provides a
    general framework for loading data into DeepChem. This class should
    never be instantiated directly.  To load your own type of data, make
    a subclass of `DataLoader` and provide your own implementation for
    the `create_dataset()` method.

    To construct a `Dataset` from input data, first instantiate a
    concrete data loader (that is, an object which is an instance of a
    subclass of `DataLoader`) with a given `Featurizer` object. Then
    call the data loader's `create_dataset()` method on a list of input
    files that hold the source data to process. Note that each subclass
    of `DataLoader` is specialized to handle one type of input data so
    you will have to pick the loader class suitable for your input data
    type.

    Note that it isn't necessary to use a data loader to process input
    data. You can directly use `Featurizer` objects to featurize
    provided input into numpy arrays, but note that this calculation
    will be performed in memory, so you will have to write generators
    that walk the source files and write featurized data to disk
    yourself. `DataLoader` and its subclasses make this process easier
    for you by performing this work under the hood.
    """

    def __init__(self,
                 tasks: List[str],
                 featurizer: Featurizer,
                 id_field: Optional[str] = None,
                 log_every_n: int = 1000):
        """Construct a DataLoader object.

        This constructor is provided as a template mainly. You
        shouldn't ever call this constructor directly as a user.

        Parameters
        ----------
        tasks: List[str]
            List of task names
        featurizer: Featurizer
            Featurizer to use to process data.
        id_field: str, optional (default None)
            Name of field that holds sample identifier. Note that the
            meaning of "field" depends on the input data type and can have a
            different meaning in different subclasses. For example, a CSV
            file could have a field as a column, and an SDF file could have
            a field as molecular property.
        log_every_n: int, optional (default 1000)
            Writes a logging statement this often.
        """
        if self.__class__ is DataLoader:
            raise ValueError(
                "DataLoader should never be instantiated directly. Use a subclass instead."
            )
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list.")
        self.tasks = tasks
        self.id_field = id_field
        self.user_specified_features = None
        if isinstance(featurizer, UserDefinedFeaturizer):
            self.user_specified_features = featurizer.feature_fields
        self.featurizer = featurizer
        self.log_every_n = log_every_n

    def featurize(self,
                  inputs: OneOrMany[Any],
                  data_dir: Optional[str] = None,
                  shard_size: Optional[int] = 8192) -> Dataset:
        """Featurize provided files and write to specified location.

        DEPRECATED: This method is now a wrapper for `create_dataset()`
        and calls that method under the hood.

        For large datasets, automatically shards into smaller chunks
        for convenience. This implementation assumes that the helper
        methods `_get_shards` and `_featurize_shard` are implemented and
        that each shard returned by `_get_shards` is a pandas dataframe.
        You may choose to reuse or override this method in your subclass
        implementations.

        Parameters
        ----------
        inputs: List
            List of inputs to process. Entries can be filenames or arbitrary objects.
        data_dir: str, default None
            Directory to store featurized dataset.
        shard_size: int, optional (default 8192)
            Number of examples stored in each shard.

        Returns
        -------
        Dataset
            A `Dataset` object containing a featurized representation of data
            from `inputs`.
        """
        warnings.warn(
            "featurize() is deprecated and has been renamed to create_dataset()."
            "featurize() will be removed in DeepChem 3.0", FutureWarning)
        return self.create_dataset(inputs, data_dir, shard_size)

    def create_dataset(self,
                       inputs: OneOrMany[Any],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 8192) -> Dataset:
        """Creates and returns a `Dataset` object by featurizing provided files.

        Reads in `inputs` and uses `self.featurizer` to featurize the
        data in these inputs.  For large files, automatically shards
        into smaller chunks of `shard_size` datapoints for convenience.
        Returns a `Dataset` object that contains the featurized dataset.

        This implementation assumes that the helper methods `_get_shards`
        and `_featurize_shard` are implemented and that each shard
        returned by `_get_shards` is a pandas dataframe.  You may choose
        to reuse or override this method in your subclass implementations.

        Parameters
        ----------
        inputs: List
            List of inputs to process. Entries can be filenames or arbitrary objects.
        data_dir: str, optional (default None)
            Directory to store featurized dataset.
        shard_size: int, optional (default 8192)
            Number of examples stored in each shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `inputs`.
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))

        # Special case handling of single input
        if not isinstance(inputs, list):
            inputs = [inputs]

        def shard_generator():
            for shard_num, shard in enumerate(
                    self._get_shards(inputs, shard_size)):
                time1 = time.time()
                X, valid_inds = self._featurize_shard(shard)
                ids = shard[self.id_field].values
                ids = ids[valid_inds]
                if len(self.tasks) > 0:
                    # Featurize task results iff they exist.
                    y, w = _convert_df_to_numpy(shard, self.tasks)
                    # Filter out examples where featurization failed.
                    y, w = (y[valid_inds], w[valid_inds])
                    assert len(X) == len(ids) == len(y) == len(w)
                else:
                    # For prospective data where results are unknown, it
                    # makes no sense to have y values or weights.
                    y, w = (None, None)
                    assert len(X) == len(ids)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir,
                                          self.tasks)

    def _get_shards(self, inputs: List, shard_size: Optional[int]) -> Iterator:
        """Stub for children classes.

        Should implement a generator that walks over the source data in
        `inputs` and returns a "shard" at a time. Here a shard is a
        chunk of input data that can reasonably be handled in memory. For
        example, this may be a set of rows from a CSV file or a set of
        molecules from a SDF file. To re-use the
        `DataLoader.create_dataset()` method, each shard must be a pandas
        dataframe.

        If you chose to override `create_dataset()` directly you don't
        need to override this helper method.

        Parameters
        ----------
        inputs: list
            List of inputs to process. Entries can be filenames or arbitrary objects.
        shard_size: int, optional
            Number of examples stored in each shard.
        """
        raise NotImplementedError

    def _featurize_shard(self, shard: Any):
        """Featurizes a shard of input data.

        Recall a shard is a chunk of input data that can reasonably be
        handled in memory. For example, this may be a set of rows from a
        CSV file or a set of molecules from a SDF file. Featurize this
        shard in memory and return the results.

        Parameters
        ----------
        shard: Any
            A chunk of input data
        """
        raise NotImplementedError


class CSVLoader(DataLoader):
    """
    Creates `Dataset` objects from input CSV files.

    This class provides conveniences to load data from CSV files.
    It's possible to directly featurize data from CSV files using
    pandas, but this class may prove useful if you're processing
    large CSV files that you don't want to manipulate directly in
    memory. Note that samples which cannot be featurized are filtered
    out in the creation of final dataset.

    Examples
    --------
    Let's suppose we have some smiles and labels

    >>> smiles = ["C", "CCC"]
    >>> labels = [1.5, 2.3]

    Let's put these in a dataframe.

    >>> import pandas as pd
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])

    Let's now write this to disk somewhere. We can now use `CSVLoader` to
    process this CSV dataset.

    >>> import tempfile
    >>> import deepchem as dc
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
    ...                              featurizer=dc.feat.CircularFingerprint())
    ...     dataset = loader.create_dataset(tmpfile.name)
    >>> len(dataset)
    2

    Of course in practice you should already have your data in a CSV file if
    you're using `CSVLoader`. If your data is already in memory, use
    `InMemoryLoader` instead.

    Sometimes there will be datasets without specific tasks, for example
    datasets which are used in unsupervised learning tasks. Such datasets
    can be loaded by leaving the `tasks` field empty.

    Example
    -------

    >>> x1, x2 = [2, 3, 4], [4, 6, 8]
    >>> df = pd.DataFrame({"x1":x1, "x2": x2}).reset_index()
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(tasks=[], id_field="index", feature_field=["x1", "x2"],
    ...                              featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)
    >>> len(dataset)
    3

    """

    def __init__(self,
                 tasks: List[str],
                 featurizer: Featurizer,
                 feature_field: Optional[str] = None,
                 id_field: Optional[str] = None,
                 smiles_field: Optional[str] = None,
                 log_every_n: int = 1000):
        """Initializes CSVLoader.

        Parameters
        ----------
        tasks: List[str]
            List of task names
        featurizer: Featurizer
            Featurizer to use to process data.
        feature_field: str, optional (default None)
            Field with data to be featurized.
        id_field: str, optional, (default None)
            CSV column that holds sample identifier
        smiles_field: str, optional (default None) (DEPRECATED)
            Name of field that holds smiles string.
        log_every_n: int, optional (default 1000)
            Writes a logging statement this often.
        """
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list.")
        if smiles_field is not None:
            logger.warning(
                "smiles_field is deprecated and will be removed in a future version of DeepChem."
                "Use feature_field instead.")
            if feature_field is not None and smiles_field != feature_field:
                raise ValueError(
                    "smiles_field and feature_field if both set must have the same value."
                )
            elif feature_field is None:
                feature_field = smiles_field

        self.tasks = tasks
        self.feature_field = feature_field
        self.id_field = id_field
        if id_field is None:
            self.id_field = feature_field  # Use features as unique ids if necessary
        else:
            self.id_field = id_field
        self.user_specified_features = None
        if isinstance(featurizer, UserDefinedFeaturizer):
            self.user_specified_features = featurizer.feature_fields
        self.featurizer = featurizer
        self.log_every_n = log_every_n

    def _get_shards(self, input_files: List[str],
                    shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of filenames to process
        shard_size: int, optional
            The size of a shard of data to process at a time.

        Returns
        -------
        Iterator[pd.DataFrame]
            Iterator over shards
        """
        return load_csv_files(input_files, shard_size)

    def _featurize_shard(self,
                         shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Featurizes a shard of an input dataframe.

        Parameters
        ----------
        shard: pd.DataFrame
            DataFrame that holds a shard of the input CSV file

        Returns
        -------
        features: np.ndarray
            Features computed from CSV file.
        valid_inds: np.ndarray
            Indices of rows in source CSV with valid data.
        """
        logger.info("About to featurize shard.")
        if self.featurizer is None:
            raise ValueError(
                "featurizer must be specified in constructor to featurizer data/"
            )
        features = [elt for elt in self.featurizer(shard[self.feature_field])]
        valid_inds = np.array(
            [1 if np.array(elt).size > 0 else 0 for elt in features],
            dtype=bool)
        features = [
            elt for (is_valid, elt) in zip(valid_inds, features) if is_valid
        ]
        return np.array(features), valid_inds


class UserCSVLoader(CSVLoader):
    """
    Handles loading of CSV files with user-defined features.

    This is a convenience class that allows for descriptors already present in a
    CSV file to be extracted without any featurization necessary.

    Examples
    --------
    Let's suppose we have some descriptors and labels. (Imagine that these
    descriptors have been computed by an external program.)

    >>> desc1 = [1, 43]
    >>> desc2 = [-2, -22]
    >>> labels = [1.5, 2.3]
    >>> ids = ["cp1", "cp2"]

    Let's put these in a dataframe.

    >>> import pandas as pd
    >>> df = pd.DataFrame(list(zip(ids, desc1, desc2, labels)), columns=["id", "desc1", "desc2", "task1"])

    Let's now write this to disk somewhere. We can now use `UserCSVLoader` to
    process this CSV dataset.

    >>> import tempfile
    >>> import deepchem as dc
    >>> featurizer = dc.feat.UserDefinedFeaturizer(["desc1", "desc2"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.UserCSVLoader(["task1"], id_field="id",
    ...                              featurizer=featurizer)
    ...     dataset = loader.create_dataset(tmpfile.name)
    >>> len(dataset)
    2
    >>> dataset.X[0, 0]
    1

    The difference between `UserCSVLoader` and `CSVLoader` is that our
    descriptors (our features) have already been computed for us, but are spread
    across multiple columns of the CSV file.

    Of course in practice you should already have your data in a CSV file if
    you're using `UserCSVLoader`. If your data is already in memory, use
    `InMemoryLoader` instead.
    """

    def _get_shards(self, input_files: List[str],
                    shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of filenames to process
        shard_size: int, optional
            The size of a shard of data to process at a time.

        Returns
        -------
        Iterator[pd.DataFrame]
            Iterator over shards
        """
        return load_csv_files(input_files, shard_size)

    def _featurize_shard(self,
                         shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Featurizes a shard of an input dataframe.

        Parameters
        ----------
        shard: pd.DataFrame
            DataFrame that holds a shard of the input CSV file

        Returns
        -------
        features: np.ndarray
            Features extracted from CSV file.
        valid_inds: np.ndarray
            Indices of rows in source CSV with valid data.
        """
        assert isinstance(self.featurizer, UserDefinedFeaturizer)
        time1 = time.time()
        feature_fields = self.featurizer.feature_fields
        shard[feature_fields] = shard[feature_fields].apply(pd.to_numeric)
        X_shard = shard[feature_fields].to_numpy()
        time2 = time.time()
        logger.info("TIMING: user specified processing took %0.3f s" %
                    (time2 - time1))
        return (X_shard, np.ones(len(X_shard), dtype=bool))


class JsonLoader(DataLoader):
    """
    Creates `Dataset` objects from input json files.

    This class provides conveniences to load data from json files.
    It's possible to directly featurize data from json files using
    pandas, but this class may prove useful if you're processing
    large json files that you don't want to manipulate directly in
    memory.

    It is meant to load JSON files formatted as "records" in line
    delimited format, which allows for sharding.
    ``list like [{column -> value}, ... , {column -> value}]``.

    Examples
    --------
    Let's create the sample dataframe.

    >>> composition = ["LiCoO2", "MnO2"]
    >>> labels = [1.5, 2.3]
    >>> import pandas as pd
    >>> df = pd.DataFrame(list(zip(composition, labels)), columns=["composition", "task"])

    Dump the dataframe to the JSON file formatted as "records" in line delimited format and
    load the json file by JsonLoader.

    >>> import tempfile
    >>> import deepchem as dc
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_json(tmpfile.name, orient='records', lines=True)
    ...     featurizer = dc.feat.ElementPropertyFingerprint()
    ...     loader = dc.data.JsonLoader(["task"], feature_field="composition", featurizer=featurizer)
    ...     dataset = loader.create_dataset(tmpfile.name)
    >>> len(dataset)
    2
    """

    def __init__(self,
                 tasks: List[str],
                 feature_field: str,
                 featurizer: Featurizer,
                 label_field: Optional[str] = None,
                 weight_field: Optional[str] = None,
                 id_field: Optional[str] = None,
                 log_every_n: int = 1000):
        """Initializes JsonLoader.

        Parameters
        ----------
        tasks: List[str]
            List of task names
        feature_field: str
            JSON field with data to be featurized.
        featurizer: Featurizer
            Featurizer to use to process data
        label_field: str, optional (default None)
            Field with target variables.
        weight_field: str, optional (default None)
            Field with weights.
        id_field: str, optional (default None)
            Field for identifying samples.
        log_every_n: int, optional (default 1000)
            Writes a logging statement this often.
        """
        if not isinstance(tasks, list):
            raise ValueError("Tasks must be a list.")
        self.tasks = tasks
        self.feature_field = feature_field
        self.label_field = label_field
        self.weight_field = weight_field
        self.id_field = id_field

        self.user_specified_features = None
        if isinstance(featurizer, UserDefinedFeaturizer):
            self.user_specified_features = featurizer.feature_fields
        self.featurizer = featurizer
        self.log_every_n = log_every_n

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 8192) -> DiskDataset:
        """Creates a `Dataset` from input JSON files.

        Parameters
        ----------
        input_files: OneOrMany[str]
            List of JSON filenames.
        data_dir: Optional[str], default None
            Name of directory where featurized data is stored.
        shard_size: int, optional (default 8192)
            Shard size when loading data.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `input_files`.
        """
        if not isinstance(input_files, list):
            try:
                if isinstance(input_files, str):
                    input_files = [input_files]
                else:
                    input_files = list(input_files)
            except TypeError:
                raise ValueError(
                    "input_files is of an unrecognized form. Must be one filename or a list of filenames."
                )

        def shard_generator():
            """Yield X, y, w, and ids for shards."""
            for shard_num, shard in enumerate(
                    self._get_shards(input_files, shard_size)):

                time1 = time.time()
                X, valid_inds = self._featurize_shard(shard)
                if self.id_field:
                    ids = shard[self.id_field].values
                else:
                    ids = np.ones(len(valid_inds))
                ids = ids[valid_inds]

                if len(self.tasks) > 0:
                    # Featurize task results if they exist.
                    y, w = _convert_df_to_numpy(shard, self.tasks)

                    if self.label_field:
                        y = shard[self.label_field]
                    if self.weight_field:
                        w = shard[self.weight_field]

                    # Filter out examples where featurization failed.
                    y, w = (y[valid_inds], w[valid_inds])
                    assert len(X) == len(ids) == len(y) == len(w)
                else:
                    # For prospective data where results are unknown, it
                    # makes no sense to have y values or weights.
                    y, w = (None, None)
                    assert len(X) == len(ids)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir,
                                          self.tasks)

    def _get_shards(self, input_files: List[str],
                    shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of filenames to process
        shard_size: int, optional
            The size of a shard of data to process at a time.

        Returns
        -------
        Iterator[pd.DataFrame]
            Iterator over shards
        """
        return load_json_files(input_files, shard_size)

    def _featurize_shard(self,
                         shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Featurizes a shard of an input dataframe.

        Helper that computes features for the given shard of data.

        Parameters
        ----------
        shard: pd.DataFrame
            DataFrame that holds data to be featurized.

        Returns
        -------
        features: np.ndarray
            Array of feature vectors. Note that samples for which featurization has
            failed will be filtered out.
        valid_inds: np.ndarray
            Boolean values indicating successful featurization for corresponding
            sample in the source.
        """
        logger.info("About to featurize shard.")
        if self.featurizer is None:
            raise ValueError(
                "featurizer must be specified in constructor to featurizer data/"
            )
        features = [elt for elt in self.featurizer(shard[self.feature_field])]
        valid_inds = np.array(
            [1 if np.array(elt).size > 0 else 0 for elt in features],
            dtype=bool)
        features = [
            elt for (is_valid, elt) in zip(valid_inds, features) if is_valid
        ]
        return np.array(features), valid_inds


class SDFLoader(DataLoader):
    """Creates a `Dataset` object from SDF input files.

    This class provides conveniences to load and featurize data from
    Structure Data Files (SDFs). SDF is a standard format for structural
    information (3D coordinates of atoms and bonds) of molecular compounds.

    Examples
    --------
    >>> import deepchem as dc
    >>> import os
    >>> current_dir = os.path.dirname(os.path.realpath(__file__))
    >>> featurizer = dc.feat.CircularFingerprint(size=16)
    >>> loader = dc.data.SDFLoader(["LogP(RRCK)"], featurizer=featurizer, sanitize=True)
    >>> dataset = loader.create_dataset(os.path.join(current_dir, "tests", "membrane_permeability.sdf")) # doctest:+ELLIPSIS
    >>> len(dataset)
    2
    """

    def __init__(self,
                 tasks: List[str],
                 featurizer: Featurizer,
                 sanitize: bool = False,
                 log_every_n: int = 1000):
        """Initialize SDF Loader

        Parameters
        ----------
        tasks: list[str]
            List of tasknames. These will be loaded from the SDF file.
        featurizer: Featurizer
            Featurizer to use to process data
        sanitize: bool, optional (default False)
            Whether to sanitize molecules.
        log_every_n: int, optional (default 1000)
            Writes a logging statement this often.
        """
        self.featurizer = featurizer
        self.sanitize = sanitize
        self.tasks = tasks
        # The field in which dc.utils.save.load_sdf_files stores RDKit mol objects
        self.mol_field = "mol"
        # The field in which load_sdf_files return value stores smiles
        self.id_field = "smiles"
        self.log_every_n = log_every_n

    def create_dataset(self,
                       inputs: OneOrMany[Any],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 8192) -> Dataset:
        """Creates and returns a `Dataset` object by featurizing provided sdf files.

        Parameters
        ----------
        inputs: List
            List of inputs to process. Entries can be filenames or arbitrary objects.
            Each file should be supported format (.sdf) or compressed folder of
            .sdf files
        data_dir: str, optional (default None)
            Directory to store featurized dataset.
        shard_size: int, optional (default 8192)
            Number of examples stored in each shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `inputs`.
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))

        # Special case handling of single input
        if not isinstance(inputs, list):
            inputs = [inputs]

        processed_files = []
        for input_file in inputs:
            filename, extension = os.path.splitext(input_file)
            extension = extension.lower()
            if extension == ".sdf":
                processed_files.append(input_file)
            elif extension == ".zip":
                zip_dir = tempfile.mkdtemp()
                unzip_file(input_file, zip_dir)
                zip_files = [
                    os.path.join(zip_dir, name) for name in os.listdir(zip_dir)
                ]
                for zip_file in zip_files:
                    _, extension = os.path.splitext(zip_file)
                    extension = extension.lower()
                    if extension in [".sdf"]:
                        processed_files.append(zip_file)
            else:
                raise ValueError("Unsupported file format")

        inputs = processed_files

        def shard_generator():
            for shard_num, shard in enumerate(
                    self._get_shards(inputs, shard_size)):
                time1 = time.time()
                X, valid_inds = self._featurize_shard(shard)
                ids = shard[self.id_field].values
                ids = ids[valid_inds]
                if len(self.tasks) > 0:
                    # Featurize task results iff they exist.
                    y, w = _convert_df_to_numpy(shard, self.tasks)
                    # Filter out examples where featurization failed.
                    y, w = (y[valid_inds], w[valid_inds])
                    assert len(X) == len(ids) == len(y) == len(w)
                else:
                    # For prospective data where results are unknown, it
                    # makes no sense to have y values or weights.
                    y, w = (None, None)
                    assert len(X) == len(ids)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir,
                                          self.tasks)

    def _get_shards(self, input_files: List[str],
                    shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of filenames to process
        shard_size: int, optional
            The size of a shard of data to process at a time.

        Returns
        -------
        Iterator[pd.DataFrame]
            Iterator over shards
        """
        return load_sdf_files(input_files=input_files,
                              clean_mols=self.sanitize,
                              tasks=self.tasks,
                              shard_size=shard_size)

    def _featurize_shard(self,
                         shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Featurizes a shard of an input dataframe.

        Helper that computes features for the given shard of data.

        Parameters
        ----------
        shard: pd.DataFrame
            DataFrame that holds data to be featurized.

        Returns
        -------
        features: np.ndarray
            Array of feature vectors. Note that samples for which featurization has
            failed will be filtered out.
        valid_inds: np.ndarray
            Boolean values indicating successful featurization for corresponding
            sample in the source.
        """
        pos_cols = ['pos_x', 'pos_y', 'pos_z']
        if set(pos_cols).issubset(shard.columns):
            features = [
                elt for elt in self.featurizer(shard[self.mol_field],
                                               pos_x=shard['pos_x'],
                                               pos_y=shard['pos_y'],
                                               pos_z=shard['pos_z'])
            ]
        else:
            features = [elt for elt in self.featurizer(shard[self.mol_field])]
        valid_inds = np.array(
            [1 if np.array(elt).size > 0 else 0 for elt in features],
            dtype=bool)
        features = [
            elt for (is_valid, elt) in zip(valid_inds, features) if is_valid
        ]
        try:
            return np.array(features), valid_inds
        except ValueError as e:
            logger.warning("Exception message: {}".format(e))
            return np.asarray(features, dtype=object), valid_inds


class FASTALoader(DataLoader):
    """Handles loading of FASTA files.

    FASTA files are commonly used to hold sequence data. This
    class provides convenience files to lead FASTA data and
    one-hot encode the genomic sequences for use in downstream
    learning tasks.
    """

    def __init__(self,
                 featurizer: Optional[Featurizer] = None,
                 auto_add_annotations: bool = False,
                 legacy: bool = True):
        """Initialize FASTALoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded FASTA data.

            If featurizer is None and legacy is True, the original featurization
            logic is used, creating a one hot encoding of all included FASTA strings
            of shape
            (number of FASTA sequences, number of channels + 1, sequence length, 1).

            If featurizer is None and legacy is False, the featurizer is initialized
            as a OneHotFeaturizer object with charset ("A", "C", "T", "G") and
            max_length = None.

        auto_add_annotations: bool (default False)
            Whether create_dataset will automatically add [CLS] and [SEP] annotations
            to the sequences it reads in order to assist tokenization.
            Keep False if your FASTA file already includes [CLS] and [SEP] annotations.

        legacy: bool (default True)
            Whether to use legacy logic for featurization. Legacy mode will create
            a one hot encoding of the FASTA content of shape
            (number of FASTA sequences, number of channels + 1, max length, 1).

            Legacy mode is only tested for ACTGN charsets, and will be deprecated.
       """

        # Process legacy toggle
        if legacy:
            warnings.warn(
                """
                    Legacy mode is deprecated and will be removed in
                    DeepChem 3.0. Disable legacy mode by passing legacy=False
                    during construction of FASTALoader object.
                    """, FutureWarning)
            if featurizer is not None or auto_add_annotations:
                raise ValueError(f"""
                          featurizer option must be None and
                          auto_add_annotations must be false when legacy mode
                          is enabled. You set featurizer to {featurizer} and
                          auto_add_annotations to {auto_add_annotations}.
                          """)

        # Set attributes
        self.legacy = legacy
        self.auto_add_annotations = auto_add_annotations

        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            featurizer = OneHotFeaturizer(charset=["A", "C", "T", "G"],
                                          max_length=None)

        # Set self.featurizer
        self.featurizer = featurizer

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = None) -> DiskDataset:
        """Creates a `Dataset` from input FASTA files.

        At present, FASTA support is limited and doesn't allow for sharding.

        Parameters
        ----------
        input_files: List[str]
            List of fasta files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
        shard_size: int, optional (default None)
            For now, this argument is ignored and each FASTA file gets its
            own shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `input_files`.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():  # TODO Enable sharding with shard size parameter
            for input_file in input_files:
                if self.legacy:
                    X = encode_bio_sequence(input_file)
                else:
                    sequences = _read_file(input_file)
                    X = self.featurizer(sequences)
                ids = np.ones(len(X))
                # (X, y, w, ids)
                yield X, None, None, ids

        def _read_file(input_file: str):
            """
            Convert the FASTA file to a numpy array of FASTA-format strings.
            """

            # TODO don't convert all sequences into np array (allow shards)
            def _generate_sequences(fasta_file, header_mark=">") -> np.ndarray:
                """
                Uses a fasta_file to create a numpy array of annotated FASTA-format strings
                """
                sequences: np.ndarray = np.array([])
                sequence: np.ndarray = np.array([])
                header_read = False
                for line in fasta_file:
                    # Check if line is a header
                    if line.startswith(header_mark):  # New header line
                        header_read = True
                        sequences = _add_sequence(sequences, sequence)
                        sequence = np.array([])
                    elif header_read:  # Line contains sequence in FASTA format
                        if line[-1:] == '\n':  # Check last character in string
                            line = line[0:-1]  # Remove last character
                        sequence = np.append(sequence, line)
                sequences = _add_sequence(sequences,
                                          sequence)  # Add last sequence
                return sequences

            def _add_sequence(sequences: np.ndarray,
                              sequence: np.ndarray) -> np.ndarray:
                # Handle empty sequence
                if sequence is None or len(sequence) <= 0:
                    return np.array([])
                # Annotate start/stop of sequence
                if self.auto_add_annotations:
                    sequence = np.insert(sequence, 0, "[CLS]")
                    sequence = np.append(sequence, "[SEP]")
                new_sequence = ''.join(sequence)
                new_sequences = np.append(sequences, new_sequence)
                return new_sequences

            with open(input_file, 'r') as f:  # Read FASTA file
                return _generate_sequences(f)

        return DiskDataset.create_dataset(shard_generator(), data_dir)


def _fastq_load_files(input_files: List[str],
                      shard_size: Optional[int] = 4096) -> Iterator:
    """Load data as Iterator.

    Parameters
    ----------
    input_files: List[str]
        List of fastq filenames.
    shard_size: int, optional (default 4096)
        Chunksize for reading fastq files.

    Yields
    -------
    Iterator
        Generator which yields the data which is the same shard size.
    """
    shard_num = 0
    for input_file in input_files:
        logger.info("About to start loading fastq from %s." % input_file)
        # Open index file
        with open(input_file, 'r') as f:
            # create an empty list to store lines in files.
            df = []
            line_number = 0
            # iterate through each line in the input file
            for num, line in enumerate(f):
                # If the number of lines iterated through is equal or less than the shard size:
                if (shard_size is not None) and ((num + 1) - line_number <=
                                                 (shard_size * 4)):
                    # append to list
                    df.append(line)
                else:
                    # else yield the list
                    shard_num += 1
                    logger.info("Loading shard %d of size %s." %
                                (shard_num, str(shard_size)))
                    # set the line_number variable to the last line number (num) before 'yield' was called
                    line_number = num
                    # yield list (shard/batch)
                    yield df
                    # Re-initialize list with the index line to begin a new shard.
                    df = [line]
            if len(df) > 0:
                yield df


class FASTQLoader(DataLoader):
    """Handles loading of FASTQ files.

    FASTQ files are commonly used to hold very large sequence data. It is a variant of FASTA format.
    This class provides convenience files to load FASTQ data and one-hot encode
    the genomic sequences for use in downstream learning tasks.

    Example
    -------
    >>> import os
    >>> from deepchem.feat.molecule_featurizers import OneHotFeaturizer
    >>> from deepchem.data.data_loader import FASTQLoader
    >>> current_dir = os.path.dirname(os.path.abspath(__file__))
    >>> input_file = os.path.join(current_dir, "tests", "sample1.fastq")
    >>> loader = FASTQLoader()
    >>> sequences = loader.create_dataset(input_file)

    See Also
    --------
    `Info on the structure of FASTQ files <https://support.illumina.com/bulletins/2016/04/fastq-files-explained.html>`
    """

    def __init__(self,
                 featurizer: Optional[Featurizer] = None,
                 auto_add_annotations: bool = False,
                 return_quality_scores: bool = False):
        """Initialize FASTQLoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded FASTQ data.
            The featurizer is initialized as a OneHotFeaturizer object with charset ("A", "C", "T", "G") and
            max_length = None.
        auto_add_annotations: bool (default False)
            Whether create_dataset will automatically add [CLS] and [SEP] annotations
            to the sequences it reads in order to assist tokenization.
            Keep False if your FASTQ file already includes [CLS] and [SEP] annotations.
        return_quality_scores: bool (default True)
            returns the quality (likelihood) score of the nucleotides in the sequence.
       """

        # Set attributes
        self.auto_add_annotations = auto_add_annotations

        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            featurizer = OneHotFeaturizer(charset=["A", "C", "T", "G"],
                                          max_length=None)

        # Set self.featurizer
        self.featurizer = featurizer
        # Set self.return_quality_scores
        self.return_quality_scores = return_quality_scores

    def _get_shards(self,
                    input_files: List[str],
                    shard_size: Optional[int] = 4096) -> Iterator:
        """Defines a generator which returns data for each shard

        Parameters
        ----------
        input_files: List[str]
            List of file names to process
        n_samples:  int, optional
            The number of samples to extract from each variant in the data
        shard_size: int, optional (default 4096)
            The size of a shard of data to process at a time. Here, shard_size is equal to the
            number of variants to fetch. You can think of them as number of rows to get from the
            full dataset.

        Yields
        -------
        Iterator
            Iterator over shards
        """

        return _fastq_load_files(input_files, shard_size)

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 4096) -> DiskDataset:
        """Creates a `Dataset` from input FASTQ files.

        Parameters
        ----------
        input_files: List[str]
            List of fastQ files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
          shard_size: int, optional (default 4096)

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `input_files`.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():

            for shard_num, shard in enumerate(
                    self._get_shards(input_files, shard_size)):
                if self.return_quality_scores:
                    sequences, quality_scores = _generate_sequences(shard)
                    # Featurize sequences
                    X = self.featurizer(sequences)
                    ids = np.ones(len(X))
                    # (X, y , w, ids)
                    yield X, None, quality_scores, ids
                else:
                    sequences = _generate_sequences(shard)
                    # Featurize sequences
                    X = self.featurizer(sequences)
                    ids = np.ones(len(X))
                    # (X, y , w, ids)
                    yield X, None, None, ids

        def _generate_sequences(shard: List) -> OneOrMany[np.ndarray]:
            """
            Creates a numpy array of annotated FASTQ-format strings.
            """
            assert len(
                shard
            ) % 4 == 0, f'Sharded length not divisible by four: Length of shard = {len(shard)}. File is possibly incomplete'
            sequences: np.ndarray = np.array([], dtype='object')
            if self.return_quality_scores:
                quality_scores: np.ndarray = np.array([], dtype='object')

            # Go through each sequence entity in the fastq_file: each sequence consists of 4 lines
            # First line : header description
            # second line : sequence
            # third line : more description usually the same as the first line
            # fourth line: quality scores of the sequence
            for start_index in range(0, len(shard), 4):
                each_sequence = shard[start_index:start_index + 4]

                # Second line : add sequence to the sequence array
                sequences = _add_sequence(
                    sequences, np.array([each_sequence[1].strip("\n")]))

                # Fourth line
                if self.return_quality_scores:
                    quality_scores = _add_sequence(
                        quality_scores,
                        np.array([each_sequence[3].strip("\n")]))

            if self.return_quality_scores:
                return sequences, quality_scores
            else:
                return sequences

        def _add_sequence(sequences: np.ndarray,
                          sequence: np.ndarray) -> np.ndarray:
            # Handle empty sequence
            if sequence is None or len(sequence) <= 0:
                return np.array([])
            # Annotate start/stop of sequence
            if self.auto_add_annotations:
                sequence = np.insert(sequence, 0, "[CLS]")
                sequence = np.append(sequence, "[SEP]")
            new_sequence = ''.join(sequence)
            new_sequences = np.append(sequences, new_sequence)
            return new_sequences

        return DiskDataset.create_dataset(shard_generator(), data_dir)


class ImageLoader(DataLoader):
    """Creates `Dataset` objects from input image files.

    This class allows for loading of images in various formats.
    For user convenience, also accepts zip-files and directories
    of images and uses some limited intelligence to attempt to
    traverse subdirectories which contain images.

    Currently, only .png and .tif files are supported. If the
    inputs or labels are given as a list of files, the list must contain
    only image files.

    Examples
    --------
    For this example, we will be using the BBBC001 Dataset. This dataset contains 6 images
    of human HT29 colon cancer cells. We will use the images as inputs and we will assign
    the labels as integers ranging from 1 to 6 for the sake of simplicity.

    To learn more about this dataset, please visit: https://data.broadinstitute.org/bbbc/BBBC001/
    and also see our loader for this dataset: `deepchem.molnet.loadbbbc001`.

    Let's begin by importing the necessary modules and downloading the dataset.
    >>> import os
    >>> import deepchem as dc
    >>> data_dir = dc.utils.data_utils.get_data_dir()
    >>> dataset_file = os.path.join(data_dir, "BBBC001_v1_images_tif.zip")
    >>> BBBC1_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip'
    >>> if not os.path.exists(dataset_file):
    ...    dc.utils.data_utils.download_url(url=BBBC1_IMAGE_URL, dest_dir=data_dir)

    Now that we have the dataset, let's create a list of labels for each image.

    >>> labels = np.array([1,2,3,4,5,6])

    Let's now write this to disk somewhere. We can now use `ImageLoader` to process
    this Image dataset. We do not use a featurizer here, hence the `UserDefinedFeaturizer`
    with an empty list.

    >>> featurizer = dc.feat.UserDefinedFeaturizer([])
    >>> loader = dc.data.ImageLoader(tasks=['demo-task'], sorting=False)
    >>> dataset = loader.create_dataset(inputs=(dataset_file, labels),
    ...                                 in_memory=False)

    We can confirm that we have 6 images in our dataset and 6 labels. The images are
    of size 512x512 while the labels are just integers.

    >>> len(dataset)
    6
    >>> dataset.X.shape
    (6, 512, 512)
    >>> dataset.y.shape
    (6,)

    The label files can also be images similar to the inputs, in which case we
    can provide a list of label files instead of a list of labels.

    To show this, we will use the input data as the ground truths, this is often
    seen when making autoencoders. Similar to the above example, let's use `ImageLoader`
    to process this Image dataset.

    >>> featurizer = dc.feat.UserDefinedFeaturizer([])
    >>> loader = dc.data.ImageLoader(tasks=['demo-task'], sorting=False)
    >>> dataset = loader.create_dataset(inputs=(dataset_file, dataset_file),
    ...                                 in_memory=False)

    We can confirm that we have 6 images in our dataset and 6 labels. The images are
    of size 512x512 while the labels are also images of size 512x512.

    >>> len(dataset)
    6
    >>> dataset.X.shape
    (6, 512, 512)
    >>> dataset.y.shape
    (6, 512, 512)
    """

    def __init__(self, tasks: Optional[List[str]] = None, sorting: bool = True):
        """Initialize image loader.

        At present, custom image featurizers aren't supported by this
        loader class.

        Parameters
        ----------
        tasks: List[str], optional (default None)
            List of task names for image labels.
        sorting: bool, optional (default True)
            Whether to sort image files by filename.
        """
        if tasks is None:
            tasks = []
        self.tasks = tasks
        self.sorting = sorting

    def create_dataset(self,
                       inputs: Union[OneOrMany[str], Tuple[Any], Tuple[str,
                                                                       Any]],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 8192,
                       in_memory: bool = False) -> Dataset:
        """Creates and returns a `Dataset` object by featurizing provided image files and labels/weights.

        Parameters
        ----------
        inputs: `Union[OneOrMany[str], Tuple[Any]]`
            The inputs provided should be one of the following

              - filename
              - list of filenames
              - Tuple (list of filenames, labels)
              - Tuple (list of filenames, list of label filenames)
              - Tuple (list of filenames, labels, weights)
              - Tuple (list of filenames, list of label filenames, weights)

            Each file in a given list of filenames should either be of a supported
            image format (.png, .tif only for now) or of a compressed folder of
            image files (only .zip for now). If `labels` or `weights` are provided,
            they must correspond to the sorted order of all filenames provided, with
            one label/weight per file. Labels can be filenames too, in which case the
            labels are loaded as images.

        data_dir: str, optional (default None)
            Directory to store featurized dataset.
        shard_size: int, optional (default 8192)
            Shard size when loading data.
        in_memory: bool, optioanl (default False)
            If true, return in-memory `NumpyDataset`. Else return `ImageDataset`.

        Returns
        -------
        `ImageDataset` or `NumpyDataset` or `DiskDataset`
            - if `in_memory == False`, the return value is `ImageDataset`.
            - if `in_memory == True` and `data_dir is None`, the return value is `NumpyDataset`.
            - if `in_memory == True` and `data_dir is not None`, the return value is `DiskDataset`.
        """
        labels, weights = None, None
        if isinstance(inputs, tuple):
            if len(inputs) == 1:
                input_files = inputs[0]
                if isinstance(inputs, str):
                    input_files = [inputs]
            elif len(inputs) == 2:
                input_files, labels = inputs
            elif len(inputs) == 3:
                input_files, labels, weights = inputs
            else:
                raise ValueError("Input must be a tuple of length 1, 2, or 3")
        else:
            input_files = inputs
        if isinstance(input_files, str):
            input_files = [input_files]

        image_files = []
        # Sometimes zip files contain directories within. Traverse directories
        while len(input_files) > 0:
            remainder = []
            for input_file in input_files:
                filename, extension = os.path.splitext(input_file)
                extension = extension.lower()
                if os.path.isdir(input_file):
                    dirfiles = [
                        os.path.join(input_file, subfile)
                        for subfile in os.listdir(input_file)
                    ]
                    remainder += dirfiles
                elif extension == ".zip":
                    zip_dir = tempfile.mkdtemp()
                    zip_ref = zipfile.ZipFile(input_file, 'r')
                    zip_ref.extractall(path=zip_dir)
                    zip_ref.close()
                    zip_files = [
                        os.path.join(zip_dir, name)
                        for name in zip_ref.namelist()
                    ]
                    for zip_file in zip_files:
                        _, extension = os.path.splitext(zip_file)
                        extension = extension.lower()
                        if extension in [".png", ".tif"]:
                            image_files.append(zip_file)
                elif extension in [".png", ".tif"]:
                    image_files.append(input_file)
                else:
                    raise ValueError("Unsupported file format")
            input_files = remainder

        # Sort image files
        if self.sorting:
            image_files = sorted(image_files)

        if isinstance(labels, str):
            label_files = [labels]
        else:
            label_files = []

        label_image_files = []
        # Sometimes zip files contain directories within. Traverse directories
        while len(label_files) > 0:
            remainder = []
            for label_file in label_files:
                filename, extension = os.path.splitext(label_file)
                extension = extension.lower()
                if os.path.isdir(label_file):
                    dirfiles = [
                        os.path.join(label_file, subfile)
                        for subfile in os.listdir(label_file)
                    ]
                    remainder += dirfiles
                elif extension == ".zip":
                    zip_dir = tempfile.mkdtemp()
                    zip_ref = zipfile.ZipFile(label_file, 'r')
                    zip_ref.extractall(path=zip_dir)
                    zip_ref.close()
                    zip_files = [
                        os.path.join(zip_dir, name)
                        for name in zip_ref.namelist()
                    ]
                    for zip_file in zip_files:
                        _, extension = os.path.splitext(zip_file)
                        extension = extension.lower()
                        if extension in [".png", ".tif"]:
                            label_image_files.append(zip_file)
                elif extension in [".png", ".tif"]:
                    label_image_files.append(label_file)
                else:
                    raise ValueError("Unsupported file format")
            label_files = remainder

        # Sort label image files
        if self.sorting:
            label_image_files = sorted(label_image_files)

        if in_memory:
            if data_dir is None:
                if isinstance(labels, str):
                    return NumpyDataset(load_image_files(image_files),
                                        y=load_image_files(label_image_files),
                                        w=weights,
                                        ids=image_files)
                else:
                    return NumpyDataset(load_image_files(image_files),
                                        y=labels,
                                        w=weights,
                                        ids=image_files)
            else:
                if isinstance(labels, str):
                    dataset = DiskDataset.from_numpy(
                        load_image_files(image_files),
                        y=load_image_files(label_image_files),
                        w=weights,
                        ids=image_files,
                        tasks=self.tasks,
                        data_dir=data_dir)
                else:
                    dataset = DiskDataset.from_numpy(
                        load_image_files(image_files),
                        y=labels,
                        w=weights,
                        ids=image_files,
                        tasks=self.tasks,
                        data_dir=data_dir)
                if shard_size is not None:
                    dataset.reshard(shard_size)
                return dataset
        else:
            if isinstance(labels, str):
                return ImageDataset(image_files,
                                    y=label_image_files,
                                    w=weights,
                                    ids=image_files)
            else:
                return ImageDataset(image_files,
                                    y=labels,
                                    w=weights,
                                    ids=image_files)


class InMemoryLoader(DataLoader):
    """Facilitate Featurization of In-memory objects.

    When featurizing a dataset, it's often the case that the initial set of
    data (pre-featurization) fits handily within memory. (For example, perhaps
    it fits within a column of a pandas DataFrame.) In this case, it would be
    convenient to directly be able to featurize this column of data. However,
    the process of featurization often generates large arrays which quickly eat
    up available memory. This class provides convenient capabilities to process
    such in-memory data by checkpointing generated features periodically to
    disk.

    Example
    -------
    Here's an example with only datapoints and no labels or weights.

    >>> import deepchem as dc
    >>> smiles = ["C", "CC", "CCC", "CCCC"]
    >>> featurizer = dc.feat.CircularFingerprint()
    >>> loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    >>> dataset = loader.create_dataset(smiles, shard_size=2)
    >>> len(dataset)
    4

    Here's an example with both datapoints and labels

    >>> import deepchem as dc
    >>> smiles = ["C", "CC", "CCC", "CCCC"]
    >>> labels = [1, 0, 1, 0]
    >>> featurizer = dc.feat.CircularFingerprint()
    >>> loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    >>> dataset = loader.create_dataset(zip(smiles, labels), shard_size=2)
    >>> len(dataset)
    4

    Here's an example with datapoints, labels, weights and ids all provided.

    >>> import deepchem as dc
    >>> smiles = ["C", "CC", "CCC", "CCCC"]
    >>> labels = [1, 0, 1, 0]
    >>> weights = [1.5, 0, 1.5, 0]
    >>> ids = ["C", "CC", "CCC", "CCCC"]
    >>> featurizer = dc.feat.CircularFingerprint()
    >>> loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    >>> dataset = loader.create_dataset(zip(smiles, labels, weights, ids), shard_size=2)
    >>> len(dataset)
    4

    """

    def create_dataset(self,
                       inputs: Sequence[Any],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 8192) -> DiskDataset:
        """Creates and returns a `Dataset` object by featurizing provided files.

        Reads in `inputs` and uses `self.featurizer` to featurize the
        data in these input files.  For large files, automatically shards
        into smaller chunks of `shard_size` datapoints for convenience.
        Returns a `Dataset` object that contains the featurized dataset.

        This implementation assumes that the helper methods `_get_shards`
        and `_featurize_shard` are implemented and that each shard
        returned by `_get_shards` is a pandas dataframe.  You may choose
        to reuse or override this method in your subclass implementations.

        Parameters
        ----------
        inputs: Sequence[Any]
            List of inputs to process. Entries can be arbitrary objects so long as
            they are understood by `self.featurizer`
        data_dir: str, optional (default None)
            Directory to store featurized dataset.
        shard_size: int, optional (default 8192)
            Number of examples stored in each shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of data
            from `inputs`.
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))

        if not isinstance(inputs, list):
            try:
                inputs = list(inputs)
            except TypeError:
                inputs = [inputs]

        def shard_generator():
            global_index = 0
            for shard_num, shard in enumerate(
                    self._get_shards(inputs, shard_size)):
                time1 = time.time()
                X, y, w, ids = self._featurize_shard(shard, global_index)
                global_index += len(shard)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir,
                                          self.tasks)

    def _get_shards(self, inputs: List,
                    shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
        """Break up input into shards.

        Parameters
        ----------
        inputs: List
            Each entry in this list must be of the form `(featurization_input,
            label, weight, id)` or `(featurization_input, label, weight)` or
            `(featurization_input, label)` or `featurization_input` for one
            datapoint, where `featurization_input` is any input that is recognized
            by `self.featurizer`.
        shard_size: int, optional
            The size of shard to generate.

        Returns
        -------
        Iterator[pd.DataFrame]
            Iterator which iterates over shards of data.
        """
        current_shard: List = []
        for i, datapoint in enumerate(inputs):
            if i != 0 and shard_size is not None and i % shard_size == 0:
                shard_data = current_shard
                current_shard = []
                yield shard_data
            current_shard.append(datapoint)
        yield current_shard

    # FIXME: Signature of "_featurize_shard" incompatible with supertype "DataLoader"
    def _featurize_shard(  # type: ignore[override]
        self, shard: List, global_index: int) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        """Featurizes a shard of an input data.

        Parameters
        ----------
        shard: List
            List each entry of which must be of the form `(featurization_input,
            label, weight, id)` or `(featurization_input, label, weight)` or
            `(featurization_input, label)` or `featurization_input` for one
            datapoint, where `featurization_input` is any input that is recognized
            by `self.featurizer`.
        global_index: int
            The starting index for this shard in the full set of provided inputs

        Returns
        ------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The tuple is `(X, y, w, ids)`. All values are numpy arrays.
        """
        features = []
        labels = []
        weights = []
        ids = []
        n_tasks = len(self.tasks)
        for i, entry in enumerate(shard):
            if not isinstance(entry, tuple):
                entry = (entry,)
            if len(entry) > 4:
                raise ValueError(
                    "Entry is malformed and must be of length 1-4 containing featurization_input"
                    "and optionally label, weight, and id.")
            if len(entry) == 4:
                featurization_input, label, weight, entry_id = entry
            elif len(entry) == 3:
                featurization_input, label, weight = entry
                entry_id = global_index + i
            elif len(entry) == 2:
                featurization_input, label = entry
                weight = np.ones((n_tasks), np.float32)
                entry_id = global_index + i
            elif len(entry) == 1:
                featurization_input = entry
                label = np.zeros((n_tasks), np.float32)
                weight = np.zeros((n_tasks), np.float32)
                entry_id = global_index + i
            feature = self.featurizer(featurization_input)
            features.append(feature)
            weights.append(weight)
            labels.append(label)
            ids.append(entry_id)
        X = np.concatenate(features, axis=0)
        return X, np.array(labels), np.array(weights), np.array(ids)


class DFTYamlLoader(DataLoader):
    """
    Creates a `Dataset` object from YAML input files.

    This class provides methods to load and featurize data from a YAML file.
    Although, in this class, we only focus on a specfic input format
    that can be used to perform Density Functional Theory calculations.

    Examples
    --------
    >>> from deepchem.data.data_loader import DFTYamlLoader
    >>> import deepchem as dc
    >>> import pytest
    >>> inputs = 'deepchem/data/tests/dftdata.yaml'
    >>> data = DFTYamlLoader()
    >>> output = data.create_dataset(inputs)

    Notes
    -----
    Format (and example) for the YAML file:

    - e_type : 'ae'
      true_val : '0.09194410469'
      systems : [{
            'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0',
            'basis': '6-311++G(3df,3pd)'
        }]

    Each entry in the YAML file must contain the three parameters : e_type,
    true_val and systems in this particular order.
    One entry object may contain one or more systems.
    This data class does not support/ require an additional featurizer,
    since the datapoints are featurized within the methods.
    To read more about the parameters and their possible values please refer to
    deepchem.feat.dft_data.

    """

    def __init__(self):
        """
        Initialize DFTYAML loader
        """

    def create_dataset(self,
                       inputs: OneOrMany[Any],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = 1) -> Dataset:
        """
        Creates and returns a `Dataset` object by featurizing provided YAML
        files.

        Parameters
        ----------
        input_files: OneOrMany[str]
            List of YAML filenames.
        data_dir: Optional[str], default None
            Name of directory where featurized data is stored.
        shard_size: int, optional (default 1)
            Shard size when loading data.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation
            of data from `inputs`.
        """

        def shard_generator():
            entries = self._get_shards(inputs)
            for i, shard in enumerate(entries):
                X = np.array(self._featurize_shard(shard))
                y = X[0].get_true_val()
                w = np.array([X[0].get_weight()])
                ids = np.array([i])
                yield X, y, w, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir)

    def _get_shards(self, inputs):
        """
        Loads and divides the .yaml file into shards.

        Parameters
        ----------
        input_files: str
            .yaml file to be processed.

        Returns
        -------
        data
            list of dictionaries where each dictionary corresponds to one
            shard and is then featurized into one entry object.
        """
        with open(inputs) as f:
            data = yaml.load(f, Loader=SafeLoader)
        return (data)

    def _featurize_shard(self, shard):
        """
        Featurizes shards in the dataset

        Parameters
        ----------
        shard: dict
            Dictionary containing values to initialize the DFTEntry object.

        Returns
        -------
        x: featurized shard (DFTEntry objects)
        """
        from deepchem.feat.dft_data import DFTEntry
        try:
            e_type = shard['e_type']
            if 'true_val' in shard.keys():
                true_val = shard['true_val']
            else:
                true_val = '0.0'
            systems = shard['systems']
        except KeyError:
            raise ValueError(
                "Unknown key in yaml file. Please check format for correctness."
            )
        if 'weight' in shard.keys():
            weight = shard['weight']
            x = DFTEntry.create(e_type, true_val, systems, weight)
        else:
            x = DFTEntry.create(e_type, true_val, systems)
        return [x]


class SAMLoader(DataLoader):
    """Handles loading of SAM files.
    Sequence Alignment Map (SAM) is a text-based format used for storing biological sequences
    aligned to a reference sequence.It is generally used for storing nucleotide sequences,
    generated by next generation sequencing technologies, and unmapped sequences.
    SAM files have a header section and an alignment section.Alignment sections have
    11 mandatory fields, as well as a variable number of optional fields. Here, we
    extract Query Name, Query Sequence, Query Length, Reference Name,
    Reference Start, CIGAR and Mapping Quality of each read in the SAM file.
    This class provides methods to load and featurize data from SAM files.

    Examples
    --------
    >>> from deepchem.data.data_loader import SAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.sam'
    >>> data = SAMLoader()
    >>> output = data.create_dataset(inputs)

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux or MacOS X.
    To use Pysam on Windows, use Windows Subsystem for Linux(WSL).
    """

    def __init__(self, featurizer: Optional[Featurizer] = None):
        """Initialize SAMLoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded SAM data.

       """

        # Set attributes
        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            from deepchem.feat import SAMFeaturizer
            featurizer = SAMFeaturizer()

        # Set self.featurizer
        self.featurizer = featurizer

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = None) -> DiskDataset:
        """Creates a `Dataset` from input SAM files.

        Parameters
        ----------
        input_files: List[str]
            List of SAM files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
        shard_size: int, optional (default None)
            For now, this argument is ignored and each SAM file gets its
            own shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of
            data from `input_files`.

        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():  # TODO Enable sharding with shard size parameter
            for input_file in input_files:
                samfile = pysam.AlignmentFile(input_file, "r")
                X = self.featurizer._featurize(samfile)
                ids = np.ones(len(X))
                # (X, y, w, ids)
                yield X, None, None, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir)


class BAMLoader(DataLoader):
    """Handles loading of BAM files.
    Binary Alignment Map (BAM) is the comprehensive raw data of genome
    sequencing. It consists of the lossless, compressed binary representation
    of the Sequence Alignment Map files. BAM files are smaller and more
    efficient to work with than SAM files, saving time and reducing costs of
    computation and storage. BAM files store alignment data and often have
    corresponding BAM index files.The structure of BAM files include a header
    section and an alignment section.
    Here, we extract Query Name, Query Sequence, Query Length, Reference Name,
    Reference Start, CIGAR and Mapping Quality of each read in the BAM file.
    This class provides methods to load and featurize data from BAM files.
    Additionally, we can also get pileups from BAM files by setting
    get_pileup=True.

    Examples
    --------
    >>> from deepchem.data.data_loader import BAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.bam'
    >>> data = BAMLoader()
    >>> output = data.create_dataset(inputs)

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux
    or MacOS X. To use Pysam on Windows, use Windows Subsystem for Linux(WSL).
    """

    def __init__(self,
                 featurizer: Optional[Featurizer] = None,
                 get_pileup: Optional[bool] = False):
        """Initialize BAMLoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded BAM data.
        get_pileup: bool, optional (default: False)
            If True, the pileup of reads at each position is
            returned, to be used in DeepVariant.

       """

        # Set attributes
        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            from deepchem.feat import BAMFeaturizer
            if get_pileup:
                featurizer = BAMFeaturizer(max_records=None, get_pileup=True)
            else:
                featurizer = BAMFeaturizer(max_records=None)

        # Set self.featurizer
        self.featurizer = featurizer

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = None) -> DiskDataset:
        """Creates a `Dataset` from input BAM files.

        Parameters
        ----------
        input_files: List[str]
            List of BAM files, with their corresponding index files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
        shard_size: int, optional (default None)
            For now, this argument is ignored and each BAM file gets its
            own shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of
            data from `input_files`.

        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():  # TODO Enable sharding with shard size parameter
            for input_file in input_files:
                bamfile = pysam.AlignmentFile(input_file, "rb")
                X = self.featurizer._featurize(bamfile)
                ids = np.ones(len(X))
                # (X, y, w, ids)
                yield X, None, None, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir)


class CRAMLoader(DataLoader):
    """Handles loading of CRAM files.
    Compressed Reference-oriented Alignment Map (CRAM) is a compressed
    columnar file format for storing biological sequences aligned to a
    reference sequence. CRAM is an efficient reference-based alternative to
    the Sequence Alignment Map (SAM) and Binary Alignment Map (BAM) file formats.
    The basic structure of a CRAM file has a series of containers, the first of
    which holds a compressed copy of the SAM header. Subsequent containers consist
    of a container Compression Header followed by a series of slices which hold the
    alignment records, formatted as a series of blocks.
    Here, we extract Query Name, Query Sequence, Query Length, Reference Name,
    Reference Start, CIGAR and Mapping Quality of each read in the CRAM file.
    This class provides methods to load and featurize data from CRAM files.

    Examples
    --------
    >>> from deepchem.data.data_loader import CRAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.cram'
    >>> data = CRAMLoader()
    >>> output = data.create_dataset(inputs)

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux
    or MacOS X. To use Pysam on Windows, use Windows Subsystem for Linux(WSL).
    """

    def __init__(self, featurizer: Optional[Featurizer] = None):
        """Initialize CRAMLoader.

        Parameters
        ----------
        featurizer: Featurizer (default: None)
            The Featurizer to be used for the loaded CRAM data.
       """

        # Set attributes
        self.user_specified_features = None

        # Handle special featurizer cases
        if isinstance(featurizer,
                      UserDefinedFeaturizer):  # User defined featurizer
            self.user_specified_features = featurizer.feature_fields
        elif featurizer is None:  # Default featurizer
            from deepchem.feat import CRAMFeaturizer
            featurizer = CRAMFeaturizer(max_records=None)

        # Set self.featurizer
        self.featurizer = featurizer

    def create_dataset(self,
                       input_files: OneOrMany[str],
                       data_dir: Optional[str] = None,
                       shard_size: Optional[int] = None) -> DiskDataset:
        """Creates a `Dataset` from input CRAM files.

        Parameters
        ----------
        input_files: List[str]
            List of CRAM files.
        data_dir: str, optional (default None)
            Name of directory where featurized data is stored.
        shard_size: int, optional (default None)
            For now, this argument is ignored and each CRAM file gets its
            own shard.

        Returns
        -------
        DiskDataset
            A `DiskDataset` object containing a featurized representation of
            data from `input_files`.

        """
        if isinstance(input_files, str):
            input_files = [input_files]

        def shard_generator():  # TODO Enable sharding with shard size parameter
            for input_file in input_files:
                cramfile = pysam.AlignmentFile(input_file, "rc")
                X = self.featurizer._featurize(cramfile)
                ids = np.ones(len(X))
                # (X, y, w, ids)
                yield X, None, None, ids

        return DiskDataset.create_dataset(shard_generator(), data_dir)

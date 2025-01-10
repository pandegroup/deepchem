"""
Simple utils to save and load from disk.
"""
import joblib
import gzip
import pickle
import os
import tempfile
import tarfile
import zipfile
import logging
from urllib.request import urlretrieve
from deepchem.utils.rdkit_utils import get_xyz_from_mol
from typing import Any, Iterator, List, Optional, Tuple, Union, cast, IO

import pandas as pd
import numpy as np

import deepchem as dc

logger = logging.getLogger(__name__)


def pad_array(x: np.ndarray,
              shape: Union[Tuple, int],
              fill: float = 0.0,
              both: bool = False) -> np.ndarray:
    """
    Pad an array with a fill value.

    Parameters
    ----------
    x: np.ndarray
        A numpy array.
    shape: Tuple or int
        Desired shape. If int, all dimensions are padded to that size.
    fill: float, optional (default 0.0)
        The padded value.
    both: bool, optional (default False)
        If True, split the padding on both sides of each axis. If False,
        padding is applied to the end of each axis.

    Returns
    -------
    np.ndarray
        A padded numpy array
    """
    x = np.asarray(x)
    if not isinstance(shape, tuple):
        shape = tuple(shape for _ in range(x.ndim))
    pad = []
    for i in range(x.ndim):
        diff = shape[i] - x.shape[i]
        assert diff >= 0
        if both:
            a, b = divmod(diff, 2)
            b += a
            pad.append((a, b))
        else:
            pad.append((0, diff))
    pad = tuple(pad)  # type: ignore
    x = np.pad(x, pad, mode='constant', constant_values=fill)
    return x


def get_data_dir() -> str:
    """Get the DeepChem data directory.

    Returns
    -------
    str
        The default path to store DeepChem data. If you want to
        change this path, please set your own path to `DEEPCHEM_DATA_DIR`
        as an environment variable.
    """
    if 'DEEPCHEM_DATA_DIR' in os.environ:
        return os.environ['DEEPCHEM_DATA_DIR']
    return tempfile.gettempdir()


def download_url(url: str,
                 dest_dir: str = get_data_dir(),
                 name: Optional[str] = None):
    """Download a file to disk.

    Parameters
    ----------
    url: str
        The URL to download from
    dest_dir: str
        The directory to save the file in
    name: str
        The file name to save it as.  If omitted, it will try to extract a file name from the URL
    """
    if name is None:
        name = url
        if '?' in name:
            name = name[:name.find('?')]
        if '/' in name:
            name = name[name.rfind('/') + 1:]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    urlretrieve(url, os.path.join(dest_dir, name))


def untargz_file(file: str,
                 dest_dir: str = get_data_dir(),
                 name: Optional[str] = None):
    """Untar and unzip a .tar.gz file to disk.

    Parameters
    ----------
    file: str
        The filepath to decompress
    dest_dir: str
        The directory to save the file in
    name: str
        The file name to save it as.  If omitted, it will use the file name
    """
    if name is None:
        name = file
    tar = tarfile.open(name)
    tar.extractall(path=dest_dir)
    tar.close()


def unzip_file(file: str,
               dest_dir: str = get_data_dir(),
               name: Optional[str] = None):
    """Unzip a .zip file to disk.

    Parameters
    ----------
    file: str
        The filepath to decompress
    dest_dir: str
        The directory to save the file in
    name: str
        The directory name to unzip it to.  If omitted, it will use the file name
    """
    if name is None:
        name = file
    if dest_dir is None:
        dest_dir = os.path.join(get_data_dir, name)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


class UniversalNamedTemporaryFile:
    """The implementation for cross platform NamedTemporaryFile.

    `tempfile.NamedTemporaryFile` causes a permission error on Windows.
    This implementation avoids the error, please see threads on the stackoverflow [1]_.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    """

    def __init__(self, mode='wb', delete=True):
        self._mode = mode
        self._delete = delete

    def __enter__(self):
        # Generate a random temporary file name
        file_name = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        # Ensure the file is created
        open(file_name, "x").close()
        # Open the file in the given mode
        self._tempFile = open(file_name, self._mode)
        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempFile.close()
        if self._delete:
            os.remove(self._tempFile.name)


def load_image_files(input_files: List[str]) -> np.ndarray:
    """Loads a set of images from disk.

    Parameters
    ----------
    input_files: List[str]
        List of image filenames.

    Returns
    -------
    np.ndarray
        A numpy array that contains loaded images. The shape is, `(N,...)`.

    Notes
    -----
    This method requires Pillow to be installed.
    The supported file types are PNG and TIF.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError:
        raise ImportError("This function requires Pillow to be installed.")

    images = []
    for input_file in input_files:
        _, extension = os.path.splitext(input_file)
        extension = extension.lower()
        if extension == ".png":
            image = np.array(Image.open(input_file))
            images.append(image)
        elif extension == ".tif":
            im = Image.open(input_file)
            imarray = np.array(im)
            images.append(imarray)
        else:
            raise ValueError("Unsupported image filetype for %s" % input_file)
    # dtype=object allows for arrays(images here) of arbitrary size
    try:
        return np.array(images)
    except:
        return np.array(images, dtype=object)


def load_sdf_files(input_files: List[str],
                   clean_mols: bool = True,
                   tasks: List[str] = [],
                   shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Load SDF file into dataframe.

    Parameters
    ----------
    input_files: List[str]
        List of filenames
    clean_mols: bool, default True
        Whether to sanitize molecules.
    tasks: List[str], default []
        Each entry in `tasks` is treated as a property in the SDF file and is
        retrieved with `mol.GetProp(str(task))` where `mol` is the RDKit mol
        loaded from a given SDF entry.
    shard_size: int, default None
        The shard size to yield at one time.

    Returns
    -------
    Iterator[pd.DataFrame]
        Generator which yields the dataframe which is the same shard size.

    Notes
    -----
    This function requires RDKit to be installed.
    """
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    df_rows = []
    for input_file in input_files:
        # Tasks are either in .sdf.csv file or in the .sdf file itself for QM9 dataset
        has_csv = os.path.isfile(input_file + ".csv")
        # Structures are stored in .sdf file
        logger.info("Reading structures from %s." % input_file)

        suppl = Chem.SDMolSupplier(str(input_file),
                                   sanitize=clean_mols,
                                   removeHs=False,
                                   strictParsing=False)
        for ind, mol in enumerate(suppl):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            df_row = [ind, smiles, mol]
            if not has_csv:  # Get task targets from .sdf file
                for task in tasks:
                    df_row.append(mol.GetProp(str(task)))

            pos_x, pos_y, pos_z = tuple([tuple(pos_dim) for pos_dim in np.transpose(get_xyz_from_mol(mol)).tolist()])
            df_row.append(str(pos_x))
            df_row.append(str(pos_y))
            df_row.append(str(pos_z))
            df_rows.append(df_row)

            if shard_size is not None and len(df_rows) == shard_size:
                if has_csv:
                    mol_df = pd.DataFrame(df_rows,
                                          columns=('mol_id', 'smiles', 'mol',
                                                   'pos_x', 'pos_y', 'pos_z'))
                    raw_df = next(
                        load_csv_files([input_file + ".csv"], shard_size=None))
                    yield pd.concat([mol_df, raw_df], axis=1, join='inner')
                else:
                    # Note: Here, the order of columns is based on the order in which the values
                    # are appended to `df_row`. Since pos_x, pos_y, pos_z are appended after appending
                    # tasks above, they occur after `tasks` here.
                    # FIXME Ideally, we should use something like a dictionary here to keep it independent
                    # of column ordering.
                    mol_df = pd.DataFrame(df_rows,
                                          columns=('mol_id', 'smiles', 'mol') +
                                          tuple(tasks) +
                                          ('pos_x', 'pos_y', 'pos_z'))
                    yield mol_df
                # Reset aggregator
                df_rows = []

        # Handle final leftovers for this file
        if len(df_rows) > 0:
            if has_csv:
                mol_df = pd.DataFrame(df_rows,
                                      columns=('mol_id', 'smiles', 'mol',
                                               'pos_x', 'pos_y', 'pos_z'))
                raw_df = next(
                    load_csv_files([input_file + ".csv"], shard_size=None))
                yield pd.concat([mol_df, raw_df], axis=1, join='inner')
            else:
                mol_df = pd.DataFrame(df_rows,
                                      columns=('mol_id', 'smiles', 'mol') +
                                      tuple(tasks) +
                                      ('pos_x', 'pos_y', 'pos_z'))
                yield mol_df
            df_rows = []


def load_csv_files(input_files: List[str],
                   shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Load data as pandas dataframe from CSV files.

    Parameters
    ----------
    input_files: List[str]
        List of filenames
    shard_size: int, default None
        The shard size to yield at one time.

    Returns
    -------
    Iterator[pd.DataFrame]
        Generator which yields the dataframe which is the same shard size.
    """
    # First line of user-specified CSV *must* be header.
    shard_num = 1
    for input_file in input_files:
        if shard_size is None:
            yield pd.read_csv(input_file)
        else:
            logger.info("About to start loading CSV from %s" % input_file)
            for df in pd.read_csv(input_file, chunksize=shard_size):
                logger.info("Loading shard %d of size %s." %
                            (shard_num, str(shard_size)))
                df = df.replace(np.nan, str(""), regex=True)
                shard_num += 1
                yield df


def load_json_files(input_files: List[str],
                    shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Load data as pandas dataframe.

    Parameters
    ----------
    input_files: List[str]
        List of json filenames.
    shard_size: int, default None
        Chunksize for reading json files.

    Returns
    -------
    Iterator[pd.DataFrame]
        Generator which yields the dataframe which is the same shard size.

    Notes
    -----
    To load shards from a json file into a Pandas dataframe, the file
    must be originally saved with ``df.to_json('filename.json', orient='records', lines=True)``
    """
    shard_num = 1
    for input_file in input_files:
        if shard_size is None:
            yield pd.read_json(input_file, orient='records', lines=True)
        else:
            logger.info("About to start loading json from %s." % input_file)
            for df in pd.read_json(input_file,
                                   orient='records',
                                   chunksize=shard_size,
                                   lines=True):
                logger.info("Loading shard %d of size %s." %
                            (shard_num, str(shard_size)))
                df = df.replace(np.nan, str(""), regex=True)
                shard_num += 1
                yield df


def load_pickle_file(input_file: str) -> Any:
    """Load from single, possibly gzipped, pickle file.

    Parameters
    ----------
    input_file: str
        The filename of pickle file. This function can load from
        gzipped pickle file like `XXXX.pkl.gz`.

    Returns
    -------
    Any
        The object which is loaded from the pickle file.
    """
    if ".gz" in input_file:
        with gzip.open(input_file, "rb") as unzipped_file:
            return pickle.load(cast(IO[bytes], unzipped_file))
    else:
        with open(input_file, "rb") as opened_file:
            return pickle.load(opened_file)


def load_pickle_files(input_files: List[str]) -> Iterator[Any]:
    """Load dataset from pickle files.

    Parameters
    ----------
    input_files: List[str]
        The list of filenames of pickle file. This function can load from
        gzipped pickle file like `XXXX.pkl.gz`.

    Returns
    -------
    Iterator[Any]
        Generator which yields the objects which is loaded from each pickle file.
    """
    for input_file in input_files:
        yield load_pickle_file(input_file)


def load_data(input_files: List[str],
              shard_size: Optional[int] = None) -> Iterator[Any]:
    """Loads data from files.

    Parameters
    ----------
    input_files: List[str]
        List of filenames.
    shard_size: int, default None
        Size of shard to yield

    Returns
    -------
    Iterator[Any]
        Iterator which iterates over provided files.

    Notes
    -----
    The supported file types are SDF, CSV and Pickle.
    """
    if len(input_files) == 0:
        raise ValueError("The length of `filenames` must be more than 1.")

    file_type = _get_file_type(input_files[0])
    if file_type == "sdf":
        if shard_size is not None:
            logger.info("Ignoring shard_size for sdf input.")
        for value in load_sdf_files(input_files):
            yield value
    elif file_type == "csv":
        for value in load_csv_files(input_files, shard_size):
            yield value
    elif file_type == "pickle":
        if shard_size is not None:
            logger.info("Ignoring shard_size for pickle input.")
        for value in load_pickle_files(input_files):
            yield value


def _get_file_type(input_file: str) -> str:
    """Get type of input file. Must be csv/pkl/sdf/joblib file."""
    filename, file_extension = os.path.splitext(input_file)
    # If gzipped, need to compute extension again
    if file_extension == ".gz":
        filename, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        return "csv"
    elif file_extension == ".pkl":
        return "pickle"
    elif file_extension == ".joblib":
        return "joblib"
    elif file_extension == ".sdf":
        return "sdf"
    else:
        raise ValueError("Unrecognized extension %s" % file_extension)


def save_to_disk(dataset: Any, filename: str, compress: int = 3):
    """Save a dataset to file.

    Parameters
    ----------
    dataset: str
        A data saved
    filename: str
        Path to save data.
    compress: int, default 3
        The compress option when dumping joblib file.
    """
    if filename.endswith('.joblib'):
        joblib.dump(dataset, filename, compress=compress)
    elif filename.endswith('.npy'):
        np.save(filename, dataset)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)


def load_from_disk(filename: str) -> Any:
    """Load a dataset from file.

    Parameters
    ----------
    filename: str
        A filename you want to load data.

    Returns
    -------
    Any
        A loaded object from file.
    """
    name = filename
    if os.path.splitext(name)[1] == ".gz":
        name = os.path.splitext(name)[0]
    extension = os.path.splitext(name)[1]
    if extension == ".pkl":
        return load_pickle_file(filename)
    elif extension == ".joblib":
        return joblib.load(filename)
    elif extension == ".csv":
        # First line of user-specified CSV *must* be header.
        df = pd.read_csv(filename, header=0)
        df = df.replace(np.nan, str(""), regex=True)
        return df
    elif extension == ".npy":
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Unrecognized filetype for %s" % filename)


def load_dataset_from_disk(
    save_dir: str
) -> Tuple[bool, Optional[Tuple["dc.data.DiskDataset", "dc.data.DiskDataset",
                                "dc.data.DiskDataset"]],
           List["dc.trans.Transformer"]]:
    """Loads MoleculeNet train/valid/test/transformers from disk.

    Expects that data was saved using `save_dataset_to_disk` below. Expects the
    following directory structure for `save_dir`:
    save_dir/
      |
      ---> train_dir/
      |
      ---> valid_dir/
      |
      ---> test_dir/
      |
      ---> transformers.pkl

    Parameters
    ----------
    save_dir: str
        Directory name to load datasets.

    Returns
    -------
    loaded: bool
        Whether the load succeeded
    all_dataset: Tuple[DiskDataset, DiskDataset, DiskDataset]
        The train, valid, test datasets
    transformers: Transformer
        The transformers used for this dataset

    See Also
    --------
    save_dataset_to_disk
    """

    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if not os.path.exists(train_dir) or not os.path.exists(
            valid_dir) or not os.path.exists(test_dir):
        return False, None, list()
    loaded = True
    train = dc.data.DiskDataset(train_dir)
    valid = dc.data.DiskDataset(valid_dir)
    test = dc.data.DiskDataset(test_dir)
    train.memory_cache_size = 40 * (1 << 20)  # 40 MB
    all_dataset = (train, valid, test)
    transformers = load_transformers(save_dir)
    return loaded, all_dataset, transformers


def save_dataset_to_disk(save_dir: str, train: "dc.data.DiskDataset",
                         valid: "dc.data.DiskDataset",
                         test: "dc.data.DiskDataset",
                         transformers: List["dc.trans.Transformer"]):
    """Utility used by MoleculeNet to save train/valid/test datasets.

    This utility function saves a train/valid/test split of a dataset along
    with transformers in the same directory. The saved datasets will take the
    following structure:
    save_dir/
      |
      ---> train_dir/
      |
      ---> valid_dir/
      |
      ---> test_dir/
      |
      ---> transformers.pkl

    Parameters
    ----------
    save_dir: str
        Directory name to save datasets to.
    train: DiskDataset
        Training dataset to save.
    valid: DiskDataset
        Validation dataset to save.
    test: DiskDataset
        Test dataset to save.
    transformers: List[Transformer]
        List of transformers to save to disk.

    See Also
    --------
    load_dataset_from_disk
    """
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    train.move(train_dir)
    valid.move(valid_dir)
    test.move(test_dir)
    save_transformers(save_dir, transformers)


def load_transformers(save_dir: str) -> List["dc.trans.Transformer"]:
    """Load the transformers for a MoleculeNet dataset from disk."""
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
        return pickle.load(f)


def save_transformers(save_dir: str,
                      transformers: List["dc.trans.Transformer"]):
    """Save the transformers for a MoleculeNet dataset to disk."""
    with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
        pickle.dump(transformers, f)


def remove_missing_entries(dataset):
    """Remove missing entries.

    Some of the datasets have missing entries that sneak in as zero'd out
    feature vectors. Get rid of them.
    """
    for i, (X, y, w, ids) in enumerate(dataset.itershards()):
        available_rows = X.any(axis=1)
        logger.info("Shard %d has %d missing entries." %
                    (i, np.count_nonzero(~available_rows)))
        X = X[available_rows]
        y = y[available_rows]
        w = w[available_rows]
        ids = ids[available_rows]
        dataset.set_shard(i, X, y, w, ids)

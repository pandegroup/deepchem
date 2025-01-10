"""
qm8 dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

GDB8_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz"
QM8_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv"
QM8_TASKS = [
    "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
    "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
    "f1-CAM", "f2-CAM"
]


class _QM8Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "qm8.sdf")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=GDB8_URL,
                                             dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(
                os.path.join(self.data_dir, "gdb8.tar.gz"), self.data_dir)
        loader = dc.data.SDFLoader(tasks=self.tasks,
                                   featurizer=self.featurizer,
                                   sanitize=True)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_qm8(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.CoulombMatrix(26),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load QM8 dataset

    QM8 is the dataset used in a study on modeling quantum
    mechanical calculations of electronic spectra and excited
    state energy of small molecules. Multiple methods, including
    time-dependent density functional theories (TDDFT) and
    second-order approximate coupled-cluster (CC2), are applied to
    a collection of molecules that include up to eight heavy atoms
    (also a subset of the GDB-17 database). In our collection,
    there are four excited state properties calculated by four
    different methods on 22 thousand samples:

    S0 -> S1 transition energy E1 and the corresponding oscillator strength f1

    S0 -> S2 transition energy E2 and the corresponding oscillator strength f2

    E1, E2 are in atomic units. f1, f2 are in length representation

    Random splitting is recommended for this dataset.

    The source data contain:

    - qm8.sdf: molecular structures
    - qm8.sdf.csv: tables for molecular properties

    - Column 1: Molecule ID (gdb9 index) mapping to the .sdf file
    - Columns 2-5: RI-CC2/def2TZVP
    - Columns 6-9: LR-TDPBE0/def2SVP
    - Columns 10-13: LR-TDPBE0/def2TZVP
    - Columns 14-17: LR-TDCAM-B3LYP/def2TZVP

    Parameters
    ----------
    featurizer: Featurizer or str
        the featurizer to use for processing the data.  Alternatively you can pass
        one of the names from dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.
    transformers: list of TransformerGenerators or strings
        the Transformers to apply to the data.  Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
    reload: bool
        if True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
        a directory to save the raw data in
    save_dir: str
        a directory to save the dataset in

    Note
    ----
    DeepChem 2.4.0 has turned on sanitization for this dataset by
    default.  For the QM8 dataset, this means that calling this
    function will return 21747 compounds instead of 21786 in the source
    dataset file.  This appears to be due to valence specification
    mismatches in the dataset that weren't caught in earlier more lax
    versions of RDKit.  Note that this may subtly affect benchmarking
    results on this dataset.

    References
    ----------
    .. [1] Blum, Lorenz C., and Jean-Louis Reymond. "970 million druglike
        small molecules for virtual screening in the chemical universe database
        GDB-13." Journal of the American Chemical Society 131.25 (2009):
        8732-8733.
    .. [2] Ramakrishnan, Raghunathan, et al. "Electronic spectra from TDDFT
        and machine learning in chemical space." The Journal of chemical physics
        143.8 (2015): 084111.
    """
    loader = _QM8Loader(featurizer, splitter, transformers, QM8_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('qm8', reload)

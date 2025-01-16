"""
qm9 dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
QM9_TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    "h298", "g298"
]


class _QM9Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "gdb9.sdf")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=GDB9_URL,
                                             dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(
                os.path.join(self.data_dir, "gdb9.tar.gz"), self.data_dir)
        loader = dc.data.SDFLoader(tasks=self.tasks,
                                   featurizer=self.featurizer,
                                   sanitize=True)
        return loader.create_dataset(dataset_file, shard_size=4096)


def load_qm9(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.CoulombMatrix(29),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load QM9 dataset

    QM9 is a comprehensive dataset that provides geometric, energetic,
    electronic and thermodynamic properties for a subset of GDB-17
    database, comprising 134 thousand stable organic molecules with up
    to 9 heavy atoms.  All molecules are modeled using density
    functional theory (B3LYP/6-31G(2df,p) based DFT).

    Random splitting is recommended for this dataset.

    The source data contain:

    - qm9.sdf: molecular structures
    - qm9.sdf.csv: tables for molecular properties

    - "mol_id" - Molecule ID (gdb9 index) mapping to the .sdf file
    - "A" - Rotational constant (unit: GHz)
    - "B" - Rotational constant (unit: GHz)
    - "C" - Rotational constant (unit: GHz)
    - "mu" - Dipole moment (unit: D)
    - "alpha" - Isotropic polarizability (unit: Bohr^3)
    - "homo" - Highest occupied molecular orbital energy (unit: Hartree)
    - "lumo" - Lowest unoccupied molecular orbital energy (unit: Hartree)
    - "gap" - Gap between HOMO and LUMO (unit: Hartree)
    - "r2" - Electronic spatial extent (unit: Bohr^2)
    - "zpve" - Zero point vibrational energy (unit: Hartree)
    - "u0" - Internal energy at 0K (unit: Hartree)
    - "u298" - Internal energy at 298.15K (unit: Hartree)
    - "h298" - Enthalpy at 298.15K (unit: Hartree)
    - "g298" - Free energy at 298.15K (unit: Hartree)
    - "cv" - Heat capavity at 298.15K (unit: cal/(mol*K))
    - "u0_atom" - Atomization energy at 0K (unit: kcal/mol)
    - "u298_atom" - Atomization energy at 298.15K (unit: kcal/mol)
    - "h298_atom" - Atomization enthalpy at 298.15K (unit: kcal/mol)
    - "g298_atom" - Atomization free energy at 298.15K (unit: kcal/mol)

    "u0_atom" ~ "g298_atom" (used in MoleculeNet) are calculated from the
    differences between "u0" ~ "g298" and sum of reference energies of all
    atoms in the molecules, as given in
    https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643

    Parameters
    ----------
    featurizer: Featurizer or str
        the featurizer to use for processing the data.  Alternatively you can pass
        one of the names from dc.molnet.featurizers as a shortcut.
        You can check the description of all the available featurizers here: https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html
    splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.
        You can check the description of all the available splitters here: https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html
    transformers: list of TransformerGenerators or strings
        the Transformers to apply to the data.  Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
        You can check the description of all the available transformers here: https://deepchem.readthedocs.io/en/latest/api_reference/transformers.html
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
    default.  For the QM9 dataset, this means that calling this
    function will return 132480 compounds instead of 133885 in the
    source dataset file. This appears to be due to valence
    specification mismatches in the dataset that weren't caught in
    earlier more lax versions of RDKit. Note that this may subtly
    affect benchmarking results on this dataset.

    References
    ----------
    .. [1] Blum, Lorenz C., and Jean-Louis Reymond. "970 million druglike small
        molecules for virtual screening in the chemical universe database GDB-13."
        Journal of the American Chemical Society 131.25 (2009): 8732-8733.
    .. [2] Ramakrishnan, Raghunathan, et al. "Quantum chemistry structures and
        properties of 134 kilo molecules." Scientific data 1 (2014): 140022.
    """
    loader = _QM9Loader(featurizer, splitter, transformers, QM9_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('qm9', reload)

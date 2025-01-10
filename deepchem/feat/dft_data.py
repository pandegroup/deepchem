"""
Density Functional Theory Data
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/entry.py
"""
from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List, Dict, Optional
import numpy as np
from deepchem.utils.dftutils import KSCalc
from deepchem.utils.dft_utils import Mol, BaseSystem, parse_moldesc, BaseGrid


class DFTSystem():
    """
    The DFTSystem class creates and returns the various systems in an entry object as dictionaries.

    Examples
    --------
    >>> from deepchem.feat.dft_data import DFTSystem
    >>> systems = {'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0','basis': '6-311++G(3df,3pd)'}
    >>> output = DFTSystem(systems)

    Returns
    -------
    DFTSystem object for all the individual atoms/ions/molecules in an entry object.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.

    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/system/base_system.py
    """

    def __init__(self, system: Dict):
        self.system = system
        self.moldesc = system["moldesc"]
        self.basis = system["basis"]
        self.spin = 0
        self.charge = 0
        self.no = 1
        if 'spin' in system.keys():
            self.spin = int(system["spin"])
        if 'charge' in system.keys():
            self.charge = int(system["charge"])
        if 'number' in system.keys():
            self.no = int(system["number"])
        """
        Parameters
        ----------
        system: Dict
            system is a dictionary containing information on the atomic positions,
            atomic numbers, and basis set used for the DFT calculation.
        """

    def get_dqc_mol(self, pos_reqgrad: bool = False) -> BaseSystem:
        """
        This method converts the system dictionary to a DQC system and returns it.
        Parameters
        ----------
        pos_reqgrad: bool
            decides if the atomic position require gradient calculation.
        Returns
        -------
        mol
            DQC mol object
        """
        atomzs, atomposs = parse_moldesc(self.moldesc)
        if pos_reqgrad:
            atomposs.requires_grad_()
        mol = Mol(self.moldesc, self.basis, spin=self.spin,
                  charge=self.charge)  # type: ignore[abstract]
        return mol


class DFTEntry():
    """
    Handles creating and initialising DFTEntry objects from the dataset. This object contains information    about the various systems in the datapoint (atoms, molecules and ions) along with the ground truth
    values.
    Notes
    -----
    Entry class should not be initialized directly, but created through
    ``Entry.create``

    Example
    -------
    >>> from deepchem.feat.dft_data import DFTEntry
    >>> e_type= 'dm'
    >>> true_val= 'deepchem/data/tests/dftHF_output.npy'
    >>> systems = [{'moldesc': 'H 0.86625 0 0; F -0.86625 0 0','basis': '6-311++G(3df,3pd)'}]
    >>> dm_entry_for_HF = DFTEntry.create(e_type, true_val, systems)
    """

    @classmethod
    def create(self,
               e_type: str,
               true_val: Optional[str],
               systems: List[Dict],
               weight: Optional[int] = 1):
        """
        This method is used to initialise the DFTEntry class. The entry objects are created
        based on their entry type.

        Parameters
        ----------
        e_type: str
            Determines the type of calculation to be carried out on the entry
            object. Accepts the following values: "ae", "ie", "dm", "dens", that stand for atomization energy,
            ionization energy, density matrix and density profile respectively.
        true_val: str
            Ground state energy values for the entry object as a string (for ae
            and ie), or a .npy file containing a matrix ( for dm and dens).
        systems: List[Dict]
            List of dictionaries contains "moldesc", "basis" and "spin"
            of all the atoms/molecules. These values are to be entered in
            the DQC or PYSCF format. The systems needs to be entered in a
            specific order, i.e ; the main atom/molecule needs to be the
            first element. (This is for objects containing equations, such
            as ae and ie entry objects). Spin and charge of the system are
            optional parameters and are considered '0' if not specified.
            The system number refers to the number of times the systems is
            present in the molecule - this is for polyatomic molecules and the
            default value is 1. For example ; system number of Hydrogen in water
            is 2.
        weight: int
            Weight of the entry object.
        Returns
        -------
        DFTEntry object based on entry type

        """
        if true_val is None:
            true_val = '0.0'
        if e_type == "ae":
            return _EntryAE(e_type, true_val, systems, weight)
        elif e_type == "ie":
            return _EntryIE(e_type, true_val, systems, weight)
        elif e_type == "dm":
            return _EntryDM(e_type, true_val, systems, weight)
        elif e_type == "dens":
            return _EntryDens(e_type, true_val, systems, weight)
        else:
            raise NotImplementedError("Unknown entry type: %s" % e_type)

    def __init__(self,
                 e_type: str,
                 true_val: Optional[str],
                 systems: List[Dict],
                 weight: Optional[int] = 1):
        self._systems = [DFTSystem(p) for p in systems]
        self._weight = weight

    def get_systems(self) -> List[DFTSystem]:
        """
        Parameters
        ----------
        systems:  List[DFTSystem]

        Returns
        -------
        List of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returns
        -------
        The type of entry ;
        1) Atomic Ionization Potential (IP/IE)
        2) Atomization Energy (AE)
        3) Density Profile (DENS)
        4) Density Matrix (DM)
        """
        pass

    def get_true_val(self) -> np.ndarray:
        """
        Get the true value of the DFTEntry.
        For the AE and IP entry types, the experimental values are collected from the NIST CCCBDB/ASD
        databases.
        The true values of density profiles are calculated using PYSCF-CCSD calculations. This method            simply loads the value, no calculation is performed.
        """
        return np.array(0)

    @abstractmethod
    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        Return the energy value of the entry, using a DQC-DFT calculation, where the XC has been
        replaced by the trained neural network. This method does not carry out any calculations, it is
        an interface to the KSCalc utility.
        """
        pass

    def get_weight(self):
        """
        Returns
        -------
        Weight of the entry object
        """
        return self._weight


class _EntryDM(DFTEntry):
    """
    Entry for Density Matrix (DM)
    The density matrix is used to express total energy of both non-interacting and
    interacting systems.

    Notes
    -----
    dm entry can only have 1 system
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
           must be a .npy file containing the pre-calculated density matrix
        systems: List[Dict]

        """
        super().__init__(e_type, true_val, systems)
        self.true_val = true_val
        assert len(self.get_systems()) == 1
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "dm"

    def get_true_val(self) -> np.ndarray:
        # get the density matrix from PySCF's CCSD calculation
        dm = np.load(self.true_val)
        return dm

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        val = qcs[0].aodmtot()
        return np.array([val.tolist()])


class _EntryDens(DFTEntry):
    """
    Entry for density profile (dens), compared with CCSD calculation
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
           must be a .npy file containing the pre-calculated density profile.
        systems: List[Dict]
        """
        super().__init__(e_type, true_val, systems)
        self.true_val = true_val
        assert len(self.get_systems()) == 1
        self._grid: Optional[BaseGrid] = None
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "dens"

    def get_true_val(self) -> np.ndarray:
        dens = np.load(self.true_val)
        return dens

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        This method calculates the integration grid which is then used to calculate the
        density profile of an entry object.

        Parameters
        ----------
        qcs: List[KSCalc]

        Returns
        -------
        Numpy array containing calculated density profile
        """
        qc = qcs[0]

        grid = self.get_integration_grid()
        rgrid = grid.get_rgrid()
        val = qc.dens(rgrid)
        return np.array(val.tolist())

    def get_integration_grid(self) -> BaseGrid:
        """
        This method is used to calculate the integration grid required for a system
        in order to calculate it's density profile using Differentiable DFT.

        Returns
        -------
        grid: BaseGrid

        References
        ----------
        https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/grid/base_grid.py
        """
        if self._grid is None:
            system = self.get_systems()[0]
            dqc_mol = system.get_dqc_mol()
            dqc_mol.setup_grid()
            grid = dqc_mol.get_grid()
            assert grid.coord_type == "cart"
            self._grid = grid

        return self._grid


class _EntryIE(DFTEntry):
    """
    Entry for Ionization Energy (IE)
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
        systems: List[Dict]
        """
        super().__init__(e_type, true_val, systems)
        self.true_val = float(true_val)
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "ie"

    def get_true_val(self) -> np.ndarray:
        return np.array([self.true_val])

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        This method calculates the energy of an entry based on the systems and command present
        in the data object. For example; for a Lithium hydride molecule the total energy
        would be ; energy(Li) + energy(H) - energy(LiH)

        Parameters
        ----------
        qcs: List[KSCalc]

        Returns
        -------
        Total Energy of a data object for entry types IE and AE
        """
        systems = [i.no for i in self.get_systems()]
        e_1 = [m.energy() for m in qcs]
        e = [item1 * item2 for item1, item2 in zip(systems, e_1)]
        val = sum(e) - 2 * e[0]
        return np.array([val.tolist()])


class _EntryAE(_EntryIE):
    """
    Entry for Atomization Energy (AE)
    """

    @property
    def entry_type(self) -> str:
        return "ae"

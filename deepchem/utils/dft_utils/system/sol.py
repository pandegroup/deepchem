from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict
import torch
import numpy as np
import scipy.special
from deepchem.utils.cache_utils import Cache
from deepchem.utils.dft_utils.api.parser import parse_moldesc
from deepchem.utils.dft_utils.data.datastruct import AtomCGTOBasis, CGTOBasis, DensityFitInfo, ZType
from deepchem.utils.dft_utils.grid.factory import get_predefined_grid
from deepchem.utils.dft_utils.hamilton.base_hamilton import BaseHamilton
from deepchem.utils.dft_utils.hamilton.hcgto_pbc import HamiltonCGTO_PBC
from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
from deepchem.utils.dft_utils.hamilton.intor.pbcintor import PBCIntOption
from deepchem.utils.dft_utils.system.base_system import BaseSystem
from deepchem.utils.dft_utils.system.mol import _get_nelecs_spin, _get_orb_weights, _parse_basis
from deepchem.utils.dft_utils import BaseGrid, SpinParam, AtomZsType, AtomPosType, BasisInpType
from deepchem.utils.periodic_table_utils import get_atom_mass
from deepchem.utils.safeops_utils import safe_cdist

__all__ = ["Sol"]


class Sol(BaseSystem):
    """
    Describe the system of a solid (i.e. periodic boundary condition system).
    """

    def __init__(
            self,
            soldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
            alattice: torch.Tensor,
            basis: Union[str, List[CGTOBasis], List[str],
                         List[List[CGTOBasis]]],
            *,
            grid: Union[int, str] = "sg3",
            spin: Optional[ZType] = None,
            lattsum_opt: Optional[Union[PBCIntOption, Dict]] = None,
            dtype: torch.dtype = torch.float64,
            device: torch.device = torch.device('cpu'),
    ):
        """Initialises the System.

        Parameters
        ----------
        soldesc: str or 2-elements tuple
            Description of the molecule system.
            If string, it can be described like ``"H 1 0 0; H -1 0 0"``.
            If tuple, the first element of the tuple is the Z number of the atoms while
            the second element is the position of the atoms: ``(atomzs, atomposs)``.
        basis: str, CGTOBasis, list of str, or CGTOBasis
            The string describing the gto basis. If it is a list, then it must have
            the same length as the number of atoms.
        grid: int
            Describe the grid.
            If it is an integer, then it uses the default grid with specified level
            of accuracy.
        spin: int, float, torch.Tensor, or None
            The difference between spin-up and spin-down electrons.
            It must be an integer or ``None``.
            If ``None``, then it is ``num_electrons % 2``.
            For floating point atomzs and/or charge, the ``spin`` must be specified.
        charge: int, float, or torch.Tensor
            The charge of the molecule.
        orb_weights: SpinParam[torch.Tensor] or None
            Specifiying the orbital occupancy (or weights) directly. If specified,
            ``spin`` and ``charge`` arguments are ignored.
        dtype: torch.dtype
            The data type of tensors in this class.
        device: torch.device
            The device on which the tensors in this class are stored.
        """
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._basis_inp = basis
        self._grid: Optional[BaseGrid] = None
        charge = 0  # we can't have charged solids for now

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,) dtype: torch.int or dtype for floating point
        # atompos: (natoms, ndim)
        atomzs, atompos = parse_moldesc(soldesc, dtype, device)
        allbases = _parse_basis(atomzs, basis)  # list of list of CGTOBasis
        atombases = [
            AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
            for (atz, bas, atpos) in zip(atomzs, allbases, atompos)
        ]
        self._atombases = atombases
        self._atompos = atompos  # (natoms, ndim)
        self._atomzs = atomzs  # (natoms,) int-type
        nelecs_tot: torch.Tensor = torch.sum(atomzs)

        # get the number of electrons and spin and orbital weights
        nelecs, spin, frac_mode = _get_nelecs_spin(nelecs_tot, spin, charge)
        assert not frac_mode, "Fractional Z mode for pbc is not supported"
        _orb_weights, _orb_weights_u, _orb_weights_d = _get_orb_weights(
            nelecs, spin, frac_mode, dtype, device)

        # initialize cache
        self._cache = Cache()

        # save the system's properties
        self._spin = spin
        self._charge = charge
        self._numel = nelecs
        self._orb_weights = _orb_weights
        self._orb_weights_u = _orb_weights_u
        self._orb_weights_d = _orb_weights_d
        self._alattice_inp = alattice
        self._lattice = Lattice(self._alattice_inp)
        self._lattsum_opt = PBCIntOption.get_default(lattsum_opt)

    def densityfit(self,
                   method: Optional[str] = None,
                   auxbasis: Optional[BasisInpType] = None) -> BaseSystem:
        """
        Indicate that the system's Hamiltonian uses density fit for its integral.

        Parameters
        ----------
        method: Optional[str]
            Density fitting method. Available methods in this class are:

            * ``"gdf"``: Density fit with gdf compensating charge to perform
                the lattice sum. Ref https://doi.org/10.1063/1.4998644 (default)

        auxbasis: Optional[BasisInpType]
            Auxiliary basis for the density fit. If not specified, then it uses
            ``"cc-pvtz-jkfit"``.

        Returns
        -------
        BaseSystem
            System with density fitting.

        """
        if method is None:
            method = "gdf"
        if auxbasis is None:
            auxbasis = "cc-pvtz-jkfit"

        # get the auxiliary basis
        assert auxbasis is not None
        auxbasis_lst = _parse_basis(self._atomzs, auxbasis)
        atomauxbases = [
            AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
            for (atz, bas,
                 atpos) in zip(self._atomzs, auxbasis_lst, self._atompos)
        ]

        # change the hamiltonian to have density fit
        df = DensityFitInfo(method=method, auxbases=atomauxbases)
        self._hamilton = HamiltonCGTO_PBC(
            self._atombases,
            df=df,
            latt=self._lattice,
            lattsum_opt=self._lattsum_opt,
            cache=self._cache.add_prefix("hamilton"))
        return self

    def get_hamiltonian(self) -> BaseHamilton:
        """
        Returns
        -------
        Hamiltonian that corresponds to the system, i.e.
        :class:`~dqc.hamilton.HamiltonCGTO_PBC`
        """
        return self._hamilton

    def set_cache(self,
                  fname: str,
                  paramnames: Optional[List[str]] = None) -> BaseSystem:
        """
        Setup the cache of some parameters specified by `paramnames` to be read/written
        on a file.
        If the file exists, then the parameters will not be recomputed, but just
        loaded from the cache instead.
        Cache is usually used for repeated calculations where the cached parameters
        are not changed (e.g. running multiple systems with slightly different environment.)

        Parameters
        ----------
        fname: str
            The file to store the cache.
        paramnames: list of str or None
            List of parameter names to be read/write from the cache.

        Returns
        -------
        BaseSystem
            The system with the cache setup.

        """
        self._cache.set(fname, paramnames)
        return self

    def get_orbweight(
        self,
        polarized: bool = False
    ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Get the orbital weights of the system.

        Parameters
        ----------
        polarized: bool (default=False)
            If True, then return the polarized orbital weights.

        Returns
        -------
        torch.Tensor or SpinParam[torch.Tensor]
            The orbital weights of the system.

        """
        if not polarized:
            return self._orb_weights
        else:
            return SpinParam(u=self._orb_weights_u, d=self._orb_weights_d)

    def get_nuclei_energy(self) -> torch.Tensor:
        """Get the nuclear-nuclear energy of the system.

        Returns
        -------
        torch.Tensor
            Nuclear-nuclear energy of the system.

        """
        # self._atomzs: (natoms,)
        # self._atompos: (natoms, ndim)

        # r12: (natoms, natoms)
        r12_inf = safe_cdist(self._atompos,
                             self._atompos,
                             add_diag_eps=True,
                             diag_inf=True)
        r12 = safe_cdist(self._atompos, self._atompos, add_diag_eps=True)
        z12 = self._atomzs.unsqueeze(-2) * self._atomzs.unsqueeze(
            -1)  # (natoms, natoms)

        precision = self._lattsum_opt.precision
        eta = self._lattice.estimate_ewald_eta(precision) * 2
        vol = self._lattice.volume()
        rcut = scipy.special.erfcinv(
            float(vol.detach()) * eta * eta / (2 * np.pi) * precision) / eta
        gcut = scipy.special.erfcinv(
            precision * np.sqrt(np.pi) / 2 / eta) * 2 * eta

        # get the shift vector in real space and in reciprocal space
        ls = self._lattice.get_lattice_ls(rcut=rcut,
                                          exclude_zeros=True)  # (nls, ndim)
        # gv: (ngv, ndim), gvweights: (ngv,)
        gv, gvweights = self._lattice.get_gvgrids(gcut=gcut, exclude_zeros=True)
        gv_norm2 = torch.einsum("gd,gd->g", gv, gv)  # (ngv)

        # get the shift in position
        atpos_shift = self._atompos - ls.unsqueeze(-2)  # (nls, natoms, ndim)
        r12_ls = safe_cdist(atpos_shift, self._atompos)  # (nls, natoms, natoms)

        # calculate the short range
        short_range_comp1 = torch.erfc(
            eta * r12_ls) / r12_ls  # (nls, natoms, natoms)
        short_range_comp2 = torch.erfc(eta * r12) / r12_inf  # (natoms, natoms)
        short_range1 = torch.sum(z12 * short_range_comp1)  # scalar
        short_range2 = torch.sum(z12 * short_range_comp2)  # scalar
        short_range = short_range1 + short_range2

        # calculate the long range sum
        coul_g = 4 * np.pi / gv_norm2 * gvweights  # (ngv,)
        # this part below is quicker, but raises warning from pytorch
        si = torch.exp(
            1j *
            torch.matmul(self._atompos, gv.transpose(-2, -1)))  # (natoms, ngv)
        zsi = torch.einsum("a,ag->g", self._atomzs.to(si.dtype), si)  # (ngv,)
        zexpg2 = zsi * torch.exp(-gv_norm2 / (4 * eta * eta))
        long_range = torch.einsum("a,a,a->", zsi.conj(), zexpg2,
                                  coul_g.to(zsi.dtype)).real  # (scalar)

        # # alternative way to compute the long-range part
        # r12_pair = self._atompos.unsqueeze(-2) - self._atompos  # (natoms, natoms, ndim)
        # long_range_exp = torch.exp(-gv_norm2 / (4 * eta * eta)) * coul_g  # (ngv,)
        # long_range_cos = torch.cos(torch.einsum("gd,abd->gab", gv, -r12_pair))  # (ngv, natoms, natoms)
        # long_range = torch.sum(long_range_exp[:, None, None] * long_range_cos * z12)  # scalar

        # background interaction
        vbar1 = -torch.sum(self._atomzs**2) * (2 * eta / np.sqrt(np.pi))
        vbar2 = -torch.sum(self._atomzs)**2 * np.pi / (eta * eta * vol)
        vbar = vbar1 + vbar2  # (scalar)

        eii = short_range + long_range + vbar
        return eii * 0.5

    def setup_grid(self) -> None:
        """Automatically setup the grid for the system."""
        self._grid = get_predefined_grid(self._grid_inp,
                                         self._atomzs,
                                         self._atompos,
                                         lattice=self._lattice,
                                         dtype=self._dtype,
                                         device=self._device)

    def get_grid(self) -> BaseGrid:
        """Get the grid of the system.

        Returns
        -------
        BaseGrid
            Grid of the system.

        """
        if self._grid is None:
            raise RuntimeError(
                "Please run mol.setup_grid() first before calling get_grid()")
        return self._grid

    def requires_grid(self) -> bool:
        """Check if the system requires grid for its calculation.

        Returns
        -------
        bool
            True if the system requires grid, False otherwise.

        """
        return False

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Get the parameter names of the system.

        Parameters
        ----------
        methodname: str
            Method name.
        prefix: str
            Prefix to add to the parameter names.

        Returns
        -------
        List[str]
            The list of parameter names.

        """
        pass

    def make_copy(self, **kwargs) -> Sol:
        """
        Returns a copy of the system identical to the orginal except for new
        parameters set in the kwargs.

        Parameters
        ----------
        **kwargs
            Must be the same kwargs as Sol.

        Returns
        -------
        Sol
            New system with the new parameters set

        """
        # create dictionary of all parameters
        parameters = {
            'soldesc': (self.atomzs, self.atompos),
            'alattice': self._alattice_inp,
            'basis': self._basis_inp,
            'grid': self._grid_inp,
            'spin': self._spin,
            'lattsum_opt': self._lattsum_opt,
            'dtype': self._dtype,
            'device': self._device
        }
        # update dictionary with provided kwargs
        parameters.update(kwargs)
        # create new system
        return Sol(**parameters)

    # properties
    @property
    def atompos(self) -> torch.Tensor:
        """Get the atomic positions of the system.

        Returns
        -------
        torch.Tensor
            Atomic positions of the system.

        """
        return self._atompos

    @property
    def atomzs(self) -> torch.Tensor:
        """Get the atomic numbers of the system.

        Returns
        -------
        torch.Tensor
            Atomic numbers of the system.

        """
        return self._atomzs

    @property
    def atommasses(self) -> torch.Tensor:
        """Get the atomic masses of the system.

        Returns
        -------
        torch.Tensor
            Atomic masses of the system.

        """
        # returns the atomic mass (only for non-isotope for now)
        return torch.tensor(
            [get_atom_mass(int(atomz)) for atomz in self._atomzs],
            dtype=self._dtype,
            device=self._device)

    @property
    def spin(self) -> ZType:
        """Get the spin of the system.

        Returns
        -------
        ZType
            Spin of the system.

        """
        return self._spin

    @property
    def charge(self) -> ZType:
        """Get the charge of the system.

        Returns
        -------
        ZType
            Charge of the system.

        """
        return self._charge

    @property
    def numel(self) -> ZType:
        """Get the number of electrons of the system.

        Returns
        -------
        ZType
            Number of electrons of the system.

        """
        return self._numel

    @property
    def efield(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get the external electric field of the system.

        Returns
        -------
        Optional[Tuple[torch.Tensor, ...]]
            External electric field of the system.

        """
        return None

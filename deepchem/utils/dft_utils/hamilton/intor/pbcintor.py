from __future__ import annotations
import ctypes
from functools import reduce
import operator
import warnings
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
from deepchem.utils.dft_utils.hamilton.intor.utils import NDIM, int2ctypes, np2ctypes, CPBC, CGTO, c_null_ptr
from deepchem.utils.dft_utils import LibcintWrapper
from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager
from deepchem.utils.dft_utils.hamilton.intor.molintor import _check_and_set, _get_intgl_optimizer
from deepchem.utils.pytorch_utils import estimate_ovlp_rcut, get_complex_dtype


@dataclass
class PBCIntOption:
    """PBCIntOption is a class that contains parameters for the PBC integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import PBCIntOption
    >>> pbc = PBCIntOption()
    >>> pbc.get_default()
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)

    Attributes
    ----------
    precision: float (default 1e-8)
        Precision of the integral to limit the lattice sum.
    kpt_diff_tol: float (default 1e-6)
        Difference between k-points to be regarded as the same.

    """
    precision: float = 1e-8
    kpt_diff_tol: float = 1e-6

    @staticmethod
    def get_default(
        lattsum_opt: Optional[Union[PBCIntOption,
                                    Dict]] = None) -> PBCIntOption:
        """Get the default PBCIntOption object.

        Parameters
        ----------
        lattsum_opt: Optional[Union[PBCIntOption, Dict]]
            The lattice sum option. If it is a dictionary, then it will be
            converted to a PBCIntOption object. If it is None, then just use
            the default value of PBCIntOption.

        Returns
        -------
        PBCIntOption
            The default PBCIntOption object.

        """
        if lattsum_opt is None:
            return PBCIntOption()
        elif isinstance(lattsum_opt, dict):
            return PBCIntOption(**lattsum_opt)
        else:
            return lattsum_opt

def pbc_int1e(shortname: str, wrapper: LibcintWrapper,
              other: Optional[LibcintWrapper] = None,
              kpts: Optional[torch.Tensor] = None,  # (nkpts, ndim)
              options: Optional[PBCIntOption] = None,
              ):
    """
    Performing the periodic boundary condition (PBC) on 1-electron integrals.

    Arguments
    ---------
    shortname: str
        The shortname of the integral (i.e. without the prefix `int1e_` or else)
    wrapper: LibcintWrapper
        The environment wrapper containing the basis
    other: Optional[LibcintWrapper]
        Another environment wrapper containing the basis. This environment
        must have the same complete environment as `wrapper` (e.g. `other` can be
        a subset of `wrapper`). If unspecified, then `other = wrapper`.
    kpts: Optional[torch.Tensor]
        k-points where the integration is supposed to be performed. If specified,
        it should have the shape of `(nkpts, ndim)`. Otherwise, it is assumed
        to be all zeros.
    options: Optional[PBCIntOption]
        The integration options. If unspecified, then just use the default
        value of `PBCIntOption`.

    Returns
    -------
    torch.Tensor
        A complex tensor representing the 1-electron integral with shape
        `(nkpts, *ncomp, nwrapper, nother)` where `ncomp` is the Cartesian
        components of the integral, e.g. `"ipovlp"` integral will have 3
        components each for x, y, and z.
    """

    # check and set the default values
    other1 = check_and_set_pbc(wrapper, other)
    options1 = get_default_options(options)
    kpts1 = get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)

    assert isinstance(wrapper.lattice, Lattice)  # check if wrapper has a lattice
    return _PBCInt2cFunction.apply(
        *wrapper.params,
        *wrapper.lattice.params,
        kpts1,
        [wrapper, other1],
        IntorNameManager("int1e", shortname), options1)

def pbc_int2c2e(shortname: str, wrapper: LibcintWrapper,
                other: Optional[LibcintWrapper] = None,
                kpts: Optional[torch.Tensor] = None,  # (nkpts, ndim)
                options: Optional[PBCIntOption] = None,
                ):
    """
    Performing the periodic boundary condition (PBC) on 2-centre 2-electron
    integrals.

    Arguments
    ---------
    shortname: str
        The shortname of the integral (i.e. without the prefix `int2c2e_` or else)
    wrapper: LibcintWrapper
        The environment wrapper containing the basis
    other: Optional[LibcintWrapper]
        Another environment wrapper containing the basis. This environment
        must have the same complete environment as `wrapper` (e.g. `other` can be
        a subset of `wrapper`). If unspecified, then `other = wrapper`.
    kpts: Optional[torch.Tensor]
        k-points where the integration is supposed to be performed. If specified,
        it should have the shape of `(nkpts, ndim)`. Otherwise, it is assumed
        to be all zeros.
    options: Optional[PBCIntOption]
        The integration options. If unspecified, then just use the default
        value of `PBCIntOption`.

    Returns
    -------
    torch.Tensor
        A complex tensor representing the 1-electron integral with shape
        `(nkpts, *ncomp, nwrapper, nother)` where `ncomp` is the Cartesian
        components of the integral, e.g. `"ipr12"` integral will have 3
        components each for x, y, and z.
    """

    # check and set the default values
    other1 = check_and_set_pbc(wrapper, other)
    options1 = get_default_options(options)
    kpts1 = get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)

    assert isinstance(wrapper.lattice, Lattice)  # check if wrapper has a lattice
    return _PBCInt2cFunction.apply(
        *wrapper.params,
        *wrapper.lattice.params,
        kpts1,
        [wrapper, other1],
        IntorNameManager("int2c2e", shortname), options1)

def pbc_int3c2e(shortname: str, wrapper: LibcintWrapper,
                other1: Optional[LibcintWrapper] = None,
                other2: Optional[LibcintWrapper] = None,
                kpts_ij: Optional[torch.Tensor] = None,  # (nkpts, 2, ndim)
                options: Optional[PBCIntOption] = None,
                ):
    """
    Performing the periodic boundary condition (PBC) on 3-electron integrals.

    Arguments
    ---------
    shortname: str
        The shortname of the integral (i.e. without the prefix `int1e_` or else)
    wrapper: LibcintWrapper
        The environment wrapper containing the basis. `wrapper` and `other1`
        correspond to the first electron.
    other1: Optional[LibcintWrapper]
        Another environment wrapper containing the basis. This environment
        must have the same complete environment as `wrapper` (e.g. `other1` can be
        a subset of `wrapper`). If unspecified, then `other1 = wrapper`.
        `wrapper` and `other1` correspond to the first electron.
    other2: Optional[LibcintWrapper]
        Another environment wrapper containing the basis. This environment
        must have the same complete environment as `wrapper` (e.g. `other2` can be
        a subset of `wrapper`). If unspecified, then `other2 = wrapper`.
        `other2` corresponds to the second electron (usually density-fitted).
    kpts_ij: Optional[torch.Tensor]
        k-points where the integration is supposed to be performed. If specified,
        it should have the shape of `(nkpts_ij, 2, ndim)`. Otherwise, it is assumed
        to be all zeros.
    options: Optional[PBCIntOption]
        The integration options. If unspecified, then just use the default
        value of `PBCIntOption`.

    Returns
    -------
    torch.Tensor
        A complex tensor representing the 3-centre integral with shape
        `(nkpts_ij, *ncomp, nwrapper, nother1, nother2)` where `ncomp` is the Cartesian
        components of the integral, e.g. `"ipovlp"` integral will have 3
        components each for x, y, and z.

    Note
    ----
    The default 2-electron integrals are non-convergence for non-neutral,
    non-zero dipole and quadrupole `other2` environment, so make sure the
    `other2` environment is electrically neutral and free from dipole and
    quadrupole.
    """
    other1w = check_and_set_pbc(wrapper, other1)
    other2w = check_and_set_pbc(wrapper, other2)
    options1 = get_default_options(options)
    kpts_ij1 = get_default_kpts_ij(kpts_ij, dtype=wrapper.dtype, device=wrapper.device)

    # check if wrapper has a defined lattice
    assert isinstance(wrapper.lattice, Lattice)
    return _PBCInt3cFunction.apply(
        *wrapper.params,
        *wrapper.lattice.params,
        kpts_ij1,
        [wrapper, other1w, other2w],
        IntorNameManager("int3c2e", shortname), options1)

# shortcuts
def pbc_overlap(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
                kpts: Optional[torch.Tensor] = None,
                options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return pbc_int1e("ovlp", wrapper, other=other, kpts=kpts, options=options)

def pbc_kinetic(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
                kpts: Optional[torch.Tensor] = None,
                options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return pbc_int1e("kin", wrapper, other=other, kpts=kpts, options=options)

def pbc_coul2c(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
               kpts: Optional[torch.Tensor] = None,
               options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # 2-centre integral for electron repulsion
    return pbc_int2c2e("r12", wrapper, other=other, kpts=kpts, options=options)

def pbc_coul3c(wrapper: LibcintWrapper, other1: Optional[LibcintWrapper] = None,
               other2: Optional[LibcintWrapper] = None,
               kpts_ij: Optional[torch.Tensor] = None,
               options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return pbc_int3c2e("ar12", wrapper, other1=other1, other2=other2,
                       kpts_ij=kpts_ij, options=options)

################# torch autograd function wrappers #################
class _PBCInt2cFunction(torch.autograd.Function):
    # wrapper class for the periodic boundary condition 2-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                # basis params
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                # lattice params
                alattice: torch.Tensor,
                # other parameters
                kpts: torch.Tensor,
                wrappers: List[LibcintWrapper], int_nmgr: IntorNameManager,
                options: PBCIntOption) -> torch.Tensor:
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)

        out_tensor = PBCIntor(int_nmgr, wrappers, kpts, options).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, alattice, kpts)
        ctx.other_info = (wrappers, int_nmgr, options)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        raise NotImplementedError("gradients of PBC 2-centre integrals are not implemented")

class _PBCInt3cFunction(torch.autograd.Function):
    # wrapper class for the periodic boundary condition 3-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                # basis params
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                # lattice params
                alattice: torch.Tensor,
                # other parameters
                kpts_ij: torch.Tensor,
                wrappers: List[LibcintWrapper], int_nmgr: IntorNameManager,
                options: PBCIntOption) -> torch.Tensor:
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)
        # kpts_ij: (nkpts, 2, ndim)

        out_tensor = PBCIntor(int_nmgr, wrappers, kpts_ij, options).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, alattice, kpts_ij)
        ctx.other_info = (wrappers, int_nmgr, options)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        raise NotImplementedError("gradients of PBC 3-centre integrals are not implemented")

################# integrator object (direct interface to lib*) #################
class PBCIntor(object):
    def __init__(self, int_nmgr: IntorNameManager, wrappers: List[LibcintWrapper],
                 kpts_inp: torch.Tensor, options: PBCIntOption):
        # This is a class for once integration only
        # I made a class for refactoring reason because the integrals share
        # some parameters
        # No gradients propagated in the methods of this class

        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        kpts_inp_np = kpts_inp.detach().numpy()  # (nk, ndim)
        opname = int_nmgr.get_intgl_name(wrapper0.spherical)
        lattice = wrapper0.lattice
        assert isinstance(lattice, Lattice)

        # get the output's component shape
        comp_shape = int_nmgr.get_intgl_components_shape()
        ncomp = reduce(operator.mul, comp_shape, 1)

        # estimate the rcut and the lattice translation vectors
        coeffs, alphas, _ = wrapper0.params
        rcut = estimate_ovlp_rcut(options.precision, coeffs, alphas)
        ls = np.asarray(lattice.get_lattice_ls(rcut=rcut))

        self.int_type = int_nmgr.int_type
        self.wrappers = wrappers
        self.kpts_inp_np = kpts_inp_np
        self.opname = opname
        self.dtype = wrapper0.dtype
        self.device = wrapper0.device
        self.comp_shape = comp_shape
        self.ncomp = ncomp
        self.ls = ls
        self.options = options

        # this class is meant to be used once
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e" or self.int_type == "int2c2e":
            return self._int2c()
        elif self.int_type == "int3c2e":
            return self._int3c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> torch.Tensor:
        # 2-centre integral
        # this function works mostly in numpy
        # no gradients propagated in this function (and it's OK)
        # this function mostly replicate the `intor_cross` function in pyscf
        # https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/gto/cell.py
        # https://github.com/pyscf/pyscf/blob/f1321d5dd4fa103b5b04f10f31389c408949269d/pyscf/pbc/gto/cell.py#L345
        assert len(self.wrappers) == 2

        # libpbc will do in-place shift of the basis of one of the wrappers, so
        # we need to make a concatenated copy of the wrapper's atm_bas_env
        atm, bas, env, ao_loc = concat_atm_bas_env(self.wrappers[0], self.wrappers[1])
        i0, i1 = self.wrappers[0].shell_idxs
        j0, j1 = self.wrappers[1].shell_idxs
        nshls0 = len(self.wrappers[0].parent)
        shls_slice = (i0, i1, j0 + nshls0, j1 + nshls0)

        # prepare the output
        nkpts = len(self.kpts_inp_np)
        outshape = (nkpts,) + self.comp_shape + tuple(w.nao() for w in self.wrappers)
        out = np.empty(outshape, dtype=np.complex128)

        # TODO: add symmetry here
        fill = CPBC().PBCnr2c_fill_ks1
        fintor = getattr(CGTO(), self.opname)
        # TODO: use proper optimizers
        cintopt = _get_intgl_optimizer(self.opname, atm, bas, env)
        cpbcopt = c_null_ptr()

        # get the lattice translation vectors and the exponential factors
        expkl = np.asarray(np.exp(1j * np.dot(self.kpts_inp_np, self.ls.T)), order='C')

        # if the ls is too big, it might produce segfault
        if (self.ls.shape[0] > 1e6):
            warnings.warn("The number of neighbors in the integral is too many, "
                          "it might causes segfault")

        # perform the integration
        drv = CPBC().PBCnr2c_drv
        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(nkpts), int2ctypes(self.ncomp), int2ctypes(len(self.ls)),
            np2ctypes(self.ls),
            np2ctypes(expkl),
            (ctypes.c_int * len(shls_slice))(*shls_slice),
            np2ctypes(ao_loc),
            cintopt, cpbcopt,
            np2ctypes(atm), int2ctypes(atm.shape[0]),
            np2ctypes(bas), int2ctypes(bas.shape[0]),
            np2ctypes(env), int2ctypes(env.size))

        out_tensor = torch.as_tensor(out, dtype=get_complex_dtype(self.dtype),
                                     device=self.device)
        return out_tensor

    def _int3c(self) -> torch.Tensor:
        # 3-centre integral
        # this function works mostly in numpy
        # no gradients propagated in this function (and it's OK)
        # this function mostly replicate the `aux_e2` and `wrap_int3c` functions in pyscf
        # https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/df/incore.py
        # https://github.com/pyscf/pyscf/blob/f1321d5dd4fa103b5b04f10f31389c408949269d/pyscf/pbc/df/incore.py#L46
        assert len(self.wrappers) == 3

        # libpbc will do in-place shift of the basis of one of the wrappers, so
        # we need to make a concatenated copy of the wrapper's atm_bas_env
        atm, bas, env, ao_loc = concat_atm_bas_env(*self.wrappers)
        i0, i1 = self.wrappers[0].shell_idxs
        j0, j1 = self.wrappers[1].shell_idxs
        k0, k1 = self.wrappers[2].shell_idxs
        nshls0 = len(self.wrappers[0].parent)
        nshls01 = len(self.wrappers[1].parent) + nshls0
        shls_slice = (i0, i1, j0 + nshls0, j1 + nshls0, k0 + nshls01, k1 + nshls01)

        # kpts is actually kpts_ij in this function
        nkpts_ij = len(self.kpts_inp_np)
        outshape = (nkpts_ij,) + self.comp_shape + tuple(w.nao() for w in self.wrappers)
        out = np.empty(outshape, dtype=np.complex128)

        # get the unique k-points
        kpts_i = self.kpts_inp_np[:, 0, :]  # (nkpts, NDIM)
        kpts_j = self.kpts_inp_np[:, 1, :]
        kpts_stack = np.concatenate((kpts_i, kpts_j), axis=0)
        kpt_diff_tol = self.options.kpt_diff_tol
        _, kpts_idxs = np.unique(np.floor(kpts_stack / kpt_diff_tol) * kpt_diff_tol,
                                 axis=0, return_index=True)
        kpts = kpts_stack[kpts_idxs, :]
        nkpts = len(kpts)
        expkl = np.asarray(np.exp(1j * np.dot(kpts, self.ls.T)), order="C")

        # get the kpts_ij_idxs
        # TODO: check if it is the index inverse from unique
        wherei = np.where(np.abs(kpts_i.reshape(-1, 1, 3) - kpts).sum(axis=2) < kpt_diff_tol)[1]
        wherej = np.where(np.abs(kpts_j.reshape(-1, 1, 3) - kpts).sum(axis=2) < kpt_diff_tol)[1]
        kpts_ij_idxs = np.asarray(wherei * nkpts + wherej, dtype=np.int32)

        # prepare the optimizers
        # TODO: use proper optimizers
        # NOTE: using _get_intgl_optimizer in this case produce wrong results (I don't know why)
        cintopt = c_null_ptr()  # _get_intgl_optimizer(self.opname, atm, bas, env)
        cpbcopt = c_null_ptr()

        # do the integration
        drv = CPBC().PBCnr3c_drv
        fill = CPBC().PBCnr3c_fill_kks1  # TODO: optimize the kk-type and symmetry
        fintor = getattr(CPBC(), self.opname)
        drv(fintor, fill, np2ctypes(out),
            int2ctypes(nkpts_ij),
            int2ctypes(nkpts),
            int2ctypes(self.ncomp), int2ctypes(len(self.ls)),
            np2ctypes(self.ls),
            np2ctypes(expkl),
            np2ctypes(kpts_ij_idxs),
            (ctypes.c_int * len(shls_slice))(*shls_slice),
            np2ctypes(ao_loc),
            cintopt, cpbcopt,
            np2ctypes(atm), int2ctypes(atm.shape[0]),
            np2ctypes(bas), int2ctypes(bas.shape[0]),
            np2ctypes(env), int2ctypes(env.size))

        out_tensor = torch.as_tensor(out, dtype=get_complex_dtype(self.dtype),
                                     device=self.device)
        return out_tensor

################# helper functions #################
def check_and_set_pbc(wrapper: LibcintWrapper, other: Optional[LibcintWrapper]) -> LibcintWrapper:
    # check the `other` parameter if it is compatible to `wrapper`, then return
    # the `other` parameter (set to wrapper if it is `None`)
    other1 = _check_and_set(wrapper, other)
    assert other1.lattice is wrapper.lattice
    return other1

def get_default_options(options: Optional[PBCIntOption]) -> PBCIntOption:
    # if options is None, then set the default option.
    # otherwise, just return the input options
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    return options1

def get_default_kpts(kpts: Optional[torch.Tensor], dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
    # if kpts is None, then set the default kpts (k = zeros)
    # otherwise, just return the input kpts in the correct dtype and device
    if kpts is None:
        kpts1 = torch.zeros((1, NDIM), dtype=dtype, device=device)
    else:
        kpts1 = kpts.to(dtype).to(device)
        assert kpts1.ndim == 2
        assert kpts1.shape[-1] == NDIM
    return kpts1


def get_default_kpts_ij(kpts_ij: Optional[torch.Tensor], dtype: torch.dtype,
                         device: torch.device) -> torch.Tensor:
    # if kpts_ij is None, then set the default kpts_ij (k = zeros)
    # otherwise, just return the input kpts_ij in the correct dtype and device
    if kpts_ij is None:
        kpts1 = torch.zeros((1, 2, NDIM), dtype=dtype, device=device)
    else:
        kpts1 = kpts_ij.to(dtype).to(device)
        assert kpts1.ndim == 3
        assert kpts1.shape[-1] == NDIM
        assert kpts1.shape[-2] == 2
    return kpts1

def concat_atm_bas_env(*wrappers: LibcintWrapper) -> Tuple[np.ndarray, ...]:
    # make a copy of the concatenated atm, bas, env, and also return the new
    # ao_location
    assert len(wrappers) >= 2

    PTR_COORD = 1
    PTR_ZETA = 3
    ATOM_OF = 0
    PTR_EXP = 5
    PTR_COEFF = 6

    atm0, bas0, env0 = wrappers[0].atm_bas_env
    atms = [atm0]
    bass = [bas0]
    envs = [env0]
    ao_locs = [wrappers[0].full_shell_to_aoloc]

    offset = len(env0)
    natm_offset = len(atm0)
    for i in range(1, len(wrappers)):

        wrapper = wrappers[i]
        atm, bas, env = wrapper.atm_bas_env
        atm = np.copy(atm)
        bas = np.copy(bas)
        atm[:, PTR_COORD] += offset
        atm[:, PTR_ZETA] += offset
        bas[:, ATOM_OF] += natm_offset
        bas[:, PTR_EXP] += offset
        bas[:, PTR_COEFF] += offset

        ao_locs.append(wrapper.full_shell_to_aoloc[1:] + ao_locs[-1][-1])
        atms.append(atm)
        bass.append(bas)
        envs.append(env)

        offset += len(env)
        natm_offset += len(atm)

    all_atm = np.asarray(np.vstack(atms), dtype=np.int32)
    all_bas = np.asarray(np.vstack(bass), dtype=np.int32)
    all_env = np.hstack(envs)
    ao_loc = np.concatenate(ao_locs)

    return (all_atm, all_bas, all_env, ao_loc)

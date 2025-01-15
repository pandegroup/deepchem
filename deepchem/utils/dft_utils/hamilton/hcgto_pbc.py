from typing import List, Optional, Union, overload, Dict
import torch
import numpy as np

from deepchem.utils.cache_utils import Cache
from deepchem.utils.dft_utils.data.datastruct import AtomCGTOBasis, CGTOBasis, DensityFitInfo, ValGrad
from deepchem.utils.dft_utils.df.base_df import BaseDF
from deepchem.utils.dft_utils.df.dfpbc import DFPBC
from deepchem.utils.dft_utils.df.pbc_utils import get_gcut, unweighted_coul_ft
from deepchem.utils.dft_utils.hamilton.base_hamilton import BaseHamilton
from deepchem.utils.dft_utils.hamilton.hcgto import HamiltonCGTO
from deepchem.utils.dft_utils.hamilton.intor.gtoeval import pbc_eval_gradgto, pbc_eval_gto, pbc_eval_laplgto
from deepchem.utils.dft_utils.hamilton.intor.gtoft import eval_gto_ft
from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
from deepchem.utils.dft_utils.hamilton.intor.lcintwrap import LibcintWrapper
from deepchem.utils.dft_utils.hamilton.intor.pbcftintor import pbcft_overlap
from deepchem.utils.dft_utils.hamilton.intor.pbcintor import PBCIntOption, pbc_coul3c, pbc_kinetic, pbc_overlap
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils import BaseGrid, SpinParam
from deepchem.utils.differentiation_utils.linop import LinearOperator
from deepchem.utils.pytorch_utils import get_complex_dtype


class HamiltonCGTO_PBC(HamiltonCGTO):
    """
    Hamiltonian with contracted Gaussian type orbitals in a periodic boundary
    condition systems.
    The calculation of Hamiltonian components follow the reference:
    Sun, et al., J. Chem. Phys. 147, 164119 (2017)
    https://doi.org/10.1063/1.4998644
    """
    def __init__(self, atombases: List[AtomCGTOBasis],
                 latt: Lattice,
                 *,
                 kpts: Optional[torch.Tensor] = None,
                 wkpts: Optional[torch.Tensor] = None,  # weights of k-points to get the density
                 spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 lattsum_opt: Optional[Union[PBCIntOption, Dict]] = None,
                 cache: Optional[Cache] = None) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._lattice = latt
        # alpha for the compensating charge
        # TODO: calculate eta properly or put it in lattsum_opt
        self._eta = 0.2
        self._eta = 0.46213127322256375  # temporary to follow pyscf.df
        # lattice sum integral options
        self._lattsum_opt = PBCIntOption.get_default(lattsum_opt)

        self._basiswrapper = LibcintWrapper(
            atombases, spherical=spherical, lattice=latt)
        self.dtype = self._basiswrapper.dtype
        self.cdtype = get_complex_dtype(self.dtype)
        self.device = self._basiswrapper.device

        # set the default k-points and their weights
        self._kpts = kpts if kpts is not None else \
            torch.zeros((1, 3), dtype=self.dtype, device=self.device)
        nkpts = self._kpts.shape[0]
        # default weights are just 1/nkpts (nkpts,)
        self._wkpts = wkpts if wkpts is not None else \
            torch.ones((nkpts,), dtype=self.dtype, device=self.device) / nkpts

        assert self._wkpts.shape[0] == self._kpts.shape[0]
        assert self._wkpts.ndim == 1
        assert self._kpts.ndim == 2

        # initialize cache
        self._cache = cache if cache is not None else Cache.get_dummy()
        self._cache.add_cacheable_params(["overlap", "kinetic", "nuclattr"])

        if df is None:
            self._df: Optional[BaseDF] = None
        else:
            self._df = DFPBC(dfinfo=df, wrapper=self._basiswrapper, kpts=self._kpts,
                             wkpts=self._wkpts, eta=self._eta,
                             lattsum_opt=self._lattsum_opt,
                             cache=self._cache.add_prefix("df"))

        self._is_built = False

    @property
    def nao(self) -> int:
        return self._basiswrapper.nao()

    @property
    def kpts(self) -> torch.Tensor:
        return self._kpts

    @property
    def df(self) -> Optional[BaseDF]:
        return self._df

    ############ setups ############
    def build(self) -> BaseHamilton:
        if self._df is None:
            raise NotImplementedError(
                "Periodic boundary condition without density fitting is not implemented")
        assert isinstance(self._df, BaseDF)
        # (nkpts, nao, nao)
        with self._cache.open():

            # check the signature
            self._cache.check_signature({
                "atombases": self._atombases,
                "spherical": self._spherical,
                "lattice": self._lattice.lattice_vectors().detach(),
            })

            self._olp_mat = self._cache.cache(
                "overlap", lambda: pbc_overlap(self._basiswrapper, kpts=self._kpts,
                                                     options=self._lattsum_opt))
            self._kin_mat = self._cache.cache(
                "kinetic", lambda: pbc_kinetic(self._basiswrapper, kpts=self._kpts,
                                                     options=self._lattsum_opt))
            self._nucl_mat = self._cache.cache("nuclattr", self._calc_nucl_attr)

        self._kinnucl_mat = self._kin_mat + self._nucl_mat
        self._df.build()
        self._is_built = True
        return self

    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        # save the family and save the xc
        self.xc = xc
        if xc is None:
            self.xcfamily = 1
        else:
            self.xcfamily = xc.family

        # save the grid
        self.grid = grid
        self.rgrid = grid.get_rgrid()
        assert grid.coord_type == "cart"

        # setup the basis as a spatial function
        self.is_ao_set = True
        self.basis = pbc_eval_gto(  # (nkpts, nao, ngrid)
            self._basiswrapper, self.rgrid, kpts=self._kpts, options=self._lattsum_opt)
        basis_dvolume = self.basis * self.grid.get_dvolume()  # (nkpts, nao, ngrid)
        self.basis_dvolume_conj = basis_dvolume.conj()

        if self.xcfamily == 1:  # LDA
            return

        # setup the gradient of the basis
        self.is_grad_ao_set = True
        self.grad_basis = pbc_eval_gradgto(  # (ndim, nkpts, nao, ngrid)
            self._basiswrapper, self.rgrid, kpts=self._kpts, options=self._lattsum_opt)
        if self.xcfamily == 2:  # GGA
            return

        # setup the laplacian of the basis
        self.is_lapl_ao_set = True
        self.lapl_basis = pbc_eval_laplgto(  # (nkpts, nao, ngrid)
            self._basiswrapper, self.rgrid, kpts=self._kpts, options=self._lattsum_opt)

    ############ fock matrix components ############
    def get_nuclattr(self) -> LinearOperator:
        # return: (nkpts, nao, nao)
        return LinearOperator.m(self._nucl_mat, is_hermitian=True)

    def get_kinnucl(self) -> LinearOperator:
        # kinnucl_mat: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        return LinearOperator.m(self._kinnucl_mat, is_hermitian=True)

    def get_overlap(self) -> LinearOperator:
        # olp_mat: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        return LinearOperator.m(self._olp_mat, is_hermitian=True)

    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        # dm: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        assert self._df is not None
        return self._df.get_elrep(dm)

    @overload
    def get_exchange(self, dm: torch.Tensor) -> LinearOperator:
        ...

    @overload
    def get_exchange(self, dm: SpinParam[torch.Tensor]) -> SpinParam[LinearOperator]:
        ...

    def get_exchange(self, dm):
        msg = "Exact exchange for periodic boundary conditions has not been implemented"
        raise NotImplementedError(msg)

    def get_vext(self, vext: torch.Tensor) -> LinearOperator:
        # vext: (*BR, ngrid)
        # return: (*BR, nkpts, nao, nao)
        if not self.is_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, xc)` to call this function")
        mat = torch.einsum("...r,kbr,kcr->...kbc", vext, self.basis_dvolume_conj, self.basis)  # (*BR, nao, nao)
        mat = (mat + mat.transpose(-2, -1).conj()) * 0.5  # ensure the hermitianness and reduce numerical instability
        return LinearOperator.m(mat, is_hermitian=True)

    @overload
    def get_vxc(self, dm: SpinParam[torch.Tensor]) -> SpinParam[LinearOperator]:
        ...

    @overload
    def get_vxc(self, dm: torch.Tensor) -> LinearOperator:
        ...

    def get_vxc(self, dm):
        # dm: (*BD, nao, nao)
        return super().get_vxc(dm)

    ############### interface to dm ###############
    def ao_orb2dm(self, orb: torch.Tensor, orb_weight: torch.Tensor) -> torch.Tensor:
        # convert the atomic orbital to the density matrix

        # orb: (nkpts, nao, norb)
        # orb_weight: (norb)
        # return: (nkpts, nao, nao)
        dtype = orb.dtype
        res = torch.einsum("kao,o,kbo->kab", orb, orb_weight.to(dtype), orb.conj())
        return res

    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (*BR, ndim)
        # dm: (*BD, nkpts, nao, nao)
        # returns: (*BRD)

        nao = dm.shape[-1]
        nkpts = self._kpts.shape[0]
        xyzshape = xyz.shape  # (*BR, ndim)

        # basis: (nkpts, nao, *BR)
        xyz1 = xyz.reshape(-1, xyzshape[-1])  # (BR=ngrid, ndim)
        # ao1: (nkpts, nao, ngrid)
        ao1 = pbc_eval_gto(self._basiswrapper, xyz1, kpts=self._kpts, options=self._lattsum_opt)
        ao1 = torch.movedim(ao1, -1, 0).reshape(*xyzshape[:-1], nkpts, nao)  # (*BR, nkpts, nao)

        # dens = torch.einsum("...ka,...kb,...kab,k->...", ao1, ao1.conj(), dm, self._wkpts)
        densk = torch.matmul(dm, ao1.conj().unsqueeze(-1))  # (*BRD, nkpts, nao, 1)
        densk = torch.matmul(ao1.unsqueeze(-2), densk).squeeze(-1).squeeze(-1)  # (*BRD, nkpts)
        assert densk.imag.abs().max() < 1e-9, "The density should be real at this point"

        dens = torch.einsum("...k,k->...", densk.real, self._wkpts)  # (*BRD)
        return dens

    ############### energy of the Hamiltonian ###############
    def get_e_hcore(self, dm: torch.Tensor) -> torch.Tensor:
        # get the energy from one electron operator
        return torch.einsum("...kij,...kji,k->...", self._kinnucl_mat, dm, self._wkpts)

    def get_e_elrep(self, dm: torch.Tensor) -> torch.Tensor:
        # get the energy from two electron repulsion operator
        elrep_mat = self.get_elrep(dm).fullmatrix()
        return 0.5 * torch.einsum("...kij,...kji,k->...", elrep_mat, dm, self._wkpts)

    def get_e_exchange(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # get the energy from two electron exchange operator
        exc_mat = self.get_exchange(dm)
        ene = SpinParam.apply_fcn(
            lambda exc_mat, dm:
                0.5 * torch.einsum("...kij,...kji,k->...", exc_mat.fullmatrix(), dm, self._wkpts),
            exc_mat, dm)
        enetot = SpinParam.sum(ene)
        return enetot

    def get_e_xc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        return super().get_e_xc(dm)

    ################ xc-related ################
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        # getparamnames to list the name of parameters affecting the method
        if methodname == "get_kinnucl":
            return [prefix + "_kinnucl_mat"]
        elif methodname == "get_nuclattr":
            return [prefix + "_nucl_mat"]
        elif methodname == "get_overlap":
            return [prefix + "_olp_mat"]
        elif methodname == "get_elrep":
            assert self._df is not None
            return self._df.getparamnames("get_elrep", prefix=prefix + "_df.")
        elif methodname == "ao_orb2dm":
            return []
        elif methodname == "get_vext":
            return [prefix + "basis_dvolume_conj", prefix + "basis"]
        elif methodname == "get_grad_vext":
            return [prefix + "basis_dvolume_conj", prefix + "grad_basis"]
        elif methodname == "get_lapl_kin_vext":
            raise NotImplementedError()
        elif methodname == "get_vxc":
            return super().getparamnames("get_vxc", prefix=prefix)
        elif methodname == "_get_vxc_from_potinfo":
            params = [prefix + "basis", prefix + "basis_dvolume_conj"]
            if self.xcfamily in [2, 4]:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis", prefix + "dvolume"]
            return params
        elif methodname == "_get_dens_at_grid":
            return [prefix + "basis"]
        elif methodname == "_get_grad_dens_at_grid":
            return [prefix + "basis", prefix + "grad_basis"]
        elif methodname == "_get_lapl_dens_at_grid":
            return [prefix + "basis", prefix + "lapl_basis"]
        elif methodname == "_dm2densinfo":
            params = [prefix + "basis"]
            if self.xcfamily == 2 or self.xcfamily == 4:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis"]
            return params
        else:
            return super().getparamnames(methodname, prefix=prefix)

    ################ private methods ################
    def _calc_nucl_attr(self) -> torch.Tensor:
        # calculate the nuclear attraction matrix
        # this follows the equation (31) in Sun, et al., J. Chem. Phys. 147 (2017)

        # construct the fake nuclei atombases for nuclei
        # (in this case, we assume each nucleus is a very sharp s-type orbital)
        nucl_atbases = self._create_fake_nucl_bases(alpha=1e16, chargemult=1)
        # add a compensating charge
        cnucl_atbases = self._create_fake_nucl_bases(alpha=self._eta, chargemult=-1)
        # real charge + compensating charge
        nucl_atbases_all = nucl_atbases + cnucl_atbases
        nucl_wrapper = LibcintWrapper(
            nucl_atbases_all, spherical=self._spherical, lattice=self._lattice)
        cnucl_wrapper = LibcintWrapper(
            cnucl_atbases, spherical=self._spherical, lattice=self._lattice)
        natoms = nucl_wrapper.nao() // 2

        # construct the k-points ij
        # duplicating kpts to have shape of (nkpts, 2, ndim)
        kpts_ij = self._kpts.unsqueeze(-2) * torch.ones((2, 1), dtype=self.dtype, device=self.device)

        ############# 1st part of nuclear attraction: short range #############
        # get the 1st part of the nuclear attraction: the charge and compensating charge
        # nuc1: (nkpts, nao, nao, 2 * natoms)
        # nuc1 is not hermitian
        basiswrapper1, nucl_wrapper1 = LibcintWrapper.concatenate(self._basiswrapper, nucl_wrapper)
        nuc1_c = pbc_coul3c(basiswrapper1, other1=basiswrapper1,
                                  other2=nucl_wrapper1, kpts_ij=kpts_ij,
                                  options=self._lattsum_opt)
        nuc1 = -nuc1_c[..., :natoms] + nuc1_c[..., natoms:]
        nuc1 = torch.sum(nuc1, dim=-1)  # (nkpts, nao, nao)

        # add vbar for 3 dimensional cell
        # vbar is the interaction between the background charge and the
        # compensating function.
        # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/df/aft.py#L239
        nucbar = sum([-atb.atomz / self._eta for atb in self._atombases])
        nuc1_b = -nucbar * np.pi / self._lattice.volume() * self._olp_mat
        nuc1 = nuc1 + nuc1_b

        ############# 2nd part of nuclear attraction: long range #############
        # get the 2nd part from the Fourier Transform
        # get the G-points, choosing min because the two FTs are multiplied
        gcut = get_gcut(self._lattsum_opt.precision,
                        wrappers=[cnucl_wrapper, self._basiswrapper],
                        reduce="min")
        # gvgrids: (ngv, ndim), gvweights: (ngv,)
        gvgrids, gvweights = self._lattice.get_gvgrids(gcut)

        # the compensating charge's Fourier Transform
        # TODO: split gvgrids and gvweights to reduce the memory usage
        cnucl_ft = eval_gto_ft(cnucl_wrapper, gvgrids)  # (natoms, ngv)
        # overlap integral of the electron basis' Fourier Transform
        cbas_ft = pbcft_overlap(
            self._basiswrapper, gvgrid=-gvgrids, kpts=self._kpts,
            options=self._lattsum_opt)  # (nkpts, nao, nao, ngv)
        # coulomb kernel Fourier Transform
        coul_ft = unweighted_coul_ft(gvgrids) * gvweights  # (ngv,)
        coul_ft = coul_ft.to(cbas_ft.dtype)  # cast to complex

        # optimized by opt_einsum
        # nuc2 = -torch.einsum("tg,kabg,g->kab", cnucl_ft, cbas_ft, coul_ft)
        nuc2_temp = torch.einsum("g,tg->g", coul_ft, cnucl_ft)
        nuc2 = -torch.einsum("g,kabg->kab", nuc2_temp, cbas_ft)  # (nkpts, nao, nao)
        # print((nuc2 - nuc2.conj().transpose(-2, -1)).abs().max())  # check hermitian-ness

        # get the total contribution from the short range and long range
        nuc = nuc1 + nuc2

        # symmetrize for more stable numerical calculation
        nuc = (nuc + nuc.conj().transpose(-2, -1)) * 0.5
        return nuc

    def _create_fake_nucl_bases(self, alpha: float, chargemult: int) -> List[AtomCGTOBasis]:
        # create a list of basis (of s-type) at every nuclei positions
        res: List[AtomCGTOBasis] = []
        alphas = torch.tensor([alpha], dtype=self.dtype, device=self.device)
        # normalizing so the integral of the cgto is 1
        # 0.5 / np.sqrt(np.pi) * 2 / scipy.special.gamma(1.5) * alphas ** 1.5
        norm_coeff = 0.6366197723675814 * alphas ** 1.5
        for atb in self._atombases:
            # put the charge in the coefficients
            coeffs = atb.atomz * norm_coeff
            basis = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs, normalized=True)
            res.append(AtomCGTOBasis(atomz=0, bases=[basis], pos=atb.pos))
        return res

    def _get_vxc_from_potinfo(self, potinfo: ValGrad) -> LinearOperator:
        # overloading from hcgto

        vext = potinfo.value
        vb = potinfo.value * self.basis

        if self.xcfamily in [2, 4]:  # GGA or MGGA
            assert potinfo.grad is not None  # (..., ndim, nrgrid)
            vgrad = potinfo.grad * 2
            vb += torch.einsum("...r,kar->...kar", vgrad[..., 0, :], self.grad_basis[0])
            vb += torch.einsum("...r,kar->...kar", vgrad[..., 1, :], self.grad_basis[1])
            vb += torch.einsum("...r,kar->...kar", vgrad[..., 2, :], self.grad_basis[2])
        if self.xcfamily == 4:  # MGGA
            assert potinfo.lapl is not None  # (..., nrgrid)
            assert potinfo.kin is not None
            vb += 2 * potinfo.lapl.unsqueeze(-2).unsqueeze(-2) * self.lapl_basis

        # calculating the matrix from multiplication with the basis
        mat = torch.matmul(vb, self.basis_dvolume_conj.transpose(-2, -1))

        if self.xcfamily == 4:  # MGGA
            assert potinfo.lapl is not None  # (..., nrgrid)
            assert potinfo.kin is not None
            lapl_kin_dvol = (2 * potinfo.lapl + 0.5 * potinfo.kin) * self.dvolume
            mat += torch.einsum("...r,kbr,kcr->...kbc", lapl_kin_dvol, self.grad_basis[0], self.grad_basis[0])
            mat += torch.einsum("...r,kbr,kcr->...kbc", lapl_kin_dvol, self.grad_basis[1], self.grad_basis[1])
            mat += torch.einsum("...r,kbr,kcr->...kbc", lapl_kin_dvol, self.grad_basis[2], self.grad_basis[2])

        mat = (mat + mat.transpose(-2, -1).conj()) * 0.5
        return LinearOperator.m(mat, is_hermitian=True)

    def _dm2densinfo(self, dm: torch.Tensor) -> ValGrad:
        # overloading from hcgto
        # dm: (*BD, nkpts, nao, nao), Hermitian
        # family: 1 for LDA, 2 for GGA, 3 for MGGA
        # self.basis: (nkpts, nao, ngrid)

        # dm @ ao will be used in every case
        dmdmh = (dm + dm.transpose(-2, -1).conj()) * 0.5  # (*BD, nao, nao)
        dmao = torch.matmul(dmdmh, self.basis.conj())  # (*BD, nao, nr)
        dmao2 = 2 * dmao

        # calculate the density
        dens = torch.einsum("...kir,kir->...r", dmao, self.basis)

        # calculate the density gradient
        gdens: Optional[torch.Tensor] = None
        if self.xcfamily == 2 or self.xcfamily == 4:  # GGA or MGGA
            if not self.is_grad_ao_set:
                msg = "Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient"
                raise RuntimeError(msg)

            gdens = torch.zeros((*dm.shape[:-3], 3, self.basis.shape[-1]),
                                dtype=dm.dtype, device=dm.device)  # (..., ndim, ngrid)
            gdens[..., 0, :] = torch.einsum("...kir,kir->...r", dmao2, self.grad_basis[0])
            gdens[..., 1, :] = torch.einsum("...kir,kir->...r", dmao2, self.grad_basis[1])
            gdens[..., 2, :] = torch.einsum("...kir,kir->...r", dmao2, self.grad_basis[2])

        lapldens: Optional[torch.Tensor] = None
        kindens: Optional[torch.Tensor] = None
        # if self.xcfamily == 4:  # TODO: to be completed
        #     # calculate the laplacian of the density and kinetic energy density at the grid
        #     if not self.is_lapl_ao_set:
        #         msg = "Please call `setup_grid(grid, gradlevel>=2)` to calculate the density gradient"
        #         raise RuntimeError(msg)
        #     lapl_basis = torch.einsum("...kir,kir->...r", dmao2, self.lapl_basis)
        #     grad_grad = torch.einsum("...kij,kir,kjr->...r", dmdmt, self.grad_basis[0], self.grad_basis[0].conj())
        #     grad_grad += torch.einsum("...kij,kir,kjr->...r", dmdmt, self.grad_basis[1], self.grad_basis[1].conj())
        #     grad_grad += torch.einsum("...kij,kir,kjr->...r", dmdmt, self.grad_basis[2], self.grad_basis[2].conj())
        #     lapldens = lapl_basis + 2 * grad_grad
        #     kindens = grad_grad * 0.5

        # dens: (*BD, ngrid)
        # gdens: (*BD, ndim, ngrid)
        res = ValGrad(value=dens, grad=gdens, lapl=lapldens, kin=kindens)
        return res

    def _get_dens_at_grid(self, dm: torch.Tensor) -> torch.Tensor:
        # get the density at the grid
        return torch.einsum("...kij,kir,kjr->...r", dm, self.basis, self.basis.conj())

    def _get_grad_dens_at_grid(self, dm: torch.Tensor) -> torch.Tensor:
        # get the gradient of density at the grid
        if not self.is_grad_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient")
        # gdens = torch.einsum("...kij,dkir,kjr->...dr", dm, self.grad_basis, self.basis.conj())
        gdens = torch.zeros((*dm.shape[:-3], 3, self.basis.shape[-1]), device=self.device,
                            dtype=self.cdtype)
        basis_conj = self.basis.conj()
        gdens[..., 0, :] = torch.einsum("...kij,kir,kjr->...r", dm, self.grad_basis[0], basis_conj)
        gdens[..., 1, :] = torch.einsum("...kij,kir,kjr->...r", dm, self.grad_basis[1], basis_conj)
        gdens[..., 2, :] = torch.einsum("...kij,kir,kjr->...r", dm, self.grad_basis[2], basis_conj)
        return gdens + gdens.conj()  # + complex conjugate
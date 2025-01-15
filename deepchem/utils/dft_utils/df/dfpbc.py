from typing import List, Optional
import numpy as np
import torch
from deepchem.utils.dft_utils.hamilton.intor.gtoft import eval_gto_ft
from deepchem.utils.dft_utils.hamilton.intor.pbcftintor import pbcft_overlap
from deepchem.utils.dft_utils.hamilton.intor.pbcintor import pbc_coul2c, pbc_coul3c, pbc_overlap
from deepchem.utils.differentiation_utils import LinearOperator
from deepchem.utils.dft_utils import BaseDF, CGTOBasis, AtomCGTOBasis, DensityFitInfo, \
    LibcintWrapper, PBCIntOption
from deepchem.utils import get_complex_dtype, gaussian_integral
from deepchem.utils.cache_utils import Cache
from deepchem.utils.dft_utils.df.pbc_utils import unweighted_coul_ft, get_gcut


class DFPBC(BaseDF):
    """
    DFPBC represents a class for gaussian density fitting of systems with
    periodic boundary condition.
    """

    def __init__(self,
                 dfinfo: DensityFitInfo,
                 wrapper: LibcintWrapper,
                 kpts: torch.Tensor,
                 wkpts: torch.Tensor,
                 eta: float,
                 lattsum_opt: PBCIntOption,
                 *,
                 cache: Optional[Cache] = None):
        """Initialize the DFPBC class.

        Parameters
        ----------
        dfinfo: DensityFitInfo
            Density fitting information.
        wrapper: LibcintWrapper
            Wrapper containing the basis information.
        kpts: torch.Tensor
            K-points of the system.
        wkpts: torch.Tensor
            Weights of each k-points.
        eta: float
            Eta value.
        lattsum_opt: PBCIntOption
            Lattice sum option.
        cache: Optional[Cache], optional
            Cache object, by default None.
        """
        self._dfinfo = dfinfo
        self._wrapper = wrapper
        self._eta = eta
        self._kpts = kpts
        self._wkpts = wkpts  # weights of each k-points
        self._lattsum_opt = lattsum_opt
        self.dtype = wrapper.dtype
        self.device = wrapper.device
        assert wrapper.lattice is not None
        self._lattice = wrapper.lattice
        self._is_built = False

        # set up cache
        self._cache = cache if cache is not None else Cache.get_dummy()
        self._cache.add_cacheable_params(["j2c", "j3c", "el_mat"])

    def build(self) -> BaseDF:
        """Build the DFPBC class.

        Returns
        -------
        BaseDF
            DFPBC class.
        """
        self._is_built = True
        df = self._dfinfo

        # calculate the matrices required to calculate the electron repulsion operator
        # i.e. the 3-centre 2-electron integrals (short + long range) and j3c @ (j2c^-1)
        method = df.method.lower()
        df_auxbases = _renormalize_auxbases(df.auxbases)
        aux_comp_bases = self._create_compensating_bases(df_auxbases,
                                                         eta=self._eta)
        fuse_aux_bases = df_auxbases + aux_comp_bases
        fuse_aux_wrapper = LibcintWrapper(fuse_aux_bases,
                                          spherical=self._wrapper.spherical,
                                          lattice=self._lattice)
        aux_comp_wrapper = LibcintWrapper(aux_comp_bases,
                                          spherical=self._wrapper.spherical,
                                          lattice=self._lattice)
        aux_wrapper = LibcintWrapper(df_auxbases,
                                     spherical=self._wrapper.spherical,
                                     lattice=self._lattice)
        nxcao = aux_comp_wrapper.nao(
        )  # number of aux compensating basis wrapper
        nxao = fuse_aux_wrapper.nao() - nxcao  # number of aux basis wrapper
        assert nxcao == nxao

        # only gaussian density fitting is implemented at the moment
        if method != "gdf":
            raise NotImplementedError(
                "Density fitting %s is not implemented (only gdf)" % df.method)

        # get the k-points needed for the integrations
        nkpts = self._kpts.shape[0]
        kpts_ij = _combine_kpts_to_kpts_ij(self._kpts)  # (nkpts_ij, 2, ndim)
        kpts_reduce = _reduce_kpts_ij(kpts_ij)  # (nkpts_ij, ndim)
        nkpts_ij = kpts_ij.shape[0]
        kpts_j = kpts_ij[..., 1, :]  # (nkpts_ij, ndim)

        def _calc_integrals():
            """Calculate the integrals.

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                Tuple of 3 tensors.

            """
            # short-range integrals
            # 3-centre 2-electron integral
            _basisw, _fusew = LibcintWrapper.concatenate(
                self._wrapper, fuse_aux_wrapper)
            # (nkpts_ij, nao, nao, nxao+nxcao)
            j3c_short_f = pbc_coul3c(_basisw,
                                     other2=_fusew,
                                     kpts_ij=kpts_ij,
                                     options=self._lattsum_opt)
            j3c_short = j3c_short_f[..., :nxao] - j3c_short_f[
                ..., nxao:]  # (nkpts_ij, nao, nao, nxao)

            # 2-centre 2-electron integrals
            # (nkpts_unique, nxao+nxcao, nxao+nxcao)
            j2c_short_f = pbc_coul2c(fuse_aux_wrapper,
                                     kpts=kpts_reduce,
                                     options=self._lattsum_opt)
            # j2c_short: (nkpts_unique, nxao, nxao)
            j2c_short = j2c_short_f[..., :nxao, :nxao] + j2c_short_f[..., nxao:, nxao:] \
                - j2c_short_f[..., :nxao, nxao:] - j2c_short_f[..., nxao:, :nxao]

            # long-range integrals
            # only use the compensating wrapper as the gcut
            gcut = get_gcut(self._lattsum_opt.precision, [aux_comp_wrapper])
            # gvgrids: (ngv, ndim), gvweights: (ngv,)
            gvgrids, gvweights = self._lattice.get_gvgrids(gcut)
            ngv = gvgrids.shape[0]
            gvk = gvgrids.unsqueeze(-2) + kpts_reduce  # (ngv, nkpts_ij, ndim)
            gvk = gvk.view(-1, gvk.shape[-1])  # (ngv * nkpts_ij, ndim)

            # get the fourier transform variables
            # TODO: iterate over ngv axis
            # ft of the compensating basis
            comp_ft = eval_gto_ft(aux_comp_wrapper,
                                  gvk)  # (nxcao, ngv * nkpts_ij)
            comp_ft = comp_ft.view(-1, ngv, nkpts_ij)  # (nxcao, ngv, nkpts_ij)
            # ft of the auxiliary basis
            auxb_ft_c = eval_gto_ft(aux_wrapper, gvk)  # (nxao, ngv * nkpts_ij)
            auxb_ft_c = auxb_ft_c.view(-1, ngv,
                                       nkpts_ij)  # (nxao, ngv, nkpts_ij)
            auxb_ft = auxb_ft_c - comp_ft  # (nxao, ngv, nkpts_ij)
            # ft of the overlap integral of the basis (nkpts_ij, nao, nao, ngv)
            aoao_ft = self._get_pbc_overlap_with_kpts_ij(
                gvgrids, kpts_reduce, kpts_j)
            # ft of the coulomb kernel
            coul_ft = unweighted_coul_ft(gvk)  # (ngv * nkpts_ij,)
            coul_ft = coul_ft.to(comp_ft.dtype).view(
                ngv, nkpts_ij) * gvweights.unsqueeze(-1)  # (ngv, nkpts_ij)

            # 1: (nkpts_ij, nxao, nxao)
            pattern = "gi,xgi,ygi->ixy"
            j2c_long = torch.einsum(pattern, coul_ft, comp_ft.conj(), auxb_ft)
            # 2: (nkpts_ij, nxao, nxao)
            j2c_long += torch.einsum(pattern, coul_ft, auxb_ft.conj(), comp_ft)
            # 3: (nkpts_ij, nxao, nxao)
            j2c_long += torch.einsum(pattern, coul_ft, comp_ft.conj(), comp_ft)

            # calculate the j3c long-range
            patternj3 = "gi,xgi,iyzg->iyzx"
            # (nkpts_ij, nao, nao, nxao)
            j3c_long = torch.einsum(patternj3, coul_ft, comp_ft.conj(), aoao_ft)

            # get the average potential
            auxbar_f = self._auxbar(
                kpts_reduce, fuse_aux_wrapper)  # (nkpts_ij, nxao + nxcao)
            auxbar = auxbar_f[:, :nxao] - auxbar_f[:, nxao:]  # (nkpts_ij, nxao)
            auxbar = auxbar.reshape(nkpts, nkpts,
                                    auxbar.shape[-1])  # (nkpts, nkpts, nxao)
            olp_mat = pbc_overlap(
                self._wrapper, kpts=self._kpts,
                options=self._lattsum_opt)  # (nkpts, nao, nao)
            j3c_bar = auxbar[:, :, None, None, :] * olp_mat[
                ..., None]  # (nkpts, nkpts, nao, nao, nxao)
            j3c_bar = j3c_bar.reshape(
                -1, *j3c_bar.shape[2:])  # (nkpts_ij, nao, nao, nxao)

            # combining integrals
            j2c = j2c_short + j2c_long  # (nkpts_ij, nxao, nxao)
            j3c = j3c_short + j3c_long - j3c_bar  # (nkpts_ij, nao, nao, nxao)
            el_mat = torch.einsum("kxy,kaby->kabx", torch.inverse(j2c),
                                  j3c)  # (nkpts_ij, nao, nao, nxao)
            return j2c, j3c, el_mat

        with self._cache.open():

            # check the signature
            self._cache.check_signature({
                "dfinfo": self._dfinfo,
                "kpts": self._kpts.detach(),
                "wkpts": self._wkpts.detach(),
                "atombases": self._wrapper.atombases,
                "alattice": self._lattice.lattice_vectors().detach(),
            })

            j2c, j3c, el_mat = self._cache.cache_multi(["j2c", "j3c", "el_mat"],
                                                       _calc_integrals)

        self._j2c = j2c
        self._j3c = j3c
        self._el_mat = el_mat

        return self

    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """Return the electron repulsion operator given the density matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix.

        Returns
        -------
        LinearOperator
            Electron repulsion operator.
        """
        # return the electron repulsion operator given the density matrix
        # dm: (nkpts, nao, nao)
        # self._el_mat: (nkpts_ij, nao, nao, nxao)
        # self._j3c: (nkpts_ij, nao, nao, nxao)
        # return: (nkpts, nao, nao)
        nkpts = dm.shape[-3]
        el_mat = self._el_mat.view(
            nkpts, nkpts,
            *self._el_mat.shape[1:])  # (nkpts, nkpts, nao, nao, nxao)
        j3c = self._j3c.view(
            nkpts, nkpts,
            *self._j3c.shape[1:])  # (nkpts, nkpts, nao, nao, nxao)
        fitcoeffs = torch.einsum("llabx,lab,l->x", el_mat, dm,
                                 self._wkpts.to(dm.dtype))  # (nxao,)
        elrep_mat = torch.einsum("x,llabx->lab", fitcoeffs,
                                 j3c.conj())  # (nkpts, nao, nao)

        # check hermitianness
        # assert torch.allclose(elrep_mat, elrep_mat.conj().transpose(-2, -1))
        elrep_mat = (elrep_mat + elrep_mat.conj().transpose(-2, -1)) * 0.5

        return LinearOperator.m(elrep_mat, is_hermitian=True)

    @property
    def j2c(self) -> torch.Tensor:
        """Return the 2-centre 2-electron integrals.

        Returns
        -------
        torch.Tensor
            2-centre 2-electron integrals.
        """
        return self._j2c

    @property
    def j3c(self) -> torch.Tensor:
        """Return the 3-centre 2-electron integrals.

        Returns
        -------
        torch.Tensor
            3-centre 2-electron integrals.
        """
        return self._j3c

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Return the parameter names.

        Parameters
        ----------
        methodname: str
            Method name.
        prefix: str, optional
            Prefix, by default

        Returns
        -------
        List[str]
            List of parameter names.

        Raises
        ------
        KeyError
            If the method name is not found.

        """
        if methodname == "get_elrep":
            return [prefix + "_el_mat", prefix + "_j3c"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)

    # private functions
    def _create_compensating_bases(self, atombases: List[AtomCGTOBasis],
                                   eta: float) -> List[AtomCGTOBasis]:
        """create the list of atom bases containing the compensating basis with
        given `eta` as the exponentials, see make_modchg_basis in
        https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/df/df.py#L116

        Parameters
        ----------
        atombases: List[AtomCGTOBasis]
            List of atom bases.
        eta: float
            Eta value.

        Returns
        -------
        List[AtomCGTOBasis]
            List of atom bases containing the compensating basis
            with given `eta` as the exponentials.

        """

        # pre-calculate the norms up to angmom 6
        half_sph_norm = 0.5 / np.sqrt(np.pi)
        norms = [
            half_sph_norm / gaussian_integral(2 * angmom + 2, eta)
            for angmom in range(7)
        ]
        norms_t = [
            torch.tensor([nrm], dtype=self.dtype, device=self.device)
            for nrm in norms
        ]

        res: List[AtomCGTOBasis] = []
        alphas = torch.tensor([eta], dtype=self.dtype, device=self.device)
        for atb in atombases:
            # TODO: use reduced bases to optimize the integration time
            # angmoms = set(bas.angmom for bas in atb.bases)
            # bases = [
            #     CGTOBasis(angmom=angmom, alphas=alphas, coeffs=norms[angmom], normalized=True) \
            #     for angmom in angmoms
            # ]
            bases: List[CGTOBasis] = []
            for bas in atb.bases:
                # calculate the integral of the basis
                int1 = gaussian_integral(bas.angmom * 2 + 2, bas.alphas)
                s = torch.sum(bas.coeffs * int1) / half_sph_norm

                # set the coefficients of the compensating basis to have
                # the same integral
                coeffs = s * norms_t[bas.angmom]
                b2 = CGTOBasis(angmom=bas.angmom,
                               alphas=alphas,
                               coeffs=coeffs,
                               normalized=True)
                bases.append(b2)
            res.append(AtomCGTOBasis(atomz=0, bases=bases, pos=atb.pos))
        return res

    def _get_pbc_overlap_with_kpts_ij(self, gvgrids: torch.Tensor,
                                      kpts_reduce: torch.Tensor,
                                      kpts_j: torch.Tensor) -> torch.Tensor:
        """Compute the pbcft overlap integral for the basis by filling in
        the k-points one by one.
        this is to compute eq. (16) on Sun et al., https://doi.org/10.1063/1.4998644

        Parameters
        ----------
        gvgrids: torch.Tensor
            Grid vectors.
        kpts_reduce: torch.Tensor
            Reduced k-points.
        kpts_j: torch.Tensor
            K-points.

        Returns
        -------
        torch.Tensor
            PBCFT overlap integral for the basis.

        """
        nkpts_ij = kpts_reduce.shape[0]
        nao = self._wrapper.nao()
        ngv = gvgrids.shape[0]
        dctype = get_complex_dtype(self.dtype)
        res = torch.empty((nkpts_ij, nao, nao, ngv),
                          dtype=dctype,
                          device=self.device)
        for i in range(nkpts_ij):
            kpt_ij = kpts_reduce[i]  # (ndim,)
            kpt_j = kpts_j[i:i + 1]  # (1, ndim)
            gvk = gvgrids + kpt_ij  # (ngv, ndim)
            aoao_ft_i = pbcft_overlap(
                self._wrapper,
                gvgrid=gvk,
                kpts=kpt_j,
                options=self._lattsum_opt)  # (1, nao, nao, ngv)
            res[i] = aoao_ft_i[0]
        return res

    def _auxbar(self, kpts: torch.Tensor,
                fuse_wrapper: LibcintWrapper) -> torch.Tensor:
        """Compute the single basis integral in equation (14) in
        Sun et al., https://doi.org/10.1063/1.4998644

        Parameters
        ----------
        kpts: torch.Tensor
            K-points.
        fuse_wrapper: LibcintWrapper
            Wrapper containing the basis information.

        Returns
        -------
        torch.Tensor
            Single basis integral.

        """

        # retrieve the parameters needed from the wrapper
        ao_to_shell = torch.as_tensor(
            fuse_wrapper.full_ao_to_shell)  # (nao_tot,)
        ao_idx0, ao_idx1 = fuse_wrapper.ao_idxs()
        coeffs, alphas, _ = fuse_wrapper.params  # coeffs, alphas: (ngauss_tot,)
        angmoms = torch.as_tensor(fuse_wrapper.full_angmoms)  # (ngauss_tot,)
        gauss_to_shell = torch.as_tensor(fuse_wrapper.full_gauss_to_shell,
                                         dtype=torch.int64)  # (ngauss_tot,)

        # calculate the vbar for each gauss
        half_sph_norm = 0.5 / np.sqrt(np.pi)
        bar = -1.0 / alphas  # (ngauss_tot,)
        norms = half_sph_norm / gaussian_integral(2, alphas)  # (ngauss_tot,)
        vbar = coeffs * (angmoms == 0) / norms * bar  # (ngauss_tot,)

        # scatter the vbar to the appropriate shell
        nshells_tot = len(ao_to_shell)
        vbar_shell = torch.zeros((nshells_tot,),
                                 dtype=self.dtype,
                                 device=self.device)
        vbar_shell.scatter_add_(dim=0, index=gauss_to_shell,
                                src=vbar)  # (nshells_tot,)

        # gather vbar to ao
        vbar_ao = torch.gather(vbar_shell, dim=0,
                               index=ao_to_shell)  # (nao_tot,)
        vbar_ao = vbar_ao[ao_idx0:ao_idx1]  # (nao,)
        vbar_ao = vbar_ao * (np.pi / self._lattice.volume())

        # gather the results to the indices where k-points are 0
        nkpts = kpts.shape[0]
        kpts_zero = kpts.norm(dim=-1) < 1e-9  # (nkpts,)
        res = torch.zeros((nkpts, vbar_ao.shape[0]),
                          dtype=self.dtype,
                          device=self.device)
        res[kpts_zero, :] = vbar_ao

        return res


def _combine_kpts_to_kpts_ij(kpts: torch.Tensor) -> torch.Tensor:
    """Combine the k-points into pair of k-points.

    Parameters
    ----------
    kpts: torch.Tensor
        K-points of the system. (nkpts, ndim)

    Returns
    -------
    torch.Tensor
        Pair of k-points. (nkpts_ij, 2, ndim) where nkpts_ij = nkpts ** 2
    """
    nkpts, ndim = kpts.shape
    kpts_ij = torch.zeros((nkpts, nkpts, 2, ndim),
                          dtype=kpts.dtype,
                          device=kpts.device)
    kpts_ij[:, :, 0, :] = kpts.unsqueeze(1)
    kpts_ij[:, :, 1, :] = kpts.unsqueeze(0)
    kpts_ij = kpts_ij.view(-1, 2, ndim)
    return kpts_ij


def _reduce_kpts_ij(kpts_ij: torch.Tensor) -> torch.Tensor:
    """Reduce the k-points.
    Get the value of kpts_reduce = kpts_i - kpts_j.
    however, as it might contain repeated k-points, only returns the unique
    value of kpts_reduce and the inverse index that can be used to reconstruct

    Parameters
    ----------
    kpts_ij: torch.Tensor
        K-points. (nkpts_ij, 2, ndim)

    Returns
    -------
    torch.Tensor
        Reduced k-points. (nkpts_ij, ndim)
    """
    kpts_reduce = -kpts_ij[..., 0, :] + kpts_ij[..., 1, :]  # (nkpts_ij, ndim)
    # inverse_idxs = torch.arange(kpts_reduce.shape[0], device=kpts_ij.device)
    # return kpts_reduce, inverse_idxs
    return kpts_reduce


def _renormalize_auxbases(auxbases: List[AtomCGTOBasis]) -> List[AtomCGTOBasis]:
    """density basis renormalization, following pyscf here:
    https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/pbc/df/df.py#L95
    this renormalization makes the integral of auxbases (not auxbases * auxbases) to be 1

    Parameters
    ----------
    auxbases: List[AtomCGTOBasis]
        List of atom bases.

    Returns
    -------
    List[AtomCGTOBasis]
        Renormalized auxbases.
    """
    res: List[AtomCGTOBasis] = []
    # libcint multiply np.sqrt(4*pi) to the basis
    half_sph_norm = 0.5 / np.sqrt(np.pi)
    for atb in auxbases:  # atb is AtomCGTOBasis
        bases: List[CGTOBasis] = []
        for bas in atb.bases:  # bas is CGTOBasis
            assert bas.normalized
            int1 = gaussian_integral(bas.angmom * 2 + 2, bas.alphas)
            s = torch.sum(bas.coeffs * int1)
            coeffs = bas.coeffs * (half_sph_norm / s)
            b2 = CGTOBasis(angmom=bas.angmom,
                           coeffs=coeffs,
                           alphas=bas.alphas,
                           normalized=True)
            bases.append(b2)
        atb2 = AtomCGTOBasis(atomz=atb.atomz, bases=bases, pos=atb.pos)
        res.append(atb2)
    return res

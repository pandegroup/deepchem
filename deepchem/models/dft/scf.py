from __future__ import annotations
from abc import abstractmethod
from typing import Union
import torch
import numpy as np
import warnings
try:
    from deepchem.utils.dft_utils import KS
    from deepchem.feat.dft_data import DFTEntry, DFTSystem
    from deepchem.utils.dftutils import KSCalc, hashstr, SpinParam
    from deepchem.models.dft.nnxc import BaseNNXC, HybridXC
except Exception as e:
    warnings.warn(f"Import error: {e}")


class XCNNSCF(torch.nn.Module):
    """
    Exchange Correlation Neural Network - Self Consistent Iterations

    In the Kohn-Sham theory, the inclusion of the noninteracting kinetic energy     functional results in a set of one-particle equations with Kohn-Sham
    orbitals as their solutions after functional differentiation. It is a
    variational approach that determines the lowest energy and the related
    molecular orbitals and orbital energies by using the electron-electron
    interaction potential. To learn more about Density Functional Theory
    and the Kohn-Sham approach please use the references below.

    The XCNNSCF is used for performing self-consistent iterations. The
    XC functional in the Kohn-Sham model implementation is replaced by a
    neural network.

    Examples
    --------
    >>> from deepchem.models.dft.scf import XCNNSCF
    >>> import torch
    >>> from deepchem.feat.dft_data import DFTEntry, DFTSystem
    >>> from deepchem.models.dft.nnxc import HybridXC
    >>> nnmodel = (torch.nn.Sequential(
    ...         torch.nn.Linear(2, 10),
    ...         torch.nn.Tanh(),
    ...         torch.nn.Linear(10, 1))).to(torch.double)
    >>> hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    >>> e_type = 'dm'
    >>> true_val = 'deepchem/feat/tests/data/dftHF_output.npy'
    >>> systems = [{
    ...     'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
    ...     'basis': '6-311++G(3df,3pd)'
    ... }]
    >>> entry = DFTEntry.create(e_type, true_val, systems)
    >>> evl = XCNNSCF(hybridxc, entry)
    >>> system = DFTSystem(systems[0])
    >>> run = evl.run(system)
    The 6-311++G(3df,3pd) basis for atomz 9 does not exist, but we will download it
    Downloaded to /home/runner/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/09.gaussian94
    >>> output = run.energy()

    Notes
    -----
    This code is derived from https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/evaluator.py

    References
    ----------
    deepchem.models.dft.nnxc
    Kohn, W. and Sham, L.J., 1965. Self-consistent equations including
    exchange and correlation effects. Physical review, 140(4A), p.A1133.
    """

    def __init__(self, xc: Union[BaseNNXC, HybridXC], entry: DFTEntry):
        super().__init__()
        """
        Parameters
        ----------
        xc: Union[BaseNNXC, HybridXC]
            exchange correlation functional that has been replaced by a
            neural network.
        entry: DFTEntry
        """
        self.xc = xc

    @abstractmethod
    def get_xc(self) -> HybridXC:
        """
        Returns
        -------
        Exchange correlation functional that has been replaced by a
        neural network, based on a BaseNNXC model.
        """
        return self.xc

    @abstractmethod
    def run(self, system: DFTSystem) -> KSCalc:
        """
        Kohn Sham Model
        This method runs the Quantum Chemistry calculation (Differentiable
        DFT) of the given system and returns the post-run object. This method
        starts with an intial density matrix, the new density matrix can be
        obtained from the post-run object.

        Parameters
        ----------
        system: DFTSystem

        Returns
        -------
        KSCalc object

        """
        dm0, dmname = self._get_dm0(system)
        mol = system.get_dqc_mol()
        qc = KS(mol, xc=self.xc).run(dm0=dm0, bck_options={"max_niter": 50})
        return KSCalc(qc)

    def _dm0_name(self, obj) -> str:
        """
        Returns
        -------
        dm0 followed by the name of the system
        """
        return "dm0_" + hashstr(str(obj))

    def _get_dm0(self, system: DFTSystem):
        """
        This method calculates and retuns the density matrix of a system.
        The matrix will vary depending on the atomic numbers, positions, and
        spins.

        Parameters
        ----------
        system: DFTSystem
        """
        dm_name = self._dm0_name(system)
        dm0: torch.Tensor
        get_dm = np.array(getattr(self, dm_name, None), dtype=bool)
        dm0 = torch.Tensor(get_dm)
        dm_exists = dm0 is not None
        dm_written = dm_exists and torch.any(dm0 != 0.0)
        if not dm_written:
            dm0_res: Union[None, torch.Tensor, SpinParam[torch.Tensor]] = None
        elif system.get_dqc_mol().spin != 0:
            dm0_res = SpinParam(u=dm0[0].detach(), d=dm0[1].detach())
        else:
            dm0_res = dm0

        return dm0_res, (dm_name if dm_exists else None)

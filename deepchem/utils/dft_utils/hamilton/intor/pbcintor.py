from __future__ import annotations
from typing import Optional, List, Tuple, Union, Dict
import ctypes
import operator
import warnings
from dataclasses import dataclass
from functools import reduce
import numpy as np
import torch
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, CPBC, CGTO, NDIM, c_null_ptr
from deepchem.utils.pytorch_utils import get_complex_dtype
from deepchem.utils.misc_utils import estimate_ovlp_rcut
from deepchem.utils.dft_utils.hamilton.intor.molintor import _check_and_set, _get_intgl_optimizer
from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager
from deepchem.utils.dft_utils import LibcintWrapper, Lattice


@dataclass
class PBCIntOption:
    """PBCIntOption is a class that contains parameters for the PBC integrals.

    Examples
    --------
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


# helper functions
def get_default_options(options: Optional[PBCIntOption] = None) -> PBCIntOption:
    """if options is None, then set the default option.
    otherwise, just return the input options.

    Examples
    --------
    >>> get_default_options()
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)

    Parameters
    ----------
    options: Optional[PBCIntOption]
        Input options object

    Returns
    -------
    PBCIntOption
        The options object

    """
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    return options1


def get_default_kpts(kpts: Optional[torch.Tensor], dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
    """if kpts is None, then set the default kpts (k = zeros)
    otherwise, just return the input kpts in the correct dtype and device

    Examples
    --------
    >>> get_default_kpts(torch.tensor([[1, 1, 1]]), torch.float64, 'cpu')
    tensor([[1., 1., 1.]], dtype=torch.float64)

    Parameters
    ----------
    kpts: Optional[torch.Tensor]
        Input k-points
    dtype: torch.dtype
        The dtype of the kpts
    device: torch.device
        Device on which the tensord are located. Ex: cuda, cpu

    Returns
    -------
    torch.Tensor
        Default k-points

    """
    if kpts is None:
        kpts1 = torch.zeros((1, NDIM), dtype=dtype, device=device)
    else:
        kpts1 = kpts.to(dtype).to(device)
        assert kpts1.ndim == 2
        assert kpts1.shape[-1] == NDIM
    return kpts1

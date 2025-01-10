from abc import abstractmethod
from typing import Union, List
import torch
import warnings
try:
    from deepchem.utils.dft_utils import ValGrad, get_xc, BaseXC
    from deepchem.utils.dftutils import SpinParam
    from deepchem.utils import safenorm, safepow
except Exception as e:
    warnings.warn(f"Import error: {e}")


class BaseNNXC(BaseXC, torch.nn.Module):
    """
    Base class for the Neural Network XC (NNXC)  and HybridXC classes.

    Density-functional theory (DFT) is a theory used to calculate the
    electronic structure of atoms, molecules, and solids. Its objective is
    to use the fundamental laws of quantum mechanics to quantitatively
    comprehend the properties of materials.

    There are serious limitations to the tradional methods used to approximate
    solutions to the Schrödinger equation of N interacting electrons moving in
    an external potential. Whereas in DFT, instead of the many-body wave
    function, the density (n(r)) is a function of three spatial coordinates.

    The many-body electronic ground state can be described using single-particle    equations and an effective potential thanks to the Kohn-Sham theory. The
    exchange-correlation potential, which accounts for many-body effects, the
    Hartree potential, which describes the electrostatic electron-electron
    interaction, and the ionic potential resulting from the atomic cores make
    up the effective potential.

    The difference between the total exact energy and the total of the rest
    of the energy terms (such as kinetic energy), is known as the
    exchange-correlation energy. The exchange-correlation functional is obtained    by calculating the functional derivate of the XC energy w.r.t the
    electron density function. In this model, we are trying to build a neural
    network that can be trained to calculate an exchange-correlation functional
    based on a specific set of molecules/atoms/ions.

    This base class can be used to build layers such as the NNLDA layer, where
    the exchange correlation functional is trained based on the pre-defined LDA     class of functionals. The methods in this class take the electron density as
    the input and transform it accordingly. For example; The NNLDA layer
    requires only the density to build an NNXC whereas a GGA based model would
    require the density gradient as well. This method also takes polarization
    into account.

    References
    ----------
    Encyclopedia of Condensed Matter Physics, 2005.
    Mark R. Pederson, Tunna Baruah, in Advances In Atomic, Molecular, and
    Optical Physics, 2015
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    """

    @abstractmethod
    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """
        This method is used to transform the electron density. The output
        of this method varies depending on the layer.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.
        """
        pass

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method is implemented only to avoid errors while passing the
        get_edensityxc method values to DQC and Xitorch.
        """
        # torch.nn.module prefix has no ending dot, while xt prefix has
        nnprefix = prefix if prefix == "" else prefix[:-1]
        return [
            name for (name, param) in self.named_parameters(prefix=nnprefix)
        ]


class NNLDA(BaseNNXC):
    """
    Neural network xc functional of LDA

    Local-density approximations (LDA) are a class of approximations to the
    exchange–correlation (XC) energy. The LDA assumes variations of the
    density to be gradual, i.e, it is  based on the homogeneous electron
    gas model. Which is why it is regarded as the simplest approach to the
    exchange correlation functional. This class of functionals depend only
    upon the value of the electronic density at each point in space.

    Hence, in this model, we only input the density and not other components
    such as the gradients of the density (which is used in other functionals
    such as the GGA class).

    Examples
    --------
    >>> from deepchem.models.dft.nnxc import NNLDA
    >>> import torch
    >>> import torch.nn as nn
    >>> n_input, n_hidden = 2, 1
    >>> nnmodel = (nn.Linear(n_input, n_hidden))
    >>> output = NNLDA(nnmodel)

    References
    ----------
    Density-Functional Theory of the Electronic Structure of Molecules,Robert
    G. Parr and Weitao Yang, Annual Review of Physical Chemistry 1995 46:1,
    701-728
    R. O. Jones and O. Gunnarsson, Rev. Mod. Phys. 61, 689 (1989)
    """

    def __init__(self, nnmodel: torch.nn.Module):
        super().__init__()
        """
        Parameters
        ----------
        nnmodel: torch.nn.Module
            Neural network for xc functional
        """
        self.nnmodel = nnmodel

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """
        This method transform the local electron density (n) and the spin
        density (xi) for polarized and unpolarized cases, to be the input of
        the neural network.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.

        Returns
        -------
        res
            Neural network output by calculating total density (n) and the spin
            density (xi). The shape of res is (ninp , ) where ninp is the number            of layers in nnmodel ; which is user defined.
        """
        if isinstance(densinfo, ValGrad):  # unpolarized case
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
        else:  # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            xi = (nu - nd) / (n + 1e-18)  # avoiding nan

        ninp = n

        x = torch.cat((ninp, xi), dim=-1)  # (*BD, nr, 2)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = nnout * n  # (*BD, nr, 1)
        res = res.squeeze(-1)
        return res


class NNPBE(BaseNNXC):
    # neural network xc functional of GGA (receives the density and grad as inputs)
    """
    Neural network xc functional of GGA class of functionals

    The Purdew-Burke-Ernzerhof (PBE) functional is a popular non-empirical
    functional which falls under the Generalized Gradient Approximation (GGA)
    class of functionals. GGA differs from LDA by incorporating nonlocal
    corrections involving gradients (or derivatives) of the electron density.
    In other words, the XC potential is a complicated function in 3D space
    since it depends on the electron density and its gradient.

    Hence, in this model we input the electron density as well as the
    calculated gradients. We calculate these values using the DQC and Xitorch
    libraries.

    Examples
    --------
    >>> from deepchem.models.dft.nnxc import NNPBE
    >>> import torch
    >>> import torch.nn as nn
    >>> n_input, n_hidden = 3, 3
    >>> nnmodel = (nn.Linear(n_input, n_hidden))
    >>> output = NNPBE(nnmodel)

    References
    ----------
    Perdew, John P., Kieron Burke, and Matthias Ernzerhof. "Generalized gradient approximation made simple." Physical review letters 77.18 (1996): 3865.
    A.W. Ghosh,
    5.09 - Electronics with Molecules,
    Editor(s): Pallab Bhattacharya, Roberto Fornari, Hiroshi Kamimura,
    Comprehensive Semiconductor Science and Technology,
    Elsevier,
    2011,
    Pages 383-479,
    ISBN 9780444531537,
    https://doi.org/10.1016/B978-0-44-453153-7.00033-X.
    """

    def __init__(self, nnmodel: torch.nn.Module):
        """
        Parameters
        ----------
        nnmodel: torch.nn.Module
            Neural network for xc functional. Shape; (3,...). This is because
            the nnmodel receives an input with the shape (....,3). This
            dimension is for;
            (0) total density (n): (n_up + n_dn), and
            (1) spin density (xi): (n_up - n_dn) / (n_up + n_dn)
            (2) normalized gradients (s): |del(n)| / [2(3*pi^2)^(1/3) * n^(4/3)]
        """
        super().__init__()
        self.nnmodel = nnmodel

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """
        This method transform the local electron density (n), its gradient
        and the spin density (xi) for polarized and unpolarized cases, to be
        the input of the neural network.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.

        Returns
        -------
        res
            Neural network output by calculating total density (n) and the spin
            density (xi). The shape of res is (ninp , ) where ninp is the number            of layers in nnmodel ; which is 3 for NNPBE. The shape of the output is (....,1) and it represents the energy density per density per unit volume.
        """
        # densinfo.value: (*BD, nr)
        # densinfo.grad : (*BD, nr, 3)

        # collect the total density (n), spin density (xi), and normalized gradients (s)
        a = 6.187335452560271  # 2 * (3 * np.pi ** 2) ** (1.0 / 3)
        if isinstance(densinfo, ValGrad):  # unpolarized case
            assert densinfo.grad is not None
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
            n_offset = n + 1e-18  # avoiding nan
            s = safenorm(densinfo.grad, dim=0).unsqueeze(-1)
        else:  # polarized case
            assert densinfo.u.grad is not None
            assert densinfo.d.grad is not None
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            n_offset = n + 1e-18  # avoiding nan
            xi = (nu - nd) / n_offset
            s = safenorm(densinfo.u.grad + densinfo.d.grad, dim=0).unsqueeze(-1)

        # normalize the gradient
        s = s / a * safepow(n, -4.0 / 3)

        # decide how to transform the density to be the input of nn
        ninp = n
        sinp = s

        # get the neural network output
        x = torch.cat((ninp, xi, sinp), dim=-1)  # (*BD, nr, 3)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = nnout * n
        res = res.squeeze(-1)
        return res


class HybridXC(BaseNNXC):
    """
    The HybridXC module computes XC energy by summing XC energy computed
    from libxc(any conventional DFT functional) and the trainable neural
    network with tunable weights.
    This layer constructs a hybrid functional based on the user's choice
    of what model is to be used to train the functional. (Currently, we only
    support an LDA based model). This hybrid functional is a combination of
    the xc that is trained by a neural network, and a conventional DFT
    functional.

    Examples
    --------
    >>> from deepchem.models.dft.nnxc import HybridXC
    >>> import torch
    >>> import torch.nn as nn
    >>> n_input, n_hidden = 2, 1
    >>> nnmodel = (nn.Linear(n_input, n_hidden))
    >>> output = HybridXC("lda_x", nnmodel, aweight0=0.0)
    """

    def __init__(self,
                 xcstr: str,
                 nnmodel: torch.nn.Module,
                 aweight0: float = 0.0,
                 bweight0: float = 1.0):

        super().__init__()
        """
        Parameters
        ----------
        xcstr: str
            The choice of xc to use. Some of the commonly used ones are:
            lda_x, lda_c_pw, lda_c_ow, lda_c_pz, lda_xc_lp_a, lda_xc_lp_b.
            The rest of the possible values can be found under the
            "LDA Functionals" section in the reference given below.
        nnmodel: nn.Module
            trainable neural network for prediction xc energy.
        aweight0: float
            weight of the neural network
        bweight0: float
            weight of the default xc

        References
        ----------
        https://tddft.org/programs/libxc/functionals/
        """
        self.xc = get_xc(xcstr)
        family = self.xc.family
        if family == 1:
            self.nnxc = NNLDA(nnmodel)
        else:
            self.nnxc = NNPBE(nnmodel)
        self.aweight = torch.nn.Parameter(
            torch.tensor(aweight0, requires_grad=True))
        self.bweight = torch.nn.Parameter(
            torch.tensor(bweight0, requires_grad=True))
        self.weight_activation = torch.nn.Identity()

    @property
    def family(self) -> int:
        """
        This method determines the type of model to be used, to train the
        neural network. Currently we only support an LDA based model and will
        implement more in subsequent iterations.
        Returns
        -------
        xc.family
        """
        return self.xc.family

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Get electron density from xc
        This function reflects eqn. 4 in the `paper <https://arxiv.org/abs/2102.04229>_`.
        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.

        Returns
        -------
        Total calculated electron density with tunable weights.
        """
        nnxc_ene = self.nnxc.get_edensityxc(densinfo)
        xc_ene = self.xc.get_edensityxc(densinfo)
        aweight = self.weight_activation(self.aweight)
        bweight = self.weight_activation(self.bweight)
        return nnxc_ene * aweight + xc_ene * bweight

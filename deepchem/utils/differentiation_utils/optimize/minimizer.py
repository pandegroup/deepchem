import warnings
import torch
from typing import Callable, List, Optional


def gd(
        fcn: Callable[..., torch.Tensor],
        x0: torch.Tensor,
        params: List,
        # gd parameters
        step: float = 1e-3,
        gamma: float = 0.9,
        # stopping conditions
        maxiter: int = 1000,
        f_tol: float = 0.0,
        f_rtol: float = 1e-8,
        x_tol: float = 0.0,
        x_rtol: float = 1e-8,
        # misc parameters
        verbose=False,
        terminate=False,
        **unused):
    r"""
    Vanilla gradient descent with momentum. The stopping conditions use OR criteria.
    The update step is following the equations below.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils.optimize.minimizer import gd
    >>> def fcn(x):
    ...     return (x - 2) ** 2, 2 * (x - 2)
    >>> x0 = torch.tensor(0.0, requires_grad=True)
    >>> x = gd(fcn, x0, [])
    >>> x
    tensor(2.0000)


    .. math::
        \mathbf{v}_{t+1} &= \gamma \mathbf{v}_t - \eta \nabla_{\mathbf{x}} f(\mathbf{x}_t) \\
        \mathbf{x}_{t+1} &= \mathbf{x}_t + \mathbf{v}_{t+1}

    Parameters
    ----------
    fcn: callable
        The objective function to minimize. It should take a tensor and return a
        tensor and its gradient.
    x0: torch.Tensor
        The initial guess.
    step: float
        The step size towards the steepest descent direction, i.e. :math:`\eta` in
        the equations above.
    gamma: float
        The momentum factor, i.e. :math:`\gamma` in the equations above.
    maxiter: int
        Maximum number of iterations.
    f_tol: float or None
        The absolute tolerance of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    terminate: bool (default False)
        Whether to use termination condition or, keep on running the minimizer.

    """

    x = x0.clone()
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)

        # update the step
        v = (gamma * v - step * dfdx).detach()
        xprev = x.detach()
        x = (xprev + v).detach()

        # check the stopping conditions
        to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

        if to_stop and terminate:
            break

        fprev = f
    if terminate:
        x = stop_cond.get_best_x(x)
    return x


def adam(
        fcn: Callable[..., torch.Tensor],
        x0: torch.Tensor,
        params: List,
        # gd parameters
        step: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        # stopping conditions
        maxiter: int = 1000,
        f_tol: float = 0.0,
        f_rtol: float = 1e-8,
        x_tol: float = 0.0,
        x_rtol: float = 1e-8,
        # misc parameters
        verbose=False,
        terminate=False,
        **unused):
    r"""
    Adam optimizer by Kingma & Ba (2015). The stopping conditions use OR criteria.
    The update step is following the equations below.

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.minimizer import adam
    >>> def fcn(x):
    ...     return (x - 4) * 2, (x * 2) + 3
    >>> x0 = torch.tensor(0.0, requires_grad=True)
    >>> x = adam(fcn, x0, [], maxiter=10000)
    >>> x
    tensor(-1.5000)

    .. math::
        \mathbf{g}_t &= \nabla_{\mathbf{x}} f(\mathbf{x}_{t-1}) \\
        \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
        \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
        \hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t) \\
        \hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t) \\
        \mathbf{x}_t &= \mathbf{x}_{t-1} - \alpha \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)

    Parameters
    ----------
    step: float
        The step size towards the descent direction, i.e. :math:`\alpha` in
        the equations above.
    beta1: float
        Exponential decay rate for the first moment estimate.
    beta2: float
        Exponential decay rate for the first moment estimate.
    eps: float
        Small number to prevent division by 0.
    maxiter: int
        Maximum number of iterations.
    f_tol: float or None
        The absolute tolerance of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    terminate: bool (default False)
        Whether to use the termination condition, or keep running the minimizer.
    """

    x = x0.clone()
    stop_cond = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, verbose)
    fprev = torch.tensor(0.0, dtype=x0.dtype, device=x0.device)
    v = torch.zeros_like(x)
    m = torch.zeros_like(x)
    beta1t = beta1
    beta2t = beta2
    for i in range(maxiter):
        f, dfdx = fcn(x, *params)
        f = f.detach()
        dfdx = dfdx.detach()

        # update the step
        m = beta1 * m + (1 - beta1) * dfdx
        v = beta2 * v + (1 - beta2) * dfdx**2
        mhat = m / (1 - beta1t)
        vhat = v / (1 - beta2t)
        beta1t *= beta1
        beta2t *= beta2
        xprev = x.detach()
        x = (xprev - step * mhat / (vhat**0.5 + eps)).detach()

        # check the stopping conditions
        to_stop = stop_cond.to_stop(i, x, xprev, f, fprev)

        if to_stop and terminate:
            break

        fprev = f
    if terminate:
        x = stop_cond.get_best_x(x)
    return x


class TerminationCondition(object):
    """The class to handle the stopping conditions.

    Examples
    --------
    >>> stop_cond = TerminationCondition(1e-8, 1e-8, 1e-8, 1e-8, True)

    """

    def __init__(self, f_tol: float, f_rtol: float, x_tol: float, x_rtol: float,
                 verbose: bool):
        """Initialize the TerminationCondition.

        Parameters
        ----------
        f_tol: float or None
            Absolute tolerance of the output ``f``.
        f_rtol: float or None
            Relative tolerance of the output ``f``.
        x_tol: float or None
            Absolute tolerance of the norm of the input ``x``.
        x_rtol: float or None
            Relative tolerance of the norm of the input ``x``.
        verbose: bool
            Whether to print the iteration information.

        """
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.verbose = verbose

        self._ever_converge = False
        self._max_i = -1
        self._best_dxnorm = float("inf")
        self._best_df = float("inf")
        self._best_f = float("inf")
        self._best_x: Optional[torch.Tensor] = None

    def to_stop(self, i: int, xnext: torch.Tensor, x: torch.Tensor,
                f: torch.Tensor, fprev: torch.Tensor) -> bool:
        """Check if the stopping conditions are met.

        Parameters
        ----------
        i: int
            The iteration number.
        xnext: torch.Tensor
            The next input.
        x: torch.Tensor
            The current input.
        f: torch.Tensor
            The current output.
        fprev: torch.Tensor
            The previous output.

        Returns
        -------
        bool
            Whether to stop the iteration.

        """
        xnorm: float = float(x.detach().norm().item())
        dxnorm: float = float((x - xnext).detach().norm().item())
        fabs: float = float(f.detach().abs().item())
        df: float = float((fprev - f).detach().abs().item())
        fval: float = float(f.detach().item())

        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = df < self.f_tol
        yrcheck = df < self.f_rtol * fabs
        converge = xtcheck or xrcheck or ytcheck or yrcheck
        if self.verbose:
            if i == 0:
                print("   #:             f |        dx,        df")
            if converge:
                print("Finish with convergence")
            if i == 0 or ((i + 1) % 10) == 0 or converge:
                print("%4d: %.6e | %.3e, %.3e" % (i + 1, f, dxnorm, df))

        res = (i > 0 and converge)

        # get the best values
        if not self._ever_converge and res:
            self._ever_converge = True
        if i > self._max_i:
            self._max_i = i
        if fval < self._best_f:
            self._best_f = fval
            self._best_x = x
            self._best_dxnorm = dxnorm
            self._best_df = df
        return res

    def get_best_x(self, x: torch.Tensor) -> torch.Tensor:
        """Get the best input.

        Parameters
        ----------
        x: torch.Tensor
            The current input.

        Returns
        -------
        torch.Tensor
            The best input.

        """
        # usually user set maxiter == 0 just to wrap the minimizer backprop
        if not self._ever_converge and self._max_i > -1:
            msg = (
                "The minimizer does not converge after %d iterations. "
                "Best |dx|=%.4e, |df|=%.4e, f=%.4e" %
                (self._max_i, self._best_dxnorm, self._best_df, self._best_f))
            warnings.warn(msg)
            assert isinstance(self._best_x, torch.Tensor)
            return self._best_x
        else:
            return x

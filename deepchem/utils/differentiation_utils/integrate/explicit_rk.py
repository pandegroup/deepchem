from typing import List, Callable, Sequence, NamedTuple
import torch


# list of tableaus
class _Tableau(NamedTuple):
    """To specify a particular method, one needs to provide the integer s
    (the number of stages), and the coefficients a[i,j] (for 1 ≤ j < i ≤ s),
    b[i] (for i = 1, 2, ..., s) and c[i] (for i = 2, 3, ..., s). The matrix
    [aij] is called the Runge–Kutta matrix, while the b[i] and c[i] are known
    as the weights and the nodes. These data are usually arranged in a
    mnemonic device, known as a Butcher tableau (after John C. Butcher):

    Examples
    --------
    >>> euler = _Tableau(c=[0.0],
    ...                  b=[1.0],
    ...                  a=[[0.0]]
    ... )
    >>> euler.c
    [0.0]

    Attributes
    ----------
    c: List[float]
        The nodes
    b: List[float]
        The weights
    a: List[List[float]]
        The Runge-Kutta matrix

    """
    c: List[float]
    b: List[float]
    a: List[List[float]]


rk4_tableau = _Tableau(c=[0.0, 0.5, 0.5, 1.0],
                       b=[1 / 6., 1 / 3., 1 / 3., 1 / 6.],
                       a=[[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
rk38_tableau = _Tableau(c=[0.0, 1 / 3, 2 / 3, 1.0],
                        b=[1 / 8, 3 / 8, 3 / 8, 1 / 8],
                        a=[[0.0, 0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0, 0.0],
                           [-1 / 3, 1.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0]])
fwd_euler_tableau = _Tableau(c=[0.0], b=[1.0], a=[[0.0]])
midpoint_fableau = _Tableau(c=[0.0, 0.5],
                            b=[0.0, 1.0],
                            a=[[0.0, 0.0], [0.5, 0.0]])


def explicit_rk(tableau: _Tableau,
                fcn: Callable[..., torch.Tensor],
                y0: torch.Tensor,
                t: torch.Tensor,
                params: torch.Tensor,
                batch_size: int = 1,
                device="cpu"):
    """The family of explicit Runge–Kutta methods is a generalization
    of the RK4 method mentioned above.

    Examples
    --------
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 100)
    >>> y_init = torch.rand(2, 1)
    >>> solver_param = [rk4_tableau,
    ...                 lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 torch.tensor([1.1, 0.4, 0.1, 0.4])]
    >>> sol = explicit_rk(*solver_param)
    >>> sol[-1]
    tensor([[0.0567], [2.3627]])

    For solving multiple ODEs, we can use the GPU and feed in multiple initial conditions
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 100)
    >>> batch_size = 10
    >>> y_init = torch.rand(2, batch_size)
    >>> params = torch.rand(4, batch_size)
    >>> solver_param = [rk4_tableau,
    ...                 lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 params]
    >>> sol = explicit_rk(*solver_param, batch_size=batch_size, device='cuda')
    >>> print(sol.shape)
    torch.Size([100, 2, 10])

    Parameters
    ----------
    fcn: callable dy/dt = fcn(t, y, *params)
        The function to be integrated. It should produce output of list of
        tensors following the shapes of tuple `y`. `t` should be a single element.
    t: torch.Tensor (nt,)
        The time values to integrate over. Can be generated by `linspace` (Refer example above).
    y0: list of torch.Tensor (*ny)
        The list of initial values.
    params: list
        List of input parameters for the function `fcn`.
    batch_size: int
        The batch size to compute the RK method. Default is 1.
    device: str
        The device to compute the RK method. Default is "cpu".

    Returns
    -------
    yt: list of torch.Tensor (nt, *ny, batch_size)
        The value of `y` at the given time `t` for each batch.

    Notes
    -----
    We require the function `fcn` to have t as the first argument, y as the second argument and
    the rest of the arguments as `params`. The function should return a tensor of the same shape as `y`.
    Even though t may not be used in the function, it is required to have it as the first argument. For
    multiple derivatives, pass the y as a list of tensors. If multiple parameters are required,
    they should be passed as a list. Refer example above.

    References
    ----------
    [1].. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods

    """

    assert y0.shape[-1] == batch_size, "Number of initial conditions should match the batch size"

    c, a, b = torch.tensor(tableau.c, device=device), torch.tensor(
        tableau.a, device=device), torch.tensor(tableau.b, device=device)
    t = torch.tensor(t).clone().detach().to(device)
    y0 = torch.tensor(y0).clone().detach().to(device)
    if params is not None:
        params = torch.tensor(params).to(device)
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)
    s = len(c)
    num_steps = len(t)
    yt_list = [y0]
    y = y0.clone()
    h = (t[-1] - t[0]) / (num_steps - 1)

    for i in range(1, num_steps):
        t0 = t[i]
        k = torch.zeros((s,) + y.shape, device=device)

        # Vectorize stage computation by stacking inputs
        for i in range(s):
            y_sum = y + h * torch.sum(
                a[i, :i].unsqueeze(-1).unsqueeze(-1) * k[:i], dim=0)
            if len(y0.shape) == 1:
                y_sum = y_sum.squeeze(0)
            k[i] = fcn(t=t0 + c[i] * h, y=y_sum.to(device), params=params)
        y = y + h * torch.sum(b.unsqueeze(-1).unsqueeze(-1) * k, dim=0)
        yt_list.append(y)

    yt = torch.stack(yt_list)
    return yt


# list of methods
def rk38_ivp(fcn: Callable[..., torch.Tensor],
             y0: torch.Tensor,
             t: torch.Tensor,
             params: torch.Tensor,
             batch_size: int = 1,
             device="cpu",
             **kwargs):
    """A slight variation of "the" Runge–Kutta method is also due to
    Kutta in 1901 and is called the 3/8-rule.[19] The primary advantage
    this method has is that almost all of the error coefficients are
    smaller than in the popular method, but it requires slightly more
    FLOPs (floating-point operations) per time step.

    Examples
    --------
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 100)
    >>> y_init = torch.rand(2, 1)
    >>> solver_param = [lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 torch.tensor([1.1, 0.4, 0.1, 0.4])]
    >>> sol = rk38_ivp(*solver_param)
    >>> print(sol[-1])
    tensor([0.3483, 3.2585])

    Parameters
    ----------
    fcn: callable dy/dt = fcn(t, y, *params)
        The function to be integrated. It should produce output of list of
        tensors following the shapes of tuple `y`. `t` should be a single element.
    t: torch.Tensor (nt,)
        The integrated times
    y0: list of torch.Tensor (*ny)
        The list of initial values
    params: list
        List of any other parameters
    batch_size: int
        The batch size to compute the RK method. Default is 1.
    device: str
        The device to compute the RK method. Default is "cpu".
    **kwargs: dict
        Any other keyword arguments

    Returns
    -------
    yt: list of torch.Tensor (nt,*ny)
        The value of `y` at the given time `t`

    """
    return explicit_rk(rk38_tableau, fcn, y0, t, params, batch_size, device)


def fwd_euler_ivp(fcn: Callable[..., torch.Tensor],
                  y0: torch.Tensor,
                  t: torch.Tensor,
                  params: torch.Tensor,
                  batch_size: int = 1,
                  device="cpu",
                  **kwargs):
    """However, the simplest Runge–Kutta method is the (forward) Euler method,
    given by the formula $y_{n+1} = y_{n} + hf(t_{n}, y_{n}). This is the only
    consistent explicit Runge–Kutta method with one stage.

    Examples
    --------
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 1000)
    >>> y_init = torch.randn(2, 1)
    >>> solver_param = [lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 torch.tensor([1.1, 0.4, 0.1, 0.4])]
    >>> sol = fwd_euler_ivp(*solver_param)
    >>> print(sol[-1])
    tensor([[4.6852], [0.5726]])

    Parameters
    ----------
    fcn: callable dy/dt = fcn(t, y, *params)
        The function to be integrated. It should produce output of list of
        tensors following the shapes of tuple `y`. `t` should be a single element.
    t: torch.Tensor (nt,)
        The integrated times
    y0: list of torch.Tensor (*ny)
        The list of initial values
    params: list
        List of any other parameters
    batch_size: int
        The batch size to compute the RK method. Default is 1.
    device: str
        The device to compute the RK method. Default is "cpu".
    **kwargs: dict
        Any other keyword arguments

    Returns
    -------
    yt: list of torch.Tensor (nt,*ny)
        The value of `y` at the given time `t`

    """
    return explicit_rk(fwd_euler_tableau, fcn, y0, t, params, batch_size,
                       device)


def rk4_ivp(fcn: Callable[..., torch.Tensor],
            y0: torch.Tensor,
            t: torch.Tensor,
            params: torch.Tensor,
            batch_size: int = 1,
            device="cpu",
            **kwargs):
    """The most commonly used Runge Kutta method to find the solution
    of a differential equation is the RK4 method, i.e., the fourth-order
    Runge-Kutta method. The Runge-Kutta method provides the approximate
    value of y for a given point x. Only the first order ODEs can be
    solved using the Runge Kutta RK4 method.

    Examples
    --------
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 100)
    >>> y_init = torch.rand(2, 1)
    >>> solver_param = [lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 torch.tensor([1.1, 0.4, 0.1, 0.4])]
    >>> sol = rk4_ivp(*solver_param)
    >>> print(sol[-1])
    tensor([0.3459, 3.2954])

    Parameters
    ----------
    fcn: callable dy/dt = fcn(t, y, *params)
        The function to be integrated. It should produce output of list of
        tensors following the shapes of tuple `y`. `t` should be a single element.
    t: torch.Tensor (nt,)
        The integrated times
    y0: list of torch.Tensor (*ny)
        The list of initial values
    params: list
        List of any other parameters
    batch_size: int
        The batch size to compute the RK method. Default is 1.
    device: str
        The device to compute the RK method. Default is "cpu".
    **kwargs: dict
        Any other keyword arguments

    Returns
    -------
    yt: list of torch.Tensor (nt,*ny)
        The value of `y` at the given time `t`

    """
    return explicit_rk(rk4_tableau, fcn, y0, t, params, batch_size, device)


def mid_point_ivp(fcn: Callable[..., torch.Tensor],
                  y0: torch.Tensor,
                  t: torch.Tensor,
                  params: torch.Tensor,
                  batch_size: int = 1,
                  device="cpu",
                  **kwargs):
    """The explicit midpoint method is sometimes also known as the
    modified Euler method, the implicit method is the most simple
    collocation method, and, applied to Hamiltonian dynamics, a
    symplectic integrator.

    Examples
    --------
    >>> def lotka_volterra(t, y, params):
    ...     X, Y = y
    ...     alpha, beta, delta, gamma = params
    ...     dx_dt = alpha * X - beta * X * Y
    ...     dy_dt = delta * X * Y - gamma * Y
    ...     return torch.stack([dx_dt, dy_dt])
    >>> t = torch.linspace(0, 50, 100)
    >>> y_init = torch.rand(2, 1)
    >>> solver_param = [lotka_volterra,
    ...                 y_init,
    ...                 t,
    ...                 torch.tensor([1.1, 0.4, 0.1, 0.4])]
    >>> sol = rk4_ivp(*solver_param)
    >>> sol[-1]
    tensor([0.3459, 3.2954])

    Parameters
    ----------
    fcn: callable dy/dt = fcn(t, y, *params)
        The function to be integrated. It should produce output of list of
        tensors following the shapes of tuple `y`. `t` should be a single element.
    t: torch.Tensor (nt,)
        The integrated times
    y0: list of torch.Tensor (*ny)
        The list of initial values
    params: list
        List of any other parameters
    batch_size: int
        The batch size to compute the RK method. Default is 1.
    device: str
        The device to compute the RK method. Default is "cpu".
    **kwargs: dict
        Any other keyword arguments

    Returns
    -------
    yt: list of torch.Tensor (nt,*ny)
        The value of `y` at the given time `t`

    """
    return explicit_rk(midpoint_fableau, fcn, y0, t, params, batch_size, device)

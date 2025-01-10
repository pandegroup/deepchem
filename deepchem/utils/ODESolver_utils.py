class ODESolver:
  def __init__(self, f):
    self.f = f
  def euler_method(self, f, x0, y0, h, x_end):
    """
    Euler's method to approximate the solution of a differential equation.
    Args:
        f: A function that defines the differential equation (dy/dx).
        x0: The initial x-value.
        y0: The initial y-value.
        h: The step size.
        x_end: The end point for the x-values.
    Returns:
        A list of x-values and a list of corresponding approximate y-values.
    """
    x = [x0]
    y = [y0]
    while x[-1] < x_end:
      y_new = y[-1] + h * f(x[-1], y[-1])
      x.append(x[-1] + h)
      y.append(y_new)
    return x, y

  def runge_kutta_4(self, f, x0, y0, h, x_end):
    """
      fourth-order Runge-Kutta method to approximate
      the solution of a differential equation.
      Args:
          f: A function that defines the differential equation (dy/dx).
          x0: The initial x-value.
          y0: The initial y-value.
          h: The step size.
          x_end: The end point for the x-values.
      Returns:
         A list of x-values and a list of corresponding approximate y-values.
    """
    x = [x0]
    y = [y0]
    while x[-1] < x_end:
      k1 = h * f(x[-1], y[-1])
      k2 = h * f(x[-1] + h/2, y[-1] + k1/2)
      k3 = h * f(x[-1] + h/2, y[-1] + k2/2)
      k4 = h * f(x[-1] + h, y[-1] + k3)
      y_new = y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
      x.append(x[-1] + h)
      y.append(y_new)
    return x, y

  def runge_kutta_3(self, f, x0, y0, h, x_end):
    """
        third-order Runge-Kutta method with four stages to approximate
        the solution of a differential equation.
        Args:
            f: A function that defines the differential equation (dy/dx).
            x0: The initial x-value.
            y0: The initial y-value.
            h: The step size.
            x_end: The end point for the x-values.
        Returns:
            A list of x-values and a list of corresponding approximate y-values.
    """
    x = [x0]
    y = [y0]
    while x[-1] < x_end:
      k1 = h * f(x[-1], y[-1])
      k2 = h * f(x[-1] + h/2, y[-1] + k1/2)
      k3 = h * f(x[-1] + h/2, y[-1] + k2/2)
      k4 = h * f(x[-1] + h, y[-1] + k3)
      y_new = y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
      x.append(x[-1] + h)
      y.append(y_new)
      return x, y


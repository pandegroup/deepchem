import unittest
import pytest
import torch
import math
class ODESolver:
  def __init__(self, f):
    self.f = f
  def euler_method(self, f, x0, y0, h, x_end):
    """Euler's method to approximate the solution of a differential equation.
    Args:
        f: A function that defines the differential equation (dy/dx).
        x0: The initial x-value.
        y0: The initial y-value.
        h: The step size.
        x_end: The end point for the x-values.
    Returns:
        A list of x-values and a list of corresponding approximate y-values. """
    x = torch.tensor([x0])
    y = torch.tensor([y0])
    while x[-1] < x_end:
      slope = self.f(x[-1], y[-1])
      y_new = torch.tensor(y[-1] + h * (slope))
      if y_new.dim() == 0:
            y_new = y_new.unsqueeze(dim=0)
      x = torch.cat((x, torch.tensor([x[-1] + h])), dim=0)
      y = torch.cat((y, y_new), dim=0)
    return x, y
  
  def runge_kutta_4(self, f, x0, y0, h, x_end):
    """fourth-order Runge-Kutta method to approximate
      the solution of a differential equation.
      Args:
          f: A function that defines the differential equation (dy/dx).
          x0: The initial x-value.
          y0: The initial y-value.
          h: The step size.
          x_end: The end point for the x-values.
      Returns:
         A list of x-values and a list of corresponding approximate y-values."""
    if torch.cuda.is_available():
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")
    x = torch.tensor([x0])
    y = torch.tensor([y0])
    while x[-1] < x_end:
      k1 = h * f(x[-1], y[-1])
      k2 = h * f(x[-1] + h/2, y[-1] + k1/2)
      k3 = h * f(x[-1] + h/2, y[-1] + k2/2)
      k4 = h * f(x[-1] + h, y[-1] + k3)
      y_new = torch.tensor(y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6)
      if y_new.dim() == 0:
            y_new = y_new.unsqueeze(dim=0)
      x = torch.cat((x, torch.tensor([x[-1] + h])), dim=0)
      y = torch.cat((y, y_new), dim=0)
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
    x = torch.tensor([x0])
    y = torch.tensor([y0])
    while x[-1] < x_end:
      k1 = h * self.f(x[-1], y[-1])
      k2 = h * self.f(x[-1] + h/2, y[-1] + k1/2)
      k3 = h * self.f(x[-1] + h, y[-1] + 2*k2 - k1)
      y_new = y[-1].clone().detach() + (k1 + 4*k2 + k3) / 6
      y_new = torch.tensor(y_new)
      if y_new.dim() == 0:
            y_new = y_new.unsqueeze(dim=0)
      x = torch.cat((x, torch.tensor([x[-1] + h])), dim=0)
      y = torch.cat((y, y_new), dim=0)
      return x, y
    
class TestODESolver(unittest.TestCase):
  def test_euler_method_simple(self):
    """Tests Euler's method with a simple differential equation (dy/dx = y)."""
    def f(x, y):
      return y
    solver = ODESolver(f)
    x, y = solver.euler_method(f, 1, 2, 0.1, 2)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], 2 * math.exp(x[i] - 1), delta = 0.5)  
  def test_euler_method_custom_f(self):
    """Tests Euler's method with a custom differential equation (dy/dx = x^2 - y)."""
    def f(x, y):
      return x**2 - y
    solver = ODESolver(f)
    x, y = solver.euler_method(f, 0, 1, 0.1, 1)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], solver.euler_method(f, 0.0, 1.0, 0.01, x[i])[1][-1], delta=0.5)  
      
  def test_runge_kutta_4_simple(self):
    """Tests Runge-Kutta-4 with a simple differential equation (dy/dx = y)."""

    def f(x, y):
      return y

    solver = ODESolver(f)
    x, y = solver.runge_kutta_4(f, 1, 2, 0.1, 2)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], 2 * math.exp(x[i] - 1), delta = 0.15)  
  def test_runge_kutta_4_custom_f(self):
    """Tests Runge-Kutta-4 with a custom differential equation (dy/dx = x^2 - y)."""

    def f(x, y):
      return x**2 - y

    solver = ODESolver(f)
    x, y = solver.runge_kutta_4(f, 0, 1, 0.1, 1)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], solver.runge_kutta_4(f, 0, 1, 0.01, x[i])[1][-1], delta = 0.15)  

  def test_runge_kutta_3_simple(self):
    """Tests Runge-Kutta-3 with a simple differential equation (dy/dx = y)."""
    def f(x, y):
      return y
    solver = ODESolver(f)
    x, y = solver.runge_kutta_3(f, 1, 2, 0.1, 2)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], 2 * math.exp(x[i] - 1), delta=0.15)  

  def test_runge_kutta_3_custom_f(self):
    """Tests Runge-Kutta-3 with a custom differential equation (dy/dx = x^2 - y)."""
    def f(x, y):
      return x**2 - y

    solver = ODESolver(f)
    x, y = solver.runge_kutta_3(f, 0, 1, 0.1, 1)
    self.assertEqual(len(x), len(y))
    for i in range(len(x)):
      self.assertAlmostEqual(y[i], solver.runge_kutta_4(f, 0, 1, 0.01, x[i])[1][-1], delta = 0.15)  


if __name__ == "__main__":
  unittest.main()

 
 
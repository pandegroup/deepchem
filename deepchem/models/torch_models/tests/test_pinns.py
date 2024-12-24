import pytest
import torch
import numpy as np
from deepchem.models.torch_models.pinns_model import PINNModel
import deepchem as dc


@pytest.mark.torch
def test_pinns_overfit():
    model = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.Tanh(),
                                torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                torch.nn.Linear(64, 1))
    pinn = PINNModel(
        model=model,
        pde_fn=lambda u, x: torch.zeros_like(u),
        boundary_data={
            'dirichlet': {
                'points': torch.tensor([[0.0], [1.0]], dtype=torch.float32),
                'values': torch.tensor([[0.0], [1.0]], dtype=torch.float32)
            }
        })
    dataset = dc.data.NumpyDataset(X=torch.tensor([[0.0], [1.0]]),
                                   y=torch.tensor([[0.0], [1.0]]))
    loss = pinn.fit(dataset, nb_epoch=300)
    assert loss < 1e-3, "Model can't overfit"


@pytest.mark.torch
def test_pinn_default_model():
    """Test if default model works correctly"""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pinn = PINNModel(in_channels=x.shape[1])

    assert pinn.model is not None, "Default model is not initialized"
    output = pinn.predict(x)
    assert output is not None, "Error in prediction using default untrained model"
    assert output.shape[0] == x.shape[0], "Output shape mismatch"


@pytest.mark.torch
def test_pinn_custom_eval():
    """Test if custom evaluation function works correctly"""
    model = torch.nn.Linear(1, 1)

    def custom_eval(x):
        return 2 * model(x)

    pinn = PINNModel(model=model, eval_fn=custom_eval)

    test_dataset = torch.tensor([[1.0]], dtype=torch.float32)

    with torch.no_grad():
        y_standard = model(test_dataset)
        y_custom = pinn.predict(test_dataset)

    assert np.abs(y_custom - 2 * y_standard.numpy()) < 1e-6


@pytest.mark.torch
def test_pinn_heat_equation():
    """
    Test PINNModel by solving the 1D steady-state heat equation:
    d²u/dx² = 0
    with boundary conditions:
    u(0) = 0
    u(1) = 1
    """

    class HeatNet(torch.nn.Module):

        def __init__(self):
            super(HeatNet, self).__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(1, 64),
                                           torch.nn.Tanh(),
                                           torch.nn.Linear(64, 64),
                                           torch.nn.Tanh(),
                                           torch.nn.Linear(64, 1))

        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            return self.net(x)

    def heat_equation_residual(model, x):
        """
        The heat equation is given by du/dx = alpha * d²u/dx²

        The goal is to minimize d²u/dx² - alpha * du/dx."""
        x.requires_grad_(True)
        u = model(x)

        du_dx = torch.autograd.grad(u.sum(),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)[0]

        d2u_dx2 = torch.autograd.grad(du_dx.sum(),
                                      x,
                                      create_graph=True,
                                      retain_graph=True)[0]

        return du_dx - d2u_dx2 # Let alpha be 1.0
    
    def custom_loss(outputs, labels, weights=None):
        outputs = outputs[0]
        labels = labels[0]
        
        data_loss = torch.mean(torch.square(outputs - labels))
        pde_residuals = heat_equation_residual(model, labels)
        pde_loss = torch.mean(torch.abs(pde_residuals))
        boundary_loss = 0.0
        for _, value in boundary_data.items():
            if isinstance(value, dict):
                points = value.get('points')
                values = value.get('values')
                if points is not None and values is not None:
                    pred = model(points)
                    boundary_loss += torch.mean(torch.square(pred - values))
        return data_loss + pde_loss + 10*boundary_loss

    def generate_data(n_points: int = 200):
        x_interior = torch.linspace(0, 1, n_points)[1:-1].reshape(-1, 1)
        x_boundary = torch.tensor([[0.0], [1.0]])
        x = torch.cat([x_interior, x_boundary], dim=0)
        y = x.clone()

        return x, y

    x_train, y_train = generate_data(n_points=200)
    dataset = dc.data.NumpyDataset(X=x_train.numpy(), y=y_train.numpy())

    # Boundary conditions: u(0) = 0, u(1) = 1
    boundary_data = {
        'dirichlet': {
            'points': torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            'values': torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        }
    }

    model = HeatNet()
    pinn = PINNModel(model=model,
                     pde_fn=lambda u, x: heat_equation_residual(model, x),
                     loss_fn=custom_loss,
                     boundary_data=boundary_data,
                     learning_rate=0.001,
                     batch_size=32,
                     data_weight=1.0,
                     physics_weight=1.0)

    pinn.fit(dataset, nb_epoch=1000)

    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = pinn.predict(x_test)

    y_true = x_test  # analytical solution: u(x) = x
    mse = torch.mean((y_pred - y_true)**2)

    assert mse < 1e-2, f"MSE {mse} is too high"
    assert torch.abs(
        y_pred[0]) < 1e-2, "Boundary condition at x=0 not satisfied"
    assert torch.abs(y_pred[-1] -
                     1.0) < 1e-2, "Boundary condition at x=1 not satisfied"

    x_interior = x_test[1:-1].clone().requires_grad_(True)
    residuals = heat_equation_residual(model, x_interior)
    pde_error = torch.mean(torch.abs(residuals))
    assert pde_error < 1e-2, "PDE residuals are too high"

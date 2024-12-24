import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Union
from deepchem.models.torch_models.torch_model import TorchModel
import logging

logger = logging.getLogger(__name__)


class PINNModel(TorchModel):
    """
    This model is recommended for linear partial differential equations. When accurately write
    the gradient function in PyTorch depending on your use case, then it will work as well.

    This is class is derived from the TorchModel class and methods are also very similar to TorchModel,

    This class requires two important arguments.

    [1] pde_fn : Each PINNs may have a different strategy for calculating the physics losses. This
    function tells the PINNModel how to go about computing the physics PDE residuals.
    It should follow this format:

    >>>
    >> def heat_equation_residual(model, x):
    >>    x.requires_grad_(True)
    >>    u = model(x)
    >>    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    >>    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True, retain_graph=True)[0]
    >>    return d2u_dx2

    The model argument is the neural network model that is being trained. The x argument is the input.
    You can either use the default neural network defined in the class or pass your own model as argument to the class.

    [2] boundary_data: This is a dictionary containing the boundary conditions data.
    The dictionary should have the following format:
    boundary_data = {
        'dirichlet': {
            'points': torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            'values': torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        }
    }

    References
    ----------
    .. [1] Raissi et. al. "Physics-informed neural networks: A deep learning framework for solving
        forward and inverse problems involving nonlinear partial differential equations" Journal of
        Computational Physics https://doi.org/10.1016/j.jcp.2018.10.045

    .. [2] Raissi et. al. "Physics Informed Deep Learning (Part I): Data-driven
        Solutions of Nonlinear Partial Differential Equations" arXiv preprint arXiv:1711.10561

    Notes
    -----
    This class requires PyTorch to be installed.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 in_channels: Optional[int] = None,
                 loss_fn: Optional[Callable] = None,
                 data_weight: float = 1.0,
                 boundary_data: Dict = {},
                 pde_fn: Union[List,
                               Callable] = [lambda u, x: torch.zeros_like(u)],
                 pde_weights: Union[List, float] = [1.0],
                 physics_weight: float = 1.0,
                 eval_fn: Optional[Callable] = None,
                 **kwargs) -> None:
        """Initialize PINNModel.

        Parameters
        ----------
        model: nn.Module
            PyTorch neural network model for training
        loss: Loss or LossFn
            Loss function for the data-driven part
        pde_fn: Callable
            Function that computes the PDE residuals. Should take model predictions
            and input coordinates as arguments and return the PDE residuals
        pde_weights: List[float]
            List of weights for the PDE loss terms
        boundary_data: Dict
            Dictionary containing boundary conditions data
        eval_fn: Callable, optional (default=None)
            Custom function for model evaluation during inference
        data_weight: float, optional (default=1.0)
            Weight for the data-driven loss term
        physics_weight: float, optional (default=1.0)
            Weight for the physics-informed loss term
        **kwargs:
            Additional arguments to pass to TorchModel
        """

        if model is None:
            if in_channels is not None:
                model = NeuralNet(in_channels=in_channels)
            else:
                model = NeuralNet(in_channels=1)

        if not isinstance(pde_fn, list):
            pde_fn = [pde_fn]
        if not isinstance(pde_weights, list):
            pde_weights = [pde_weights] * len(pde_fn)

        self.loss_fn = loss_fn or self._loss_fn
        self.pde_fn = [pde_fn] if not isinstance(pde_fn, list) else pde_fn
        self.boundary_data = boundary_data
        self.eval_fn = eval_fn or self._default_eval_fn
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.pde_weights = ([pde_weights] * len(self.pde_fn) if
                            not isinstance(pde_weights, list) else pde_weights)
        super(PINNModel, self).__init__(model=model,
                                        loss=self.loss_fn,
                                        **kwargs)

    def _compute_pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the physics-informed loss using PDE residuals.

        Parameters
        ----------
        x: torch.Tensor
            Input coordinates where PDE residuals should be computed

        Returns
        -------
        torch.Tensor
            Computed PDE loss
        """
        x.requires_grad_(True)
        u_pred = self.model(x)
        residual_loss = torch.tensor(0.0)
        for pde, weight in zip(self.pde_fn, self.pde_weights):
            residuals = pde(u_pred, x)
            residual_loss += weight * torch.mean(torch.square(residuals))
        return residual_loss

    def _compute_boundary_loss(self) -> torch.Tensor:
        """Compute loss at boundary points for Dirichlet, Neumann and Robin conditions.

        Returns
        -------
        torch.Tensor
            Computed boundary loss
        """
        if not self.boundary_data:
            return torch.tensor(0.0)

        boundary_loss = torch.tensor(0.0)
        for key, value in self.boundary_data.items():
            if isinstance(value, dict):
                points = value.get('points')
                values = value.get('values')
                alpha = value.get('alpha', 1.0)
                beta = value.get('beta', 1.0)
                if points is not None and values is not None:
                    pred = self.model(points)
                    if key.lower() == 'dirichlet':
                        boundary_loss += torch.mean(torch.square(pred - values))
                    elif key.lower() == 'neumann':
                        pred_grad = torch.autograd.grad(
                            pred,
                            points,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True)[0]
                        boundary_loss += torch.mean(
                            torch.square(pred_grad - values))
                    elif key.lower() == 'robin':
                        pred_grad = torch.autograd.grad(
                            pred,
                            points,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True)[0]
                        robin_loss = alpha * pred + beta * pred_grad - values
                        boundary_loss += torch.mean(torch.square(robin_loss))
                    else:
                        logger.warning(
                            f"Unknown boundary condition type: {key}. Skipping boundary value."
                        )
                        continue

        return boundary_loss

    def _default_eval_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Default evaluation function for inference."""
        with torch.no_grad():
            return self.model(x)

    def _loss_fn(self, outputs: list, labels: list,
                 weights: list) -> torch.Tensor:
        """
        This is a custom loss function that combines data, physics, and boundary losses.
        The additional physics and boundary loss are constraints that are used to enforce the model to learn the underlying physics of the system.
        The physics loss is computed using the PDE residuals, and the boundary loss is computed using the boundary conditions.

        Total loss = data_weight * data_loss + physics_weight * physics_loss + boundary_loss

        data_weight and physics_weight are the weights for the data and physics loss terms, respectively. By default both are set to 1.0.
        """
        outputs = outputs[0]
        labels = labels[0]
        data_loss = nn.MSELoss()(outputs, labels)
        physics_loss = self._compute_pde_loss(torch.Tensor(outputs))
        boundary_loss = self._compute_boundary_loss()

        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss + boundary_loss

        return total_loss

    def predict(self, dataset):
        """
        Makes predictions on dataset using the evaluation function.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make predictions on

        Returns
        -------
        Predicted values
        """
        eval_fn = self.eval_fn or self._default_eval_fn
        return eval_fn(dataset)


class NeuralNet(nn.Module):
    """A simple neural network that will be used by the PINNModel as the default model when no model is provided.
    """

    def __init__(self, in_channels: int = 1):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)

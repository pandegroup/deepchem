import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Union
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.layers import MultilayerPerceptron
import logging

logger = logging.getLogger(__name__)


class PINNModel(TorchModel):
    """
    This model is designed for solving linear partial differential equations (PDEs) using
    Physics-Informed Neural Networks (PINNs). It extends the TorchModel class, and its
    methods are similar to those in TorchModel, with additional functionality for handling
    physics-based constraints.

    Parameters
    ----------
    pde_fn : callable
        A function that defines the physics PDE residuals. Each PINN may have a unique
        strategy for calculating the physics losses, and this function specifies how
        the PINNModel computes the PDE residuals. The function should follow this format:

        >>> def heat_equation_residual(model, x):
        >>>     x.requires_grad_(True)
        >>>     u = model(x)
        >>>     du_dx = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
        >>>     d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True, retain_graph=True)[0]
        >>>     return d2u_dx2

        Here, `model` is the neural network being trained, and `x` is the input.

    boundary_data : dict
        A dictionary containing the boundary condition data. It should have the following format:

        >>> boundary_data = {
        >>>     'dirichlet': {
        >>>         'points': torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        >>>         'values': torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        >>>     }
        >>> }

        - `points`: Tensor of input points where boundary conditions are defined.
        - `values`: Tensor of target values at the boundary points.

    loss_fn : callable
        A custom loss function that combines data, physics, and boundary losses.
        An example is shown below:

        >>> def custom_loss(outputs, labels, weights=None):
        >>>     outputs = outputs[0]
        >>>     labels = labels[0]
        >>>     data_loss = torch.mean(torch.square(outputs - labels))
        >>>     pde_residuals = heat_equation_residual(model, labels)
        >>>     pde_loss = torch.mean(torch.abs(pde_residuals))
        >>>     boundary_loss = 0.0
        >>>     for _, value in boundary_data.items():
        >>>         if isinstance(value, dict):
        >>>             points = value.get('points')
        >>>             values = value.get('values')
        >>>             if points is not None and values is not None:
        >>>                 pred = model(points)
        >>>                 boundary_loss += torch.mean(torch.square(pred - values))
        >>>     return data_loss + pde_loss + 10 * boundary_loss

    References
    ----------
    [1] Raissi et al. "Physics-informed neural networks: A deep learning framework for solving
        forward and inverse problems involving nonlinear partial differential equations."
        Journal of Computational Physics, https://doi.org/10.1016/j.jcp.2018.10.045

    [2] Raissi et al. "Physics-Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear
        Partial Differential Equations." arXiv preprint arXiv:1711.10561

    Notes
    -----
    - This class requires PyTorch to be installed.
    - Users can use the default neural network provided by the class or pass a custom model.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 in_features: Optional[int] = None,
                 loss_fn: Optional[Callable] = None,
                 pde_fn: Union[List,
                               Callable] = [lambda u, x: torch.zeros_like(u)],
                 pde_weights: Union[List, float] = [1.0],
                 boundary_data: Dict = {},
                 data_weight: float = 1.0,
                 physics_weight: float = 1.0,
                 eval_fn: Optional[Callable] = None,
                 **kwargs) -> None:
        """
        Initialize PINNModel.

        Parameters
        ----------
        model : nn.Module, optional
            PyTorch neural network model for training. If not provided, a default neural network is used for regression.
        in_features : int, optional
            Number of input features for the default model. Ignored if a custom model is provided.
        loss_fn : Callable
            Loss function for the data-driven part of the training.
        pde_fn : Callable or List[Callable]
            Function(s) that compute the PDE residuals. Should take model predictions and input
            coordinates as arguments and return the PDE residuals.
        pde_weights : float or List[float]
            Weights for each PDE when there are multiple. If a single value is provided, it is applied to all PDE terms.
        boundary_data : Dict
            Dictionary containing boundary condition data.
        data_weight : float, optional, default=1.0
            Weight for the data-driven loss term in the total loss computation.
        physics_weight : float, optional, default=1.0
            Weight for the physics-informed loss term in the total loss computation.
        eval_fn : Callable, optional
            Custom function for model evaluation during inference. If not provided, a default
            evaluation function is used.
        **kwargs :
            Additional arguments passed to the parent `TorchModel` class.
        """

        if model is None:
            if in_features is not None:
                model = MultilayerPerceptron(
                    d_input=in_features,
                    d_hidden=(100, 100, 100, 100, 100),
                    d_output=1,
                    activation_fn='tanh')  # Regression by default
            else:
                model = MultilayerPerceptron(d_input=1,
                                             d_hidden=(100, 100, 100, 100, 100),
                                             d_output=1,
                                             activation_fn='tanh')

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
        """
        Compute the physics-informed loss using PDE residuals.

        The physics loss ensures that the model satisfies the given PDEs by minimizing
        the residuals computed for the input coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates where PDE residuals are computed. Gradients are enabled
            on `x` to allow for backpropagation through the PDE computations.

        Returns
        -------
        torch.Tensor
            The computed physics-informed loss.
        """

        x.requires_grad_(True)
        u_pred = self.model(x)
        residual_loss = torch.tensor(0.0)
        for pde, weight in zip(self.pde_fn, self.pde_weights):
            residuals = pde(u_pred, x)
            residual_loss += weight * torch.mean(torch.square(residuals))
        return residual_loss

    def _compute_boundary_loss(self) -> torch.Tensor:
        """
        Compute the loss at boundary points for Dirichlet, Neumann, and Robin conditions.

        This loss ensures that the model satisfies the specified boundary conditions:
            - Dirichlet: Model outputs match the target values at boundary points.
            - Neumann: Gradients of model outputs match target values at boundary points.
            - Robin: A linear combination of model outputs and their gradients matches target values.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            The computed boundary loss.
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
        Custom loss function that combines data, physics, and boundary losses.

        The total loss is calculated as:
            Total loss = data_weight * data_loss + physics_weight * physics_loss + boundary_loss

        Parameters
        ----------
        outputs : list
            The model's predictions.
        labels : list
            The ground truth values corresponding to the predictions.
        weights : list
            Optional weights for the loss terms.

        Returns
        -------
        torch.Tensor
            The computed total loss.
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
        return self.eval_fn(dataset)

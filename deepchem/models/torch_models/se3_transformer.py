import torch
import torch.nn as nn
import deepchem as dc
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from typing import Tuple, Iterable, List
from deepchem.models.torch_models.layers import SE3Attention


class SE3TransformerLayers(nn.Module):
    """
    Multi-layer SE(3) Transformer for processing 3D molecular data.

    Parameters
    ----------
    embed_dim: int
        Dimensionality of feature embeddings.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of SE(3) Attention layers.

    Example
    -------
    >>> from deepchem.models.torch_models import SE3TransformerLayers
    >>> embed_dim, num_heads, num_layers = 64, 4, 3
    >>> transformer = SE3TransformerLayers(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
    >>> x = torch.randn(1, 10, embed_dim)  # Input feature tensor
    >>> coords = torch.randn(1, 10, 3)     # Input coordinates
    >>> features, coords = transformer(x, coords)
    >>> features.shape, coords.shape
    (torch.Size([1, 10, 64]), torch.Size([1, 10, 3]))
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super(SE3TransformerLayers, self).__init__()
        self.layers = nn.ModuleList(
            [SE3Attention(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor,
                coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SE(3) Transformer.

        Parameters
        ----------
        x: torch.Tensor
            Input feature tensor of shape `(B, N, embed_dim)`.
        coords: torch.Tensor
            Input coordinate tensor of shape `(B, N, 3)`.

        Returns
        -------

        Tuple of Updated feature tensor of shape `(B, N, embed_dim)` and updated coordinate tensor of shape `(B, N, 3)`.
        """
        for layer in self.layers:
            x, coords = layer(x, coords)
        return x, coords


class SE3Transformer(nn.Module):
    """
    SE(3) Transformer model with a final output layer.

    Parameters
    ----------
    embed_dim: int
        Dimensionality of feature embeddings.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of SE(3) Attention layers.

    Example
    -------
    >>> from deepchem.models.torch_models import SE3Transformer
    >>> embed_dim, num_heads, num_layers = 64, 4, 3
    >>> transformer = SE3Transformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
    >>> x = torch.randn(1, 10, embed_dim + 3)  # Combined input features and coordinates
    >>> predictions = transformer(x)
    >>> predictions.shape
    torch.Size([1, 1])
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super(SE3Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.model = SE3TransformerLayers(embed_dim, num_heads, num_layers)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DeepChem-compatible SE(3) Transformer.

        Parameters
        ----------
        X: torch.Tensor
            Combined input tensor of shape `(batch_size, num_atoms, embed_dim + 3)`.

        Returns
        -------
        torch.Tensor
            Predicted output of shape `(batch_size, 1)`.
        """
        features = X[..., :self.embed_dim]
        coords = X[..., self.embed_dim:]
        updated_features, _ = self.model(features, coords)
        pooled_features = updated_features.mean(dim=1)
        outputs = self.output_layer(pooled_features)
        return outputs


class SE3TransformerModel(TorchModel):
    """
    SE(3) Transformer wrapper. This model employs SE(3) equivariant layers to predict
    molecular properties by learning features that respect the 3D spatial structure of
    molecules. Inspired by [equi1], the model leverages SE(3)-equivariant architectures
    to address 3D spatial symmetries, with the potential to further integrate graph-based
    molecular representations to capture relational and topological data effectively.

    Parameters
    ----------
    embed_dim: int
        Dimensionality of feature embeddings.
    num_heads: int
        Number of attention heads.
    num_layers: int
        Number of SE(3) Attention layers.
    mode: str
        Mode of operation ('fit', 'predict', or 'evaluate').

    References
    ----------
    .. [equi1] Ganea, O. E., Grote, A., & Barzilay, R. (2021).
    Independent SE(3)-Equivariant Models for End-to-End Rigid Protein Docking.
    arXiv preprint arXiv:2006.10503.

    Example
    -------
    >>> from deepchem.models.torch_models import SE3TransformerModel
    >>> import numpy as np
    >>> embed_dim, num_heads, num_layers, batch_size = 64, 4, 3, 32
    >>> model = SE3TransformerModel(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, batch_size=batch_size)
    >>> X = np.random.rand(100, 10, embed_dim + 3).astype(np.float32)  # Features + coordinates
    >>> y = np.random.rand(100, 1).astype(np.float32)  # Target values
    >>> w = np.ones((100, 1)).astype(np.float32)
    >>> train_dataset = dc.data.NumpyDataset(X=X, y=y, w=w)
    >>> _ = model.fit(train_dataset, nb_epoch=10)
    >>> preds = model.predict(train_dataset)
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 mode: str = 'fit',
                 **kwargs) -> None:
        loss = L2Loss()
        model = SE3Transformer(embed_dim, num_heads, num_layers)
        super(SE3TransformerModel, self).__init__(model, loss=loss, **kwargs)

    def default_generator(
        self,
        dataset: dc.data.Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = True,
    ) -> Iterable[Tuple[List, List, List]]:
        """
        Generator for batching data during training, validation, or prediction.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Input dataset.
        epochs: int
            Number of epochs.
        mode: str
            Operation mode ('fit', 'predict', or 'evaluate').
        deterministic: bool
            Whether to shuffle data deterministically.
        pad_batches: bool
            Whether to pad batches to ensure consistent sizes.

        Returns
        -------
        Iterator which walks over the batches
        """
        for epoch in range(epochs):
            for X_b, y_b, w_b, _ in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):
                y_b = y_b.reshape(-1, 1)
                w_b = w_b.reshape(-1, 1)
                yield ([X_b], [y_b], [w_b])

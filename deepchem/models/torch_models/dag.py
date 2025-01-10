from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss
import torch
import deepchem as dc
from deepchem.models.losses import _make_pytorch_shapes_consistent
from deepchem.models.torch_models.layers import DAGLayer, DAGGather
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.metrics import to_one_hot


class DAG(nn.Module):
    """Directed Acyclic Graph models for molecular property prediction.
    
    PyTorch implementation of the DAG model described in:
    Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. "Deep architectures and deep learning in 
    chemoinformatics: the prediction of aqueous solubility for drug-like molecules." 
    Journal of chemical information and modeling 53.7 (2013): 1563-1575.
    """

    def __init__(self,
                 n_tasks,
                 max_atoms=50,
                 n_atom_feat=75,
                 n_graph_feat=30,
                 n_outputs=30,
                 layer_sizes=[100],
                 layer_sizes_gather=[100],
                 dropout=None,
                 mode="classification",
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        max_atoms: int, optional
            Maximum number of atoms in a molecule, should be defined based on dataset.
        n_atom_feat: int, optional
            Number of features per atom.
        n_graph_feat: int, optional
            Number of features for atom in the graph.
        n_outputs: int, optional
            Number of features for each molecule.
        layer_sizes: list of int, optional
            List of hidden layer size(s) in the propagation step.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs to enable uncertainty prediction
        """
        super(DAG, self).__init__()

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        if uncertainty and mode != "regression":
            raise ValueError("Uncertainty is only supported in regression mode")
        if uncertainty and (dropout is None or dropout == 0.0):
            raise ValueError('Dropout must be included to predict uncertainty')

        self.losses = []
        self.n_tasks = n_tasks
        self.max_atoms = max_atoms
        self.n_atom_feat = n_atom_feat
        self.n_graph_feat = n_graph_feat
        self.n_outputs = n_outputs
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty

        # DAG layers
        self.dag_layer = DAGLayer(n_graph_feat=self.n_graph_feat,
                                  n_atom_feat=self.n_atom_feat,
                                  max_atoms=self.max_atoms,
                                  layer_sizes=layer_sizes,
                                  dropout=dropout,
                                  batch_size=batch_size)

        # Gather layer
        self.dag_gather = DAGGather(n_graph_feat=self.n_graph_feat,
                                    n_outputs=self.n_outputs,
                                    max_atoms=self.max_atoms,
                                    layer_sizes=layer_sizes_gather,
                                    dropout=dropout)

        # Output layers
        if self.mode == 'classification':
            self.dense = nn.Linear(n_outputs, n_tasks * n_classes)
        else:
            self.dense = nn.Linear(n_outputs, n_tasks)
            if uncertainty:
                self.log_var = nn.Linear(n_outputs, n_tasks)

    def forward(self, inputs):
        """Forward pass through the model."""
        atoms_all, parents_all, calculation_orders, calculation_masks, membership, n_atoms = inputs
        # Propagate information through the graph
        daglayer = self.dag_layer([
            atoms_all, parents_all, calculation_orders, calculation_masks,
            n_atoms
        ])
        membership = membership.long()
        # Gather information from the graph
        dagather = self.dag_gather([daglayer, membership])

        # Output layer
        output = self.dense(dagather)

        if self.mode == 'classification':
            logits = output.view(-1, self.n_tasks, self.n_classes)
            output = F.softmax(logits, dim=-1)
            return [output, logits]
        else:
            if self.uncertainty:
                log_var = self.log_var(dagather)
                var = torch.exp(log_var)
                return [output, var, output, log_var]
            else:
                return [output]


class DAGModel(TorchModel):

    def __init__(self,
                 n_tasks,
                 max_atoms=50,
                 n_atom_feat=75,
                 n_graph_feat=30,
                 n_outputs=30,
                 layer_sizes=[100],
                 layer_sizes_gather=[100],
                 dropout=None,
                 mode="classification",
                 n_classes=2,
                 uncertainty=False,
                 batch_size=100,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        max_atoms: int, optional
            Maximum number of atoms in a molecule, should be defined based on dataset.
        n_atom_feat: int, optional
            Number of features per atom.
        n_graph_feat: int, optional
            Number of features for atom in the graph.
        n_outputs: int, optional
            Number of features for each molecule.
        layer_sizes: list of int, optional
            List of hidden layer size(s) in the propagation step.
        layer_sizes_gather: list of int, optional
            List of hidden layer size(s) in the gather step.
        dropout: None or float, optional
            Dropout probability.
        mode: str, optional
            Either "classification" or "regression" for type of model.
        n_classes: int, optional
            the number of classes to predict (only used in classification mode)
        uncertainty: bool, optional
            if True, include extra outputs to enable uncertainty prediction
        """
        self.model = DAG(n_tasks=n_tasks,
                         max_atoms=max_atoms,
                         n_atom_feat=n_atom_feat,
                         n_graph_feat=n_graph_feat,
                         n_outputs=n_outputs,
                         layer_sizes=layer_sizes,
                         layer_sizes_gather=layer_sizes_gather,
                         dropout=dropout,
                         mode=mode,
                         n_classes=n_classes,
                         uncertainty=uncertainty,
                         batch_size=batch_size)
        self.n_tasks = n_tasks
        self.mode = mode
        self.max_atoms = max_atoms
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        if mode == 'classification':
            self.output_types = ['prediction', 'loss']
            self.loss = SoftmaxCrossEntropy()
        else:
            if uncertainty:
                self.log_var = nn.Linear(n_outputs, n_tasks)
                self.output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    # Ensure outputs and labels are shape-consistent
                    output, labels = _make_pytorch_shapes_consistent(
                        outputs[0], labels[0])
                    # Compute the losses
                    losses = (output - labels)**2 / torch.exp(
                        outputs[1]) + outputs[1]

                    # Handle weights reshaping if necessary
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        shape = tuple(w.shape)
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = w.view(*shape,
                                   *([1] * (len(losses.shape) - len(w.shape))))

                    # Compute the weighted mean loss and add model-specific losses
                    return torch.mean(losses * w) + sum(self.model.losses)

                self.loss = loss
            else:
                self.output_types = ['prediction']
                self.loss = L2Loss()
        super(DAGModel, self).__init__(self.model,
                                       loss=self.loss,
                                       output_types=self.output_types,
                                       batch_size=batch_size,
                                       **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        """Convert a dataset into the tensors needed for learning.
        
        Parameters
        ----------
        dataset : object
            The dataset to iterate over
        epochs : int, optional
            Number of epochs to iterate
        deterministic : bool, optional
            Whether to iterate over the dataset deterministically
        pad_batches : bool, optional
            Whether to pad the batches to the same size
            
        Yields
        ------
        tuple
            A tuple of (inputs, labels, weights) as expected by TorchModel.fit()
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):

                # Convert labels for classification
                if y_b is not None and self.mode == 'classification':
                    y_b = np.array(y_b.flatten())
                    y_b = to_one_hot(y_b, self.n_classes)
                    y_b = np.reshape(y_b, (-1, self.n_tasks, self.n_classes))

                # Process molecular graphs
                atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
                n_atoms = sum(atoms_per_mol)
                start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

                atoms_all = []
                parents_all = []
                calculation_orders = []
                calculation_masks = []
                membership = []

                for idm, mol in enumerate(X_b):
                    atoms_all.append(mol.get_atom_features())
                    parents = mol.parents
                    parents_all.extend(parents)
                    calculation_index = np.array(parents)[:, :, 0]
                    mask = np.array(calculation_index - self.max_atoms,
                                    dtype=bool)
                    calculation_orders.append(calculation_index +
                                              start_index[idm])
                    calculation_masks.append(mask)
                    membership.extend([idm] * atoms_per_mol[idm])

                # Create inputs tuple
                inputs = [
                    np.concatenate(atoms_all, axis=0),
                    np.stack(parents_all, axis=0),
                    np.concatenate(calculation_orders, axis=0),
                    np.concatenate(calculation_masks, axis=0),
                    np.array(membership),
                    np.array(n_atoms)
                ]

                yield inputs, [y_b], [w_b]

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss, SparseSoftmaxCrossEntropy
import dgl
from dgl.nn import GraphConv, GATv2Conv

class ChemBERTaGNN(nn.Module):
    """
    A multimodal neural network for molecular property prediction, combining SMILES-based embeddings
    (from ChemBERTa-2) and graph-based embeddings (from a GNN like GCN or GATv2).

    The model integrates the following components:
    - A ChemBERTa-2 model to process SMILES representations into latent embeddings.
    - A GNN (GCN or GATv2) to process molecular graphs into graph-level embeddings.
    - A fusion mechanism to combine SMILES and graph embeddings.
    - A final feedforward neural network to predict molecular properties.

    This model can be used for regression or classification tasks.

    Examples
    --------
    >>> from deepchem.models.torch_models.torch_model import TorchModel
    >>> from deepchem.models.losses import L2Loss
    >>> import torch
    >>> import dgl
    >>> smiles_inputs = {'input_ids': torch.randint(0, 50265, (4, 128)), 'attention_mask': torch.ones((4, 128))}
    >>> graph = dgl.batch([dgl.graph(([0, 1], [1, 2])) for _ in range(4)])
    >>> graph.ndata['x'] = torch.rand(6, 30)  # Node features
    >>> model = ChemBERTaGNN(bert_model_name="seyonec/ChemBERTa-2", gnn_type="GATv2", mode="regression", n_tasks=1)
    >>> outputs = model(smiles_inputs, graph)
    >>> print(outputs.shape)  # Output shape is (batch_size, n_tasks) for regression.

    Parameters
    ----------
    bert_model_name : str
        Name of the pretrained ChemBERTa-2 model from HuggingFace.
    gnn_type : str
        Type of GNN layer to use ('GCN' or 'GATv2').
    node_feat_dim : int
        Number of input features for graph nodes.
    edge_feat_dim : int
        Number of input features for graph edges (if used).
    hidden_dim : int
        Dimension of the hidden layers in the GNN and fusion networks.
    n_tasks : int
        Number of tasks for the output layer.
    mode : str
        The task type: 'regression' or 'classification'.
    n_classes : int
        Number of output classes (for classification tasks only).

    Returns
    -------
    torch.Tensor
        The predictions of the model. For regression, this has shape (batch_size, n_tasks).
        For classification, this has shape (batch_size, n_tasks, n_classes).
    """
    def __init__(self, bert_model_name, gnn_type="GCN", node_feat_dim=30, edge_feat_dim=11,
                 hidden_dim=128, n_tasks=1, mode='regression', n_classes=2):
        super(ChemBERTaGNN, self).__init__()
        assert mode in ['regression', 'classification'], "Mode must be 'regression' or 'classification'"
        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        # Initialize ChemBERTa for SMILES-based embeddings
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert_model.config.hidden_size

        # Initialize the GNN (GCN or GATv2) for graph-based embeddings
        if gnn_type == "GCN":
            self.gnn_layer = GraphConv(node_feat_dim, hidden_dim)
        elif gnn_type == "GATv2":
            self.gnn_layer = GATv2Conv(node_feat_dim, hidden_dim, num_heads=4)
        else:
            raise ValueError("Invalid GNN type. Choose 'GCN' or 'GATv2'.")

        # Fusion layer to combine SMILES and graph embeddings
        self.fusion_layer = nn.Linear(self.bert_hidden_size + hidden_dim, hidden_dim)

        # Output layer for regression or classification
        self.prediction_layer = nn.Linear(hidden_dim, n_tasks * n_classes if mode == 'classification' else n_tasks)

    def forward(self, smiles, graph):
        """
        Forward pass through the ChemBERTaGNN model.

        Parameters
        ----------
        smiles : dict
            Dictionary of tokenized SMILES inputs with keys 'input_ids' and 'attention_mask'.
        graph : DGLGraph
            Batched DGLGraph object with node features stored in `graph.ndata['x']`.

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, n_tasks) for regression or
            (batch_size, n_tasks, n_classes) for classification.
        """
        # Extract SMILES embeddings using ChemBERTa
        bert_output = self.bert_model(**smiles)
        smiles_embeddings = bert_output.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Process graph embeddings using the GNN
        node_features = graph.ndata['x']
        gnn_output = self.gnn_layer(graph, node_features)
        graph_embeddings = dgl.mean_nodes(graph, 'x')  # Graph-level embedding via mean pooling

        # Combine SMILES and graph embeddings
        fused_representation = torch.cat([smiles_embeddings, graph_embeddings], dim=-1)
        fused_representation = F.relu(self.fusion_layer(fused_representation))

        # Make predictions
        predictions = self.prediction_layer(fused_representation)
        if self.mode == 'classification':
            predictions = predictions.view(-1, self.n_tasks, self.n_classes)
            probabilities = F.softmax(predictions, dim=-1)
            return probabilities, predictions
        return predictions


class MolPROPModel(TorchModel):
    """
    TorchModel wrapper for ChemBERTaGNN to integrate with DeepChem's API.
    """

    def __init__(self, bert_model_name="seyonec/ChemBERTa-2", gnn_type="GCN", mode="regression",
                 n_tasks=1, n_classes=2, **kwargs):
        """
        Parameters
        ----------
        bert_model_name : str
            HuggingFace model name for ChemBERTa-2.
        gnn_type : str
            Type of GNN layer ('GCN' or 'GATv2').
        mode : str
            'regression' or 'classification'.
        n_tasks : int
            Number of tasks for prediction.
        n_classes : int
            Number of classes for classification.
        kwargs : dict
            Additional arguments for TorchModel.
        """
        model = ChemBERTaGNN(bert_model_name=bert_model_name, gnn_type=gnn_type, mode=mode,
                             n_tasks=n_tasks, n_classes=n_classes)
        loss = L2Loss() if mode == "regression" else SparseSoftmaxCrossEntropy()
        output_types = ['prediction'] if mode == "regression" else ['prediction', 'loss']
        super(MolPROPModel, self).__init__(model, loss=loss, output_types=output_types, **kwargs)

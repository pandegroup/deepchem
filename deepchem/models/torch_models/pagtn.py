"""
DGL-based PAGTN for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class Pagtn(nn.Module):
  """Model for Graph Property Prediction

  This model proceeds as follows:

  * Update node representations in graphs with a variant of GAT, where a
    linear additive form of attention is applied. Attention Weights are derived
    by concatenating the node and edge features for each bond.
  * Update node representations with multiple rounds of message passing.
  * For each layer has, residual connections with its previous layer.
  * The final molecular representation is computed by combining the representations
    of all nodes in the molecule.
  * Perform the final prediction using a linear layer

  Examples
  --------

  >>> import deepchem as dc
  >>> import dgl
  >>> from deepchem.models import Pagtn
  >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
  >>> featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
  >>> graphs = featurizer.featurize(smiles)
  >>> print(type(graphs[0]))
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> dgl_graphs = [graphs[i].to_dgl_graph() for i in range(len(graphs))]
  >>> batch_dgl_graph = dgl.batch(dgl_graphs)
  >>> model = Pagtn(n_tasks=1, mode='regression')
  >>> preds = model(batch_dgl_graph)
  >>> print(type(preds))
  <class 'torch.Tensor'>
  >>> preds.shape == (2, 1)
  True

  References
  ----------
  .. [1] Benson Chen, Regina Barzilay, Tommi Jaakkola. "Path-Augmented
         Graph Transformer Network." arXiv:1905.12712

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               number_atom_features: int = 94,
               number_bond_features: int = 42,
               mode: str = 'regression',
               n_classes: int = 2,
               output_node_features: int = 256,
               hidden_features: int = 32,
               num_layers: int = 5,
               num_heads: int = 1,
               dropout: float = 0.1,
               nfeat_name: str = 'x',
               efeat_name: str = 'edge_attr',
               pool_mode: str = 'sum'):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    number_atom_features : int
      Size for the input node features. Default to 94.
    number_bond_features : int
      Size for the input edge features. Default to 42.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    output_node_features : int
      Size for the output node features in PAGTN layers. Default to 256.
    hidden_features : int
      Size for the hidden node features in PAGTN layers. Default to 32.
    num_layers : int
      Number of PAGTN layers to be applied. Default to 5.
    num_heads : int
      Number of attention heads. Default to 1.
    dropout : float
      The probability for performing dropout. Default to 0.1
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
    efeat_name: str
      For an input graph ``g``, the model assumes that it stores edge features in
      ``g.edata[efeat_name]`` and will retrieve input edge features from that.
      Default to 'edge_attr'.
    pool_mode : 'max' or 'mean' or 'sum'
      Whether to compute elementwise maximum, mean or sum of the node representations.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')
    try:
      import dgllife
    except:
      raise ImportError('This class requires dgllife.')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    super(Pagtn, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    self.efeat_name = efeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import PAGTNPredictor as DGLPAGTNPredictor

    self.model = DGLPAGTNPredictor(
        node_in_feats=number_atom_features,
        node_out_feats=output_node_features,
        node_hid_feats=hidden_features,
        edge_feats=number_bond_features,
        depth=num_layers,
        nheads=num_heads,
        dropout=dropout,
        n_tasks=out_size,
        mode=pool_mode)

  def forward(self, g):
    """Predict graph labels

    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
      ``dgl_graph.edata[self.efeat_name]``.

    Returns
    -------
    torch.Tensor
      The model output.

      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be
        ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
        its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
    torch.Tensor, optional
      This is only returned when self.mode = 'classification', the output consists of the
      logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    edge_feats = g.edata[self.efeat_name]
    out = self.model(g, node_feats, edge_feats)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    else:
      return out


class PagtnModel(TorchModel):
  """Model for Graph Property Prediction.

  This model proceeds as follows:

  * Update node representations in graphs with a variant of GAT, where a
    linear additive form of attention is applied. Attention Weights are derived
    by concatenating the node and edge features for each bond.
  * Update node representations with multiple rounds of message passing.
  * For each layer has, residual connections with its previous layer.
  * The final molecular representation is computed by combining the representations
    of all nodes in the molecule.
  * Perform the final prediction using a linear layer

  Examples
  --------

  >>>
  >> import deepchem as dc
  >> from deepchem.models import PagtnModel
  >> featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
  >> tasks, datasets, transformers = dc.molnet.load_tox21(
  ..     reload=False, featurizer=featurizer, transformers=[])
  >> train, valid, test = datasets
  >> model = PagtnModel(mode='classification', n_tasks=len(tasks),
  ..                    batch_size=16, learning_rate=0.001)
  >> model.fit(train, nb_epoch=50)

  References
  ----------
  .. [1] Benson Chen, Regina Barzilay, Tommi Jaakkola. "Path-Augmented
         Graph Transformer Network." arXiv:1905.12712

  Notes
  -----
  This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
  (https://github.com/awslabs/dgl-lifesci) to be installed.
  """

  def __init__(self,
               n_tasks: int,
               number_atom_features: int = 94,
               number_bond_features: int = 42,
               mode: str = 'regression',
               n_classes: int = 2,
               output_node_features: int = 256,
               hidden_features: int = 32,
               num_layers: int = 5,
               num_heads: int = 1,
               dropout: float = 0.1,
               pool_mode: str = 'sum',
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    number_atom_features : int
      Size for the input node features. Default to 94.
    number_bond_features : int
      Size for the input edge features. Default to 42.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    output_node_features : int
      Size for the output node features in PAGTN layers. Default to 256.
    hidden_features : int
      Size for the hidden node features in PAGTN layers. Default to 32.
    num_layers: int
      Number of graph neural network layers, i.e. number of rounds of message passing.
      Default to 2.
    num_heads : int
      Number of attention heads. Default to 1.
    dropout: float
      Dropout probability. Default to 0.1
    pool_mode : 'max' or 'mean' or 'sum'
      Whether to compute elementwise maximum, mean or sum of the node representations.
    kwargs
      This can include any keyword argument of TorchModel.
    """
    model = Pagtn(
        n_tasks=n_tasks,
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        mode=mode,
        n_classes=n_classes,
        output_node_features=output_node_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pool_mode=pool_mode)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(PagtnModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    """Create batch data for Pagtn.

    Parameters
    ----------
    batch: tuple
      The tuple is ``(inputs, labels, weights)``.

    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: list of torch.Tensor or None
      The graph labels.
    weights: list of torch.Tensor or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [graph.to_dgl_graph() for graph in inputs[0]]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(PagtnModel, self)._prepare_batch(([], labels,
                                                                 weights))
    return inputs, labels, weights

from typing import Optional, List, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from deepchem.models.torch_models import WGANModel
from deepchem.models.torch_models.layers import MolGANEncoderLayer


class BasicMolGANModel(WGANModel):
    """
    Model for de-novo generation of small molecules based on work of Nicola De Cao et al. [molgan1]_.
    It uses a GAN directly on graph data and a reinforcement learning objective to induce the network to generate molecules with certain chemical properties.
    Utilizes WGAN infrastructure; uses adjacency matrix and node features as inputs.
    Inputs need to be one-hot representation.

    Examples
    --------

    Import necessary libraries and modules

    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import BasicMolGANModel as MolGAN
    >>> from deepchem.models.optimizers import ExponentialDecay
    >>> import torch
    >>> import torch.nn.functional as F

    Load dataset and featurize molecules
    We will use a small dataset for this example.
    We will be using `MolGanFeaturizer` to featurize the molecules.

    >>> smiles = ['CCC', 'C1=CC=CC=C1', 'CNC' ]
    >>> # create featurizer
    >>> feat = dc.feat.MolGanFeaturizer()
    >>> # featurize molecules
    >>> features = feat.featurize(smiles)
    >>> # Remove empty objects
    >>> features = list(filter(lambda x: x is not None, features))

    Create and train the model

    >>> # create model
    >>> gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000))
    >>> dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],[x.node_features for x in features])
    >>> def iterbatches(epochs):
    ...     for i in range(epochs):
    ...         for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
    ...             adjacency_tensor = F.one_hot(
    ...                     torch.Tensor(batch[0]).to(torch.int64),
    ...                     gan.edges).to(torch.float32)
    ...             node_tensor = F.one_hot(
    ...                     torch.Tensor(batch[1]).to(torch.int64),
    ...                     gan.nodes).to(torch.float32)
    ...             yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}
    >>> # train model
    >>> gan.fit_gan(iterbatches(8), generator_steps=0.2, checkpoint_interval=0)

    You can change the above parameters to get better results. The above example is just a simple example to show how to use the model.
    You can try `iterbatches(1000)` for better results.

    Now, let's generate some molecules using the trained model
    We will generate 10 molecules and then convert them to RDKit molecules.

    >>> generated_data = gan.predict_gan_generator(10)
    Generating 10 samples
    >>> # convert graphs to RDKitmolecules
    >>> nmols = feat.defeaturize(generated_data)
    >>> print("{} molecules generated".format(len(nmols)))
    10 molecules generated

    You can increase the number of generated molecules by changing the parameter in `predict_gan_generator` function.
    Generated molecules are in the form of GraphMatrix. You can convert them to RDKit molecules using `defeaturize` function of MolGanFeaturizer.

    Now, let's remove invalid molecules from the generated molecules.

    >>> # remove invalid moles
    >>> nmols = list(filter(lambda x: x is not None, nmols))
    >>> print ("{} valid molecules".format(len(nmols)))
    0 valid molecules

    We can see that currently training is unstable and 0 is a common outcome. You can try training the model with different parameters to get better results.

    References
    ----------
    .. [molgan1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 edges: int = 5,
                 vertices: int = 9,
                 nodes: int = 5,
                 embedding_dim: int = 10,
                 dropout_rate: float = 0.0,
                 mode: str = "",
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize the model

        Parameters
        ----------
        edges: int, default 5
            Number of bond types includes BondType.Zero
        vertices: int, default 9
            Max number of atoms in adjacency and node features matrices
        nodes: int, default 5
            Number of atom types in node features matrix
        embedding_dim: int, default 10
            Size of noise input array
        dropout_rate: float, default = 0.
            Rate of dropout used across whole model
        mode: str, default = ""
            The mode for generation used during training,
            "" => normal softmax
            "gumbel" => softmax + gumbel noise
            "straight" => straight-through estimator
        name: str, default ''
            Name of the model
        """

        self.edges = edges
        self.vertices = vertices
        self.nodes = nodes
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.mode = mode
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        super(BasicMolGANModel, self).__init__(device=device, **kwargs)

    def get_noise_input_shape(self) -> Tuple[int, int]:
        """
        Return shape of the noise input used in generator

        Returns
        -------
        Tuple
            Shape of the noise input
        """

        return (
            1,
            self.embedding_dim,
        )

    def get_data_input_shapes(self) -> List:
        """
        Return input shape of the discriminator

        Returns
        -------
        List
            List of shapes used as an input for distriminator.
        """
        return [
            (1, self.vertices, self.vertices, self.edges),
            (1, self.vertices, self.nodes),
        ]

    def create_generator(self):
        """
        Create generator model.
        Take noise data as an input and processes it through number of
        dense and dropout layers. Then data is converted into two forms
        one used for training and other for generation of compounds.
        The model has two outputs:
            1. edges
            2. nodes

        The format differs depending on intended use (training or sample generation).
        For sample generation use flag, sample_generation=True while calling generator
        i.e. gan.generators[0](noise_input, training=False, sample_generation=True).
        For training the model, set `sample_generation=False`
        """
        return BasicMolGANGenerator(vertices=self.vertices,
                                    edges=self.edges,
                                    nodes=self.nodes,
                                    dropout_rate=self.dropout_rate,
                                    embedding_dim=self.embedding_dim,
                                    mode=self.mode)

    def create_discriminator(self,
                             units: List[Union[Tuple[int, int],
                                               int]] = [(128, 64), 64]):
        """
        Create discriminator model based on MolGAN layers.
        Takes two inputs:
            1. adjacency tensor, containing bond information
            2. nodes tensor, containing atom information

        The input vectors need to be in one-hot encoding format.
        Use MolGAN featurizer for that purpose. It will be simplified
        in the future release.
        """

        return Discriminator(dropout_rate=self.dropout_rate,
                             units=units,
                             edges=self.edges,
                             nodes=self.nodes,
                             device=self.device)

    def predict_gan_generator(self,
                              batch_size: int = 1,
                              noise_input: Optional[Union[List,
                                                          torch.Tensor]] = None,
                              conditional_inputs: List = [],
                              generator_index: int = 0) -> List[GraphMatrix]:
        """
        Use the GAN to generate a batch of samples.

        Parameters
        ----------
        batch_size: int
            the number of samples to generate.  If either noise_input or
            conditional_inputs is specified, this argument is ignored since the batch
            size is then determined by the size of that argument.
        noise_input: array
            the value to use for the generator's noise input.  If None (the default),
            get_noise_batch() is called to generate a random input, so each call will
            produce a new set of samples.
        conditional_inputs: list of arrays
            NOT USED.
            the values to use for all conditional inputs.  This must be specified if
            the GAN has any conditional inputs.
        generator_index: int
            NOT USED.
            the index of the generator (between 0 and n_generators-1) to use for
            generating the samples.

        Returns
        -------
        List[GraphMatrix]
            Returns a list of GraphMatrix object that can be converted into
            RDKit molecules using MolGANFeaturizer defeaturize function.
        """
        if noise_input is not None:
            batch_size = len(noise_input)
        if noise_input is None:
            noise_input = self.get_noise_batch(batch_size)  # type: ignore
        noise_input = torch.tensor(noise_input,
                                   dtype=torch.float32,
                                   device=self.device)
        print(f"Generating {batch_size} samples")
        adjacency_matrix, nodes_features = self.generators[0](
            noise_input, training=False, sample_generation=True)
        graphs = [
            GraphMatrix(i, j)
            for i, j in zip(adjacency_matrix.cpu().detach().numpy(),
                            nodes_features.cpu().detach().numpy())
        ]
        return graphs


class BasicMolGANGenerator(nn.Module):
    """
    Generator class for BasicMolGAN model.
    Using subclassing rather than functional API due to requirement
    to swap between two outputs depending on situation.
    In order to get output that used for sample generation
    (conversion to rdkit molecules) pass sample_generation=True argument while
    calling the model i.e. adjacency_matrix, nodes_features = self.generators[0](
    noise_input, training=False, sample_generation=True)
    This is automatically done in predict_gan_generator().
    """

    def __init__(self,
                 vertices: int = 9,
                 edges: int = 4,
                 nodes: int = 5,
                 dropout_rate: float = 0.0,
                 embedding_dim: int = 10,
                 mode: str = "",
                 name: str = "SimpleMolGANGenerator",
                 **kwargs):
        """
        Initialize model.

        Parameters
        ----------
        vertices : int, optional
            number of max atoms dataset molecules (incl. empty atom), by default 9
        edges : int, optional
            number of bond types in molecules, by default 5
        nodes : int, optional
            number of atom types in molecules, by default 5
        dropout_rate : float, optional
            rate of dropout, by default 0.
        embedding_dim : int, optional
            noise input dimensions, by default 10
        name : str, optional
            name of the model, by default "SimpleMolGANGenerator"
        """
        super(BasicMolGANGenerator, self).__init__(**kwargs)
        self.vertices = vertices
        self.edges = edges
        self.nodes = nodes
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.mode = mode

        self.dense1 = nn.Linear(self.embedding_dim, 128)  # tanh
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(128, 256)  # tanh
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense3 = nn.Linear(256, 512)  # tanh
        self.dropout3 = nn.Dropout(dropout_rate)

        # edges logits used during training
        self.edges_dense = nn.Linear(512,
                                     self.edges * self.vertices * self.vertices)
        self.edges_dropout = nn.Dropout(dropout_rate)

        # nodes logits used during training
        self.nodes_dense = nn.Linear(512, self.vertices * self.nodes)
        self.nodes_dropout = nn.Dropout(self.dropout_rate)

    def forward(self,
                inputs: Any,
                training: bool = False,
                sample_generation: bool = False) -> List[Any]:
        """
        Call generator model

        Parameters
        ----------
        inputs : Any
            List of inputs, typically noise_batch
        training : bool, optional
            used by dropout layers, by default False
        sample_generation : bool, optional
            decide which output to use, by default False

        Returns
        -------
        List[Any, Any]
            Tensors containing either softmax values for training
            or argmax for sample generation (used for creation of rdkit molecules).
        """

        if isinstance(inputs, list):
            inputs = inputs[0]
        x = F.tanh(self.dense1(inputs))
        if training:
            x = self.dropout1(x)
        x = F.tanh(self.dense2(x))
        if training:
            x = self.dropout2(x)
        x = F.tanh(self.dense3(x))
        if training:
            x = self.dropout3(x)

        # edges logits
        edges_logits = self.edges_dense(x)
        edges_logits = edges_logits.view(-1, self.edges, self.vertices,
                                         self.vertices)
        matrix_transpose = edges_logits.permute(0, 1, 3, 2)
        edges_logits = (edges_logits + matrix_transpose) / 2
        edges_logits = edges_logits.permute(0, 2, 3, 1)
        if training:
            edges_logits = self.edges_dropout(edges_logits)

        # nodes logits
        nodes_logits = self.nodes_dense(x)
        nodes_logits = nodes_logits.view(-1, self.vertices, self.nodes)
        if training:
            nodes_logits = self.nodes_dropout(nodes_logits)

        if sample_generation is False:
            # For training
            if self.mode == "":
                edges = F.softmax(edges_logits, dim=-1)
                nodes = F.softmax(nodes_logits, dim=-1)
            elif self.mode == "gumbel":
                edges = F.gumbel_softmax(edges_logits, dim=-1)
                nodes = F.gumbel_softmax(nodes_logits, dim=-1)
            elif self.mode == "straight":
                edges = F.gumbel_softmax(edges_logits, hard=True, dim=-1)
                nodes = F.gumbel_softmax(nodes_logits, hard=True, dim=-1)
            else:
                raise NotImplementedError
        else:
            # For sample generation
            e_gumbel_logits = edges_logits - torch.log(-torch.log(
                torch.rand_like(edges_logits, dtype=edges_logits.dtype)))
            e_gumbel_argmax = F.one_hot(torch.argmax(e_gumbel_logits, dim=-1),
                                        num_classes=e_gumbel_logits.shape[-1])
            edges = torch.argmax(e_gumbel_argmax, dim=-1)

            n_gumbel_logits = nodes_logits - torch.log(-torch.log(
                torch.rand_like(nodes_logits, dtype=nodes_logits.dtype)))
            n_gumbel_argmax = F.one_hot(torch.argmax(n_gumbel_logits, dim=-1),
                                        num_classes=n_gumbel_logits.shape[-1])
            nodes = torch.argmax(n_gumbel_argmax, dim=-1)
        return [edges, nodes]


class Discriminator(nn.Module):
    """A discriminator for the MolGAN model."""

    def __init__(
        self,
        dropout_rate: float,
        units: List = [(64, 32), 128], # rn from paper, OG - [(128, 64), 64], from repo - [(128, 64), 128]
        edges: int = 4,
        nodes: int = 5,
        device: Optional[torch.device] = torch.device('cpu')
    ) -> None:
        """Initialize the discriminator.

        Parameters
        ----------
        dropout_rate : float
            Rate of dropout used across whole model
        units : List, optional
            Units for MolGAN encoder layer, by default [(128, 64), 64]
        edges : int, optional
            Edge types, by default 5
        device : Optional[torch.device], optional
            Device to use, by default torch.device('cpu')
        """
        super(Discriminator, self).__init__()
        self.dropout_rate = dropout_rate
        self.edges = edges
        self.units = units
        self.nodes = nodes
        self.device: torch.device = device  # type: ignore
        self.graph = MolGANEncoderLayer(units=self.units,
                                        dropout_rate=self.dropout_rate,
                                        edges=self.edges,
                                        nodes=self.nodes,
                                        device=self.device)

        # Define the dense layers
        self.dense1 = nn.Linear(units[1], 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(128, 1)
        # self.dense2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(dropout_rate)
        # self.dense3 = nn.Linear(64, 1)

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the discriminator.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            List of inputs, typically adjacency_tensor and node_tensor

        Returns
        -------
        torch.Tensor
            Output tensor of the discriminator
        """
        adjacency_tensor, node_tensor = inputs

        if isinstance(adjacency_tensor, list):
            adjacency_tensor = adjacency_tensor[0]
        adjacency_tensor = adjacency_tensor.to(device=self.device,
                                               dtype=torch.float32)
        node_tensor = node_tensor.to(device=self.device, dtype=torch.float32)

        graph = self.graph([adjacency_tensor, node_tensor])

        graph = graph.to(device=self.device, dtype=torch.float32)
        output = self.dense1(graph)
        output = F.tanh(output)
        output = self.dropout1(output)
        output = self.dense2(output)
        # rn from paper
        # OG + repo -
        # output = F.tanh(output)
        # output = self.dropout2(output)
        # output = self.dense3(output)
        return output

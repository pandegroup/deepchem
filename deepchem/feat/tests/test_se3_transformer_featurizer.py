import unittest
from rdkit import Chem
from deepchem.feat import SE3TransformerFeaturizer
from deepchem.feat.graph_data import GraphData
import numpy as np


class TestSE3TransformerFeaturizer(unittest.TestCase):
    """
    Tests for SE3TransformerFeaturizer.
    """

    def setUp(self):
        """
        Set up tests with a simple molecule.
        """
        smiles = 'CCO'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_bonded_graph_featurization(self):
        """
        Test featurization with bonded edges.
        """
        featurizer = SE3TransformerFeaturizer(fully_connected=False,
                                              embeded=True)
        graphs = featurizer.featurize([self.mol])

        assert len(graphs) == 1
        graph = graphs[0]

        assert isinstance(graph[0], GraphData)

        assert graph[0].node_features.shape[0] == self.mol.GetNumAtoms()
        assert graph[0].positions.shape[0] == self.mol.GetNumAtoms()
        assert graph[0].edge_index.shape[1] > 0

    def test_fully_connected_graph_featurization(self):
        """
        Test featurization with fully connected edges.
        """
        featurizer = SE3TransformerFeaturizer(fully_connected=True,
                                              embeded=True)
        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1

        graph = graphs[0]
        assert isinstance(graph[0], GraphData)

        num_atoms = self.mol.GetNumAtoms()

        expected_edges = num_atoms * (num_atoms - 1)  # Fully connected graph
        assert graph[0].edge_index.shape[1] == expected_edges

    def test_embedded_coordinates(self):
        """
        Test featurization with embedded 3D coordinates.
        """
        featurizer = SE3TransformerFeaturizer(embeded=True)
        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1

        graph = graphs[0]
        assert isinstance(graph[0], GraphData)
        # 3D positions
        assert graph[0].positions.shape[1] == 3

    def test_edge_weight_discretization(self):
        """
        Test discretization of edge weights.
        """
        featurizer = SE3TransformerFeaturizer(weight_bins=[1.0, 2.0, 3.0],
                                              embeded=True)

        graphs = featurizer.featurize([self.mol])
        assert len(graphs) == 1
        graph = graphs[0]
        assert isinstance(graph[0], GraphData)

        one_hot_weights = graph[0].edge_weights
        assert one_hot_weights.shape[1] == len(
            featurizer.weight_bins) + 1  # Bin count + 1
        assert np.all(np.sum(one_hot_weights, axis=1) == 1)

    def test_multiple_molecules(self):
        """
        Test featurization of multiple molecules.
        """
        smiles_list = ['CCO', 'CCC']
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        featurizer = SE3TransformerFeaturizer(fully_connected=False,
                                              embeded=True)
        graphs = featurizer.featurize(mols)
        assert len(graphs) == len(smiles_list)

        for graph, mol in zip(graphs, mols):
            assert graph[0].node_features.shape[0] == mol.GetNumAtoms()

import logging
import numpy as np
from deepchem.utils.typing import RDKitBond, RDKitMol, List
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import OneOrMany

from typing import Optional

logger = logging.getLogger(__name__)


class GraphMatrix:
    """
    This is class used to store data for MolGAN neural networks.

    Parameters
    ----------
    node_features: np.ndarray
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_features: np.ndarray,
        Edge feature matrix with shape [num_nodes, num_nodes]

    Returns
    -------
    graph: GraphMatrix
        A molecule graph with some features.

    """

    def __init__(self, adjacency_matrix: np.ndarray, node_features: np.ndarray):
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features


class MolGanFeaturizer(MolecularFeaturizer):
    """
    Featurizer for MolGAN de-novo molecular generation [1]_.
    The default representation is in form of GraphMatrix object.
    It is wrapper for two matrices containing atom and bond type information.
    The class also provides reverse capabilities.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> rdkit_mol, smiles_mol = Chem.MolFromSmiles('CCC'), 'C1=CC=CC=C1'
    >>> molecules = [rdkit_mol, smiles_mol]
    >>> featurizer = dc.feat.MolGanFeaturizer()
    >>> features = featurizer.featurize(molecules)
    >>> len(features) # 2 molecules
    2
    >>> type(features[0])
    <class 'deepchem.feat.molecule_featurizers.molgan_featurizer.GraphMatrix'>
    >>> molecules = featurizer.defeaturize(features) # defeaturization
    >>> type(molecules[0])
    <class 'rdkit.Chem.rdchem.Mol'>

    """

    def __init__(
        self,
        max_atom_count: int = 9,
        kekulize: bool = True,
        bond_labels: Optional[List[RDKitBond]] = None,
        atom_labels: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        max_atom_count: int, default 9
            Maximum number of atoms used for creation of adjacency matrix.
            Molecules cannot have more atoms than this number
            Implicit hydrogens do not count.
        kekulize: bool, default True
            Should molecules be kekulized.
            Solves number of issues with defeaturization when used.
        bond_labels: List[RDKitBond]
            List of types of bond used for generation of adjacency matrix
        atom_labels: List[int]
            List of atomic numbers used for generation of node features

        References
        ---------
        .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model for
            small molecular graphs" (2018), https://arxiv.org/abs/1805.11973

        """

        self.max_atom_count = max_atom_count
        self.kekulize = kekulize

        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        # bond labels
        if bond_labels is None:
            # defaults from the paper
            # section 5, "generator architecture"
            self.bond_labels = [
                Chem.rdchem.BondType.ZERO,
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
            ]
        else:
            self.bond_labels = bond_labels

        # atom labels
        if atom_labels is None:
            self.atom_labels = [0, 6, 7, 8, 9]  # C,N,O,F
        else:
            self.atom_labels = atom_labels

        # create bond encoders and decoders
        self.bond_encoder = {l: i for i, l in enumerate(self.bond_labels)}
        self.bond_decoder = {i: l for i, l in enumerate(self.bond_labels)}
        # create atom encoders and decoders
        self.atom_encoder = {l: i for i, l in enumerate(self.atom_labels)}
        self.atom_decoder = {i: l for i, l in enumerate(self.atom_labels)}

    def _featurize(self, datapoint: RDKitMol,
                   **kwargs) -> Optional[GraphMatrix]:
        """
        Calculate adjacency matrix and nodes features for RDKitMol.
        It strips any chirality and charges

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit mol object.

        Returns
        -------
        graph: GraphMatrix
            A molecule graph with some features.

        """

        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        if self.kekulize:
            Chem.Kekulize(datapoint)

        A = np.zeros(shape=(self.max_atom_count, self.max_atom_count),
                     dtype=np.float32)
        bonds = datapoint.GetBonds()

        begin, end = [b.GetBeginAtomIdx() for b in bonds
                     ], [b.GetEndAtomIdx() for b in bonds]
        bond_type = [self.bond_encoder[b.GetBondType()] for b in bonds]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:datapoint.GetNumAtoms(), :datapoint.GetNumAtoms()],
                        axis=-1)
        X = np.array(
            [
                self.atom_encoder[atom.GetAtomicNum()]
                for atom in datapoint.GetAtoms()
            ] + [0] * (self.max_atom_count - datapoint.GetNumAtoms()),
            dtype=np.int32,
        )
        graph = GraphMatrix(A, X)

        return graph if (degree > 0).all() else None

    def _defeaturize(self,
                     graph_matrix: GraphMatrix,
                     sanitize: bool = True,
                     cleanup: bool = True) -> RDKitMol:
        """
        Recreate RDKitMol from GraphMatrix object.
        Same featurizer need to be used for featurization and defeaturization.
        It only recreates bond and atom types, any kind of additional features
        like chirality or charge are not included.
        Therefore, any checks of type: original_smiles == defeaturized_smiles
        will fail on chiral or charged compounds.

        Parameters
        ----------
        graph_matrix: GraphMatrix
            GraphMatrix object.
        sanitize: bool, default True
            Should RDKit sanitization be included in the process.
        cleanup: bool, default True
            Splits salts and removes compounds with "*" atom types

        Returns
        -------
        mol: RDKitMol object
            RDKitMol object representing molecule.

        """

        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")

        if not isinstance(graph_matrix, GraphMatrix):
            return None

        node_labels = graph_matrix.node_features
        edge_labels = graph_matrix.adjacency_matrix

        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end),
                            self.bond_decoder[edge_labels[start, end]])

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None

        if cleanup:
            try:
                smiles = Chem.MolToSmiles(mol)
                smiles = max(smiles.split("."), key=len)
                if "*" not in smiles:
                    mol = Chem.MolFromSmiles(smiles)
                else:
                    mol = None
            except Exception:
                mol = None

        return mol

    def defeaturize(self,
                    graphs: OneOrMany[GraphMatrix],
                    log_every_n: int = 1000) -> np.ndarray:
        """
        Calculates molecules from corresponding GraphMatrix objects.

        Parameters
        ----------
        graphs: GraphMatrix / iterable
            GraphMatrix object or corresponding iterable
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
            A numpy array containing RDKitMol objext.

        """

        # Special case handling of single molecule
        if isinstance(graphs, GraphMatrix):
            graphs = [graphs]
        else:
            # Convert iterables to list
            graphs = list(graphs)

        molecules = []
        for i, gr in enumerate(graphs):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)

            try:
                molecules.append(self._defeaturize(gr))
            except Exception as e:
                logger.warning(
                    "Failed to defeaturize datapoint %d, %s. Appending empty array",
                    i,
                    gr,
                )
                logger.warning("Exception message: {}".format(e))
                molecules.append(np.array([]))

        return np.asarray(molecules)

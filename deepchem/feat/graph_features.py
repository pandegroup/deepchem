# flake8: noqa

import numpy as np
import deepchem as dc
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.data import DiskDataset
import logging
from typing import Optional, List, Tuple, Union, Iterable
from deepchem.utils.typing import RDKitMol, RDKitAtom
from deepchem.utils.molecule_feature_utils import one_hot_encode


def one_of_k_encoding(x, allowable_set):
    """Encodes elements of a provided set as integers.

    Parameters
    ----------
    x: object
        Must be present in `allowable_set`.
    allowable_set: list
        List of allowable quantities.

    Example
    -------
    >>> import deepchem as dc
    >>> dc.feat.graph_features.one_of_k_encoding("a", ["a", "b", "c"])
    [True, False, False]

    Raises
    ------
    `ValueError` if `x` is not in `allowable_set`.
    """
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element.

    Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
    pretends that `x` is the last element of `allowable_set`.

    Parameters
    ----------
    x: object
        Must be present in `allowable_set`.
    allowable_set: list
        List of allowable quantities.

    Examples
    --------
    >>> dc.feat.graph_features.one_of_k_encoding_unk("s", ["a", "b", "c"])
    [False, False, True]

    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths

    Note that we add 1 to the lengths of all lists (to avoid an empty list
    propagating a 0).

    Parameters
    ----------
    l: list of lists
        Returns the cumulative product of these lengths.

    Examples
    --------
    >>> dc.feat.graph_features.get_intervals([[1], [1, 2], [1, 2, 3]])
    [1, 3, 12]

    >>> dc.feat.graph_features.get_intervals([[1], [], [1, 2], [1, 2, 3]])
    [1, 1, 3, 12]

    """
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

    return intervals


def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found

    Parameters
    ----------
    l: list
        List of values
    e: object
        Object to check whether `e` is in `l`

    Examples
    --------
    >>> dc.feat.graph_features.safe_index([1, 2, 3], 1)
    0
    >>> dc.feat.graph_features.safe_index([1, 2, 3], 7)
    3

    """
    try:
        return l.index(e)
    except ValueError:
        return len(l)


class GraphConvConstants(object):
    """This class defines a collection of constants which are useful for graph convolutions on molecules."""
    possible_atom_list = [
        'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
        'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
    ]
    """Allowed Numbers of Hydrogens"""
    possible_numH_list = [0, 1, 2, 3, 4]
    """Allowed Valences for Atoms"""
    possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
    """Allowed Formal Charges for Atoms"""
    possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
    """This is a placeholder for documentation. These will be replaced with corresponding values of the rdkit HybridizationType"""
    possible_hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    """Allowed number of radical electrons."""
    possible_number_radical_e_list = [0, 1, 2]
    """Allowed types of Chirality"""
    possible_chirality_list = ['R', 'S']
    """The set of all values allowed."""
    reference_lists = [
        possible_atom_list, possible_numH_list, possible_valence_list,
        possible_formal_charge_list, possible_number_radical_e_list,
        possible_hybridization_list, possible_chirality_list
    ]
    """The number of different values that can be taken. See `get_intervals()`"""
    intervals = get_intervals(reference_lists)
    """Possible stereochemistry. We use E-Z notation for stereochemistry
https://en.wikipedia.org/wiki/E%E2%80%93Z_notation"""
    possible_bond_stereo = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    """Number of different bond types not counting stereochemistry."""
    bond_fdim_base = 6


def get_feature_list(atom):
    """Returns a list of possible features for this atom.

    Parameters
    ----------
    atom: RDKit.Chem.rdchem.Atom
        Atom to get features for

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C")
    >>> atom = mol.GetAtoms()[0]
    >>> features = dc.feat.graph_features.get_feature_list(atom)
    >>> type(features)
    <class 'list'>
    >>> len(features)
    6

    Note
    ----
    This method requires RDKit to be installed.

    Returns
    -------
    features: list
        List of length 6. The i-th value in this list provides the index of the
        atom in the corresponding feature value list. The 6 feature values lists
        for this function are `[GraphConvConstants.possible_atom_list,
        GraphConvConstants.possible_numH_list,
        GraphConvConstants.possible_valence_list,
        GraphConvConstants.possible_formal_charge_list,
        GraphConvConstants.possible_num_radical_e_list]`.
    """
    possible_atom_list = GraphConvConstants.possible_atom_list
    possible_numH_list = GraphConvConstants.possible_numH_list
    possible_valence_list = GraphConvConstants.possible_valence_list
    possible_formal_charge_list = GraphConvConstants.possible_formal_charge_list
    possible_number_radical_e_list = GraphConvConstants.possible_number_radical_e_list
    possible_hybridization_list = GraphConvConstants.possible_hybridization_list
    # Replace the hybridization
    from rdkit import Chem
    #global possible_hybridization_list
    possible_hybridization_list = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features = 6 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list,
                             atom.GetFormalCharge())
    features[4] = safe_index(possible_number_radical_e_list,
                             atom.GetNumRadicalElectrons())

    features[5] = safe_index(possible_hybridization_list,
                             atom.GetHybridization())
    return features


def features_to_id(features, intervals):
    """Convert list of features into index using spacings provided in intervals

    Parameters
    ----------
    features: list
        List of features as returned by `get_feature_list()`
    intervals: list
        List of intervals as returned by `get_intervals()`

    Returns
    -------
    id: int
        The index in a feature vector given by the given set of features.
    """
    id = 0
    for k in range(len(intervals)):
        id += features[k] * intervals[k]

    # Allow 0 index to correspond to null molecule 1
    id = id + 1
    return id


def id_to_features(id, intervals):
    """Given an index in a feature vector, return the original set of features.

    Parameters
    ----------
    id: int
        The index in a feature vector given by the given set of features.
    intervals: list
        List of intervals as returned by `get_intervals()`

    Returns
    -------
    features: list
        List of features as returned by `get_feature_list()`
    """
    features = 6 * [0]

    # Correct for null
    id -= 1

    for k in range(0, 6 - 1):
        # print(6-k-1, id)
        features[6 - k - 1] = id // intervals[6 - k - 1]
        id -= features[6 - k - 1] * intervals[6 - k - 1]
    # Correct for last one
    features[0] = id
    return features


def atom_to_id(atom):
    """Return a unique id corresponding to the atom type

    Parameters
    ----------
    atom: RDKit.Chem.rdchem.Atom
        Atom to convert to ids.

    Returns
    -------
    id: int
        The index in a feature vector given by the given set of features.
    """
    features = get_feature_list(atom)
    return features_to_id(features, intervals)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    """Helper method used to compute per-atom feature vectors.

    Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

    Parameters
    ----------
    atom: RDKit.Chem.rdchem.Atom
        Atom to compute features on.
    bool_id_feat: bool, optional
        Return an array of unique identifiers corresponding to atom type.
    explicit_H: bool, optional
        If true, model hydrogens explicitly
    use_chirality: bool, optional
        If true, use chirality information.

    Returns
    -------
    features: np.ndarray
        An array of per-atom features.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCC')
    >>> atom = mol.GetAtoms()[0]
    >>> features = dc.feat.graph_features.atom_features(atom)
    >>> type(features)
    <class 'numpy.ndarray'>
    >>> features.shape
    (75,)

    """
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        from rdkit import Chem
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'C',
            'N',
            'O',
            'S',
            'F',
            'Si',
            'P',
            'Cl',
            'Br',
            'Mg',
            'Na',
            'Ca',
            'Fe',
            'As',
            'Al',
            'I',
            'B',
            'V',
            'K',
            'Tl',
            'Yb',
            'Sb',
            'Sn',
            'Ag',
            'Pd',
            'Co',
            'Se',
            'Ti',
            'Zn',
            'H',  # H?
            'Li',
            'Ge',
            'Cu',
            'Au',
            'Ni',
            'Cd',
            'In',
            'Mn',
            'Zr',
            'Cr',
            'Pt',
            'Hg',
            'Pb',
            'Unknown'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                    ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


def bond_features(bond, use_chirality=False, use_extended_chirality=False):
    """Helper method used to compute bond feature vectors.

    Many different featurization methods compute bond features
    such as WeaveFeaturizer. This method computes such features.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        Bond to compute features on.
    use_chirality: bool, optional
        If true, use chirality information.
    use_extended_chirality: bool, optional
        If true, use chirality information with upto 6 different types.

    Note
    ----
    This method requires RDKit to be installed.

    Returns
    -------
    bond_feats: np.ndarray
        Array of bond features. This is a 1-D array of length 6 if `use_chirality`
        is `False` else of length 10 with chirality encoded.

    bond_feats: Sequence[Union[bool, int, float]]
        List of bond features returned if `use_extended_chirality` is `True`.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCC')
    >>> bond = mol.GetBonds()[0]
    >>> bond_features = dc.feat.graph_features.bond_features(bond)
    >>> type(bond_features)
    <class 'numpy.ndarray'>
    >>> bond_features.shape
    (6,)

    Note
    ----
    This method requires RDKit to be installed.

    """
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        raise ImportError("This method requires RDKit to be installed.")
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()), GraphConvConstants.possible_bond_stereo)

    if use_extended_chirality:
        stereo = one_hot_encode(int(bond.GetStereo()), list(range(6)), True)
        stereo = [int(feature) for feature in stereo]
        bond_feats = bond_feats + stereo
        return bond_feats

    return np.array(bond_feats)


def max_pair_distance_pairs(mol: RDKitMol,
                            max_pair_distance: Optional[int] = None
                           ) -> np.ndarray:
    """Helper method which finds atom pairs within max_pair_distance graph distance.

    This helper method is used to find atoms which are within max_pair_distance
    graph_distance of one another. This is done by using the fact that the
    powers of an adjacency matrix encode path connectivity information. In
    particular, if `adj` is the adjacency matrix, then `adj**k` has a nonzero
    value at `(i, j)` if and only if there exists a path of graph distance `k`
    between `i` and `j`. To find all atoms within `max_pair_distance` of each
    other, we can compute the adjacency matrix powers `[adj, adj**2,
    ...,adj**max_pair_distance]` and find pairs which are nonzero in any of
    these matrices. Since adjacency matrices and their powers are positive
    numbers, this is simply the nonzero elements of `adj + adj**2 + ... +
    adj**max_pair_distance`.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        RDKit molecules
    max_pair_distance: Optional[int], (default None)
        This value can be a positive integer or None. This
        parameter determines the maximum graph distance at which pair
        features are computed. For example, if `max_pair_distance==2`,
        then pair features are computed only for atoms at most graph
        distance 2 apart. If `max_pair_distance` is `None`, all pairs are
        considered (effectively infinite `max_pair_distance`)

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCC')
    >>> features = dc.feat.graph_features.max_pair_distance_pairs(mol, 1)
    >>> type(features)
    <class 'numpy.ndarray'>
    >>> features.shape  # (2, num_pairs)
    (2, 7)

    Returns
    -------
    np.ndarray
        Of shape `(2, num_pairs)` where `num_pairs` is the total number of pairs
        within `max_pair_distance` of one another.
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    N = len(mol.GetAtoms())
    if (max_pair_distance is None or max_pair_distance >= N):
        max_distance = N
    elif max_pair_distance is not None and max_pair_distance <= 0:
        raise ValueError(
            "max_pair_distance must either be a positive integer or None")
    elif max_pair_distance is not None:
        max_distance = max_pair_distance
    adj = rdmolops.GetAdjacencyMatrix(mol)
    # Handle edge case of self-pairs (i, i)
    sum_adj = np.eye(N)
    for i in range(max_distance):
        # Increment by 1 since we don't want 0-indexing
        power = i + 1
        sum_adj += np.linalg.matrix_power(adj, power)
    nonzero_locs = np.where(sum_adj != 0)
    num_pairs = len(nonzero_locs[0])
    # This creates a matrix of shape (2, num_pairs)
    pair_edges = np.reshape(np.array(list(zip(nonzero_locs))), (2, num_pairs))
    return pair_edges


def pair_features(
        mol: RDKitMol,
        bond_features_map: dict,
        bond_adj_list: List,
        bt_len: int = 6,
        graph_distance: bool = True,
        max_pair_distance: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper method used to compute atom pair feature vectors.

    Many different featurization methods compute atom pair features
    such as WeaveFeaturizer. Note that atom pair features could be
    for pairs of atoms which aren't necessarily bonded to one
    another.

    Parameters
    ----------
    mol: RDKit Mol
        Molecule to compute features on.
    bond_features_map: dict
        Dictionary that maps pairs of atom ids (say `(2, 3)` for a bond between
        atoms 2 and 3) to the features for the bond between them.
    bond_adj_list: list of lists
        `bond_adj_list[i]` is a list of the atom indices that atom `i` shares a
        bond with . This list is symmetrical so if `j in bond_adj_list[i]` then `i
        in bond_adj_list[j]`.
    bt_len: int, optional (default 6)
        The number of different bond types to consider.
    graph_distance: bool, optional (default True)
        If true, use graph distance between molecules. Else use euclidean
        distance. The specified `mol` must have a conformer. Atomic
        positions will be retrieved by calling `mol.getConformer(0)`.
    max_pair_distance: Optional[int], (default None)
        This value can be a positive integer or None. This
        parameter determines the maximum graph distance at which pair
        features are computed. For example, if `max_pair_distance==2`,
        then pair features are computed only for atoms at most graph
        distance 2 apart. If `max_pair_distance` is `None`, all pairs are
        considered (effectively infinite `max_pair_distance`)

    Note
    ----
    This method requires RDKit to be installed.

    Returns
    -------
    features: np.ndarray
        Of shape `(N_edges, bt_len + max_distance + 1)`. This is the array
        of pairwise features for all atom pairs, where N_edges is the
        number of edges within max_pair_distance of one another in this
        molecules.
    pair_edges: np.ndarray
        Of shape `(2, num_pairs)` where `num_pairs` is the total number of
        pairs within `max_pair_distance` of one another.
    """
    if graph_distance:
        max_distance = 7
    else:
        max_distance = 1
    N = mol.GetNumAtoms()
    pair_edges = max_pair_distance_pairs(mol, max_pair_distance)
    num_pairs = pair_edges.shape[1]
    N_edges = pair_edges.shape[1]
    features = np.zeros((N_edges, bt_len + max_distance + 1))
    # Get mapping
    mapping = {}
    for n in range(N_edges):
        a1, a2 = pair_edges[:, n]
        mapping[(int(a1), int(a2))] = n
    num_atoms = mol.GetNumAtoms()
    rings = mol.GetRingInfo().AtomRings()
    # Save coordinates
    if not graph_distance:
        coords = np.zeros((N, 3))
        for atom in range(N):
            pos = mol.GetConformer(0).GetAtomPosition(atom)
            coords[atom, :] = [pos.x, pos.y, pos.z]
    for a1 in range(num_atoms):
        for a2 in bond_adj_list[a1]:
            # first `bt_len` features are bond features(if applicable)
            if (int(a1), int(a2)) not in mapping:
                raise ValueError(
                    "Malformed molecule with bonds not in specified graph distance."
                )
            else:
                n = mapping[(int(a1), int(a2))]
            features[n, :bt_len] = np.asarray(bond_features_map[tuple(
                sorted((a1, a2)))],
                                              dtype=float)
        for ring in rings:
            if a1 in ring:
                for a2 in ring:
                    if (int(a1), int(a2)) not in mapping:
                        # For ring pairs outside max pairs distance continue
                        continue
                    else:
                        n = mapping[(int(a1), int(a2))]
                    # `bt_len`-th feature is if the pair of atoms are in the same ring
                    if a2 == a1:
                        features[n, bt_len] = 0
                    else:
                        features[n, bt_len] = 1
        # graph distance between two atoms
        if graph_distance:
            # distance is a matrix of 1-hot encoded distances for all atoms
            distance = find_distance(a1,
                                     num_atoms,
                                     bond_adj_list,
                                     max_distance=max_distance)
        else:
            # Euclidean distance between atoms
            distance = np.sqrt(
                np.sum(np.square(np.stack([coords[a1]] * N, axis=0) - coords),
                       axis=1))
        for a2 in range(num_atoms):
            if (int(a1), int(a2)) not in mapping:
                # For ring pairs outside max pairs distance continue
                continue
            else:
                n = mapping[(int(a1), int(a2))]
                features[n, bt_len + 1:] = distance[a2]

    return features, pair_edges


def find_distance(a1: RDKitAtom,
                  num_atoms: int,
                  bond_adj_list,
                  max_distance=7) -> np.ndarray:
    """Computes distances from provided atom.

    Parameters
    ----------
    a1: RDKit atom
        The source atom to compute distances from.
    num_atoms: int
        The total number of atoms.
    bond_adj_list: list of lists
        `bond_adj_list[i]` is a list of the atom indices that atom `i` shares a
        bond with. This list is symmetrical so if `j in bond_adj_list[i]` then `i in
        bond_adj_list[j]`.
    max_distance: int, optional (default 7)
        The max distance to search.

    Returns
    -------
    distances: np.ndarray
        Of shape `(num_atoms, max_distance)`. Provides a one-hot encoding of the
        distances. That is, `distances[i]` is a one-hot encoding of the distance
        from `a1` to atom `i`.
    """
    distance = np.zeros((num_atoms, max_distance))
    radial = 0
    # atoms `radial` bonds away from `a1`
    adj_list = set(bond_adj_list[a1])
    # atoms less than `radial` bonds away
    all_list = set([a1])
    while radial < max_distance:
        distance[list(adj_list), radial] = 1
        all_list.update(adj_list)
        # find atoms `radial`+1 bonds away
        next_adj = set()
        for adj in adj_list:
            next_adj.update(bond_adj_list[adj])
        adj_list = next_adj - all_list
        radial = radial + 1
    return distance


class ConvMolFeaturizer(MolecularFeaturizer):
    """This class implements the featurization to implement Duvenaud graph convolutions.

    Duvenaud graph convolutions [1]_ construct a vector of descriptors for each
    atom in a molecule. The featurizer computes that vector of local descriptors.

    Examples
    ---------
    >>> import deepchem as dc
    >>> smiles = ["C", "CCC"]
    >>> featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    >>> f = featurizer.featurize(smiles)
    >>> # Using ConvMolFeaturizer to create featurized fragments derived from molecules of interest.
    ... # This is used only in the context of performing interpretation of models using atomic
    ... # contributions (atom-based model interpretation)
    ... smiles = ["C", "CCC"]
    >>> featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
    >>> f = featurizer.featurize(smiles)
    >>> len(f) # contains 2 lists with  featurized fragments from 2 mols
    2

    See Also
    --------
    Detailed examples of `GraphConvModel` interpretation are provided in Tutorial #28

    References
    ---------

    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information
        processing systems. 2015.

    Note
    ----
    This class requires RDKit to be installed.
    """
    name = ['conv_mol']

    def __init__(self,
                 master_atom: bool = False,
                 use_chirality: bool = False,
                 atom_properties: Iterable[str] = [],
                 per_atom_fragmentation: bool = False):
        """
        Parameters
        ----------
        master_atom: Boolean
            if true create a fake atom with bonds to every other atom.
            the initialization is the mean of the other atom features in
            the molecule.  This technique is briefly discussed in
            Neural Message Passing for Quantum Chemistry
            https://arxiv.org/pdf/1704.01212.pdf
        use_chirality: Boolean
            if true then make the resulting atom features aware of the
            chirality of the molecules in question
        atom_properties: list of string or None
            properties in the RDKit Mol object to use as additional
            atom-level features in the larger molecular feature.  If None,
            then no atom-level properties are used.  Properties should be in the
            RDKit mol object should be in the form
            atom XXXXXXXX NAME
            where XXXXXXXX is a zero-padded 8 digit number coresponding to the
            zero-indexed atom index of each atom and NAME is the name of the property
            provided in atom_properties.  So "atom 00000000 sasa" would be the
            name of the molecule level property in mol where the solvent
            accessible surface area of atom 0 would be stored.
        per_atom_fragmentation: Boolean
            If True, then multiple "atom-depleted" versions of each molecule will be created (using featurize() method).
            For each molecule, atoms are removed one at a time and the resulting molecule is featurized.
            The result is a list of ConvMol objects,
            one with each heavy atom removed. This is useful for subsequent model interpretation: finding atoms
            favorable/unfavorable for (modelled) activity. This option is typically used in combination
            with a FlatteningTransformer to split the lists into separate samples.

            Since ConvMol is an object and not a numpy array, need to set dtype to
            object.
        """
        self.dtype = object
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = list(atom_properties)
        self.per_atom_fragmentation = per_atom_fragmentation

    def featurize(self,
                  datapoints: Union[RDKitMol, str, Iterable[RDKitMol],
                                    Iterable[str]],
                  log_every_n: int = 1000,
                  **kwargs) -> np.ndarray:
        """
        Override parent: aim is to add handling atom-depleted molecules featurization

        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
            RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
            strings.
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        if 'molecules' in kwargs and datapoints is None:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning(
                'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
            )

        features = super(ConvMolFeaturizer, self).featurize(datapoints,
                                                            log_every_n=1000)
        if self.per_atom_fragmentation:
            # create temporary valid ids serving to filter out failed featurizations from every sublist
            # of features (i.e. every molecules' frags list), and also totally failed sublists.
            # This makes output digestable by Loaders
            valid_frag_inds = [[
                True if np.array(elt).size > 0 else False for elt in f
            ] for f in features]
            features = np.array(
                [[elt for (is_valid, elt) in zip(l, m) if is_valid
                 ] for (l, m) in zip(valid_frag_inds, features) if any(l)],
                dtype=object)
        return features

    def _get_atom_properties(self, atom):
        """
        For a given input RDKit atom return the values of the properties
        requested when initializing the featurize.  See the __init__ of the
        class for a full description of the names of the properties

        Parameters
        ----------
        atom: RDKit.rdchem.Atom
            Atom to get the properties of
        returns a numpy lists of floats of the same size as self.atom_properties
        """
        values = []
        for prop in self.atom_properties:
            mol_prop_name = str("atom %08d %s" % (atom.GetIdx(), prop))
            try:
                values.append(float(atom.GetOwningMol().GetProp(mol_prop_name)))
            except KeyError:
                raise KeyError("No property %s found in %s in %s" %
                               (mol_prop_name, atom.GetOwningMol(), self))
        return np.array(values)

    def _featurize(self, mol):
        """Encodes mol as a ConvMol object.
        If per_atom_fragmentation is True,
        then for each molecule a list of ConvMolObjects
        will be created"""

        def per_atom(n, a):
            """
            Enumerates fragments resulting from mol object,
            s.t. each fragment = mol with single atom removed (all possible removals are enumerated)
            Goes over nodes, deletes one at a time and updates adjacency list of lists (removes connections to that node)

            Parameters
            ----------
            n: np.array of nodes (number_of_nodes X number_of_features)
            a: list of nested lists of adjacent node pairs

            """
            for i in range(n.shape[0]):
                new_n = np.delete(n, (i), axis=0)
                new_a = []
                for j, node_pair in enumerate(a):
                    if i != j:  # don't need this pair, no more connections to deleted node
                        tmp_node_pair = []
                        for v in node_pair:
                            if v < i:
                                tmp_node_pair.append(v)
                            elif v > i:
                                tmp_node_pair.append(
                                    v - 1
                                )  # renumber node, because of offset after node deletion
                        new_a.append(tmp_node_pair)
                yield new_n, new_a

        # Get the node features
        idx_nodes = [(a.GetIdx(),
                      np.concatenate(
                          (atom_features(a, use_chirality=self.use_chirality),
                           self._get_atom_properties(a))))
                     for a in mol.GetAtoms()]

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)
        if self.master_atom:
            master_atom_features = np.expand_dims(np.mean(nodes, axis=0),
                                                  axis=0)
            nodes = np.concatenate([nodes, master_atom_features], axis=0)

        # Get bond lists with reverse edges included
        edge_list = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]
        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        if self.master_atom:
            fake_atom_index = len(nodes) - 1
            for index in range(len(nodes) - 1):
                canon_adj_list[index].append(fake_atom_index)

        if not self.per_atom_fragmentation:
            return ConvMol(nodes, canon_adj_list)
        else:
            return [ConvMol(n, a) for n, a in per_atom(nodes, canon_adj_list)]

    def feature_length(self):
        return 75 + len(self.atom_properties)

    def __hash__(self):
        atom_properties = tuple(self.atom_properties)
        return hash((self.master_atom, self.use_chirality, atom_properties))

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return self.master_atom == other.master_atom and \
               self.use_chirality == other.use_chirality and \
               tuple(self.atom_properties) == tuple(other.atom_properties)


class WeaveFeaturizer(MolecularFeaturizer):
    """This class implements the featurization to implement Weave convolutions.

    Weave convolutions were introduced in [1]_. Unlike Duvenaud graph
    convolutions, weave convolutions require a quadratic matrix of interaction
    descriptors for each pair of atoms. These extra descriptors may provide for
    additional descriptive power but at the cost of a larger featurized dataset.


    Examples
    --------
    >>> import deepchem as dc
    >>> mols = ["CCC"]
    >>> featurizer = dc.feat.WeaveFeaturizer()
    >>> features = featurizer.featurize(mols)
    >>> type(features[0])
    <class 'deepchem.feat.mol_graphs.WeaveMol'>
    >>> features[0].get_num_atoms() # 3 atoms in compound
    3
    >>> features[0].get_num_features() # feature size
    75
    >>> type(features[0].get_atom_features())
    <class 'numpy.ndarray'>
    >>> features[0].get_atom_features().shape
    (3, 75)
    >>> type(features[0].get_pair_features())
    <class 'numpy.ndarray'>
    >>> features[0].get_pair_features().shape
    (9, 14)


    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.

    Note
    ----
    This class requires RDKit to be installed.
    """

    name = ['weave_mol']

    def __init__(self,
                 graph_distance: bool = True,
                 explicit_H: bool = False,
                 use_chirality: bool = False,
                 max_pair_distance: Optional[int] = None):
        """Initialize this featurizer with set parameters.

        Parameters
        ----------
        graph_distance: bool, (default True)
            If True, use graph distance for distance features. Otherwise, use
            Euclidean distance. Note that this means that molecules that this
            featurizer is invoked on must have valid conformer information if this
            option is set.
        explicit_H: bool, (default False)
            If true, model hydrogens in the molecule.
        use_chirality: bool, (default False)
            If true, use chiral information in the featurization
        max_pair_distance: Optional[int], (default None)
            This value can be a positive integer or None. This
            parameter determines the maximum graph distance at which pair
            features are computed. For example, if `max_pair_distance==2`,
            then pair features are computed only for atoms at most graph
            distance 2 apart. If `max_pair_distance` is `None`, all pairs are
            considered (effectively infinite `max_pair_distance`)
        """
        # Distance is either graph distance(True) or Euclidean distance(False,
        # only support datasets providing Cartesian coordinates)
        self.graph_distance = graph_distance
        # Set dtype
        self.dtype = object
        # If includes explicit hydrogens
        self.explicit_H = explicit_H
        # If uses use_chirality
        self.use_chirality = use_chirality
        if isinstance(max_pair_distance, int) and max_pair_distance <= 0:
            raise ValueError(
                "max_pair_distance must either be a positive integer or None")
        self.max_pair_distance = max_pair_distance
        if self.use_chirality:
            self.bt_len = int(GraphConvConstants.bond_fdim_base) + len(
                GraphConvConstants.possible_bond_stereo)
        else:
            self.bt_len = int(GraphConvConstants.bond_fdim_base)

    def _featurize(self, mol):
        """Encodes mol as a WeaveMol object."""
        # Atom features
        idx_nodes = [(a.GetIdx(),
                      atom_features(a,
                                    explicit_H=self.explicit_H,
                                    use_chirality=self.use_chirality))
                     for a in mol.GetAtoms()]
        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)

        # Get bond lists
        bond_features_map = {}
        for b in mol.GetBonds():
            bond_features_map[tuple(
                sorted([b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx()
                       ]))] = bond_features(b, use_chirality=self.use_chirality)

        # Get canonical adjacency list
        bond_adj_list = [[] for mol_id in range(len(nodes))]
        for bond in bond_features_map.keys():
            bond_adj_list[bond[0]].append(bond[1])
            bond_adj_list[bond[1]].append(bond[0])

        # Calculate pair features
        pairs, pair_edges = pair_features(
            mol,
            bond_features_map,
            bond_adj_list,
            bt_len=self.bt_len,
            graph_distance=self.graph_distance,
            max_pair_distance=self.max_pair_distance)

        return WeaveMol(nodes, pairs, pair_edges)

import pytest

try:
    import torch
except ModuleNotFoundError:
    pass


@pytest.mark.torch
@pytest.mark.parametrize('dynamic_depth', ['none', 'uniform'])
@pytest.mark.parametrize('atom_messages', [False, True])
def testGroverMPNEncoder(dynamic_depth, atom_messages):
    from deepchem.models.torch_models.grover_layers import GroverMPNEncoder
    f_atoms = torch.randn(4, 151)
    f_bonds = torch.randn(5, 165)
    a2b = torch.Tensor([[0, 0], [2, 0], [4, 0], [1, 3]]).type(torch.int32)
    b2a = torch.Tensor([0, 1, 3, 2, 3]).type(torch.long)
    b2revb = torch.Tensor([0, 2, 1, 4, 3]).type(torch.long)
    a2a = torch.Tensor([[0, 0], [3, 0], [3, 0], [1, 2]]).type(torch.int32)

    # TODO Write tests for undirected = True case, currently fails. for this case, we have
    # to generate inputs (a2b, b2a, b2revb) for undirected graph (mol2graph returns features
    # for directed graphs)
    hidden_size = 32
    depth = 5
    undirected = False
    attach_feats = True
    if not atom_messages:
        init_message_dim = f_bonds.shape[1]
        attached_feat_fdim = f_atoms.shape[1]
        layer = GroverMPNEncoder(atom_messages=atom_messages,
                                 init_message_dim=init_message_dim,
                                 attached_feat_fdim=attached_feat_fdim,
                                 hidden_size=hidden_size,
                                 depth=depth,
                                 dynamic_depth=dynamic_depth,
                                 undirected=undirected,
                                 attach_feats=attach_feats)
        init_messages = f_bonds
        init_attached_features = f_atoms
        a2nei = a2b
        a2attached = a2a
        out = layer(init_messages, init_attached_features, a2nei, a2attached,
                    b2a, b2revb)
        assert out.shape == (f_bonds.shape[0], hidden_size)
    else:
        init_message_dim = f_atoms.shape[1]
        attached_feat_fdim = f_bonds.shape[1]

        layer = GroverMPNEncoder(atom_messages=atom_messages,
                                 init_message_dim=init_message_dim,
                                 attached_feat_fdim=attached_feat_fdim,
                                 hidden_size=hidden_size,
                                 depth=depth,
                                 dynamic_depth=dynamic_depth,
                                 undirected=undirected,
                                 attach_feats=attach_feats)
        init_attached_features = f_bonds
        init_messages = f_atoms
        a2nei = a2a
        a2attached = a2b
        out = layer(init_messages, init_attached_features, a2nei, a2attached,
                    b2a, b2revb)
        assert out.shape == (f_atoms.shape[0], hidden_size)


def testGroverAttentionHead():
    from deepchem.models.torch_models.grover_layers import GroverAttentionHead
    # FIXME It is assumed that f_atoms and f_bonds are outputs of a hidden layer rather
    # than raw inputs of molecular features.
    f_atoms = torch.randn(4, 16)
    f_bonds = torch.randn(5, 16)
    a2b = torch.Tensor([[0, 0], [2, 0], [4, 0], [1, 3]]).type(torch.int32)
    b2a = torch.Tensor([0, 1, 3, 2, 3]).type(torch.long)
    b2revb = torch.Tensor([0, 2, 1, 4, 3]).type(torch.long)
    a2a = torch.Tensor([[0, 0], [3, 0], [3, 0], [1, 2]]).type(torch.int32)

    hidden_size = 16
    atom_messages = False
    layer = GroverAttentionHead(hidden_size,
                                bias=True,
                                depth=4,
                                undirected=False,
                                atom_messages=atom_messages)
    query, key, value = layer(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
    assert query.size() == (f_bonds.shape[0], hidden_size)
    assert key.size() == (f_bonds.shape[0], hidden_size)
    assert value.size() == (f_bonds.shape[0], hidden_size)


def testGroverMTBlock():
    from deepchem.models.torch_models.grover_layers import GroverMTBlock
    f_atoms = torch.randn(4, 151)
    f_bonds = torch.randn(5, 165)
    a2b = torch.Tensor([[0, 0], [2, 0], [4, 0], [1, 3]]).type(torch.int32)
    b2a = torch.Tensor([0, 1, 3, 2, 3]).type(torch.long)
    b2revb = torch.Tensor([0, 2, 1, 4, 3]).type(torch.long)
    a_scope = torch.Tensor([[1, 3]]).type(torch.int32)
    b_scope = torch.Tensor([[1, 4]]).type(torch.int32)
    a2a = torch.Tensor([[0, 0], [3, 0], [3, 0], [1, 2]]).type(torch.int32)

    hidden_size = 16
    layer = GroverMTBlock(atom_messages=True,
                          input_dim=f_atoms.shape[1],
                          num_heads=4,
                          depth=1,
                          hidden_size=hidden_size)

    new_batch = layer(
        [f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    new_f_atoms, new_f_bonds, new_a2b, new_b2a, new_b2revb, new_a_scope, new_b_scope, new_a2a = new_batch
    assert new_f_atoms.shape == (f_atoms.shape[0], hidden_size)
    assert new_f_bonds.shape == f_bonds.shape
    assert (new_a2b == a2b).all()
    assert (new_b2a == b2a).all()
    assert (new_b2revb == b2revb).all()
    assert (new_a_scope == a_scope).all()
    assert (new_b_scope == b_scope).all()
    assert (new_a2a == a2a).all()


def testGroverTransEncoder():
    from deepchem.models.torch_models.grover_layers import GroverTransEncoder
    hidden_size = 8
    f_atoms = torch.randn(4, 151)
    f_bonds = torch.randn(5, 165)
    a2b = torch.Tensor([[0, 0], [2, 0], [4, 0], [1, 3]]).type(torch.int32)
    b2a = torch.Tensor([0, 1, 3, 2, 3]).type(torch.long)
    b2revb = torch.Tensor([0, 2, 1, 4, 3]).type(torch.long)
    a_scope = torch.Tensor([[1, 3]]).type(torch.int32)
    b_scope = torch.Tensor([[1, 4]]).type(torch.int32)
    a2a = torch.Tensor([[0, 0], [3, 0], [3, 0], [1, 2]]).type(torch.int32)
    n_atoms, n_bonds = f_atoms.shape[0], f_bonds.shape[0]
    node_fdim, edge_fdim = f_atoms.shape[1], f_bonds.shape[1]
    layer = GroverTransEncoder(hidden_size=hidden_size,
                               edge_fdim=edge_fdim,
                               node_fdim=node_fdim,
                               atom_emb_output_type='both')
    output = layer([f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    assert output[0][0].shape == (n_atoms, hidden_size)
    assert output[0][1].shape == (n_bonds, hidden_size)
    assert output[1][0].shape == (n_atoms, hidden_size)
    assert output[1][1].shape == (n_bonds, hidden_size)

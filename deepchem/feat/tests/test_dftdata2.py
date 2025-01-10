import pytest
import warnings
import numpy as np
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests. Error is: " + str(e))


@pytest.mark.torch
def test_multiatom():
    from deepchem.data.data_loader import DFTYamlLoader
    from deepchem.models.dft.scf import XCNNSCF
    from deepchem.models.dft.nnxc import HybridXC
    inputs = 'deepchem/models/tests/assets/test_beh2.yaml'
    k = DFTYamlLoader()
    data = k.create_dataset(inputs)
    nnmodel = (torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Softplus(),
                                   torch.nn.Linear(10, 1, bias=False))).to(
                                       torch.double)
    hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    entry = data.X[0]
    evl = XCNNSCF(hybridxc, entry)
    qcs = []
    for system in entry.get_systems():
        qcs.append(evl.run(system))
    val = entry.get_val(qcs)
    expected_val = np.array([0.19325158])
    assert np.allclose(val, expected_val)

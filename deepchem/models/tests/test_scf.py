import pytest
import warnings
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests." + str(e))


@pytest.mark.torch
def test_scf():
    from deepchem.models.dft.scf import XCNNSCF
    from deepchem.models.dft.nnxc import HybridXC
    from deepchem.feat.dft_data import DFTEntry, DFTSystem
    torch.manual_seed(42)
    nnmodel = (torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Softplus(),
                                   torch.nn.Linear(10, 1, bias=False))).to(
                                       torch.double)
    hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    e_type = 'ae'
    true_val = '0.09194410469'
    systems = [{
        'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0',
        'basis': '6-311++G(3df,3pd)'
    }, {
        'moldesc': 'Li 0 0 0',
        'basis': '6-311++G(3df,3pd)',
        'spin': 1
    }, {
        'moldesc': 'H 0 0 0',
        'basis': '6-311++G(3df,3pd)',
        'spin': 1
    }]
    entry = DFTEntry.create(e_type, true_val, systems)
    evl = XCNNSCF(hybridxc, entry)
    system = DFTSystem(systems[1])
    run = evl.run(system)
    output = run.energy()
    expected_output = torch.tensor(-7.1914, dtype=torch.float64)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)

import pytest
import warnings
try:
    import torch
    import torch.nn as nn
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests. Error is: " + str(e))


@pytest.mark.torch
def dummymodel():
    n = 2

    class DummyModel(torch.nn.Module):

        def __init__(self, n):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(n, 1)

        def forward(self, x):
            return self.linear(x)

    return DummyModel(n)


@pytest.mark.torch
def test_nnlda():
    from deepchem.utils.dft_utils import ValGrad
    from deepchem.models.dft.nnxc import NNLDA
    torch.manual_seed(42)
    # https://github.com/diffqc/dqc/blob/742eb2576418464609f942def4fb7c3bbdc0cd82/dqc/test/test_xc.py#L15
    n = 2
    model = dummymodel()
    k = NNLDA(model)
    densinfo = ValGrad(
        value=torch.rand((n,), dtype=torch.float32).requires_grad_())
    output = k.get_edensityxc(densinfo).detach()
    expected_output = torch.tensor([0.3386, 0.0177])
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)


@pytest.mark.dqc
def test_hybridxc():
    from deepchem.utils.dft_utils import ValGrad
    from deepchem.models.dft.nnxc import HybridXC
    torch.manual_seed(42)
    n = 2
    nnmodel = dummymodel()
    k = HybridXC("lda_x", nnmodel, aweight0=0.0)
    densinfo = ValGrad(
        value=torch.rand((n,), dtype=torch.float32).requires_grad_())
    output = k.get_edensityxc(densinfo).detach()
    expected_output = torch.tensor([-0.6988, -0.2108], dtype=torch.float64)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)

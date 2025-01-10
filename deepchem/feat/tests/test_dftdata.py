import pytest
import numpy as np
try:
    from deepchem.feat.dft_data import DFTEntry
    from deepchem.utils.dftutils import KSCalc
    from deepchem.utils.dft_utils import KS
    import torch
    has_dft_req = True
except:
    has_dft_req = False


@pytest.mark.torch
def test_entryDM():
    e_type = 'dm'
    true_val = 'deepchem/feat/tests/data/dftHF_output.npy'
    systems = [{
        'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
        'basis': '6-311++G(3df,3pd)'
    }]
    dm_entry_for_HF = DFTEntry.create(e_type, true_val, systems)
    assert dm_entry_for_HF.entry_type == 'dm'
    dm_HF_system0 = dm_entry_for_HF.get_systems()[0]
    mol_dqc = dm_HF_system0.get_dqc_mol(dm_entry_for_HF)
    hf_zs = torch.Tensor([1, 9])
    hf_pos = torch.DoubleTensor([[0.86625, 0.0000, 0.0000],
                                 [-0.86625, 0.0000, 0.0000]])
    assert (mol_dqc.atomzs == hf_zs).all()
    assert (hf_pos.numpy() == mol_dqc.atompos.numpy()).all()
    dm0 = dm_entry_for_HF.get_true_val()
    assert dm0.shape == (57, 57)


@pytest.mark.torch
def test_entryAE():
    e_type = 'ae'
    true_val = '0.09194410469'
    weight = 1340
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
    ae_entry_for_LiH = DFTEntry.create(e_type, true_val, systems, weight)
    assert ae_entry_for_LiH.entry_type == 'ae'
    assert ae_entry_for_LiH.get_true_val() == 0.09194410469
    assert ae_entry_for_LiH.get_weight() == 1340

    def run(syst):
        mol_dqc = syst.get_dqc_mol(ae_entry_for_LiH)
        qc = KS(mol_dqc, xc='lda_x').run()
        return KSCalc(qc)

    qcs = [run(syst) for syst in ae_entry_for_LiH.get_systems()]
    val = np.array(0.05362133)
    calc_val = ae_entry_for_LiH.get_val(qcs)
    assert np.allclose(val, calc_val)

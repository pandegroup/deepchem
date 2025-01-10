from deepchem.utils.jackhmmer import Jackhmmer
import pytest


@pytest.mark.fold
def test_jackhmmer():
    j = Jackhmmer(database_path='./assets/test.fasta')
    result = j.query(
        "./assets/sequence.fasta")  # sequence taken exactly from test.fasta
    exp_val = '#=GC RF                             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # reference line should match exactly, denoted by 'x'
    for line in result[0]['sto'].split('\n'):
        if line.startswith('#=GC RF'):
            assert line == exp_val

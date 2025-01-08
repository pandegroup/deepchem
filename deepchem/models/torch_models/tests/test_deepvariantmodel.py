import os
import numpy as np
import pytest
import deepchem as dc
from deepchem.models.torch_models.deepvariantmodel import DeepVariant


@pytest.mark.torch
def test_deepvariant_vcf():
    """Test VCF generation from predictions."""

    n_samples = 5
    n_features = 6
    n_tasks = 3
    img_size = 299

    # Generate dummy dataset
    X = np.random.rand(n_samples, n_features, img_size,
                       img_size).astype(np.float32)
    y = []
    for i in range(n_samples):
        y.append({
            'chrom': 'chr1',
            'pos': 1000 + i,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'depth': 30
        })

    dataset = dc.data.datasets.NumpyDataset(X, y)
    model = DeepVariant(n_tasks=n_tasks, in_channels=n_features)

    # Create temp file
    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    fname = tmpfile.name

    try:
        model.call_variants(dataset, fname, "test_sample")
        with open(fname) as f:
            lines = f.readlines()
            assert any("##fileformat=VCFv4.2" in line for line in lines)
            assert sum(
                1 for line in lines if not line.startswith("#")) == n_samples
    finally:
        os.unlink(fname)

import os
import tempfile
import deepchem as dc


def test_load_singleton_csv():
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["endpoint"]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    X = loader.create_dataset(fin.name)
    assert len(X) == 1
    os.remove(fin.name)


def test_empty_task_dataset_y():
    """Test that dataset.y doesn't throw an error when there are no tasks specified in CSVLoader"""
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles\nC\nCCCC")
    fin.close()
    featurizer = dc.feat.SNAPFeaturizer()
    loader = dc.data.CSVLoader(tasks=[],
                               feature_field="smiles",
                               featurizer=featurizer,
                               id_field="smiles")
    X = loader.create_dataset(fin.name)
    assert X.y == []
    os.remove(fin.name)

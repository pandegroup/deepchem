import os
import pytest
import numpy as np
import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer


@pytest.mark.torch
def get_classification_dataset():
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_classification.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                               np.mean,
                               mode="classification")
    return dataset, metric


@pytest.mark.torch
def get_multitask_classification_dataset():
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/multitask_example.csv')
    loader = dc.data.CSVLoader(tasks=['task0', 'task1', 'task2'],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                               np.mean,
                               mode="classification")
    return dataset, metric


@pytest.mark.torch
def get_multitask_regression_dataset():
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/multitask_regression.csv')
    loader = dc.data.CSVLoader(tasks=['task0', 'task1', 'task2'],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


@pytest.mark.torch
def get_regression_dataset():
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")

    return dataset, metric


@pytest.mark.torch
def test_infographencoder():
    import numpy as np
    import torch
    from deepchem.models.torch_models.infograph import InfoGraphEncoder
    from deepchem.feat.graph_data import GraphData, BatchGraphData
    torch.manual_seed(123)
    embedding_dim = 8
    num_nodes = 10
    num_graphs = 3
    encoder = InfoGraphEncoder(num_features=5,
                               edge_features=10,
                               embedding_dim=embedding_dim)

    data = []
    for i in range(num_graphs):
        node_features = np.random.randn(num_nodes, 5)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        edge_features = np.random.randn(3, 10)

        data.append(
            GraphData(node_features=node_features,
                      edge_index=edge_index,
                      edge_features=edge_features))
    data = BatchGraphData(data).numpy_to_torch()

    embedding, feature_map = encoder(data)

    assert embedding.shape == torch.Size([num_graphs, 2 * embedding_dim])
    assert feature_map.shape == torch.Size(
        [num_nodes * num_graphs, embedding_dim])


@pytest.mark.torch
def test_GINEcnoder():
    import numpy as np
    import torch
    from deepchem.models.torch_models.infograph import GINEncoder
    from deepchem.feat.graph_data import GraphData, BatchGraphData
    torch.manual_seed(123)
    num_gc_layers = 2
    embedding_dim = 8
    num_nodes = 10
    num_graphs = 3
    encoder = GINEncoder(num_features=5,
                         embedding_dim=embedding_dim,
                         num_gc_layers=num_gc_layers)

    data = []
    for i in range(num_graphs):
        node_features = np.random.randn(num_nodes, 5)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        edge_features = np.random.randn(3, 10)

        data.append(
            GraphData(node_features=node_features,
                      edge_index=edge_index,
                      edge_features=edge_features))
    data = BatchGraphData(data).numpy_to_torch()

    embedding, intermediate_embeddings = encoder(data)

    assert embedding.shape == torch.Size([num_graphs, embedding_dim])
    assert intermediate_embeddings.shape == torch.Size(
        [num_nodes * num_graphs, embedding_dim])


@pytest.mark.torch
def test_infographstar_regression_semisupervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=2,
                               task='semisupervised')

    model.fit(dataset, nb_epoch=50)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.2


@pytest.mark.torch
def test_infographstar_classification_semisupervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_classification_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=3,
                               task='semisupervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_infograph_finetune_classification():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_classification_dataset()
    num_feat = 30
    edge_dim = 11
    model = InfoGraphModel(num_feat,
                           edge_dim,
                           num_gc_layers=3,
                           task='classification',
                           device=torch.device('cpu'),
                           n_classes=2,
                           n_tasks=1)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_infographstar_multitask_classification_supervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_multitask_classification_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64

    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               task='supervised',
                               mode='classification',
                               num_classes=2,
                               num_tasks=3)

    model.fit(dataset, nb_epoch=200)
    scores = model.evaluate(dataset, [metric])
    # .8 to save resources for a difficult task
    assert scores['mean-roc_auc_score'] >= 0.8


@pytest.mark.torch
def test_infographstar_multitask_regression_supervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_multitask_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64

    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=3,
                               task='supervised',
                               mode='regression',
                               num_tasks=3)

    model.fit(dataset, nb_epoch=200)
    scores = model.evaluate(dataset, [metric])
    # .2 to save resources for a difficult task
    assert scores['mean_absolute_error'] < 0.2


@pytest.mark.torch
def test_infographstar_regression_supervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    dataset, metric = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=3,
                               task='supervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_infograph():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    import torch
    torch.manual_seed(123)
    dataset, _ = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    model = InfoGraphModel(num_feat, edge_dim)
    # first iteration loss is around 50
    loss = model.fit(dataset, nb_epoch=20)
    assert loss < 25


@pytest.mark.torch
def test_infograph_pretrain_overfit():
    """This tests the intended use of InfoGraph and InfoGraphStar together, with InfoGraph serving as a pretraining step for InfoGraphStar."""
    from deepchem.models.torch_models.infograph import InfoGraphModel, InfoGraphStarModel
    import torch
    torch.manual_seed(123)
    np.random.seed(123)

    dataset, _ = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 32

    infograph = InfoGraphModel(num_feat, edge_dim)
    infographstar = InfoGraphStarModel(num_feat,
                                       edge_dim,
                                       dim,
                                       num_gc_layers=2,
                                       task='semisupervised')

    loss1 = infographstar.fit(dataset, nb_epoch=10)
    infograph.fit(dataset, nb_epoch=20)
    infographstar.load_from_pretrained(infograph, ['unsup_encoder'])
    loss2 = infographstar.fit(dataset, nb_epoch=10)
    infographstar.fit(dataset, nb_epoch=200)
    prediction = infographstar.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(np.round(dataset.y), np.round(prediction))
    assert loss1 > loss2


@pytest.mark.torch
def test_infographstar_fit_restore():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, _ = get_classification_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64

    model = InfoGraphStarModel(num_feat, edge_dim, dim, task='supervised')

    model.fit(dataset, nb_epoch=100)

    model2 = InfoGraphStarModel(num_feat,
                                edge_dim,
                                dim,
                                training_mode='supervised',
                                model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = model2.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(dataset.y, np.round(prediction))


@pytest.mark.torch
def test_infograph_pretrain_finetune(tmpdir):
    from deepchem.models.torch_models.infograph import InfoGraphModel
    import torch
    torch.manual_seed(123)
    np.random.seed(123)

    dataset, _ = get_regression_dataset()
    num_feat = 30
    edge_dim = 11

    pretrain_model = InfoGraphModel(num_feat,
                                    edge_dim,
                                    num_gc_layers=1,
                                    model_dir=tmpdir,
                                    device=torch.device('cpu'))
    pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)
    assert pretraining_loss
    pretrain_model.save_checkpoint()

    finetune_model = InfoGraphModel(num_feat,
                                    edge_dim,
                                    num_gc_layers=1,
                                    task='regression',
                                    n_tasks=1,
                                    model_dir=tmpdir,
                                    device=torch.device('cpu'))
    finetune_model.restore(components=['encoder'])
    finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)
    assert finetuning_loss

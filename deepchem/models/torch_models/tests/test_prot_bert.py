import os

import deepchem as dc
import pytest

try:
    import torch
    from deepchem.models.torch_models.prot_bert import ProtBERT
    # import torch.nn.functional as F
    # import torch.nn as nn
except ModuleNotFoundError:
    pass

# @pytest.mark.torch
# def test_prot_bert_pretraining_mlm(protein_classification_dataset):
#     model_path = 'Rostlab/prot_bert'
#     model = ProtBERT(task='mlm', model_path=model_path, n_tasks=1)
#     loss = model.fit(protein_classification_dataset, nb_epoch=1)
#     assert loss

#     model_path = 'Rostlab/prot_bert_BFD'
#     model = ProtBERT(task='mlm', model_path=model_path, n_tasks=1)
#     loss = model.fit(protein_classification_dataset, nb_epoch=1)
#     assert loss

# @pytest.mark.torch
# def test_prot_bert_finetuning(protein_classification_dataset):

#    model_path = 'Rostlab/prot_bert'

#    model = ProtBERT(task='classification',
#                     model_path=model_path,
#                     n_tasks=1,
#                     cls_name="LogReg")
#    loss = model.fit(protein_classification_dataset, nb_epoch=1)
#    eval_score = model.evaluate(protein_classification_dataset,
#                                metrics=dc.metrics.Metric(
#                                    dc.metrics.accuracy_score))
#    assert eval_score, loss
#    prediction = model.predict(protein_classification_dataset)
#    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)

#    model = ProtBERT(task='classification',
#                     model_path=model_path,
#                     n_tasks=1,
#                     cls_name="FFN")
#    loss = model.fit(protein_classification_dataset, nb_epoch=1)
#    eval_score = model.evaluate(protein_classification_dataset,
#                                metrics=dc.metrics.Metric(
#                                    dc.metrics.accuracy_score))
#    assert eval_score, loss
#    prediction = model.predict(protein_classification_dataset)
#    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)

#    class SimpleCNN(nn.Module):

#        def __init__(self, input_dim=1024, num_classes=2):
#            super(SimpleCNN, self).__init__()
#            self.conv1 = nn.Conv1d(in_channels=1,
#                                   out_channels=32,
#                                   kernel_size=3,
#                                   padding=1)
#            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#            self.fc1 = nn.Linear(32 * (input_dim // 2), num_classes)

#        def forward(self, x):
#            x = x.unsqueeze(1)
#            x = self.pool(F.relu(self.conv1(x)))
#            x = x.view(x.size(0), -1)
#            x = self.fc1(x)
#            return x

#    custom_torch_CNN_network = SimpleCNN()
#    model = ProtBERT(task='classification',
#                     model_path=model_path,
#                     n_tasks=1,
#                     cls_name="custom",
#                     classifier_net=custom_torch_CNN_network)
#    loss = model.fit(protein_classification_dataset, nb_epoch=1)
#    eval_score = model.evaluate(protein_classification_dataset,
#                                metrics=dc.metrics.Metric(
#                                    dc.metrics.accuracy_score))
#    assert eval_score, loss
#    prediction = model.predict(protein_classification_dataset)
#    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)


# @pytest.mark.torch
# def test_protbert_load_from_pretrained(tmpdir):
#     pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
#     finetune_model_dir = os.path.join(tmpdir, 'finetune')
#     model_path = 'Rostlab/prot_bert'
#     pretrain_model = ProtBERT(task='mlm',
#                               model_path=model_path,
#                               n_tasks=1,
#                               model_dir=pretrain_model_dir)
#     pretrain_model.save_checkpoint()

#     finetune_model = ProtBERT(task='classification',
#                               model_path=model_path,
#                               n_tasks=1,
#                               cls_name="LogReg",
#                               model_dir=finetune_model_dir)
#     finetune_model.load_from_pretrained(pretrain_model_dir)

#     # check weights match
#     pretrain_model_state_dict = pretrain_model.model.state_dict()
#     finetune_model_state_dict = finetune_model.model.state_dict()

#     pretrain_base_model_keys = [
#         key for key in pretrain_model_state_dict.keys() if 'bert' in key
#     ]
#     matches = [
#         torch.allclose(pretrain_model_state_dict[key],
#                        finetune_model_state_dict[key])
#         for key in pretrain_base_model_keys
#     ]

#     assert all(matches)


# @pytest.mark.torch
# def test_protbert_save_reload(tmpdir):
#     model_path = 'Rostlab/prot_bert'
#     model = ProtBERT(task='classification',
#                      model_path=model_path,
#                      n_tasks=1,
#                      cls_name="FFN",
#                      model_dir=tmpdir)
#     model._ensure_built()
#     model.save_checkpoint()

#     model_new = ProtBERT(task='classification',
#                          model_path=model_path,
#                          n_tasks=1,
#                          cls_name="FFN",
#                          model_dir=tmpdir)
#     model_new.restore()

#     old_state = model.model.state_dict()
#     new_state = model_new.model.state_dict()
#     matches = [
#         torch.allclose(old_state[key], new_state[key])
#         for key in old_state.keys()
#     ]

#     # all keys values should match
#     assert all(matches)


# @pytest.mark.torch
# def test_protbert_overfit():
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     featurizer = dc.feat.DummyFeaturizer()
#     tasks = ["outcome"]
#     loader = dc.data.CSVLoader(tasks=tasks,
#                                feature_field="protein",
#                                featurizer=featurizer)
#     dataset = loader.create_dataset(
#         os.path.join(current_dir,
#                      "../../tests/assets/example_protein_classification.csv"))
#     model_path = 'Rostlab/prot_bert'
#     finetune_model = ProtBERT(task='classification',
#                               model_path=model_path,
#                               n_tasks=1,
#                               cls_name="FFN",
#                               batch_size=1,
#                               learning_rate=1e-5)
#     for param in finetune_model.model.bert.parameters():
#         param.requires_grad = False

#     classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
#     finetune_model.fit(dataset, nb_epoch=20)
#     eval_score = finetune_model.evaluate(dataset, [classification_metric])
#     assert eval_score[classification_metric.name] > 0.9

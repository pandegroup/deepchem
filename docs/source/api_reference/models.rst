Model Classes
=============

DeepChem maintains an extensive collection of models for scientific
applications. DeepChem's focus is on facilitating scientific applications, so
we support a broad range of different machine learning frameworks (currently
scikit-learn, xgboost, TensorFlow, and PyTorch) since different frameworks are
more and less suited for different scientific applications.

.. include:: model_cheatsheet.rst

Model
-----

.. autoclass:: deepchem.models.Model
  :members:

Scikit-Learn Models
===================

Scikit-learn's models can be wrapped so that they can interact conveniently
with DeepChem. Oftentimes scikit-learn models are more robust and easier to
train and are a nice first model to train.

SklearnModel
------------

.. autoclass:: deepchem.models.SklearnModel
  :members:

Gradient Boosting Models
========================

Gradient Boosting Models (LightGBM and XGBoost) can be wrapped so they can interact with DeepChem.

GBDTModel
------------

.. autoclass:: deepchem.models.GBDTModel
  :members:


Deep Learning Infrastructure
============================

DeepChem maintains a lightweight layer of common deep learning model
infrastructure that can be used for models built with different underlying
frameworks. The losses and optimizers can be used for both TensorFlow and
PyTorch models.

Losses
------

.. autoclass:: deepchem.models.losses.Loss
  :members:

.. autoclass:: deepchem.models.losses.L1Loss
  :members:

.. autoclass:: deepchem.models.losses.HuberLoss
  :members:

.. autoclass:: deepchem.models.losses.L2Loss
  :members:

.. autoclass:: deepchem.models.losses.HingeLoss
  :members:

.. autoclass:: deepchem.models.losses.SquaredHingeLoss
  :members:

.. autoclass:: deepchem.models.losses.PoissonLoss
  :members:

.. autoclass:: deepchem.models.losses.BinaryCrossEntropy
  :members:

.. autoclass:: deepchem.models.losses.CategoricalCrossEntropy
  :members:

.. autoclass:: deepchem.models.losses.SigmoidCrossEntropy
  :members:

.. autoclass:: deepchem.models.losses.SoftmaxCrossEntropy
  :members:

.. autoclass:: deepchem.models.losses.SparseSoftmaxCrossEntropy
  :members:

.. autoclass:: deepchem.models.losses.VAE_ELBO
  :members:

.. autoclass:: deepchem.models.losses.VAE_KLDivergence
  :members:

.. autoclass:: deepchem.models.losses.ShannonEntropy
  :members:

.. autoclass:: deepchem.models.losses.GlobalMutualInformationLoss
  :members:

.. autoclass:: deepchem.models.losses.LocalMutualInformationLoss
  :members:

.. autoclass:: deepchem.models.losses.GroverPretrainLoss
  :members:

.. autoclass:: deepchem.models.losses.EdgePredictionLoss
  :members:

.. autoclass:: deepchem.models.losses.GraphNodeMaskingLoss
  :members:

.. autoclass:: deepchem.models.losses.GraphEdgeMaskingLoss
  :members:

.. autoclass:: deepchem.models.losses.DeepGraphInfomaxLoss
  :members:

.. autoclass:: deepchem.models.losses.GraphContextPredLoss
  :members:

.. autoclass:: deepchem.models.losses.DensityProfileLoss
  :members:

.. autoclass:: deepchem.models.losses.NTXentMultiplePositives
  :members:

Optimizers
----------

.. autoclass:: deepchem.models.optimizers.Optimizer
  :members:

.. autoclass:: deepchem.models.optimizers.LearningRateSchedule
  :members:

.. autoclass:: deepchem.models.optimizers.AdaGrad
  :members:

.. autoclass:: deepchem.models.optimizers.Adam
  :members:

.. autoclass:: deepchem.models.optimizers.AdamW
  :members:

.. autoclass:: deepchem.models.optimizers.SparseAdam
  :members:

.. autoclass:: deepchem.models.optimizers.RMSProp
  :members:

.. autoclass:: deepchem.models.optimizers.GradientDescent
  :members:

.. autoclass:: deepchem.models.optimizers.ExponentialDecay
  :members:

.. autoclass:: deepchem.models.optimizers.PolynomialDecay
  :members:

.. autoclass:: deepchem.models.optimizers.LinearCosineDecay
  :members:

.. autoclass:: deepchem.models.optimizers.LambdaLRWithWarmup
  :members:

Keras Models
============

DeepChem extensively uses `Keras`_ to build deep learning models.


KerasModel
----------

Training loss and validation metrics can be automatically logged to `Weights & Biases`_ with the following commands::

  # Install wandb in shell
  pip install wandb

  # Login in shell (required only once)
  wandb login
  # Login in notebook (required only once)
  import wandb
  wandb.login()

  # Initialize a WandbLogger
  logger = WandbLogger(…)

  # Set `wandb_logger` when creating `KerasModel`
  import deepchem as dc
  # Log training loss to wandb
  model = dc.models.KerasModel(…, wandb_logger=logger)
  model.fit(…)

  # Log validation metrics to wandb using ValidationCallback
  import deepchem as dc
  vc = dc.models.ValidationCallback(…)
  model = KerasModel(…, wandb_logger=logger)
  model.fit(…, callbacks=[vc])
  logger.finish()

.. _`Keras`: https://keras.io/

.. _`Weights & Biases`: http://docs.wandb.com/

.. autoclass:: deepchem.models.KerasModel
  :members:

TensorflowMultitaskIRVClassifier
--------------------------------

.. autoclass:: deepchem.models.TensorflowMultitaskIRVClassifier
  :members:

RobustMultitaskClassifier
-------------------------

.. autoclass:: deepchem.models.RobustMultitaskClassifier
  :members:

RobustMultitaskRegressor
------------------------

.. autoclass:: deepchem.models.RobustMultitaskRegressor
  :members:

ProgressiveMultitaskClassifier
------------------------------

.. autoclass:: deepchem.models.ProgressiveMultitaskClassifier
  :members:

ProgressiveMultitaskRegressor
-----------------------------

.. autoclass:: deepchem.models.ProgressiveMultitaskRegressor
  :members:

WeaveModel
----------

.. autoclass:: deepchem.models.WeaveModel
  :members:

DTNNModel
---------

.. autoclass:: deepchem.models.DTNNModel
  :members:

DAGModel
--------

.. autoclass:: deepchem.models.DAGModel
  :members:

GraphConvModel
--------------

.. autoclass:: deepchem.models.GraphConvModel
  :members:

MPNNModel
---------

.. autoclass:: deepchem.models.MPNNModel
  :members:

BasicMolGANModel
----------------

.. autoclass:: deepchem.models.BasicMolGANModel
  :members:

ScScoreModel
------------

.. autoclass:: deepchem.models.ScScoreModel
  :members:

SeqToSeq
--------

.. autoclass:: deepchem.models.SeqToSeq
  :members:

GAN
---

.. autoclass:: deepchem.models.GAN
  :members:

WGAN
^^^^

.. autoclass:: deepchem.models.WGAN
  :members:

TextCNNModel
------------

.. autoclass:: deepchem.models.TextCNNModel
  :members:


AtomicConvModel
---------------

.. autoclass:: deepchem.models.AtomicConvModel
  :members:


Smiles2Vec
----------

.. autoclass:: deepchem.models.Smiles2Vec
  :members:

ChemCeption
-----------

.. autoclass:: deepchem.models.ChemCeption
  :members:

NormalizingFlowModel
--------------------
The purpose of a normalizing flow is to map a simple distribution (that is
easy to sample from and evaluate probability densities for) to a more
complex distribution that is learned from data. Normalizing flows combine the
advantages of autoregressive models (which provide likelihood estimation
but do not learn features) and variational autoencoders (which learn feature
representations but do not provide marginal likelihoods). They are effective
for any application requiring a probabilistic model with these capabilities, e.g. generative modeling, unsupervised learning, or probabilistic inference.

.. autoclass:: deepchem.models.normalizing_flows.NormalizingFlowModel
  :members:


PyTorch Models
==============

DeepChem supports the use of `PyTorch`_ to build deep learning models.

.. _`PyTorch`: https://pytorch.org/

TorchModel
----------

You can wrap an arbitrary :code:`torch.nn.Module` in a :code:`TorchModel` object.

.. autoclass:: deepchem.models.TorchModel
  :members:

ModularTorchModel
-----------------

You can modify networks for different tasks by using a :code:`ModularTorchModel`.

.. autoclass:: deepchem.models.torch_models.modular.ModularTorchModel
  :members:

CNN
---

.. autoclass:: deepchem.models.CNN
  :members:

MultitaskRegressor
------------------

.. autoclass:: deepchem.models.MultitaskRegressor
  :members:

MultitaskFitTransformRegressor
------------------------------

.. autoclass:: deepchem.models.MultitaskFitTransformRegressor
  :members:

MultitaskClassifier
-------------------

.. autoclass:: deepchem.models.MultitaskClassifier
  :members:

CGCNNModel
----------

.. autoclass:: deepchem.models.CGCNNModel
  :members:


GATModel
--------

.. autoclass:: deepchem.models.GATModel
  :members:

GCNModel
--------

.. autoclass:: deepchem.models.GCNModel
  :members:

AttentiveFPModel
----------------

.. autoclass:: deepchem.models.AttentiveFPModel
  :members:

PagtnModel
----------

.. autoclass:: deepchem.models.PagtnModel
  :members:

AtomConvModel
-------------

.. autoclass:: deepchem.models.torch_models.AtomConvModel
  :members:

MPNNModel
---------

Note that this is an alternative implementation for MPNN and currently you can only import it from
``deepchem.models.torch_models``.

.. autoclass:: deepchem.models.torch_models.MPNNModel
  :members:

InfoGraphModel
--------------

.. autoclass:: deepchem.models.torch_models.InfoGraphModel
  :members:

InfoGraphStarModel
------------------

.. autoclass:: deepchem.models.torch_models.InfoGraphStarModel
  :members:


GNNModular
----------

.. autoclass:: deepchem.models.torch_models.gnn.GNNModular
  :members:

InfoMax3DModular
----------------

.. autoclass:: deepchem.models.torch_models.gnn3d.InfoMax3DModular
  :members:


LCNNModel
---------

.. autoclass:: deepchem.models.LCNNModel
  :members:

MEGNetModel
-----------

.. autoclass:: deepchem.models.MEGNetModel
  :members:

MATModel
--------

.. autoclass:: deepchem.models.torch_models.MATModel
  :members:

NormalizingFlowModel
--------------------

.. autoclass:: deepchem.models.torch_models.flows.NormalizingFlowModel
  :members:

DMPNNModel
----------

.. autoclass:: deepchem.models.torch_models.DMPNNModel
  :members:

GroverModel
-----------

.. autoclass:: deepchem.models.torch_models.GroverModel
  :members:

DTNNModel
---------

.. autoclass:: deepchem.models.torch_models.DTNNModel
  :members:

SeqToSeqModel
-------------
.. autoclass:: deepchem.models.torch_models.SeqToSeqModel
  :members:

GAN
---

.. autoclass:: deepchem.models.torch_models.GAN
  :members:

GANModel
--------

.. autoclass:: deepchem.models.torch_models.GANModel
  :members:

WGANModel
---------

.. autoclass:: deepchem.models.torch_models.WGANModel
  :members:

BasicMolGANModel
----------------

.. autoclass:: deepchem.models.torch_models.BasicMolGANModel
  :members:

Weave
----------

.. autoclass:: deepchem.models.torch_models.Weave
  :members:

WeaveModel
----------

.. autoclass:: deepchem.models.torch_models.WeaveModel
  :members:

ProgressiveMultitaskClassifier
-------------------------

.. autoclass:: deepchem.models.torch_models.ProgressiveMultitaskClassifier
  :members:

ProgressiveMultitaskRegressor
-------------------------

.. autoclass:: deepchem.models.torch_models.ProgressiveMultitaskRegressor
  :members:

RobustMultitaskClassifier
-------------------------

.. autoclass:: deepchem.models.torch_models.RobustMultitaskClassifier
  :members:

RobustMultitaskRegressor
------------------------

.. autoclass:: deepchem.models.torch_models.RobustMultitaskRegressor
  :members:
  
Density Functional Theory Model - XCModel
-----------------------------------------

.. autoclass:: deepchem.models.dft.dftxc.XCModel
  :members:

TextCNNModel
------------

.. autoclass:: deepchem.models.torch_models.TextCNNModel
  :members:

PINNModel
---------

.. autoclass:: deepchem.models.torch_models.PINNModel
  :members:

UNetModel
------------

.. autoclass:: deepchem.models.torch_models.UNetModel
  :members:

_GraphConvTorchModel
--------------------

.. autoclass:: deepchem.models.torch_models._GraphConvTorchModel
  :members:

GraphConvModel
--------------------

.. autoclass:: deepchem.models.torch_models.GraphConvModel
  :members:

Smiles2Vec
--------------------

.. autoclass:: deepchem.models.torch_models.Smiles2Vec
  :members:

Smiles2VecModel
--------------------

.. autoclass:: deepchem.models.torch_models.Smiles2VecModel
  :members:

MXMNet
------

.. autoclass:: deepchem.models.torch_models.MXMNet
  :members:

InceptionV3Model
----------------

.. autoclass:: deepchem.models.torch_models.InceptionV3Model
  :members:

MultitaskIRVClassifier
----------------

.. autoclass:: deepchem.models.torch_models.MultitaskIRVClassifier
  :members:

PyTorch Lightning Models
========================

DeepChem supports the use of `PyTorch-Lightning`_ to build PyTorch models.

.. _`PyTorch-Lightning`: https://www.pytorchlightning.ai/

DCLightningModule
-----------------

You can wrap an arbitrary :code:`TorchModel` in a :code:`DCLightningModule` object.

.. autoclass:: deepchem.models.DCLightningModule
  :members:

Jax Models
==========

DeepChem supports the use of `Jax`_ to build deep learning models.

.. _`Jax`: https://github.com/google/jax

JaxModel
--------

.. autoclass:: deepchem.models.JaxModel
  :members:

PinnModel
---------

.. autoclass:: deepchem.models.PINNModel
  :members:

Hugging Face Models
===================

HuggingFace models from the `transformers <https://huggingface.co/models>`_ library can wrapped using the wrapper :code:`HuggingFaceModel`

.. autoclass:: deepchem.models.torch_models.hf_models.HuggingFaceModel
  :members:


---------

.. autoclass:: deepchem.models.torch_models.chemberta.Chemberta
  :members:

MoLFormer
---------

.. autoclass:: deepchem.models.torch_models.molformer.MoLFormer
  :members:

ProtBERT
---------

.. autoclass:: deepchem.models.torch_models.prot_bert.ProtBERT
  :members:

DeepAbLLM
---------

.. autoclass:: deepchem.models.torch_models.antibody_modeling.DeepAbLLM
  :members:

OneFormer
---------

.. autoclass:: deepchem.models.torch_models.oneformer.OneFormer
  :members:

Trainer
=======

A `Trainer` object automates the scaling of DeepChem model's training into multi-gpu and multi-node infrastructures.

DistributedTrainer
------------------

.. autoclass:: deepchem.trainer.DistributedTrainer
  :members:

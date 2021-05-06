Model Classes
=============

DeepChem maintains an extensive collection of models for scientific
applications. DeepChem's focus is on facilitating scientific applications, so
we support a broad range of different machine learning frameworks (currently
scikit-learn, xgboost, TensorFlow, and PyTorch) since different frameworks are
more and less suited for different scientific applications.

Model Cheatsheet
----------------
If you're just getting started with DeepChem, you're probably interested in the
basics. The place to get started is this "model cheatsheet" that lists various
types of custom DeepChem models. Note that some wrappers like :code:`SklearnModel`
and :code:`GBDTModel` which wrap external machine learning libraries are excluded,
but this table is otherwise complete.

As a note about how to read this table, each row describes what's needed to
invoke a given model. Some models must be applied with given :code:`Transformer` or
:code:`Featurizer` objects. Some models also have custom training methods. You can
read off what's needed to train the model from the table below.


+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| Model                                  | Type       | Input Type           | Transformations        | Acceptable Featurizers                                         | Fit Method           |
+========================================+============+======================+========================+================================================================+======================+
| :code:`AtomicConvModel`                | Classifier/| Tuple                |                        | :code:`ComplexNeighborListFragmentAtomicCoordinates`           | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ChemCeption`                    | Classifier/| Tensor of shape      |                        | :code:`SmilesToImage`                                          | :code:`fit`          |
|                                        | Regressor  | :code:`(N, M, c)`    |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`CNN`                            | Classifier/| Tensor of shape      |                        |                                                                | :code:`fit`          |
|                                        | Regressor  | :code:`(N, c)` or    |                        |                                                                |                      |
|                                        |            | :code:`(N, M, c)` or |                        |                                                                |                      |
|                                        |            | :code:`(N, M, L, c)` |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`DTNNModel`                      | Classifier/| Matrix of            |                        | :code:`CoulombMatrix`                                          | :code:`fit`          |
|                                        | Regressor  | shape :code:`(N, N)` |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`DAGModel`                       | Classifier/| :code:`ConvMol`      | :code:`DAGTransformer` | :code:`ConvMolFeaturizer`                                      | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`GraphConvModel`                 | Classifier/| :code:`ConvMol`      |                        | :code:`ConvMolFeaturizer`                                      | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MPNNModel`                      | Classifier/| :code:`WeaveMol`     |                        | :code:`WeaveFeaturizer`                                        | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskClassifier`            | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskRegressor`             | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskFitTransformRegressor` | Regressor  | Vector of            | Any                    | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`MultitaskIRVClassifier`         | Classifier | Vector of            | :code:`IRVTransformer` | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ProgressiveMultitaskClassifier` | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ProgressiveMultitaskRegressor`  | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`RobustMultitaskClassifier`      | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`RobustMultitaskRegressor`       | Regressor  | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`ScScoreModel`                   | Classifier | Vector of            |                        | :code:`CircularFingerprint`,                                   | :code:`fit`          | 
|                                        |            | shape :code:`(N,)`   |                        | :code:`RDKitDescriptors`,                                      |                      |
|                                        |            |                      |                        | :code:`CoulombMatrixEig`,                                      |                      |
|                                        |            |                      |                        | :code:`RdkitGridFeaturizer`,                                   |                      |
|                                        |            |                      |                        | :code:`BindingPocketFeaturizer`,                               |                      |
|                                        |            |                      |                        | :code:`ElementPropertyFingerprint`,                            |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`SeqToSeq`                       | Sequence   | Sequence             |                        |                                                                | :code:`fit_sequences`|
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`Smiles2Vec`                     | Classifier/| Sequence             |                        | :code:`SmilesToSeq`                                            | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`TextCNNModel`                   | Classifier/| String               |                        |                                                                | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`WGAN`                           | Adversarial| Pair                 |                        |                                                                | :code:`fit_gan`      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`CGCNNModel`                     | Classifier/| :code:`GraphData`    |                        | :code:`CGCNNFeaturizer`                                        | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`GATModel`                       | Classifier/| :code:`GraphData`    |                        | :code:`MolGraphConvFeaturizer`                                 | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`GCNModel`                       | Classifier/| :code:`GraphData`    |                        | :code:`MolGraphConvFeaturizer`                                 | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`AttentiveFPModel`               | Classifier/| :code:`GraphData`    |                        | :code:`MolGraphConvFeaturizer`                                 | :code:`fit`          |
|                                        | Regressor  |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+
| :code:`LCCNModel`                      | Regressor  | :code:`GraphData`    |                        | :code:`LCNNFeaturizer`                                         | :code:`fit`          |
|                                        |            |                      |                        |                                                                |                      |
+----------------------------------------+------------+----------------------+------------------------+----------------------------------------------------------------+----------------------+

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

  # Start a W&B run in your script (refer to docs for optional parameters)
  wandb.init(project="my project")

  # Set `wandb` arg when creating `KerasModel`
  model = KerasModel(…, wandb=True)

.. _`Keras`: https://keras.io/

.. _`Weights & Biases`: http://docs.wandb.com/

.. autoclass:: deepchem.models.KerasModel
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
---------

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

CNN
---

.. autoclass:: deepchem.models.CNN
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

MPNNModel
---------

Note that this is an alternative implementation for MPNN and currently you can only import it from
``deepchem.models.torch_models``.

.. autoclass:: deepchem.models.torch_models.MPNNModel
  :members:

LCNNModel
---------

.. autoclass:: deepchem.models.LCNNModel
  :members:

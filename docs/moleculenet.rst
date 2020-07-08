MoleculeNet
===========
The DeepChem library is packaged alongside the MoleculeNet suite of datasets. One of the most important parts of machine learning applications is finding a suitable dataset. The MoleculeNet suite has curated a whole range of datasets and loaded them into DeepChem :code:`dc.data.Dataset` objects for convenience.

Running Benchmark
-----------------

At present, there is only support for running benchmark models

.. autofunction:: deepchem.molnet.run_benchmark

Best Known Hyperparameters
--------------------------

MoleculeNet maintains a list of the currently best known
hyperparameters for various models on MoleculeNet benchmarks.

MoleculeNet Datasets
--------------------

MoleculeNet is actively maintained and contains a growing set of
different datasets. Here are the set of currently available
MoleculeNet datasets.

BACE Dataset
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_bace_classification

.. autofunction:: deepchem.molnet.load_bace_regression

BBBC Datasets
^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_bbbc001

.. autofunction:: deepchem.molnet.load_bbbc002

BBBP Datasets
^^^^^^^^^^^^^
BBBP stands for Blood-Brain-Barrier Penetration

.. autofunction:: deepchem.molnet.load_bbbp

Cell Counting Datasets
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_cell_counting

Chembl Datasets
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_chembl

Chembl25 Datasets
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_chembl25

Clearance Datasets
^^^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_clearance

Clintox Datasets
^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_clintox

Delaney Datasets
^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_delaney

Factors Datasets
^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_factors

HIV Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_hiv

HOPV Datasets
^^^^^^^^^^^^^
HOPV stands for the Harvard Organic Photovoltaic Dataset.

.. autofunction:: deepchem.molnet.load_hopv

HPPB Datasets
^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_hppb


KAGGLE Datasets
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_kaggle

Kinase Datasets
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_kinase


Lipo Datasets
^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_lipo

MUV Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_muv

NCI Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_nci

PCBA Datasets
^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_pcba

PDBBIND Datasets
^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_pdbbind

PPB Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_ppb

QM7 Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_qm7

.. autofunction:: deepchem.molnet.load_qm7_from_mat

.. autofunction:: deepchem.molnet.load_qm7b_from_mat

QM8 Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_qm8

QM9 Datasets
^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_qm9


SAMPL Datasets
^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_sampl


SIDER Datasets
^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_sider

SWEETLEAD Datasets
^^^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_sweetlead

Thermosol Datasets
^^^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_thermosol


Tox21 Datasets
^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_tox21

Toxcast Datasets
^^^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_toxcast

USPTO Datasets
^^^^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_uspto

UV Datasets
^^^^^^^^^^^

.. autofunction:: deepchem.molnet.load_uv

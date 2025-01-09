Contributing Models to DeepChem
=============================

This guide walks through the process of contributing new models to DeepChem, starting from existing open-source implementations.

Motivation
---------
Before starting your contribution, consider the following questions:

* Why should this model be added to DeepChem?
* What value does it bring to the community?
* Is there a need for a standardized, maintained implementation?
* How will this benefit researchers and practitioners?

Data Loading
-----------
Converting existing data loading mechanisms to DeepChem's patterns involves:

1. Understanding the original repository's data loading approach
2. Identifying or creating appropriate ``DataLoader`` objects
3. Ensuring compatibility with DeepChem's data structures

Example of creating a custom DataLoader:

.. code-block:: python

    from deepchem.data import DataLoader
    
    class CustomDataLoader(DataLoader):
        """
        Custom data loader for specific data types.
        
        Parameters
        ----------
        tasks : list[str]
            List of task names
        metadata_fn : str
            Path to metadata file
        """
        def __init__(self, tasks, metadata_fn, **kwargs):
            super(CustomDataLoader, self).__init__(tasks, **kwargs)
            self.metadata_fn = metadata_fn
        
        def create_dataset(self, data_dir, **kwargs):
            """
            Create a Dataset from input data directory.
            
            Parameters
            ----------
            data_dir : str
                Directory containing the input data
            """
            # Implementation here
            pass

Featurization
------------
Converting raw data into machine learning-ready format:

1. Review existing featurizers in DeepChem
2. Determine if a new featurizer is needed
3. Implement custom featurization if required

Example of implementing a custom featurizer:

.. code-block:: python

    from deepchem.feat import Featurizer
    
    class CustomFeaturizer(Featurizer):
        """
        Custom featurizer for specific data types.
        
        Parameters
        ----------
        kwargs : dict
            Additional arguments
        """
        def __init__(self, **kwargs):
            super(CustomFeaturizer, self).__init__(**kwargs)
        
        def _featurize(self, datapoint, **kwargs):
            """
            Featurize a single datapoint.
            
            Parameters
            ----------
            datapoint : object
                The datapoint to featurize
            """
            # Implementation here
            pass

Model Conversion
--------------
Steps for wrapping external models in DeepChem:

1. Choose appropriate base class (``KerasModel``, ``TorchModel``, or ``JaxModel``)
2. Handle Dataset conversions
3. Implement required methods

Example implementation:

.. code-block:: python

    from deepchem.models import KerasModel
    
    class CustomModel(KerasModel):
        """
        Custom model implementation.
        
        Parameters
        ----------
        n_tasks : int
            Number of tasks
        kwargs : dict
            Additional arguments for model
        """
        def __init__(self, n_tasks, **kwargs):
            super(CustomModel, self).__init__(n_tasks, **kwargs)
            
        def fit(self, dataset, **kwargs):
            """
            Train this model on a dataset.
            
            Parameters
            ----------
            dataset : Dataset
                The dataset to train on
            """
            X = dataset.X
            y = dataset.y
            # Implementation here

Metrics
-------
Consider whether custom metrics are needed:

1. Review existing metrics in ``deepchem.metrics``
2. Implement new metrics if required
3. Add appropriate tests

Example of custom metric implementation:

.. code-block:: python

    from deepchem.metrics import Metric
    
    class CustomMetric(Metric):
        """
        Custom metric implementation.
        """
        def __init__(self):
            super(CustomMetric, self).__init__()
            
        def compute_metric(self, y_true, y_pred, **kwargs):
            """
            Compute custom metric.
            
            Parameters
            ----------
            y_true : np.ndarray
                True values
            y_pred : np.ndarray
                Predicted values
            """
            # Implementation here
            pass

Datasets
--------
If contributing datasets to MoleculeNet:

1. Ensure proper formatting and documentation
2. Verify usage rights and licenses
3. Implement loading functions

Example dataset loader:

.. code-block:: python

    def load_custom_dataset(reload=True, data_dir=None):
        """
        Load custom dataset.
        
        Parameters
        ----------
        reload : bool, optional (default True)
            Whether to reload dataset from disk
        data_dir : str, optional
            Directory containing the dataset
            
        Returns
        -------
        tasks : list[str]
            List of task names
        datasets : tuple
            (train, valid, test) datasets
        """
        # Implementation here
        pass

Testing Requirements
------------------

All contributions must include:

1. Unit tests for all components
2. Integration tests
3. Documentation tests
4. Performance benchmarks

Example test structure:

.. code-block:: python

    import unittest
    import deepchem as dc
    
    class TestCustomModel(unittest.TestCase):
        """
        Tests for CustomModel implementation.
        """
        def setUp(self):
            """
            Set up test cases.
            """
            self.dataset = dc.data.NumpyDataset(...)
        
        def test_fit(self):
            """
            Test model fitting.
            """
            model = CustomModel(...)
            # Test implementation here
            
        def test_predict(self):
            """
            Test model prediction.
            """
            model = CustomModel(...)
            # Test implementation here

Contributing Process
------------------

1. Fork the DeepChem repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

Remember to:

* Follow DeepChem's code style
* Write clear commit messages
* Keep pull requests focused
* Respond to reviewer feedback
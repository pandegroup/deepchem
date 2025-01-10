
# Requirements
## env_common.yml
This yml file is  environment configuration file for the Anaconda package manager, targeting the creation of a Conda environment. This contains Python packages for scientific computations. Let's break down its components :-
- Name: deepchem is the name of the Conda environment being created.
- Channels: These are the channels from which packages will be fetched. In this case, it specifies conda-forge and defaults, which are popular channels for accessing a wide variety of packages.
- Dependencies: These are the packages that are required for the environment. They are divided into two types:
  * Direct Conda Packages:
      - openmm: A toolkit for molecular simulation.
      - mdtraj: A molecular dynamics analysis tool.
      - pdbfixer: A tool for fixing issues in PDB files.

  * PIP Packages:
       - numpy: Fundamental package for scientific computing with Python.
       - rdkit: A cheminformatics and machine learning library.
       - pre-commit: A framework for managing and maintaining multi-language pre-commit hooks.
       - biopython: A set of freely available tools for biological computation.
       - dgllife: A package for deep learning on graphs for chemical informatics.
       - lightgbm: A gradient boosting framework.
       - matminer: A data mining tool for materials science.
       - mordred: A Python library for computation of molecular descriptors.
       - networkx: A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
       - pillow: The Python Imaging Library (PIL) fork.
       - pubchempy: A Python wrapper for the PubChem PUG REST API.
       - pyGPGO: A Gaussian process optimization package.
       - pymatgen: A Python library for materials analysis.
       - simdna: A package for simulating DNA sequencing data.
       - transformers: State-of-the-art natural language processing for PyTorch and TensorFlow.
       - tokenizers: String tokenization library.
       - xgboost: A scalable and accurate implementation of gradient boosting machines.
       - gensim: A package for topic modeling and document similarity analysis.
       - tensorboard: TensorFlow's visualization toolkit.
This configuration aims at setting up an environment for deep learning and cheminformatics tasks, including molecular simulation, molecular dynamics analysis, cheminformatics, and materials science.

## env_ubuntu.yml
This yml file targets creation of conda environment with packages particularly supported by debian based ubuntu distributiion of linux.
- Name : 'deepchem' is the name of conda environment
- Channels: These are the channels from which packages will be fetched. It specifies conda-forge and bioconda, which are repositories focused on providing scientific and bioinformatics-related packages.
- Dependencies:
    * Conda Packages:
       * hhsuite: This is a suite of programs for sensitive protein sequence searching and protein structure prediction.
       * vina: This likely refers to Autodock Vina, a molecular docking program.
   * PIP Packages:
      * pysam: This is a Python module for reading and manipulating SAM (Sequence Alignment/Map) files.


## env_dqc.yml

- Name: dqc is the name of the Conda environment being created.
- Channels: These are the channels from which packages will be fetched. It includes conda-forge and defaults, which are common channels for accessing various packages.
- Dependencies:
  * Conda Packages:
    * pip: This indicates the installation of the pip package manager within the environment.
  * PIP Packages:
    * numpy: Fundamental package for scientific computing with Python.
    * scipy: Scientific computing library for Python.
    * h5py: A package for reading and writing HDF5 files from Python.
    * -f https://download.pytorch.org/whl/cpu/torch_stable.html: This is a URL specifying the location of PyTorch wheel files for CPU. It's used to specify the location from which to install PyTorch.
    * torch==2.1.0+cpu: PyTorch version 2.1.0 for CPU.
    * torch-geometric: Geometric deep learning extension library for PyTorch.
    * git+https://github.com/diffqc/dqc.git: This installs a Python package directly from the specified GitHub repository URL. In this case, it's installing a package named dqc from the repository diffqc/dqc.
    * xitorch: A library for numerical optimization and automatic differentiation in Python.
    * pylibxc2: Python bindings for the LibXC library, which provides a library of exchange-correlation functionals for density-functional theory.
    * pytest: A testing framework for Python.
    * PyYAML: A YAML parser and emitter for Python.
    * yamlloader: This might be a misspelling of yaml-loader, referring to a YAML loader library for Python.
    * pyscf[all]: Python library for quantum chemistry, with additional components installed using the [all] extra specifier.
This configuration seems to be geared towards setting up an environment for quantum chemistry computations and machine learning tasks, including libraries for deep learning, quantum chemistry, numerical optimization, testing, and YAML parsing.

## env_test.yml
This yml file defines a list of Python packages and their dependencies, specified for installation via pip.
Dependencies:
- PIP Packages:
  * flake8: A Python tool for style checking and static code analysis.
  * flaky: A plugin for Pytest that allows for resilient testing in the presence of flaky tests.
  * mypy: A static type checker for Python.
  * pytest: A testing framework for Python.
  * pytest-cov: Pytest plugin for measuring coverage.
  * types-pkg_resources: Type hints for the pkg_resources module.
  * types-setuptools: Type hints for the setuptools module.
  * yapf==0.32.0: Yet Another Python Formatter, version 0.32.0. It's a code formatter for Python files.
These dependencies are to be related to development tools and testing frameworks for Python projects, including style checking, type checking, testing, coverage measurement, and code formatting.


# jax
## env_jax.cpu.yml
- Dependencies: This section lists the dependencies of the Conda environment.
   * PIP Packages: These are Python packages installed via pip.
     * -f https://storage.googleapis.com/jax-releases/jax_releases.html: This flag specifies an extra index URL from which additional package versions can be downloaded. In this case, it points to JAX releases hosted on Google Cloud Storage.
     * jax: JAX is a Python library for composable transformations of numerical functions, including autograd, JIT (just-in-time compilation), and vectorized operations.
     * jaxlib: JAX's library for linear algebra and other numerical functions. It provides low-level functionality that JAX relies upon, such as memory management and device dispatch.
     * dm-haiku: Haiku is a neural network library for JAX, designed for clarity, flexibility, and maintainability.
     * optax: Optax is a gradient processing and optimization library for JAX, featuring various optimizers and utility functions for deep learning.
This configuration is geared toward setting up an environment for deep learning tasks utilizing JAX, Haiku, and Optax libraries.

## env_jax.gpu.yml
- Dependencies: This section lists the dependencies of the Conda environment.
  * PIP Packages: These are Python packages installed via pip.
    * -f https://storage.googleapis.com/jax-releases/jax_releases.html: This flag specifies an extra index URL from which additional package versions can be downloaded. In this case, it points to JAX releases hosted on Google Cloud Storage.
    * jax: JAX is a Python library for composable transformations of numerical functions, including autograd, JIT (just-in-time compilation), and vectorized operations.
    * jaxlib: JAX's library for linear algebra and other numerical functions. It provides low-level functionality that JAX relies upon, such as memory management and device dispatch.
This configuration is geared toward setting up an environment for deep learning tasks utilizing JAX and JAXlib libraries, with a focus on GPU acceleration.

# tensorflow
## env_tensorflow.cpu.yml

- Dependencies: This section lists the dependencies of the Python environment.
   * PIP Packages: These are Python packages installed via pip.
      * tensorflow: TensorFlow is an open-source deep learning framework developed by Google. It provides tools for building and training neural networks.
      * tensorflow_probability: TensorFlow Probability is a library for probabilistic reasoning and statistical analysis built on top of TensorFlow.
      * tensorflow_addons: TensorFlow Addons is a repository of additional functionality for TensorFlow, including custom layers, optimizers, and metrics.
This configuration sets up an environment suitable for deep learning tasks using TensorFlow, along with additional probabilistic and experimental functionalities provided by TensorFlow Probability and TensorFlow Addons, respectively.

# torch
## env_torch.cpu.yml
This yml configuration specifies dependencies for a Python environment, particularly focusing on machine learning tasks with PyTorch aand related packages
- Dependencies: This section lists the dependencies of the Python environment.
  * PIP Packages: These are Python packages installed via pip.
  * -f https://download.pytorch.org/whl/cpu/torch_stable.html: This flag specifies an extra index URL from which additional package versions of PyTorch for CPU can be downloaded.
  * -f https://data.pyg.org/whl/torch-1.12.0+cpu.html: This flag specifies an extra index URL from which additional package versions of PyTorch Geometric for CPU can be downloaded.
  * -f https://data.dgl.ai/wheels/repo.html: This flag specifies an extra index URL from which additional package versions of DGL (Deep Graph Library) can be downloaded.
  * dgl: Deep Graph Library is a Python package built for easy implementation of graph neural networks.
  * torch==2.1.0+cpu: PyTorch version 2.1.0 for CPU.
  * torch-geometric: PyTorch Geometric is a library for geometric deep learning with PyTorch, providing tools for handling geometric data, such as graphs and meshes.
  * lightning: Lightning is likely referring to PyTorch Lightning, a lightweight PyTorch wrapper for high-performance training. It simplifies the training process and supports distributed training.
This configuration sets up an environment suitable for machine learning tasks, particularly focusing on graph-based deep learning with PyTorch, DGL, and PyTorch Geometric, along with training support provided by PyTorch Lightning.


## env_torch.gpu.yml
This yml configuration defines dependencies for a Python environment, particularly focusing on machine learning tasks with PyTorch and related libraries, with CUDA support.
- Dependencies: This section lists the dependencies of the Python environment.
   * PIP Packages: These are Python packages installed via pip.
     * -f https://download.pytorch.org/whl/cu113/torch_stable.html: This flag specifies an extra index URL from which additional package versions of PyTorch for CUDA 11.3 can be downloaded.
     * -f https://data.pyg.org/whl/torch-1.11.0+cu113.html: This flag specifies an extra index URL from which additional package versions of PyTorch Geometric for CUDA 11.3 can be downloaded.
     * -f https://data.dgl.ai/wheels/repo.html: This flag specifies an extra index URL from which additional package versions of DGL (Deep Graph Library) can be downloaded.
     * dgl-cu111: Deep Graph Library with CUDA support for CUDA 11.1.
     * torch==2.1.0+cu113: PyTorch version 2.1.0 for CUDA 11.3.
     * torch-geometric: PyTorch Geometric is a library for geometric deep learning with PyTorch, providing tools for handling geometric data, such as graphs and meshes.
     * lightning: PyTorch Lightning is a lightweight PyTorch wrapper for high-performance training. It simplifies the training process and supports distributed training.
This configuration sets up an environment suitable for machine learning tasks, particularly focusing on graph-based deep learning with PyTorch, DGL, and PyTorch Geometric, along with training support provided by PyTorch Lightning, with CUDA support for GPU acceleration.


















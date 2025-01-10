Installation
============

Stable version
--------------

Install DeepChem via pip or conda by simply running,

.. code-block:: bash

    pip install deepchem

or 

.. code-block:: bash

    conda install -c conda-forge deepchem

Nightly build version
---------------------
The nightly version is built by the HEAD of DeepChem.

For using general utilites like Molnet, Featurisers, Datasets, etc, then, you install DeepChem via pip.  

.. code-block:: bash

    pip install --pre deepchem

Deepchem provides support for tensorflow, pytorch, jax and each require
a individual pip Installation.

For using models with tensorflow dependencies, you install using

.. code-block:: bash

    pip install --pre deepchem[tensorflow]

For using models with Pytorch dependencies, you install using

.. code-block:: bash

    pip install --pre deepchem[torch]

For using models with Jax dependencies, you install using

.. code-block:: bash

    pip install --pre deepchem[jax]

If GPU support is required, then make sure CUDA is installed and then install the desired deep learning framework using the links below before installing DeepChem

1. tensorflow - just cuda installed
2. pytorch - https://pytorch.org/get-started/locally/#start-locally
3. jax - https://github.com/google/jax#pip-installation-gpu-cuda

In :code:`zsh` square brackets are used for globbing/pattern matching. This means
you need to escape the square brackets in the above installation. You can do so by
including the dependencies in quotes like :code:`pip install --pre 'deepchem[jax]'`

Note: Support for jax is not available in windows (jax is not officially supported in windows).

Google Colab
------------

The fastest way to get up and running with DeepChem is to run it on
Google Colab. Check out one of the `DeepChem Tutorials`_ or this
`forum post`_ for Colab quick start guides.


Docker
------

If you want to install using a docker,
you can pull two kinds of images from `DockerHub`_.

- **deepchemio/deepchem:x.x.x**

  - Image built by using a conda (x.x.x is a version of DeepChem)
  - This image is built when we push x.x.x. tag
  - Dockerfile is put in `docker/tag`_ directory

- **deepchemio/deepchem:latest**

  - Image built from source codes
  - This image is built every time we commit to the master branch
  - Dockerfile is put in `docker/nightly`_ directory

First, you pull the image you want to use.

.. code-block:: bash

    docker pull deepchemio/deepchem:latest


Then, you create a container based on the image.

.. code-block:: bash

    docker run --rm -it deepchemio/deepchem:latest

If you want GPU support:

.. code-block:: bash

    # If nvidia-docker is installed
    nvidia-docker run --rm -it deepchemio/deepchem:latest
    docker run --runtime nvidia --rm -it deepchemio/deepchem:latest

    # If nvidia-container-toolkit is installed
    docker run --gpus all --rm -it deepchemio/deepchem:latest

You are now in a docker container which DeepChem was installed.
You can start playing with it in the command line.

.. code-block:: bash

    (deepchem) root@xxxxxxxxxxxxx:~/mydir# python
    Python 3.10.13 |Anaconda, Inc.| (default, Aug 24 2023, 12:59:26)
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import deepchem as dc

If you want to check the tox21 benchmark:

.. code-block:: bash

    # you can run our tox21 benchmark
    (deepchem) root@xxxxxxxxxxxxx:~/mydir# wget https://raw.githubusercontent.com/deepchem/deepchem/master/examples/benchmark.py
    (deepchem) root@xxxxxxxxxxxxx:~/mydir# python benchmark.py -d tox21 -m graphconv -s random

Jupyter Notebook
----------------------

**Installing via these steps will allow you to install and import DeepChem into a jupyter notebook within a conda virtual environment.**

**Prerequisite**

- Shell: Bash, Zsh, PowerShell
- Conda: >4.6


First, please create a conda virtual environment (here it's named "deepchem-test") and activate it. 

.. code-block:: bash

    conda create --name deepchem-test
    conda activate deepchem-test


Install DeepChem, Jupyter and matplotlib into the conda environment.

.. code-block:: bash

    conda install -y -c conda-forge nb_conda_kernels matplotlib
    pip install tensorflow
    pip install --pre deepchem 


You may need to use :code:`pip3` depending on your Python 3 pip installation. Install pip dependencies after deepchem-test is activated.

While the deepchem-test environment is activated, open Jupyter Notebook by running :code:`jupyter notebook`. Your terminal prompt should be prefixed with (deepchem-test).
Once Jupyter Notebook opens in a browser, select the new button, and select the environment "Python[conda env:deepchem-test]." This will open a notebook running in the deepchem-test conda virtual environment.

From source with conda
----------------------

**Installing via these steps will ensure you are installing from the source**.

**Prerequisite**

- Shell: Bash, Zsh, PowerShell
- Conda: >4.6


First, please clone the DeepChem repository from GitHub.

.. code-block:: bash

    git clone https://github.com/deepchem/deepchem.git
    cd deepchem


Then, execute the shell script. The shell scripts require two arguments,
**python version** and **gpu/cpu**.

.. code-block:: bash

    source scripts/install_deepchem_conda.sh 3.10 cpu


If you want GPU support (we supports only CUDA 11.8):

.. code-block:: bash

    source scripts/install_deepchem_conda.sh 3.10 gpu


If you are using the Windows and the PowerShell:

.. code-block:: ps1

    .\scripts\install_deepchem_conda.ps1 3.10 cpu

| Sometimes, PowerShell scripts can't be executed due to problems in Execution Policies.
| In that case, you can either change the Execution policies or use the bypass argument.


.. code-block:: ps1

    powershell -executionpolicy bypass -File .\scripts\install_deepchem_conda.ps1 3.10 cpu

| Before activating DeepChem environment, make sure conda has been initialized.
| Check if there is a :code:`(XXXX)` in your command line. 
| If not, use :code:`conda init <YOUR_SHELL_NAME>` to activate it, then:

.. code-block:: bash

    conda activate deepchem
    pip install -e .
    pytest -m "not slow" deepchem # optional


From source lightweight guide
-------------------------------------

**Installing via these steps will ensure you are installing from the source**.

**Prerequisite**

- Shell: Bash, Zsh, PowerShell
- Conda: >4.6


First, please clone the DeepChem repository from GitHub.

.. code-block:: bash

    git clone https://github.com/deepchem/deepchem.git
    cd deepchem

We would advise all users to use conda environment, following below-

.. code-block:: bash

    conda create --name deepchem python=3.10
    conda activate deepchem
    pip install -e .

DeepChem provides diffrent additional packages depending on usage & contribution
If one also wants to build the tensorflow environment, add this

.. code-block:: bash

    pip install -e .[tensorflow]

If one also wants to build the Pytorch environment, add this

.. code-block:: bash

    pip install -e .[torch]

If one also wants to build the Jax environment, add this

.. code-block:: bash

    pip install -e .[jax]

DeepChem has soft requirements, which can be installed on the fly during development inside the environment 
but if you want to install all the soft-dependencies at once, then take a look at 
`deepchem/requirements <https://github.com/deepchem/deepchem/tree/master/requirements>`_


.. _`DeepChem Tutorials`: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
.. _`forum post`: https://forum.deepchem.io/t/getting-deepchem-running-in-colab/81/7
.. _`DockerHub`: https://hub.docker.com/repository/docker/deepchemio/deepchem
.. _`docker/conda-forge`: https://github.com/deepchem/deepchem/tree/master/docker/conda-forge
.. _`docker/master`: https://github.com/deepchem/deepchem/tree/master/docker/master

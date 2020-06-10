#!/usr/bin/env bash
# Used to make a conda environment with deepchem

# This line is needed for using conda activate
# This command is nearly equal to `conda init` command
source $(conda info --root)/etc/profile.d/conda.sh

if [ -z "$python_version" ]
then
    echo "Using python 3.6 by default"
    export python_version=3.6
else
    echo "Using python "$python_version". But recommended to use python 3.6."
fi

if [ -z "$1" ];
then
    echo "Installing DeepChem in current env"
else
    export envname=$1
    conda create -y --name $envname "python>=$python_version"
    conda activate $envname
fi

yes | pip install --upgrade pip
conda install -y -q --override-channels -c deepchem -c rdkit -c conda-forge -c omnia \
    mdtraj \
    pdbfixer \
    rdkit \
    joblib \
    scikit-learn \
    networkx \
    pillow \
    pandas \
    pytest \
    pytest-cov \
    flaky \
    zlib \
    py-xgboost \
    simdna \
    setuptools \
    biopython \
    numpy
yes | pip install -U tensorflow tensorflow-probability

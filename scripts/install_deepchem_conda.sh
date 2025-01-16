# This script creates the new deepchem enviroment
# This script works on only Bash and Zsh

set -e # Exit if any command fails.

CMDNAME=`basename ${BASH_SOURCE:-$0}`
if [ $# -ne 2 ]; then
    echo "Please set two arguments."
    echo "Usage) source $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) source $CMDNAME 3.10 gpu" 1>&2
    return 1
fi

# This command is nearly equal to `conda init` command
# Need to use `conda activate` command
eval "$(conda shell.bash hook)"

# Create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$1
conda activate deepchem
conda install -c conda-forge conda-merge

dir="$PWD/requirements"
if [ "$2" = "gpu" ];
then
    # We expect the CUDA vesion is 11.8.
    conda-merge $dir/env_common.yml $dir/torch/env_torch.gpu.yml $dir/env_test.yml $dir/jax/env_jax.gpu.yml > $PWD/env.yml
    echo "Installing DeepChem in the GPU environment"
else
    if [ "$(uname)" = 'Darwin' ]; then
        if [ "$1" = "3.11" ]; then
            conda-merge $dir/env_common.yml $dir/env_mac_3_11.yml $dir/env_test.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.mac.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
        else
            if [[ $(uname -m) == 'arm64' ]]; then
                conda-merge $dir/env_common.yml $dir/env_mac_arm64.yml $dir/env_test.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.mac.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
                echo "Installing DeepChem for Apple Silicon in the CPU environment"
            else
                conda-merge $dir/env_common.yml $dir/env_mac.yml $dir/env_test.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.mac.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
            fi
        fi
    elif [ "$(uname)" = 'Linux' ]; then
        if [ "$1" = "3.11" ]; then
            sudo apt update
            sudo apt install -y libatlas-base-dev libblas-dev liblapack-dev libhdf5-dev
            conda-merge $dir/env_common.yml $dir/env_test.yml $dir/env_ubuntu_3_11.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
        else
            conda-merge $dir/env_common.yml $dir/env_test.yml $dir/env_ubuntu.yml $dir/tensorflow/env_tensorflow.cpu.yml $dir/torch/env_torch.cpu.yml $dir/jax/env_jax.cpu.yml > $PWD/env.yml
        fi
    fi
    echo "Installing DeepChem in the CPU environment"
fi

# Install all dependencies
conda env update --file $PWD/env.yml

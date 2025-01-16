"""Derived from https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/litmodule.py"""
import torch
from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Tuple, Optional, List, Any
import numpy as np
import warnings
try:
    from deepchem.models.dft.scf import XCNNSCF
    from deepchem.models.dft.nnxc import HybridXC
except Exception as e:
    warnings.warn(str(e))


class DFTXC(torch.nn.Module):
    """
    This layer initializes the neural network exchange correlation functional and
    the hybrid functional. It is then used to run the Kohn Sham iterations.

    Examples
    --------
    >>> import torch
    >>> from deepchem.feat.dft_data import DFTEntry
    >>> from deepchem.models.dft.dftxc import DFTXC
    >>> e_type = 'ie'
    >>> true_val= '0.53411947056'
    >>> systems = [{'moldesc': 'N 0 0 0',
    ...       'basis': '6-311++G(3df,3pd)',
    ...        'spin': '3'},
    ...       {'moldesc': 'N 0 0 0',
    ...       'basis': '6-311++G(3df,3pd)',
    ...       'charge': 1,
    ...        'spin': '2'}]
    >>> entry = DFTEntry.create(e_type, true_val, systems)
    >>> nnmodel = _construct_nn_model(input_size=2, hidden_size=10, n_layers=1,modeltype=1).to(torch.double)
    >>> model = DFTXC("lda_x", nnmodel)
    >>> output = model([entry])
    The 6-311++G(3df,3pd) basis for atomz 7 does not exist, but we will download it
    Downloaded to /home/runner/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/07.gaussian94

    """

    def __init__(self, xcstr: str, nnmodel: torch.nn.Module):
        """
        Parameters
        ----------
        xcstr: str
            The choice of xc to use. Some of the commonly used ones are:
            lda_x, lda_c_pw, lda_c_ow, lda_c_pz, lda_xc_lp_a, lda_xc_lp_b.
        nnmodel: torch.nn.Module
            the PyTorch model implementing the calculation

        Notes
        -----
        It is not necessary to use the default method(_construct_nn_model) with the XCModel.
        """
        super(DFTXC, self).__init__()
        self.xcstr = xcstr
        self.nnmodel = nnmodel

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: list
            list of entry objects that have been defined using DFTEntry

        Returns
        -------
        output: list of torch.Tensor
            Calculated value of the data point after running the Kohn Sham iterations
            using the neural network XC functional.
        """
        hybridxc = HybridXC(self.xcstr, self.nnmodel, aweight0=1.0)
        output = []
        for entry in inputs:
            evl = XCNNSCF(hybridxc, entry)
            qcs = []
            for system in entry.get_systems():
                qcs.append(evl.run(system))
            if entry.entry_type == 'dm':
                output.append((torch.as_tensor(entry.get_val(qcs)[0])))
            else:
                output.append(
                    torch.tensor(entry.get_val(qcs), requires_grad=True))
        return output


class XCModel(TorchModel):
    """
    This class is used to initialize and run Differentiable Quantum Chemistry (i.e,
    DFT) calculations, using an exchange correlation functional that has been replaced
    by a neural network. This model is based on the paper "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." and is listed below for reference.

    To read more about Density Functional Theory and the exchange
    correlation functional please check the references below.

    Examples
    --------
    >>> from deepchem.models.dft.dftxc import XCModel
    >>> from deepchem.data.data_loader import DFTYamlLoader
    >>> inputs = 'deepchem/models/tests/assets/test_dftxcdata.yaml'
    >>> data = DFTYamlLoader()
    >>> dataset = data.create_dataset(inputs)
    >>> dataset.get_shape()
    ((2,), (2,), (2,), (2,))
    >>> model = XCModel("lda_x", batch_size=1)
    >>> loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    The 6-311++G(3df,3pd) basis for atomz 3 does not exist, but we will download it
    Downloaded to /home/runner/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/03.gaussian94
    The 6-311++G(3df,3pd) basis for atomz 1 does not exist, but we will download it
    Downloaded to /home/runner/miniconda3/envs/deepchem/lib/python3.8/site-packages/dqc/api/.database/6-311ppg_3df_3pd_/01.gaussian94

    Notes
    -----
    There are 4 types of DFT data object implementations that are used to determine the type
    of calculation to be carried out on the entry object. These types are: "ae", "ie", "dm",    "dens", that stand for atomization energy, ionization energy, density matrix and
    density profile respectively.
    The entry type "Density Matrix" cannot be used on model.evaluate as of now.
    To run predictions on this data type, a dataset containing only "dm" entries must
    be used.

    References
    ----------
    https://github.com/deepchem/deepchem/blob/3f06168a6c9c16fd90cde7f5246b94f484ea3890/deepchem/models/dft/nnxc.py
    Encyclopedia of Condensed Matter Physics, 2005.
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    """

    def __init__(self,
                 xcstr: str,
                 nnmodel: Optional[torch.nn.Module] = None,
                 input_size: int = 2,
                 hidden_size: int = 10,
                 n_layers: int = 1,
                 modeltype: int = 1,
                 n_tasks: int = 0,
                 log_frequency: int = 0,
                 mode: str = 'classification',
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        xcstr: str
            The choice of xc to use.
        nnmodel: torch.nn.Module
            the PyTorch model implementing the calculation
        input_size: int
            size of neural network input
        hidden_size: int
            size of the hidden layers ; the number of hidden layers is fixed
            in the default method.
        n_layers: int
            number of layers in the neural network
        modeltype: int
            model type 2 includes an activation layer whereas type 1 does not.
        """
        if nnmodel is None:
            nnmodel = _construct_nn_model(input_size, hidden_size, n_layers,
                                          modeltype).to(torch.double)
        model = (DFTXC(xcstr, nnmodel)).to(device)
        self.xc = xcstr
        loss: Loss = L2Loss()
        output_types = ['loss', 'predict']
        self.mode = mode
        super(XCModel, self).__init__(model,
                                      loss=loss,
                                      output_types=output_types,
                                      **kwargs)

    def _prepare_batch(
            self,
            batch) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor]]:
        """
        Method to compute inputs, labels and weight for the Torch Model.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]

        Returns
        ------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        """
        inputs, labels, weights = batch
        if labels is not None:
            labels = [
                x.astype(np.float32) if x.dtype == np.float64 else x
                for x in labels
            ]
            label_tensors = [
                torch.as_tensor(x, dtype=torch.float64,
                                device=self.device).requires_grad_()
                for x in labels
            ]
        else:
            label_tensors = []
        if weights is not None:
            weights = [
                x.astype(np.float32) if x.dtype == np.float64 else x
                for x in weights
            ]
            weight_tensors = [
                torch.as_tensor(x, dtype=torch.float64, device=self.device)
                for x in weights
            ]
        else:
            weight_tensors = []

        return (inputs, label_tensors, weight_tensors)


class ExpM1Activation(torch.nn.Module):
    """
    This class is an activation layer that is used with model_type 2.

    Examples
    --------
    >>> from deepchem.models.dft.dftxc import ExpM1Activation
    >>> import torch
    >>> model = ExpM1Activation()
    >>> x = torch.tensor(2.5)
    >>> output = model(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1


def _construct_nn_model(input_size: int, hidden_size: int, n_layers: int,
                        modeltype: int):
    """
    Constructs Neural Network

    Parameters
    ----------
    input_size: int
        size of neural network input
    hidden_size: int
        size of the hidden layers ; there are 3 hidden layers in this method
    n_layers: int
        number of layers in the neural network
    modeltype: int
        model type 2 includes an activation layer whereas type 1 does not.

    Returns
    -------
    torch.nn.Sequential(*layers)

    Notes
    -----
    It is not necessary to use this method with the XCModel, user defined pytorch
    models will work.
    """
    if modeltype == 1:
        layers: List[Any]
        layers = []
        for i in range(n_layers):
            n1 = input_size if i == 0 else hidden_size
            layers.append(torch.nn.Linear(n1, hidden_size))
            layers.append(torch.nn.Softplus())
        layers.append(torch.nn.Linear(hidden_size, 1, bias=False))
        return torch.nn.Sequential(*layers)
    elif modeltype == 2:
        layers = []
        for i in range(n_layers):
            n1 = input_size if i == 0 else hidden_size
            layers.append(torch.nn.Linear(n1, hidden_size))
            if i < n_layers - 1:
                layers.append(torch.nn.Softplus())
            else:
                layers.append(ExpM1Activation())
        layers.append(torch.nn.Linear(hidden_size, 1, bias=False))
        return torch.nn.Sequential(*layers)

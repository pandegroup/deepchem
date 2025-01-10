import time
import logging
import os
from collections.abc import Sequence as SequenceCollection
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.utils.typing import LossFn, OneOrMany

logger = logging.getLogger(__name__)


class ModularTorchModel(TorchModel):
    """ModularTorchModel is a subclass of TorchModel that allows for components to be
    pretrained and then combined into a final model. It is designed to be subclassed
    for specific models and is not intended to be used directly. There are 3 main differences
    between ModularTorchModel and TorchModel:

    - The build_components() method is used to define the components of the model.
    - The components are combined into a final model with the build_model() method.
    - The loss function is defined with the loss_func method. This may access the
      components to compute the loss using intermediate values from the network, rather
      than just the full forward pass output.

    Here is an example of how to use ModularTorchModel to pretrain a linear layer, load
    it into another network and then finetune that network:

    >>> import numpy as np
    >>> import deepchem as dc
    >>> import torch
    >>> n_samples = 6
    >>> n_feat = 3
    >>> n_hidden = 2
    >>> n_tasks = 6
    >>> pt_tasks = 3
    >>> X = np.random.rand(n_samples, n_feat)
    >>> y_pretrain = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    >>> dataset_pt = dc.data.NumpyDataset(X, y_pretrain)
    >>> y_finetune = np.zeros((n_samples, n_tasks)).astype(np.float32)
    >>> dataset_ft = dc.data.NumpyDataset(X, y_finetune)
    >>> components = {'linear': torch.nn.Linear(n_feat, n_hidden),
    ... 'activation': torch.nn.ReLU(), 'head': torch.nn.Linear(n_hidden, n_tasks)}
    >>> model = torch.nn.Sequential(components['linear'], components['activation'],
    ... components['head'])
    >>> modular_model = dc.models.torch_models.modular.ModularTorchModel(model, components)
    >>> def example_loss_func(inputs, labels, weights):
    ...    return (torch.nn.functional.mse_loss(model(inputs), labels[0]) * weights[0]).mean()
    >>> modular_model.loss_func = example_loss_func
    >>> def example_model_build():
    ...     return torch.nn.Sequential(components['linear'], components['activation'],
    ... components['head'])
    >>> modular_model.build_model = example_model_build
    >>> pretrain_components = {'linear': torch.nn.Linear(n_feat, n_hidden),
    ... 'activation': torch.nn.ReLU(), 'head': torch.nn.Linear(n_hidden, pt_tasks)}
    >>> pretrain_model = torch.nn.Sequential(pretrain_components['linear'],
    ... pretrain_components['activation'], pretrain_components['head'])
    >>> pretrain_modular_model = dc.models.torch_models.modular.ModularTorchModel(pretrain_model,
    ... pretrain_components)
    >>> def example_pt_loss_func(inputs, labels, weights):
    ...     return (torch.nn.functional.mse_loss(pretrain_model(inputs), labels[0]) * weights[0]).mean()
    >>> pretrain_modular_model.loss_func = example_pt_loss_func
    >>> pt_loss = pretrain_modular_model.fit(dataset_pt, nb_epoch=1)
    >>> modular_model.load_from_pretrained(pretrain_modular_model, components=['linear'])
    >>> ft_loss = modular_model.fit(dataset_ft, nb_epoch=1)

    """

    def __init__(self, model: nn.Module, components: dict, **kwargs):
        """Create a ModularTorchModel.

        Parameters
        ----------
        model: nn.Module
            The model to be trained.
        components: dict
            A dictionary of the components of the model. The keys are the names of the
            components and the values are the components themselves.
        """

        self.model = model
        self.components = components
        # FIXME self.loss_func is an incorrect argument for TorchModel.loss because
        # it performs more than computing loss
        super().__init__(self.model, self.loss_func, **kwargs)
        self.model.to(self.device)
        self.components = {
            k: v.to(self.device) if isinstance(v, nn.Module) else v
            for k, v in self.components.items()
        }

    def build_model(self) -> nn.Module:
        """Builds the final model from the components."""
        layers = []
        for layer in self.components.values():
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def build_components(self) -> dict:
        """Creates the components dictionary, with the keys being the names of the
        components and the values being torch.nn.module objects."""
        raise NotImplementedError("Subclass must define the components")

    def loss_func(self, inputs: OneOrMany[torch.Tensor], labels: Sequence,
                  weights: Sequence) -> torch.Tensor:
        """Defines the loss function for the model which can access the components
        using self.components. The loss function should take the inputs, labels, and
        weights as arguments and return the loss."""
        raise NotImplementedError("Subclass must define the loss function")

    def freeze_components(self, components: List[str]):
        """Freezes or unfreezes the parameters of the specified components.

        Components string refers to keys in self.components.

        Parameters
        ----------
        components: List[str]
            The components to freeze.
        """
        for component in components:
            for param in self.components[component].parameters():
                param.requires_grad = False

    def unfreeze_components(self, components: List[str]):
        """Unfreezes the parameters of the specified components.

        Components string refers to keys in self.components.

        Parameters
        ----------
        components: List[str]
            The components to unfreeze.
        """
        for component in components:
            for param in self.components[component].parameters():
                param.requires_grad = True

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[Union[List[torch.nn.Parameter],
                                                torch.nn.ParameterList]] = None,
                      loss: Optional[LossFn] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:
        """Train this model on data from a generator. This method is similar to
        the TorchModel implementation, but it passes the inputs directly to the
        loss function, rather than passing them through the model first.  This
        enables the loss to be calculated from intermediate steps of the model
        and not just the final output.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step, **kwargs) that will be invoked
            after every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval
        """

        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        # FIXME This line is not needed as loss is computed inside the call to loss_func
        if loss is None:
            loss = self._loss_fn
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            # Execute the loss function, accumulating the gradients.

            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]

            optimizer.zero_grad()
            batch_loss = self.loss_func(inputs, labels, weights)
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += float(batch_loss)

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %.10f' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0
            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                try:
                    # NOTE In DeepChem > 2.8.0, callback signature is updated to allow
                    # variable arguments.
                    c(self, current_step, iteration_loss=batch_loss)
                except TypeError:
                    # DeepChem <= 2.8.0, the callback should have this signature.
                    c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss

    def load_from_pretrained(  # type: ignore
            self,
            source_model: Optional["ModularTorchModel"] = None,
            components: Optional[List[str]] = None,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None,
            inputs: Optional[Sequence[Any]] = None,
            **kwargs) -> None:
        """Copies parameter values from a pretrained model. The pretrained model can be loaded as a source_model (ModularTorchModel object), checkpoint (pytorch .ckpt file) or a model_dir (directory with .ckpt files).
        Specific components can be chosen by passing a list of strings with the desired component names. If both a source_model and a checkpoint/model_dir are loaded, the source_model weights will be loaded.

        Parameters
        ----------
        source_model: dc.ModularTorchModel, required
            source_model can either be the pretrained model or a dc.TorchModel with
            the same architecture as the pretrained model. It is used to restore from
            a checkpoint, if value_map is None and to create a default assignment map
            if assignment_map is None
        checkpoint: str, default None
            the path to the checkpoint file to load.  If this is None, the most recent
            checkpoint will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints
        model_dir: str, default None
            Restore source model from custom model directory if needed
        inputs: List, input tensors for model
            if not None, then the weights are built for both the source and self.
        """
        if inputs is not None:
            # Ensure weights for both models are built.
            if source_model:
                source_model.model(inputs)
            self.model(inputs)

        self._ensure_built()

        if source_model is not None:
            for name, module in source_model.components.items():
                if components is None or name in components:
                    self.components[name].load_state_dict(module.state_dict(),
                                                          strict=False)
            self.build_model()

        elif source_model is None:
            self.restore(components=components,
                         checkpoint=checkpoint,
                         model_dir=model_dir)

    def save_checkpoint(self, max_checkpoints_to_keep=5, model_dir=None):
        """
        Saves the current state of the model and its components as a checkpoint file in the specified model directory.
        It maintains a maximum number of checkpoint files, deleting the oldest one when the limit is reached.

        Parameters
        ----------
        max_checkpoints_to_keep: int, default 5
            Maximum number of checkpoint files to keep.
        model_dir: str, default None
            The directory to save the checkpoint file in. If None, the model_dir specified in the constructor is used.
        """

        if model_dir is None:
            model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        data = {
            'model': self.model.state_dict(),
            'optimizer_state_dict': self._pytorch_optimizer.state_dict(),
            'global_step': self._global_step
        }

        for name, component in self.components.items():
            if hasattr(component, 'state_dict'):
                data[name] = component.state_dict()

        temp_file = os.path.join(model_dir, 'temp_checkpoint.pt')
        torch.save(data, temp_file)

        # Rename and delete older files.

        paths = [
            os.path.join(model_dir, 'checkpoint%d.pt' % (i + 1))
            for i in range(max_checkpoints_to_keep)
        ]
        if os.path.exists(paths[-1]):
            os.remove(paths[-1])
        for i in reversed(range(max_checkpoints_to_keep - 1)):
            if os.path.exists(paths[i]):
                os.rename(paths[i], paths[i + 1])
        os.rename(temp_file, paths[0])

    def restore(  # type: ignore
            self,
            components: Optional[List[str]] = None,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None) -> None:
        """
        Restores the state of a ModularTorchModel from a checkpoint file.

        If no checkpoint file is provided, it will use the latest checkpoint found in the model directory. If a list of component names is provided, only the state of those components will be restored.

        Parameters
        ----------
        components: Optional[List[str]]
            A list of component names to restore. If None, all components will be restored.
        checkpoint: Optional[str]
            The path to the checkpoint file. If None, the latest checkpoint in the model directory will
            be used.
        model_dir: Optional[str]
            The path to the model directory. If None, the model directory used to initialize the model will be used.
        """
        logger.info('Restoring model')
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint)
        for name, state_dict in data.items():
            if name != 'model' and name in self.components.keys():
                if components is None or name in components:
                    self.components[name].load_state_dict(state_dict)

        self.build_model()
        self._ensure_built()
        self._pytorch_optimizer.load_state_dict(data['optimizer_state_dict'])
        self._global_step = data['global_step']

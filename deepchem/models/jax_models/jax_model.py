import numpy as np
import time
import logging
from collections.abc import Sequence as SequenceCollection

from deepchem.data import Dataset, NumpyDataset
from deepchem.metrics import Metric
from deepchem.models.models import Model
from deepchem.models.losses import Loss
from deepchem.models.optimizers import Optimizer, Adam
from deepchem.utils.evaluate import GeneratorEvaluator
from deepchem.trans.transformers import Transformer, undo_transforms

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Sequence
from deepchem.utils.typing import LossFn, OneOrMany

# JAX depend
import jax.numpy as jnp
import jax
import haiku as hk
import optax

import warnings

logger = logging.getLogger(__name__)


def create_default_eval_fn(forward_fn, params):
    """
    Calls the function to evaluate the model
    """

    @jax.jit
    def eval_model(batch, rng=None):
        predict = forward_fn(params, rng, batch)

        return predict

    return eval_model


def create_default_update_fn(optimizer, model_loss):
    """
    This function calls the update function, to implement the backpropogation
    """

    @jax.jit
    def update(params, opt_state, batch, target, weights,
               rng) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        batch_loss, grads = jax.value_and_grad(model_loss)(params, batch,
                                                           target, weights, rng)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss

    return update


def create_default_gradient_fn(forward_fn, loss_outputs, loss_fn):
    """
    This function calls the gradient function, to implement the backpropogation
    """

    @jax.jit
    def model_loss(params, batch, target, weights, rng):
        predict = forward_fn(params, rng, batch)
        if loss_outputs is not None:
            predict = [predict[i] for i in loss_outputs]
        return loss_fn(predict, target, weights)

    return model_loss


class JaxModel(Model):
    """This is a DeepChem model implemented by a Jax Model
    Here is a simple example of that uses JaxModel to train a
    Haiku (JAX Neural Network Library) based model on deepchem
    dataset.

    >>>
    >> def forward_model(x):
    >>   net = hk.nets.MLP([512, 256, 128, 1])
    >>   return net(x)
    >> def rms_loss(pred, tar, w):
    >>   return jnp.mean(optax.l2_loss(pred, tar))
    >> params_init, forward_fn = hk.transform(forward_model)
    >> rng = jax.random.PRNGKey(500)
    >> inputs, _, _, _ = next(iter(dataset.iterbatches(batch_size=256)))
    >> params = params_init(rng, inputs)
    >> j_m = JaxModel(forward_fn, params, rms_loss, 256, 0.001, 100)
    >> j_m.fit(train_dataset)

    All optimizations will be done using the optax library.
    """

    def __init__(self,
                 forward_fn: hk.State,
                 params: hk.Params,
                 loss: Optional[Union[Loss, LossFn]],
                 output_types: Optional[List[str]] = None,
                 batch_size: int = 100,
                 learning_rate: float = 0.001,
                 optimizer: Optional[Union[optax.GradientTransformation,
                                           Optimizer]] = None,
                 grad_fn: Callable = create_default_gradient_fn,
                 update_fn: Callable = create_default_update_fn,
                 eval_fn: Callable = create_default_eval_fn,
                 rng: Optional[jax.Array] = None,
                 log_frequency: int = 100,
                 **kwargs):
        """
        Create a new JaxModel

        Parameters
        ----------
        model: hk.State or Function
            Any Jax based model that has a `apply` method for computing the network. Currently
            only haiku models are supported.
        params: hk.Params
            The parameter of the Jax based networks
        loss: dc.models.losses.Loss or function
            a Loss or function defining how to compute the training loss for each
            batch, as described above
        output_types: list of strings, optional (default None)
            the type of each output from the model, as described above
        batch_size: int, optional (default 100)
            default batch size for training and evaluating
        learning_rate: float or LearningRateSchedule, optional (default 0.001)
            the learning rate to use for fitting.  If optimizer is specified, this is
            ignored.
        optimizer: optax object
            For the time being, it is optax object
        rng: jax.random.PRNGKey, optional (default 1)
            A default global PRNG key to use for drawing random numbers.
        log_frequency: int, optional (default 100)
            The frequency at which to log data. Data is logged using
            `logging` by default.


        Miscellanous Parameters Yet To Add
        ----------------------------------
        model_dir: str, optional (default None)
            Will be added along with the save & load method
        tensorboard: bool, optional (default False)
            whether to log progress to TensorBoard during training
        wandb: bool, optional (default False)
            whether to log progress to Weights & Biases during training


        Work in Progress
        ----------------
        [1] Integrate the optax losses, optimizers, schedulers with Deepchem
        [2] Support for saving & loading the model.
        """
        super(JaxModel, self).__init__(model=(forward_fn, params), **kwargs)
        warnings.warn(
            'JaxModel is still in active development and all features may not yet be implemented'
        )
        self._loss_fn = loss  # lambda pred, tar: jnp.mean(optax.l2_loss(pred, tar))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if optimizer is None:
            optimizer = Adam(1e-3)

        if not isinstance(optimizer, optax.GradientTransformation):
            self.optimizer = optimizer._create_jax_optimizer()
        else:
            self.optimizer = optimizer
        self.forward_fn = forward_fn
        self.params = params
        self._built = False
        self.log_frequency = log_frequency
        self.rng = jax.random.PRNGKey(1) if rng is None else rng
        self._create_gradient_fn = grad_fn
        self._create_update_fn = update_fn
        self._create_eval_fn = eval_fn

        if output_types is None:
            self._prediction_outputs = None
            self._loss_outputs = None
            self._variance_outputs = None
            self._other_outputs = None
        else:
            self._prediction_outputs = []
            self._loss_outputs = []
            self._variance_outputs = []
            self._other_outputs = []
            for i, type in enumerate(output_types):
                if type == 'prediction':
                    self._prediction_outputs.append(i)
                elif type == 'loss':
                    self._loss_outputs.append(i)
                elif type == 'variance':
                    self._variance_outputs.append(i)
                else:
                    self._other_outputs.append(i)
            if len(self._loss_outputs) == 0:
                self._loss_outputs = self._prediction_outputs

    def _ensure_built(self):
        """The first time this is called, create internal data structures.
        Work in Progress
        ----------------
        [1] Integerate the optax losses, optimizers, schedulers with Deepchem
        """
        if self._built:
            return

        self._built = True
        self._global_step = 0
        self.opt_state = self.optimizer.init(self.params)

    def fit(self,
            dataset: Dataset,
            nb_epochs: int = 10,
            deterministic: bool = False,
            loss: Optional[Union[Loss, LossFn]] = None,
            callbacks: Union[Callable, List[Callable]] = [],
            all_losses: Optional[List[float]] = None) -> float:
        """Train this model on a dataset.
        Parameters
        ----------
        dataset: Dataset
            the Dataset to train on
        nb_epoch: int
            the number of epochs to train for
        deterministic: bool
            if True, the samples are processed in order.  If False, a different random
            order is used for each epoch.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.
        Returns
        -------
        The average loss over the most recent checkpoint interval
        Miscellanous Parameters Yet To Add
        ----------------------------------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of hk.Variable
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        Work in Progress
        ----------------
        [1] Integerate the optax losses, optimizers, schedulers with Deepchem
        [2] Support for saving & loading the model.
        [3] Adding support for output types (choosing only self._loss_outputs)
        """
        return self.fit_generator(
            self.default_generator(dataset,
                                   epochs=nb_epochs,
                                   deterministic=deterministic), loss,
            callbacks, all_losses)

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      loss: Optional[Union[Loss, LossFn]] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if loss is None:
            loss = self._loss_fn
        model_loss_fn = self._create_gradient_fn(self.forward_fn,
                                                 self._loss_outputs, loss)
        grad_update = self._create_update_fn(self.optimizer, model_loss_fn)

        params, opt_state = self._get_trainable_params()
        rng = self.rng
        time1 = time.time()

        # Main training loop

        for batch in generator:
            inputs, labels, weights = self._prepare_batch(batch)

            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]

            if isinstance(labels, list) and len(labels) == 1:
                labels = labels[0]

            if isinstance(weights, list) and len(weights) == 1:
                weights = weights[0]

            params, opt_state, batch_loss = grad_update(params,
                                                        opt_state,
                                                        inputs,
                                                        labels,
                                                        weights,
                                                        rng=rng)
            rng, _ = jax.random.split(rng)
            avg_loss += jax.device_get(batch_loss)
            self._global_step += 1
            current_step = self._global_step
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)

            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0
            for c in callbacks:
                c(self, current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        self._set_trainable_params(params, opt_state)
        return last_avg_loss

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer], uncertainty: bool,
                 other_output_types: Optional[OneOrMany[str]]):
        """
        Predict outputs for data provided by a generator.
        This is the private implementation of prediction.  Do not
        call it directly. Instead call one of the public prediction
        methods.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: List[dc.trans.Transformers]
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        uncertainty: bool
            specifies whether this is being called as part of estimating uncertainty.
            If True, it sets the training flag so that dropout will be enabled, and
            returns the values of the uncertainty outputs.
        other_output_types: list, optional
            Provides a list of other output_types (strings) to predict from model.

        Returns
        -------
        A NumpyArray if the model produces a single output, or a list of arrays otherwise.
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        if uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
        if other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        self._ensure_built()
        eval_fn = self._create_eval_fn(self.forward_fn, self.params)
        rng = self.rng

        for batch in generator:
            inputs, _, _ = self._prepare_batch(batch)

            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]

            output_values = eval_fn(inputs, rng)
            if isinstance(output_values, jnp.ndarray):
                output_values = [output_values]
            output_values = [jax.device_get(t) for t in output_values]

            # Apply tranformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)

            access_values = []
            if other_output_types:
                access_values += self._other_outputs
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError(
                        "predict() does not support Transformers for models with multiple outputs."
                    )
                elif len(output_values) == 1:
                    output_values = [
                        undo_transforms(output_values[0], transformers)
                    ]

            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))
        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results

    def predict_on_generator(
            self,
            generator: Iterable[Tuple[Any, Any, Any]],
            transformers: List[Transformer] = [],
            output_types: Optional[OneOrMany[str]] = None
    ) -> OneOrMany[np.ndarray]:
        """
        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: List[dc.trans.Transformers]
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        output_types: String or list of Strings
            If specified, all outputs of this type will be retrieved
            from the model. If output_types is specified, outputs must
            be None.

        Returns
        -------
            a NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs
        """
        return self._predict(generator, transformers, False, output_types)

    def predict_on_batch(
            self,
            X: np.typing.ArrayLike,
            transformers: List[Transformer] = []) -> OneOrMany[np.ndarray]:
        """Generates predictions for input samples, processing samples in a batch.
        Parameters
        ----------
        X: ndarray
            the input data, as a Numpy array.
        transformers: List[dc.trans.Transformers]
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.

        Returns
        -------
        a NumPy array of the model produces a single output, or a list of arrays
        if it produces multiple outputs
        """
        dataset = NumpyDataset(X=X, y=None)
        return self.predict(dataset, transformers)

    def predict_uncertainty_on_batch(
            self,
            X: Sequence,
            masks: int = 50) -> OneOrMany[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(
            'Predicting uncertainity on batch is not supported currently for JAX models'
        )

    def predict(
            self,
            dataset: Dataset,
            transformers: List[Transformer] = [],
            output_types: Optional[List[str]] = None) -> OneOrMany[np.ndarray]:
        """
        Uses self to make predictions on provided Dataset object.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to make prediction on
        transformers: List[dc.trans.Transformers]
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        output_types: String or list of Strings
            If specified, all outputs of this type will be retrieved
            from the model. If output_types is specified, outputs must
            be None.

        Returns
        -------
        a NumPy array of the model produces a single output, or a list of arrays
        if it produces multiple outputs
        """
        generator = self.default_generator(dataset,
                                           mode='predict',
                                           pad_batches=False)
        return self.predict_on_generator(generator,
                                         transformers=transformers,
                                         output_types=output_types)

    def get_global_step(self) -> int:
        """Get the number of steps of fitting that have been performed."""
        return self._global_step

    def predict_embedding(self, dataset: Dataset) -> OneOrMany[np.ndarray]:
        raise NotImplementedError(
            'Predicting embedding is not supported currently for JAX models')

    # def predict_uncertainty(self, dataset: Dataset, masks: int = 50
    #                        ) -> OneOrMany[Tuple[np.ndarray, np.ndarray]]:
    #   """
    #   Predict the model's outputs, along with the uncertainty in each one.
    #   The uncertainty is computed as described in https://arxiv.org/abs/1703.04977.
    #   It involves repeating the prediction many times with different dropout masks.
    #   The prediction is computed as the average over all the predictions.  The
    #   uncertainty includes both the variation among the predicted values (epistemic
    #   uncertainty) and the model's own estimates for how well it fits the data
    #   (aleatoric uncertainty).  Not all models support uncertainty prediction.
    #   Parameters
    #   ----------
    #   dataset: dc.data.Dataset
    #     Dataset to make prediction on
    #   masks: int
    #     the number of dropout masks to average over
    #   Returns
    #   -------
    #   for each output, a tuple (y_pred, y_std) where y_pred is the predicted
    #   value of the output, and each element of y_std estimates the standard
    #   deviation of the corresponding element of y_pred
    #   """
    #   sum_pred: List[np.ndarray] = []
    #   sum_sq_pred: List[np.ndarray] = []
    #   sum_var: List[np.ndarray] = []
    #   for i in range(masks):
    #     generator = self.default_generator(
    #         dataset, mode='uncertainty', pad_batches=False)
    #     results = self._predict(generator, [], True, None)
    #     if len(sum_pred) == 0:
    #       for p, v in results:
    #         sum_pred.append(p)
    #         sum_sq_pred.append(p * p)
    #         sum_var.append(v)
    #     else:
    #       for j, (p, v) in enumerate(results):
    #         sum_pred[j] += p
    #         sum_sq_pred[j] += p * p
    #         sum_var[j] += v
    #   output = []
    #   std = []
    #   for i in range(len(sum_pred)):
    #     p = sum_pred[i] / masks
    #     output.append(p)
    #     std.append(np.sqrt(sum_sq_pred[i] / masks - p * p + sum_var[i] / masks))
    #   if len(output) == 1:
    #     return (output[0], std[0])
    #   else:
    #     return list(zip(output, std))

    def evaluate_generator(self,
                           generator: Iterable[Tuple[Any, Any, Any]],
                           metrics: List[Metric],
                           transformers: List[Transformer] = [],
                           per_task_metrics: bool = False):
        """Evaluate the performance of this model on the data produced by a generator.
        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        metric: list of deepchem.metrics.Metric
            Evaluation metric
        transformers: List[dc.trans.Transformers]
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        per_task_metrics: bool
            If True, return per-task scores.
        Returns
        -------
        dict
            Maps tasks to scores under metric.
        """
        evaluator = GeneratorEvaluator(self, generator, transformers)
        return evaluator.compute_model_performance(metrics, per_task_metrics)

    def _get_trainable_params(self):
        """
        Will be used to seperate freezing parameters while transfer learning
        """
        return self.params, self.opt_state

    def _set_trainable_params(self, params: hk.Params,
                              opt_state: optax.OptState):
        """
        A functional approach to setting the final parameters after training
        """
        self.params = params
        self.opt_state = opt_state

    def _prepare_batch(self, batch):
        inputs, labels, weights = batch
        inputs = [
            x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs
        ]
        if labels is not None:
            labels = [
                x.astype(np.float32) if x.dtype == np.float64 else x
                for x in labels
            ]
        else:
            labels = []

        if weights is not None:
            weights = [
                x.astype(np.float32) if x.dtype == np.float64 else x
                for x in weights
            ]
        else:
            weights = []

        return (inputs, labels, weights)

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Create a generator that iterates batches for a dataset.
        Subclasses may override this method to customize how model inputs are
        generated from the data.
        Parameters
        ----------
        dataset: Dataset
            the data to iterate
        epochs: int
            the number of times to iterate over the full dataset
        mode: str
            allowed values are 'fit' (called during training), 'predict' (called
            during prediction), and 'uncertainty' (called during uncertainty
            prediction)
        deterministic: bool
            whether to iterate over the dataset in order, or randomly shuffle the
            data for each epoch
        pad_batches: bool
            whether to pad each batch up to this model's preferred batch size
        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])
        """

        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 _) in dataset.iterbatches(batch_size=self.batch_size,
                                           deterministic=deterministic,
                                           pad_batches=pad_batches):
                yield ([X_b], [y_b], [w_b])

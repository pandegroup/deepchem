import numpy as np
import tensorflow as tf
import time
import logging
import os
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

from deepchem.data import Dataset, NumpyDataset
from deepchem.metrics import Metric
from deepchem.models.losses import Loss
from deepchem.models.models import Model
from deepchem.models.optimizers import Adam, Optimizer, LearningRateSchedule
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator
from deepchem.utils.openvino_model import OpenVINOModel

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany

try:
  import wandb
  wandb.ensure_configured()
  if wandb.api.api_key is None:
    _has_wandb = False
    wandb.termwarn(
        "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable."
    )
  else:
    _has_wandb = True
except (ImportError, AttributeError):
  _has_wandb = False

logger = logging.getLogger(__name__)


class KerasModel(Model):
  """This is a DeepChem model implemented by a Keras model.

  This class provides several advantages over using the Keras
  model's fitting and prediction methods directly.

  1. It provides better integration with the rest of DeepChem,
     such as direct support for Datasets and Transformers.

  2. It defines the loss in a more flexible way.  In particular,
     Keras does not support multidimensional weight matrices,
     which makes it impossible to implement most multitask
     models with Keras.

  3. It provides various additional features not found in the
     Keras Model class, such as uncertainty prediction and
     saliency mapping.

  Here is a simple example of code that uses KerasModel to train
  a Keras model on a DeepChem dataset.

  >> keras_model = tf.keras.Sequential([
  >>    tf.keras.layers.Dense(1000, activation='tanh'),
  >>    tf.keras.layers.Dense(1)
  >> ])
  >> model = KerasModel(keras_model, loss=dc.models.losses.L2Loss())
  >> model.fit(dataset)

  The loss function for a model can be defined in two different
  ways.  For models that have only a single output and use a
  standard loss function, you can simply provide a
  dc.models.losses.Loss object.  This defines the loss for each
  sample or sample/task pair.  The result is automatically
  multiplied by the weights and averaged over the batch.  Any
  additional losses computed by model layers, such as weight
  decay penalties, are also added.

  For more complicated cases, you can instead provide a function
  that directly computes the total loss.  It must be of the form
  f(outputs, labels, weights), taking the list of outputs from
  the model, the expected values, and any weight matrices.  It
  should return a scalar equal to the value of the loss function
  for the batch.  No additional processing is done to the
  result; it is up to you to do any weighting, averaging, adding
  of penalty terms, etc.

  You can optionally provide an output_types argument, which
  describes how to interpret the model's outputs.  This should
  be a list of strings, one for each output. You can use an
  arbitrary output_type for a output, but some output_types are
  special and will undergo extra processing:

  - 'prediction': This is a normal output, and will be returned by predict().
    If output types are not specified, all outputs are assumed
    to be of this type.

  - 'loss': This output will be used in place of the normal
    outputs for computing the loss function.  For example,
    models that output probability distributions usually do it
    by computing unbounded numbers (the logits), then passing
    them through a softmax function to turn them into
    probabilities.  When computing the cross entropy, it is more
    numerically stable to use the logits directly rather than
    the probabilities.  You can do this by having the model
    produce both probabilities and logits as outputs, then
    specifying output_types=['prediction', 'loss'].  When
    predict() is called, only the first output (the
    probabilities) will be returned.  But during training, it is
    the second output (the logits) that will be passed to the
    loss function.

  - 'variance': This output is used for estimating the
    uncertainty in another output.  To create a model that can
    estimate uncertainty, there must be the same number of
    'prediction' and 'variance' outputs.  Each variance output
    must have the same shape as the corresponding prediction
    output, and each element is an estimate of the variance in
    the corresponding prediction.  Also be aware that if a model
    supports uncertainty, it MUST use dropout on every layer,
    and dropout most be enabled during uncertainty prediction.
    Otherwise, the uncertainties it computes will be inaccurate.

  - other: Arbitrary output_types can be used to extract outputs
    produced by the model, but will have no additional
    processing performed.
  """

  def __init__(self,
               model: tf.keras.Model,
               loss: Union[Loss, LossFn],
               output_types: Optional[List[str]] = None,
               batch_size: int = 100,
               model_dir: Optional[str] = None,
               learning_rate: Union[float, LearningRateSchedule] = 0.001,
               optimizer: Optional[Optimizer] = None,
               tensorboard: bool = False,
               wandb: bool = False,
               log_frequency: int = 100,
               **kwargs) -> None:
    """Create a new KerasModel.

    Parameters
    ----------
    model: tf.keras.Model
      the Keras model implementing the calculation
    loss: dc.models.losses.Loss or function
      a Loss or function defining how to compute the training loss for each
      batch, as described above
    output_types: list of strings
      the type of each output from the model, as described above
    batch_size: int
      default batch size for training and evaluating
    model_dir: str
      the directory on disk where the model will be stored.  If this is None,
      a temporary directory is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for fitting.  If optimizer is specified, this is
      ignored.
    optimizer: Optimizer
      the optimizer to use for fitting.  If this is specified, learning_rate is
      ignored.
    tensorboard: bool
      whether to log progress to TensorBoard during training
    wandb: bool
      whether to log progress to Weights & Biases during training
    log_frequency: int
      The frequency at which to log data. Data is logged using
      `logging` by default. If `tensorboard` is set, data is also
      logged to TensorBoard. If `wandb` is set, data is also logged
      to Weights & Biases. Logging happens at global steps. Roughly,
      a global step corresponds to one batch of training. If you'd
      like a printout every 10 batch steps, you'd set
      `log_frequency=10` for example.
    """
    super(KerasModel, self).__init__(model=model, model_dir=model_dir, **kwargs)
    if isinstance(loss, Loss):
      self._loss_fn: LossFn = _StandardLoss(model, loss)
    else:
      self._loss_fn = loss
    self.batch_size = batch_size
    if optimizer is None:
      self.optimizer: Optimizer = Adam(learning_rate=learning_rate)
    else:
      self.optimizer = optimizer
    self.tensorboard = tensorboard

    # W&B logging
    if wandb and not _has_wandb:
      logger.warning(
          "You set wandb to True but W&B is not installed. To use wandb logging, "
          "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
      )
    self.wandb = wandb and _has_wandb

    # Backwards compatibility
    if "tensorboard_log_frequency" in kwargs:
      logger.warning(
          "tensorboard_log_frequency is deprecated. Please use log_frequency instead. This argument will be removed in a future release of DeepChem."
      )
      self.log_frequency = kwargs["tensorboard_log_frequency"]
    else:
      self.log_frequency = log_frequency
    if self.tensorboard:
      self._summary_writer = tf.summary.create_file_writer(self.model_dir)
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
    self._built = False
    self._inputs_built = False
    self._training_ops_built = False
    self._output_functions: Dict[Any, Any] = {}
    self._gradient_fn_for_vars: Dict[Any, Any] = {}

    self._use_openvino = kwargs.get('use_openvino', False)
    if self._use_openvino:
      self._openvino_model = OpenVINOModel(
          self.model_dir, self.batch_size, keras_model=self)

  def _ensure_built(self) -> None:
    """The first time this is called, create internal data structures."""
    if self._built:
      return
    self._built = True
    self._global_step = tf.Variable(0, trainable=False)
    self._tf_optimizer = self.optimizer._create_tf_optimizer(self._global_step)
    self._checkpoint = tf.train.Checkpoint(
        optimizer=self._tf_optimizer, model=self.model)

  def _create_inputs(self, example_inputs: List) -> None:
    """The first time this is called, create tensors representing the inputs and outputs."""
    if self._inputs_built:
      return
    self._ensure_built()
    self._inputs_built = True
    if (self.model.inputs is not None) and len(self.model.inputs) > 0:
      self._input_shapes = [t.shape for t in self.model.inputs]
      self._input_dtypes = [t.dtype.as_numpy_dtype for t in self.model.inputs]
    else:
      self._input_shapes = [(None,) + i.shape[1:] for i in example_inputs]
      self._input_dtypes = [
          np.float32 if x.dtype == np.float64 else x.dtype
          for x in example_inputs
      ]

  def _create_training_ops(self,
                           example_batch: Tuple[List, List, List]) -> None:
    """The first time this is called, create tensors used in optimization."""
    if self._training_ops_built:
      return
    self._create_inputs(example_batch[0])
    self._training_ops_built = True
    self._label_dtypes = [
        np.float32 if x.dtype == np.float64 else x.dtype
        for x in example_batch[1]
    ]
    self._weights_dtypes = [
        np.float32 if x.dtype == np.float64 else x.dtype
        for x in example_batch[2]
    ]

  def fit(self,
          dataset: Dataset,
          nb_epoch: int = 10,
          max_checkpoints_to_keep: int = 5,
          checkpoint_interval: int = 1000,
          deterministic: bool = False,
          restore: bool = False,
          variables: Optional[List[tf.Variable]] = None,
          loss: Optional[LossFn] = None,
          callbacks: Union[Callable, List[Callable]] = [],
          all_losses: Optional[List[float]] = None) -> float:
    """Train this model on a dataset.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    deterministic: bool
      if True, the samples are processed in order.  If False, a different random
      order is used for each epoch.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
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
   """
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epoch,
            deterministic=deterministic), max_checkpoints_to_keep,
        checkpoint_interval, restore, variables, loss, callbacks, all_losses)

  def fit_generator(self,
                    generator: Iterable[Tuple[Any, Any, Any]],
                    max_checkpoints_to_keep: int = 5,
                    checkpoint_interval: int = 1000,
                    restore: bool = False,
                    variables: Optional[List[tf.Variable]] = None,
                    loss: Optional[LossFn] = None,
                    callbacks: Union[Callable, List[Callable]] = [],
                    all_losses: Optional[List[float]] = None) -> float:
    """Train this model on data from a generator.

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
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
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
    """
    if not isinstance(callbacks, SequenceCollection):
      callbacks = [callbacks]
    self._ensure_built()
    if checkpoint_interval > 0:
      manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                           max_checkpoints_to_keep)
    avg_loss = 0.0
    last_avg_loss = 0.0
    averaged_batches = 0
    if loss is None:
      loss = self._loss_fn
    var_key = None
    if variables is not None:
      var_key = tuple(v.ref() for v in variables)

      # The optimizer creates internal variables the first time apply_gradients()
      # is called for a new set of variables.  If that happens inside a function
      # annotated with tf.function it throws an exception, so call it once here.

      zero_grads = [tf.zeros(v.shape) for v in variables]
      self._tf_optimizer.apply_gradients(zip(zero_grads, variables))
    if var_key not in self._gradient_fn_for_vars:
      self._gradient_fn_for_vars[var_key] = self._create_gradient_fn(variables)
    apply_gradient_for_batch = self._gradient_fn_for_vars[var_key]
    time1 = time.time()

    # Main training loop.

    for batch in generator:
      self._create_training_ops(batch)
      if restore:
        self.restore()
        restore = False
      inputs, labels, weights = self._prepare_batch(batch)

      # Execute the loss function, accumulating the gradients.

      if len(inputs) == 1:
        inputs = inputs[0]

      batch_loss = apply_gradient_for_batch(inputs, labels, weights, loss)
      current_step = self._global_step.numpy()

      avg_loss += batch_loss

      # Report progress and write checkpoints.
      averaged_batches += 1
      should_log = (current_step % self.log_frequency == 0)
      if should_log:
        avg_loss = float(avg_loss) / averaged_batches
        logger.info(
            'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        if all_losses is not None:
          all_losses.append(avg_loss)
        # Capture the last avg_loss in case of return since we're resetting to
        # 0 now
        last_avg_loss = avg_loss
        avg_loss = 0.0
        averaged_batches = 0

      if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
        manager.save()
      for c in callbacks:
        c(self, current_step)
      if self.tensorboard and should_log:
        self._log_scalar_to_tensorboard('loss', batch_loss, current_step)
      if self.wandb and should_log:
        wandb.log({'loss': batch_loss}, step=current_step)

    # Report final results.
    if averaged_batches > 0:
      avg_loss = float(avg_loss) / averaged_batches
      logger.info(
          'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
      if all_losses is not None:
        all_losses.append(avg_loss)
      last_avg_loss = avg_loss

    if checkpoint_interval > 0:
      manager.save()

    time2 = time.time()
    logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return last_avg_loss

  def _create_gradient_fn(self,
                          variables: Optional[List[tf.Variable]]) -> Callable:
    """Create a function that computes gradients and applies them to the model.
    Because of the way TensorFlow function tracing works, we need to create a
    separate function for each new set of variables.
    """

    @tf.function(experimental_relax_shapes=True)
    def apply_gradient_for_batch(inputs, labels, weights, loss):
      with tf.GradientTape() as tape:
        outputs = self.model(inputs, training=True)
        if tf.is_tensor(outputs):
          outputs = [outputs]
        if self._loss_outputs is not None:
          outputs = [outputs[i] for i in self._loss_outputs]
        batch_loss = loss(outputs, labels, weights)
      if variables is None:
        vars = self.model.trainable_variables
      else:
        vars = variables
      grads = tape.gradient(batch_loss, vars)
      self._tf_optimizer.apply_gradients(zip(grads, vars))
      self._global_step.assign_add(1)
      return batch_loss

    return apply_gradient_for_batch

  def fit_on_batch(self,
                   X: Sequence,
                   y: Sequence,
                   w: Sequence,
                   variables: Optional[List[tf.Variable]] = None,
                   loss: Optional[LossFn] = None,
                   callbacks: Union[Callable, List[Callable]] = [],
                   checkpoint: bool = True,
                   max_checkpoints_to_keep: int = 5) -> float:
    """Perform a single step of training.

    Parameters
    ----------
    X: ndarray
      the inputs for the batch
    y: ndarray
      the labels for the batch
    w: ndarray
      the weights for the batch
    variables: list of tf.Variable
      the variables to train.  If None (the default), all trainable variables in
      the model are used.
    loss: function
      a function of the form f(outputs, labels, weights) that computes the loss
      for each batch.  If None (the default), the model's standard loss function
      is used.
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be invoked after
      every step.  This can be used to perform validation, logging, etc.
    checkpoint: bool
      if true, save a checkpoint after performing the training step
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.

    Returns
    -------
    the loss on the batch
    """
    self._ensure_built()
    dataset = NumpyDataset(X, y, w)
    return self.fit(
        dataset,
        nb_epoch=1,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_interval=self._global_step.numpy() + 2 if checkpoint else 0,
        variables=variables,
        loss=loss,
        callbacks=callbacks)

  def _predict(
      self, generator: Iterable[Tuple[Any, Any, Any]],
      transformers: List[Transformer], outputs: Optional[OneOrMany[tf.Tensor]],
      uncertainty: bool,
      other_output_types: Optional[OneOrMany[str]]) -> OneOrMany[np.ndarray]:
    """
    Predict outputs for data provided by a generator.

    This is the private implementation of prediction.  Do not
    call it directly.  Instead call one of the public prediction
    methods.

    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.
    uncertainty: bool
      specifies whether this is being called as part of estimating uncertainty.
      If True, it sets the training flag so that dropout will be enabled, and
      returns the values of the uncertainty outputs.
    other_output_types: list, optional
      Provides a list of other output_types (strings) to predict from model.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    results: Optional[List[List[np.ndarray]]] = None
    variances: Optional[List[List[np.ndarray]]] = None
    if (outputs is not None) and (other_output_types is not None):
      raise ValueError(
          'This model cannot compute outputs and other output_types simultaneously.'
          'Please invoke one at a time.')
    if uncertainty and (other_output_types is not None):
      raise ValueError(
          'This model cannot compute uncertainties and other output types simultaneously.'
          'Please invoke one at a time.')
    if uncertainty:
      assert outputs is None
      if self._variance_outputs is None or len(self._variance_outputs) == 0:
        raise ValueError('This model cannot compute uncertainties')
      if len(self._variance_outputs) != len(self._prediction_outputs):
        raise ValueError(
            'The number of variances must exactly match the number of outputs')
    if other_output_types:
      assert outputs is None
      if self._other_outputs is None or len(self._other_outputs) == 0:
        raise ValueError(
            'This model cannot compute other outputs since no other output_types were specified.'
        )
    if (outputs is not None and self.model.inputs is not None and
        len(self.model.inputs) == 0):
      raise ValueError(
          "Cannot use 'outputs' argument with a model that does not specify its inputs."
          "Note models defined in imperative subclassing style cannot specify outputs"
      )
    if tf.is_tensor(outputs):
      outputs = [outputs]

    if self._use_openvino:
      openvino_predictions, generator = self._openvino_model(generator)

    for batch in generator:
      inputs, labels, weights = batch
      self._create_inputs(inputs)
      inputs, _, _ = self._prepare_batch((inputs, None, None))

      # Invoke the model.
      if len(inputs) == 1:
        inputs = inputs[0]
      if outputs is not None:
        outputs = tuple(outputs)
        key = tuple(t.ref() for t in outputs)
        if key not in self._output_functions:
          self._output_functions[key] = tf.keras.backend.function(
              self.model.inputs, outputs)
        output_values = self._output_functions[key](inputs)
      else:
        if self._use_openvino:
          output_values = next(openvino_predictions)
          output_values = [output_values]
        else:
          output_values = self._compute_model(inputs)
          if tf.is_tensor(output_values):
            output_values = [output_values]
          output_values = [t.numpy() for t in output_values]

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
          output_values = [undo_transforms(output_values[0], transformers)]
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

  @tf.function(experimental_relax_shapes=True)
  def _compute_model(self, inputs: Sequence):
    """Evaluate the model for a set of inputs."""
    return self.model(inputs, training=False)

  def predict_on_generator(
      self,
      generator: Iterable[Tuple[Any, Any, Any]],
      transformers: List[Transformer] = [],
      outputs: Optional[OneOrMany[tf.Tensor]] = None,
      output_types: Optional[OneOrMany[str]] = None) -> OneOrMany[np.ndarray]:
    """
    Parameters
    ----------
    generator: generator
      this should generate batches, each represented as a tuple of the form
      (inputs, labels, weights).
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's
      standard prediction outputs will be returned.
      Alternatively one or more Tensors within the model may be
      specified, in which case the output of those Tensors will
      be returned. If outputs is specified, output_types must be
      None.
    output_types: String or list of Strings
      If specified, all outputs of this type will be retrieved
      from the model. If output_types is specified, outputs must
      be None.
    Returns:
      a NumPy array of the model produces a single output, or a list of arrays
      if it produces multiple outputs
    """
    return self._predict(generator, transformers, outputs, False, output_types)

  def predict_on_batch(
      self,
      X: ArrayLike,
      transformers: List[Transformer] = [],
      outputs: Optional[OneOrMany[tf.Tensor]] = None) -> OneOrMany[np.ndarray]:
    """Generates predictions for input samples, processing samples in a batch.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    dataset = NumpyDataset(X=X, y=None)
    return self.predict(dataset, transformers, outputs)

  def predict_uncertainty_on_batch(self, X: Sequence, masks: int = 50
                                  ) -> OneOrMany[Tuple[np.ndarray, np.ndarray]]:
    """
    Predict the model's outputs, along with the uncertainty in each one.

    The uncertainty is computed as described in https://arxiv.org/abs/1703.04977.
    It involves repeating the prediction many times with different dropout masks.
    The prediction is computed as the average over all the predictions.  The
    uncertainty includes both the variation among the predicted values (epistemic
    uncertainty) and the model's own estimates for how well it fits the data
    (aleatoric uncertainty).  Not all models support uncertainty prediction.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    masks: int
      the number of dropout masks to average over

    Returns
    -------
    for each output, a tuple (y_pred, y_std) where y_pred is the predicted
    value of the output, and each element of y_std estimates the standard
    deviation of the corresponding element of y_pred
    """
    dataset = NumpyDataset(X=X, y=None)
    return self.predict_uncertainty(dataset, masks)

  def predict(
      self,
      dataset: Dataset,
      transformers: List[Transformer] = [],
      outputs: Optional[OneOrMany[tf.Tensor]] = None,
      output_types: Optional[List[str]] = None) -> OneOrMany[np.ndarray]:
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.
    outputs: Tensor or list of Tensors
      The outputs to return.  If this is None, the model's standard prediction
      outputs will be returned.  Alternatively one or more Tensors within the
      model may be specified, in which case the output of those Tensors will be
      returned.
    output_types: String or list of Strings
      If specified, all outputs of this type will be retrieved
      from the model. If output_types is specified, outputs must
      be None.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    generator = self.default_generator(
        dataset, mode='predict', deterministic=True, pad_batches=False)
    return self.predict_on_generator(
        generator,
        transformers=transformers,
        outputs=outputs,
        output_types=output_types)

  def predict_embedding(self, dataset: Dataset) -> OneOrMany[np.ndarray]:
    """
    Predicts embeddings created by underlying model if any exist.
    An embedding must be specified to have `output_type` of
    `'embedding'` in the model definition.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on

    Returns
    -------
    a NumPy array of the embeddings model produces, or a list
    of arrays if it produces multiple embeddings
    """
    generator = self.default_generator(
        dataset, mode='predict', pad_batches=False)
    return self._predict(generator, [], None, False, ['embedding'])

  def predict_uncertainty(self, dataset: Dataset, masks: int = 50
                         ) -> OneOrMany[Tuple[np.ndarray, np.ndarray]]:
    """
    Predict the model's outputs, along with the uncertainty in each one.

    The uncertainty is computed as described in https://arxiv.org/abs/1703.04977.
    It involves repeating the prediction many times with different dropout masks.
    The prediction is computed as the average over all the predictions.  The
    uncertainty includes both the variation among the predicted values (epistemic
    uncertainty) and the model's own estimates for how well it fits the data
    (aleatoric uncertainty).  Not all models support uncertainty prediction.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    masks: int
      the number of dropout masks to average over

    Returns
    -------
    for each output, a tuple (y_pred, y_std) where y_pred is the predicted
    value of the output, and each element of y_std estimates the standard
    deviation of the corresponding element of y_pred
    """
    sum_pred: List[np.ndarray] = []
    sum_sq_pred: List[np.ndarray] = []
    sum_var: List[np.ndarray] = []
    for i in range(masks):
      generator = self.default_generator(
          dataset, mode='uncertainty', pad_batches=False)
      results = self._predict(generator, [], None, True, None)
      if len(sum_pred) == 0:
        for p, v in results:
          sum_pred.append(p)
          sum_sq_pred.append(p * p)
          sum_var.append(v)
      else:
        for j, (p, v) in enumerate(results):
          sum_pred[j] += p
          sum_sq_pred[j] += p * p
          sum_var[j] += v
    output = []
    std = []
    for i in range(len(sum_pred)):
      p = sum_pred[i] / masks
      output.append(p)
      std.append(np.sqrt(sum_sq_pred[i] / masks - p * p + sum_var[i] / masks))
    if len(output) == 1:
      return (output[0], std[0])
    else:
      return list(zip(output, std))

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
    transformers: list of dc.trans.Transformers
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

  def compute_saliency(self, X: np.ndarray) -> OneOrMany[np.ndarray]:
    """Compute the saliency map for an input sample.

    This computes the Jacobian matrix with the derivative of each output element
    with respect to each input element.  More precisely,

    - If this model has a single output, it returns a matrix of shape
      (output_shape, input_shape) with the derivatives.
    - If this model has multiple outputs, it returns a list of matrices, one
      for each output.

    This method cannot be used on models that take multiple inputs.

    Parameters
    ----------
    X: ndarray
      the input data for a single sample

    Returns
    -------
    the Jacobian matrix, or a list of matrices
    """
    input_shape = X.shape
    X = np.reshape(X, [1] + list(X.shape))
    self._create_inputs([X])
    X, _, _ = self._prepare_batch(([X], None, None))

    # Use a GradientTape to compute gradients.

    X = tf.constant(X[0])
    with tf.GradientTape(
        persistent=True, watch_accessed_variables=False) as tape:
      tape.watch(X)
      outputs = self._compute_model(X)
      if tf.is_tensor(outputs):
        outputs = [outputs]
      final_result = []
      for output in outputs:
        output_shape = tuple(output.shape.as_list()[1:])
        output = tf.reshape(output, [-1])
        result = []
        for i in range(output.shape[0]):
          result.append(tape.gradient(output[i], X))
        final_result.append(
            tf.reshape(tf.stack(result), output_shape + input_shape).numpy())
    if len(final_result) == 1:
      return final_result[0]
    return final_result

  def _prepare_batch(self,
                     batch: Tuple[Any, Any, Any]) -> Tuple[List, List, List]:
    inputs, labels, weights = batch
    inputs = [
        x if x.dtype == t else x.astype(t)
        for x, t in zip(inputs, self._input_dtypes)
    ]
    if labels is not None:
      labels = [
          x if x.dtype == t else x.astype(t)
          for x, t in zip(labels, self._label_dtypes)
      ]
    if weights is not None:
      weights = [
          x if x.dtype == t else x.astype(t)
          for x, t in zip(weights, self._weights_dtypes)
      ]
    for i in range(len(inputs)):
      shape = inputs[i].shape
      dims = len(shape)
      expected_dims = len(self._input_shapes[i])
      if dims < expected_dims:
        inputs[i] = inputs[i].reshape(shape + (1,) * (expected_dims - dims))
      elif dims > expected_dims and all(d == 1 for d in shape[expected_dims:]):
        inputs[i] = inputs[i].reshape(shape[:expected_dims])
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
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield ([X_b], [y_b], [w_b])

  def save_checkpoint(self,
                      max_checkpoints_to_keep: int = 5,
                      model_dir: Optional[str] = None) -> None:
    """Save a checkpoint to disk.

    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.

    Parameters
    ----------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    model_dir: str, default None
      Model directory to save checkpoint to. If None, revert to self.model_dir
    """
    self._ensure_built()
    if model_dir is None:
      model_dir = self.model_dir
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    manager = tf.train.CheckpointManager(self._checkpoint, model_dir,
                                         max_checkpoints_to_keep)
    manager.save()

  def get_checkpoints(self, model_dir: Optional[str] = None):
    """Get a list of all available checkpoint files.

    Parameters
    ----------
    model_dir: str, default None
      Directory to get list of checkpoints from. Reverts to self.model_dir if None

    """
    if model_dir is None:
      model_dir = self.model_dir
    return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths

  def restore(self,
              checkpoint: Optional[str] = None,
              model_dir: Optional[str] = None) -> None:
    """Reload the values of all variables from a checkpoint file.

    Parameters
    ----------
    checkpoint: str
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints.
    model_dir: str, default None
      Directory to restore checkpoint from. If None, use self.model_dir.
    """
    self._ensure_built()
    if model_dir is None:
      model_dir = self.model_dir
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
      raise ValueError('No checkpoint found')
    self._checkpoint.restore(checkpoint)

  def get_global_step(self) -> int:
    """Get the number of steps of fitting that have been performed."""
    return int(self._global_step)

  def _log_scalar_to_tensorboard(self, name: str, value: Any, step: int):
    """Log a scalar value to Tensorboard."""
    with self._summary_writer.as_default():
      tf.summary.scalar(name, value, step)

  def _create_assignment_map(self,
                             source_model: "KerasModel",
                             include_top: bool = True,
                             **kwargs) -> Dict[Any, Any]:
    """
    Creates a default assignment map between variables of source and current model.
    This is used only when a custom assignment map is missing. This assumes the
    model is made of different layers followed by a dense layer for mapping to
    output tasks. include_top is used to control whether or not the final dense
    layer is used. The default assignment map is useful in cases where the type
    of task is different (classification vs regression) and/or number of tasks.

    Parameters
    ----------
    source_model: dc.models.KerasModel
        Source model to copy variable values from.
    include_top: bool, default True
        if true, copies the last dense layer
    """
    assignment_map: Dict[Any, Any] = {}
    source_vars = source_model.model.trainable_variables
    dest_vars = self.model.trainable_variables

    if not include_top:
      source_vars = source_vars[:-2]
      dest_vars = dest_vars[:-2]

    for source_var, dest_var in zip(source_vars, dest_vars):
      assignment_map[source_var.ref()] = dest_var

    return assignment_map

  def _create_value_map(self, source_model: "KerasModel",
                        **kwargs) -> Dict[Any, Any]:
    """
    Creates a value map between variables in the source model and their
    current values. This is used only when a custom value map is missing, and
    assumes the restore method has been called under self.session.

    Parameters
    ----------
    source_model: dc.models.KerasModel
        Source model to create value map from
    """
    value_map: Dict[Any, Any] = {}
    source_vars = source_model.model.trainable_variables

    for source_var in source_vars:
      value_map[source_var.ref()] = source_var.numpy()

    return value_map

  def load_from_pretrained(self,
                           source_model: "KerasModel",
                           assignment_map: Optional[Dict[Any, Any]] = None,
                           value_map: Optional[Dict[Any, Any]] = None,
                           checkpoint: Optional[str] = None,
                           model_dir: Optional[str] = None,
                           include_top: bool = True,
                           inputs: Optional[Sequence[Any]] = None,
                           **kwargs) -> None:
    """Copies variable values from a pretrained model. `source_model` can either
    be a pretrained model or a model with the same architecture. `value_map`
    is a variable-value dictionary. If no `value_map` is provided, the variable
    values are restored to the `source_model` from a checkpoint and a default
    `value_map` is created. `assignment_map` is a dictionary mapping variables
    from the `source_model` to the current model. If no `assignment_map` is
    provided, one is made from scratch and assumes the model is composed of
    several different layers, with the final one being a dense layer. include_top
    is used to control whether or not the final dense layer is used. The default
    assignment map is useful in cases where the type of task is different
    (classification vs regression) and/or number of tasks in the setting.

    Parameters
    ----------
    source_model: dc.KerasModel, required
      source_model can either be the pretrained model or a dc.KerasModel with
      the same architecture as the pretrained model. It is used to restore from
      a checkpoint, if value_map is None and to create a default assignment map
      if assignment_map is None
    assignment_map: Dict, default None
      Dictionary mapping the source_model variables and current model variables
    value_map: Dict, default None
      Dictionary containing source_model trainable variables mapped to numpy
      arrays. If value_map is None, the values are restored and a default
      variable map is created using the restored values
    checkpoint: str, default None
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints
    model_dir: str, default None
      Restore model from custom model directory if needed
    include_top: bool, default True
        if True, copies the weights and bias associated with the final dense
        layer. Used only when assignment map is None
    inputs: List, input tensors for model
        if not None, then the weights are built for both the source and self.
        This option is useful only for models that are built by
        subclassing tf.keras.Model, and not using the functional API by tf.keras
    """
    if inputs is not None:
      # Ensure weights for both models are built.
      source_model.model(inputs)
      self.model(inputs)

    self._ensure_built()
    if value_map is None:
      logger.info(
          "No value map provided. Creating default value map from restored model."
      )
      source_model.restore(model_dir=model_dir, checkpoint=checkpoint)
      value_map = self._create_value_map(source_model=source_model)

    if assignment_map is None:
      logger.info("No assignment map provided. Creating custom assignment map.")
      assignment_map = self._create_assignment_map(
          source_model=source_model, include_top=include_top)

    for source_var, dest_var in assignment_map.items():
      assert source_var.deref().shape == dest_var.shape
      dest_var.assign(value_map[source_var])


class _StandardLoss(object):
  """The implements the loss function for models that use a dc.models.losses.Loss."""

  def __init__(self, model: tf.keras.Model, loss: Loss) -> None:
    self.model = model
    self.loss = loss

  def __call__(self, outputs: List, labels: List, weights: List) -> float:
    if len(outputs) != 1 or len(labels) != 1 or len(weights) != 1:
      raise ValueError(
          "Loss functions expects exactly one each of outputs, labels, and weights"
      )
    losses = self.loss._compute_tf_loss(outputs[0], labels[0])
    w = weights[0]
    if len(w.shape) < len(losses.shape):
      if tf.is_tensor(w):
        shape = tuple(w.shape.as_list())
      else:
        shape = w.shape
      shape = tuple(-1 if x is None else x for x in shape)
      w = tf.reshape(w, shape + (1,) * (len(losses.shape) - len(w.shape)))

    loss = losses * w
    return tf.reduce_mean(loss) + sum(self.model.losses)

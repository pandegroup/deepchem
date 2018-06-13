import collections
import os
import pickle
import threading
import time

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader
import tensorflow.contrib.eager as tfe

from deepchem.data import NumpyDataset
from deepchem.models.models import Model
from deepchem.models.tensorgraph.layers import InputFifoQueue, Label, Feature, Weights, Constant, Input
from deepchem.models.tensorgraph.model_ops import create_variable
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator

logger = logging.getLogger(__name__)


class TensorGraph(Model):

  def __init__(self,
               tensorboard=False,
               tensorboard_log_frequency=100,
               batch_size=100,
               random_seed=None,
               use_queue=True,
               graph=None,
               learning_rate=0.001,
               configproto=None,
               **kwargs):
    """
    Parameters
    ----------
    tensorboard: bool
      Should we log to model_dir data for tensorboard?
    tensorboard_log_frequency: int
      How many training batches before logging tensorboard?
    batch_size: int
      default batch size for training and evaluating
    use_queue: boolean
      if True when building we will create a tf.FIFO queue, which will hold
      all features, weights, and labels.  We will feed the inputs into this
      queue in batches of self.batch_size in a separate thread from the
      thread training the model.  You cannot use a queue when
      batches are not of consistent size
    graph: tensorflow.Graph
      the Graph in which to create Tensorflow objects.  If None, a new Graph
      is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    configproto: a tf.ConfigProto() object used to create tf.Session()
    """

    # Layer Management
    self.layers = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.variances = list()
    self.task_weights = list()
    self.submodels = list()
    self.loss = Constant(0)
    self.built = False
    self.queue_installed = False
    self.optimizer = Adam(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7)
    self.configproto = configproto

    # Singular place to hold Tensor objects which don't serialize
    # These have to be reconstructed on restoring from pickle
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "FileWriter": None,
        "Graph": graph,
        "train_op": None,
        "summary_op": None,
    }
    self.tensorboard = tensorboard
    self.tensorboard_log_frequency = tensorboard_log_frequency
    self.tensorboard_step = 0
    self.global_step = 0
    self.use_queue = use_queue

    self.batch_size = batch_size
    self.random_seed = random_seed
    super(TensorGraph, self).__init__(**kwargs)
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    if self.use_queue and self.tensorboard:
      raise ValueError(
          "Currently TensorGraph cannot both use_queue and tensorboard at the same time"
      )

  def _add_layer(self, layer):
    if layer.name is None:
      layer.name = "%s_%s" % (layer.__class__.__name__, len(self.layers) + 1)
    if layer.name in self.layers:
      return
    if isinstance(layer, Feature):
      self.features.append(layer)
    if isinstance(layer, Label):
      self.labels.append(layer)
    if isinstance(layer, Weights):
      self.task_weights.append(layer)
    self.layers[layer.name] = layer
    for in_layer in layer.in_layers:
      self._add_layer(in_layer)

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          submodel=None,
          **kwargs):
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
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().
    """
    return self.fit_generator(
        self.default_generator(
            dataset, epochs=nb_epoch, deterministic=deterministic),
        max_checkpoints_to_keep, checkpoint_interval, restore, submodel)

  def fit_generator(self,
                    feed_dict_generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False,
                    submodel=None):
    """Train this model on data from a generator.
    Parameters
    ----------
    feed_dict_generator: generator
      this should generate batches, each represented as a dict that maps
      Layers to values.
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().
    Returns
    -------
    the average loss over the most recent checkpoint interval
    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      loss = self.loss
      if submodel is not None and submodel.loss is not None:
        loss = submodel.loss
      if tfe.in_eager_mode():
        # In eager mode we want an optimizer and a function to compute the
        # gradient of the loss.

        submodel_vars = None
        if submodel is None:
          optimizer = self._get_tf("Optimizer")
        else:
          optimizer = submodel.create_optimizer()
          if submodel.layers is not None:
            submodel_vars = set()
            for layer in submodel.layers:
              for var in layer.variables:
                submodel_vars.add(var)
        val_grad_fn = tfe.implicit_value_and_gradients(
            lambda x: self._run_graph([loss], x, True)[0])
      else:
        # In graph mode we want a training operation.

        if submodel is None:
          train_op = self._get_tf('train_op')
        else:
          train_op = submodel.get_train_op()
      if checkpoint_interval > 0:
        saver = tf.train.Saver(
            self.get_variables(),
            max_to_keep=max_checkpoints_to_keep,
            save_relative_paths=True)
      if restore:
        self.restore()
      avg_loss, n_averaged_batches = 0.0, 0.0
      n_samples = 0
      n_enqueued = [0]
      final_sample = [None]
      if self.queue_installed:
        enqueue_thread = threading.Thread(
            target=_enqueue_batch,
            args=(self, feed_dict_generator, self._get_tf("Graph"),
                  self.session, n_enqueued, final_sample))
        enqueue_thread.start()
      for feed_dict in self._create_feed_dicts(feed_dict_generator, True):
        if self.queue_installed:
          # Don't let this thread get ahead of the enqueue thread, since if
          # we try to read more batches than the total number that get queued,
          # this thread will hang indefinitely.
          while n_enqueued[0] <= n_samples:
            if n_samples == final_sample[0]:
              break
            time.sleep(0)
          if n_samples == final_sample[0]:
            break
        n_samples += 1
        should_log = (self.tensorboard and
                      n_samples % self.tensorboard_log_frequency == 0)
        if tfe.in_eager_mode():
          value, grads_and_vars = val_grad_fn(feed_dict)
          if submodel_vars is not None:
            grads_and_vars = [
                x for x in grads_and_vars if x[1] in submodel_vars
            ]
          optimizer.apply_gradients(grads_and_vars)
          avg_loss += value
        else:
          fetches = [train_op, loss.out_tensor]
          if should_log:
            fetches.append(self._get_tf("summary_op"))
          fetched_values = self.session.run(fetches, feed_dict=feed_dict)
          if should_log:
            self._log_tensorboard(fetched_values[2])
          avg_loss += fetched_values[1]
        n_averaged_batches += 1
        self.global_step += 1
        if checkpoint_interval > 0 and self.global_step % checkpoint_interval == checkpoint_interval - 1:
          saver.save(self.session, self.save_file, global_step=self.global_step)
          avg_loss = float(avg_loss) / n_averaged_batches
          logger.info('Ending global_step %d: Average loss %g' %
                      (self.global_step, avg_loss))
          avg_loss, n_averaged_batches = 0.0, 0.0
      if n_averaged_batches > 0:
        avg_loss = float(avg_loss) / n_averaged_batches
      if checkpoint_interval > 0:
        if n_averaged_batches > 0:
          logger.info('Ending global_step %d: Average loss %g' %
                      (self.global_step, avg_loss))
        saver.save(self.session, self.save_file, global_step=self.global_step)
        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return avg_loss

  def _log_tensorboard(self, summary):
    """
    TODO(LESWING) set epoch
    Parameters
    ----------
    Returns
    -------
    """
    global_step = int(self.global_step)
    writer = self._get_tf("FileWriter")
    writer.reopen()
    writer.add_summary(summary, global_step=global_step)
    writer.close()

  def fit_on_batch(self, X, y, w, submodel=None):
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y)
    return self.fit(dataset, nb_epoch=1, submodel=submodel)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    if len(self.features) > 1:
      raise ValueError("More than one Feature, must use generator")
    if len(self.labels) > 1:
      raise ValueError("More than one Label, must use generator")
    if len(self.task_weights) > 1:
      raise ValueError("More than one Weights, must use generator")
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if len(self.labels) == 1 and y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if len(self.features) == 1 and X_b is not None:
          feed_dict[self.features[0]] = X_b
        if len(self.task_weights) == 1 and w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        for (initial_state, zero_state) in zip(self.rnn_initial_states,
                                               self.rnn_zero_states):
          feed_dict[initial_state] = zero_state
        yield feed_dict

  def __call__(self, *inputs, **kwargs):
    """Execute the model in eager mode to compute outputs as a function of inputs.
    This is very similar to predict_on_batch(), except that it returns the outputs
    as tensors rather than numpy arrays.  That means you can compute the graph's
    outputs, then do additional calculations based on them, and gradients will
    be tracked correctly through the whole process.
    Parameters
    ----------
    inputs: tensors
      the values to use for the model's features.  The number of inputs must
      exactly match the length of the model's `features` property.  The values
      may be tensors, numpy arrays, or anything else that can be converted to
      tensors of the correct shape.
    outputs: list of Layers
      the output layers to compute.  If this is omitted, self.outputs is used
      (that is, all outputs that have been added by calling add_output()).
    Returns
    -------
    The output tensors, or a list of tensors if multiple outputs were requested.
    """
    if len(inputs) != len(self.features):
      raise ValueError('Expected %d inputs, received %d' % len(self.features),
                       len(inputs))
    # TODO Once we drop Python 2 support, turn outputs into a proper keyword arg
    # instead of using the **kwargs hack.
    if 'outputs' in kwargs:
      outputs = kwargs['outputs']
    else:
      outputs = self.outputs
    feed_dict = dict(zip(self.features, inputs))
    results = self._run_graph(outputs, feed_dict, False)
    if len(results) == 1:
      return results[0]
    return results

  def _predict(self, generator, transformers, outputs, uncertainty):
    """
    Predict outputs for data provided by a generator.
    This is the private implementation of prediction.  Do not call it directly.
    Instead call one of the public prediction methods.
    Parameters
    ----------
    generator: Generator
      Generator that constructs feed dictionaries for TensorGraph.
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs.
      If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.
    uncertainty: bool
      specifies whether this is being called as part of estimating uncertainty.
      If True, it sets the training flag so that dropout will be enabled, and
      returns the values of the uncertainty outputs.
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    if outputs is None:
      outputs = self.outputs
    elif not isinstance(outputs, collections.Sequence):
      outputs = [outputs]
    if uncertainty:
      if len(self.variances) == 0:
        raise ValueError('This model cannot compute uncertainties')
      if len(self.variances) != len(outputs):
        raise ValueError(
            'The number of variances must exactly match the number of outputs')
      tensors = outputs + self.variances
    else:
      tensors = outputs

    with self._get_tf("Graph").as_default():
      # Gather results for each output
      results = [[] for out in tensors]
      n_samples = 0
      n_enqueued = [0]
      final_sample = [None]
      if self.queue_installed:
        enqueue_thread = threading.Thread(
            target=_enqueue_batch,
            args=(self, generator, self._get_tf("Graph"), self.session,
                  n_enqueued, final_sample))
        enqueue_thread.start()
      for feed_dict in self._create_feed_dicts(generator, uncertainty):
        if self.queue_installed:
          # Don't let this thread get ahead of the enqueue thread, since if
          # we try to read more batches than the total number that get queued,
          # this thread will hang indefinitely.
          while n_enqueued[0] <= n_samples:
            if n_samples == final_sample[0]:
              break
            time.sleep(0)
          if n_samples == final_sample[0]:
            break
        n_samples += 1
        feed_results = self._run_graph(tensors, feed_dict, uncertainty)
        if tfe.in_eager_mode():
          feed_results = [f.numpy() for f in feed_results]
        if len(feed_results) > 1:
          if len(transformers):
            raise ValueError("Does not support transformations "
                             "for multiple outputs.")
        elif len(feed_results) == 1:
          result = undo_transforms(feed_results[0], transformers)
          feed_results = [result]
        for ind, result in enumerate(feed_results):
          results[ind].append(result)

      final_results = []
      for result_list in results:
        final_results.append(np.concatenate(result_list, axis=0))
      # If only one output, just return array
      if len(final_results) == 1:
        return final_results[0]
      elif uncertainty:
        return zip(final_results[:len(outputs)], final_results[len(outputs):])
      else:
        return final_results

  def predict_on_generator(self, generator, transformers=[], outputs=None):
    """
    Parameters
    ----------
    generator: Generator
      Generator that constructs feed dictionaries for TensorGraph.
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs.
      If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    return self._predict(generator, transformers, outputs, False)

  def predict_on_batch(self, X, transformers=[], outputs=None):
    """Generates predictions for input samples, processing samples in a batch.
    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    transformers: List
      List of dc.trans.Transformers
    Returns
    -------
    A Numpy array of predictions.
    """
    dataset = NumpyDataset(X=X, y=None)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

  def predict_uncertainty_on_batch(self, X, masks=50):
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

  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.
    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs=self.outputs. If outputs is
      a Layer/Tensor, then will evaluate and return as a single ndarray. If
      outputs is a list of Layers/Tensors, will return a list of ndarrays.
    Returns
    -------
    results: numpy ndarray or list of numpy ndarrays
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

  def predict_uncertainty(self, dataset, masks=50):
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
    sum_pred = []
    sum_sq_pred = []
    sum_var = []
    for i in range(masks):
      generator = self.default_generator(
          dataset, predict=True, pad_batches=False)
      results = self._predict(generator, [], self.outputs, True)
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
      return zip(output, std)

  def topsort(self):

    def add_layers_to_list(layer, sorted_layers):
      if layer in sorted_layers:
        return
      for in_layer in layer.in_layers:
        add_layers_to_list(in_layer, sorted_layers)
      sorted_layers.append(layer)

    sorted_layers = []
    for l in self.features + self.labels + self.task_weights + self.outputs + self.variances:
      add_layers_to_list(l, sorted_layers)
    add_layers_to_list(self.loss, sorted_layers)
    for submodel in self.submodels:
      if submodel.loss is not None:
        add_layers_to_list(submodel.loss, sorted_layers)
    return sorted_layers

  def build(self):
    if self.built:
      return
    if tfe.in_eager_mode():
      # In eager mode, we need to execute every layer once to ensure its variables
      # have been created.

      def build_layers(layer, tensors):
        if layer in tensors:
          return tensors[layer]
        inputs = [build_layers(input, tensors) for input in layer.in_layers]
        if isinstance(layer, Input):
          # We can't execute Input layers in eager mode, since they would try
          # to create placeholders.  Instead create a tensor of the correct
          # size and type.
          shape = [1 if s is None else s for s in layer.shape]
          tensor = tf.zeros(shape, layer.dtype)
        else:
          with tf.name_scope(layer.name):
            tensor = layer.create_tensor(in_layers=inputs, set_tensors=False)
        tensors[layer] = tensor
        return tensor

      tensors = {}
      with self._get_tf("Graph").as_default():
        # Build the layers.

        build_layers(self.loss, tensors)
        for output in self.outputs:
          build_layers(output, tensors)
        for variance in self.variances:
          build_layers(variance, tensors)
        for submodel in self.submodels:
          build_layers(submodel.loss, tensors)

        # Initialize variables.

        for layer in self.layers.values():
          if layer.variable_values is not None:
            for var, val in zip(layer.variables, layer.variable_values):
              var.assign(val)
      self.session = None
      self._training_placeholder = None
      self.built = True
      return

    # In graph mode we need to create the computation graph.

    with self._get_tf("Graph").as_default():
      self._training_placeholder = tf.placeholder(dtype=tf.float32, shape=())
      if self.random_seed is not None:
        tf.set_random_seed(self.random_seed)
      self._install_queue()
      self.built = True
      for layer in self.topsort():
        with tf.name_scope(layer.name):
          layer.create_tensor(training=self._training_placeholder)
          self.rnn_initial_states += layer.rnn_initial_states
          self.rnn_final_states += layer.rnn_final_states
          self.rnn_zero_states += layer.rnn_zero_states
          layer.add_summary_to_tg(self.get_layer_variables(layer))
      self.session = tf.Session(config=self.configproto)

      # Ensure all training operators have been created.

      self._get_tf('train_op')
      for submodel in self.submodels:
        train_op = submodel.get_train_op()

      # Initialize variables.

      self.session.run(tf.global_variables_initializer())
      for layer in self.layers.values():
        if layer.variable_values is not None:
          variables = self.get_layer_variables(layer)
          for var, val in zip(variables, layer.variable_values):
            self.session.run(var.assign(val))

    for layer in self.layers.values():
      if layer.tensorboard:
        self.tensorboard = True
    tf.summary.scalar("loss", self.loss.out_tensor)
    for layer in self.layers.values():
      if layer.tensorboard:
        tf.summary.tensor_summary(layer.name, layer.out_tensor)
    if self.tensorboard:
      writer = self._get_tf("FileWriter")
      writer.add_graph(self._get_tf("Graph"))
      writer.close()

    # As a sanity check, make sure all tensors have the correct shape.

    for layer in self.layers.values():
      try:
        assert list(layer.shape) == layer.out_tensor.get_shape().as_list(
        ), '%s: Expected shape %s does not match actual shape %s' % (
            layer.name, layer.shape, layer.out_tensor.get_shape().as_list())
      except NotImplementedError:
        pass

  def _install_queue(self):
    """
    """
    if not self.use_queue or self.queue_installed:
      for layer in self.features + self.labels + self.task_weights:
        layer.pre_queue = True
      return
    inputs = self.features + self.labels + self.task_weights
    if len(inputs) == 0:
      return
    names = []
    shapes = []
    pre_q_inputs = []
    q = InputFifoQueue(shapes, names, in_layers=pre_q_inputs)
    q.name = "%s_%s" % (q.__class__.__name__, len(self.layers) + 1)

    for layer in inputs:
      pre_q_input = layer.create_pre_q()
      shapes.append(pre_q_input.shape)
      names.append(pre_q_input.name)
      pre_q_inputs.append(pre_q_input)

      layer.in_layers.append(q)

    self._add_layer(q)
    self.input_queue = q
    self.queue_installed = True

  def set_loss(self, layer):
    self._add_layer(layer)
    self.loss = layer

  def add_output(self, layer):
    """Add an output layer that can be computed by predict()"""
    self._add_layer(layer)
    self.outputs.append(layer)

  def add_variance(self, layer):
    """Add a layer that computes the variance in an output.
    If a model supports uncertainty, it must call add_variance() once for every
    output.  Each variance layer has the same shape as the corresponding output,
    and each element computes an estimate of the variance from aleatoric
    uncertainty in the corresponding element of the output.
    In addition, if a model supports uncertainty it MUST use dropout on every
    layer.  Otherwise, the uncertainties it computes will be inaccurate.
    """
    self._add_layer(layer)
    self.variances.append(layer)

  def set_optimizer(self, optimizer):
    """Set the optimizer to use for fitting."""
    self.optimizer = optimizer

  def create_submodel(self, layers=None, loss=None, optimizer=None):
    """Create an alternate objective for training one piece of a TensorGraph.
    A TensorGraph consists of a set of layers, and specifies a loss function and
    optimizer to use for training those layers.  Usually this is sufficient, but
    there are cases where you want to train different parts of a model separately.
    For example, a GAN consists of a generator and a discriminator.  They are
    trained separately, and they use different loss functions.
    A submodel defines an alternate objective to use in cases like this.  It may
    optionally specify any of the following: a subset of layers in the model to
    train; a different loss function; and a different optimizer to use.  This
    method creates a submodel, which you can then pass to fit() to use it for
    training.
    Parameters
    ----------
    layers: list
      the list of layers to train.  If None, all layers in the model will be
      trained.
    loss: Layer
      the loss function to optimize.  If None, the model's main loss function
      will be used.
    optimizer: Optimizer
      the optimizer to use for training.  If None, the model's main optimizer
      will be used.
    Returns
    -------
    the newly created submodel, which can be passed to any of the fitting
    methods.
    """
    if self.built:
      raise ValueError('Submodels must be created before build() is called.')
    submodel = Submodel(self, layers, loss, optimizer)
    self.submodels.append(submodel)
    if loss is not None:
      self._add_layer(loss)
    return submodel

  def get_pickling_errors(self, obj, seen=None):
    if seen == None:
      seen = []
    try:
      state = obj.__getstate__()
    except AttributeError:
      return
    if state == None:
      return
    if isinstance(state, tuple):
      if not isinstance(state[0], dict):
        state = state[1]
      else:
        state = state[0].update(state[1])
    result = {}
    for i in state:
      try:
        pickle.dumps(state[i], protocol=2)
      except pickle.PicklingError:
        if not state[i] in seen:
          seen.append(state[i])
          result[i] = self.get_pickling_errors(state[i], seen)
    return result

  def save(self):
    # Remove out_tensor from the object to be pickled
    must_restore = False
    tensor_objects = self.tensor_objects
    rnn_initial_states = self.rnn_initial_states
    rnn_final_states = self.rnn_final_states
    rnn_zero_states = self.rnn_zero_states
    session = self.session
    self.tensor_objects = {}
    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    self.session = None
    out_tensors = []
    submodel_ops = []
    if self.built:
      must_restore = True
      for layer in self.topsort():
        out_tensors.append(layer.none_tensors())
      for submodel in self.submodels:
        submodel_ops.append(submodel._train_op)
        submodel._train_op = None
      training_placeholder = self._training_placeholder
      self._training_placeholder = None
      self.built = False

    # Pickle itself
    pickle_name = os.path.join(self.model_dir, "model.pickle")

    with open(pickle_name, 'wb') as fout:
      try:
        pickle.dump(self, fout)
      except Exception as e:
        logger.info(self.get_pickling_errors(self))
        raise e

    # add out_tensor back to everyone
    if must_restore:
      for index, layer in enumerate(self.topsort()):
        layer.set_tensors(out_tensors[index])
      for submodel, op in zip(self.submodels, submodel_ops):
        submodel._train_op = op
      self._training_placeholder = training_placeholder
      self.built = True
    self.tensor_objects = tensor_objects
    self.rnn_initial_states = rnn_initial_states
    self.rnn_final_states = rnn_final_states
    self.rnn_zero_states = rnn_zero_states
    self.session = session

  def evaluate_generator(self,
                         feed_dict_generator,
                         metrics,
                         transformers=[],
                         labels=None,
                         outputs=None,
                         weights=[],
                         per_task_metrics=False):

    if labels is None:
      raise ValueError
    n_tasks = len(self.outputs)
    n_classes = self.outputs[0].out_tensor.get_shape()[-1].value
    evaluator = GeneratorEvaluator(
        self,
        feed_dict_generator,
        transformers,
        labels=labels,
        outputs=outputs,
        weights=weights,
        n_tasks=n_tasks,
        n_classes=n_classes)
    if not per_task_metrics:
      scores = evaluator.compute_model_performance(metrics)
      return scores
    else:
      scores, per_task_scores = evaluator.compute_model_performance(
          metrics, per_task_metrics=per_task_metrics)
      return scores, per_task_scores

  def get_layer_variables(self, layer):
    """Get the list of trainable variables in a layer of the graph."""
    if tfe.in_eager_mode():
      return layer.variables
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      if layer.variable_scope == '':
        return []
      return tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=layer.variable_scope)

  def get_variables(self):
    """Get the list of all trainable variables in the graph."""
    if not self.built:
      self.build()
    if tfe.in_eager_mode():
      variables = []
      for layer in self.layers.values():
        variables += layer.variables
      return variables
    else:
      with self._get_tf("Graph").as_default():
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  def get_global_step(self):
    return self._get_tf("GlobalStep")

  def _get_tf(self, obj):
    """Fetches underlying TensorFlow primitives.
    Parameters
    ----------
    obj: str
      If "Graph", returns tf.Graph instance. If "FileWriter", returns
      tf.summary.FileWriter. If "Optimizer", returns the optimizer. If
      "train_op", returns the train operation. If "summary_op", returns the
      merged summary. If "GlobalStep" returns the global step.
    Returns
    -------
    TensorFlow Object
    """

    if obj in self.tensor_objects and self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects['Graph'] = tf.Graph()
    elif obj == "FileWriter":
      self.tensor_objects['FileWriter'] = tf.summary.FileWriter(self.model_dir)
    elif obj == 'Optimizer':
      self.tensor_objects['Optimizer'] = self.optimizer._create_optimizer(
          self._get_tf('GlobalStep'))
    elif obj == 'train_op':
      opt = self._get_tf('Optimizer')
      global_step = self._get_tf('GlobalStep')
      try:
        self.tensor_objects['train_op'] = opt.minimize(
            self.loss.out_tensor, global_step=global_step)
      except ValueError:
        # The loss doesn't depend on any variables.
        self.tensor_objects['train_op'] = 0
    elif obj == 'summary_op':
      self.tensor_objects['summary_op'] = tf.summary.merge_all(
          key=tf.GraphKeys.SUMMARIES)
    elif obj == 'GlobalStep':
      with self._get_tf("Graph").as_default():
        self.tensor_objects['GlobalStep'] = create_variable(0, trainable=False)
    return self._get_tf(obj)

  def save_checkpoint(self, max_checkpoints_to_keep=5):
    """Save a checkpoint to disk.
    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.
    Parameters
    ----------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    """
    saver = tf.train.Saver(
        self.get_variables(), max_to_keep=max_checkpoints_to_keep)
    saver.save(self.session, self.save_file, global_step=self.global_step)

  def get_checkpoints(self):
    """Get a list of all available checkpoint files."""
    return tf.train.get_checkpoint_state(
        self.model_dir).all_model_checkpoint_paths

  def restore(self, checkpoint=None):
    """Reload the values of all variables from a checkpoint file.
    Parameters
    ----------
    checkpoint: str
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints.
    """
    if not self.built:
      self.build()
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint is None:
      raise ValueError('No checkpoint found')
    with self._get_tf("Graph").as_default():
      reader = NewCheckpointReader(checkpoint)
      var_names = set([x for x in reader.get_variable_to_shape_map()])
      var_list = []
      for var in self.get_variables():
        name = var.name
        if ':' in name:
          name = name[:name.rfind(':')]
        if name in var_names:
          var_list.append(var)
      saver = tf.train.Saver(var_list=var_list)
      saver.restore(self.session, checkpoint)

  def get_num_tasks(self):
    return len(self.outputs)

  def get_pre_q_input(self, input_layer):
    layer_name = input_layer.name
    pre_q_name = "%s_pre_q" % layer_name
    return self.layers[pre_q_name]

  @staticmethod
  def load_from_dir(model_dir, restore=True):
    pickle_name = os.path.join(model_dir, "model.pickle")
    with open(pickle_name, 'rb') as fout:
      tensorgraph = pickle.load(fout)
      tensorgraph.built = False
      tensorgraph.model_dir = model_dir
      if restore:
        try:
          tensorgraph.restore()
        except ValueError:
          pass  # No checkpoint to load
      return tensorgraph

  def __del__(self):
    pass

  def _create_feed_dicts(self, generator, training):
    """Create feed dicts for use in fitting or prediction.
    Parameters
    ----------
    generator: Generator
      the feed dict generator that was passed to fit_generator() or predict_on_generator()
    training: bool
      True during training, False during prediction
    """
    train_value = 1.0 if training else 0.0
    if self.queue_installed:
      while True:
        yield {self._training_placeholder: train_value}
    else:
      for d in generator:
        feed_dict = {}
        for key, value in d.items():
          if isinstance(key, Input):
            value = _ensure_value_shape(value, key)
            if tfe.in_eager_mode():
              value = tf.cast(value, key.dtype)
            feed_dict[key] = value
          else:
            feed_dict[key] = value
        if not tfe.in_eager_mode():
          feed_dict[self._training_placeholder] = train_value
        yield feed_dict

  def _run_graph(self, outputs, feed_dict, training):
    """Run the calculations in the graph to compute some outputs.
    In graph mode, this just calls session.run().  In eager mode, it executes
    all required layers to compute the output.
    Parameters
    ----------
    outputs: list of Layers
      the output layers to compute
    feed_dict: dict
      maps input layers to values
    training: bool
      whether this is being executed in training mode
    """
    if not tfe.in_eager_mode():
      return self.session.run(outputs, feed_dict)

    def run_layers(layer, tensors):
      if layer in tensors:
        return tensors[layer]
      inputs = [run_layers(input, tensors) for input in layer.in_layers]
      tensor = layer.create_tensor(
          in_layers=inputs, set_tensors=False, training=training)
      tensors[layer] = tensor
      return tensor

    tensors = feed_dict.copy()
    return [run_layers(o, tensors) for o in outputs]

  def make_estimator(self,
                     feature_columns,
                     weight_column=None,
                     metrics={},
                     model_dir=None,
                     config=None):
    """Construct a Tensorflow Estimator from this model.
    tf.estimator.Estimator is the standard Tensorflow API for representing models.
    This method provides interoperability between DeepChem and other Tensorflow
    based tools by allowing any model to be used an Estimator.
    Once this method returns, the Estimator it created is independent of the model
    it was created from.  They do not share tensors, variables, save files, or any
    other resources.  The Estimator is a self contained object with its own methods
    for training, evaluation, prediction, checkpointing, etc.
    Parameters
    ----------
    feature_columns: list of tf.feature_column objects
      this describes the input features to the models.  There must be one entry
      for each Feature layer in this model's features field.
    weight_column: tf.feature_column or None
      if this model includes a Weights layer, this describes the input weights.
      Otherwise, this should be None.
    metrics: map
      metrics that should be computed in calls to evaluate().  For each entry,
      the key is the name to report for the metric, and the value is a function
      of the form f(labels, predictions, weights) that returns the tensors for
      computing the metric.  Any of the functions in tf.metrics can be used, as
      can other functions that satisfy the same interface.
    model_dir: str
      the directory in which the Estimator should save files.  If None, this
      defaults to the model's model_dir.
    config: RunConfig
      configuration options for the Estimator
    """
    # Check the inputs.

    if tfe.in_eager_mode():
      raise ValueError('make_estimator() is not supported in eager mode')
    if len(feature_columns) != len(self.features):
      raise ValueError(
          'This model requires %d feature column(s)' % len(self.features))
    if len(self.labels) != 1:
      raise ValueError(
          'Can only create an Estimator from a model with exactly one Label input'
      )
    if len(self.task_weights) > 1:
      raise ValueError(
          'Cannot create an Estimator from a model with multiple Weight inputs')
    if weight_column is None:
      if len(self.task_weights) > 0:
        raise ValueError('This model requires a weight column')
    else:
      if len(self.task_weights) == 0:
        raise ValueError(
            'Cannot specify weight_column for a model with no Weight inputs')
    if model_dir is None:
      model_dir = self.model_dir

    # Define a function that recursively creates tensors from layers.

    def create_tensors(layer, tensors, training):
      if layer in tensors:
        return tensors[layer]
      inputs = [
          create_tensors(in_layer, tensors, training)
          for in_layer in layer.in_layers
      ]
      tensor = layer.create_tensor(
          in_layers=inputs, set_tensors=False, training=training)
      tensors[layer] = tensor
      layer.add_summary_to_tg(tensor)
      return tensor

    # Define the model function.

    def model_fn(features, labels, mode):
      # Define the inputs.

      tensors = self.create_estimator_inputs(feature_columns, weight_column,
                                             features, labels, mode)
      for layer, tensor in tensors.items():
        layer.add_summary_to_tg(tensor)

      # Create the correct outputs, based on the mode.

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        for i, output in enumerate(self.outputs):
          predictions[i] = create_tensors(output, tensors, 0)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      if mode == tf.estimator.ModeKeys.EVAL:
        loss = create_tensors(self.loss, tensors, 0)
        predictions = create_tensors(self.outputs[0], tensors, 0)
        if len(self.task_weights) == 0:
          weights = None
        else:
          weights = tensors[self.task_weights[0]]
        eval_metric_ops = {}
        for name, function in metrics.items():
          eval_metric_ops[name] = function(tensors[self.labels[0]], predictions,
                                           weights)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)
      if mode == tf.estimator.ModeKeys.TRAIN:
        loss = create_tensors(self.loss, tensors, 1)
        global_step = tf.train.get_global_step()
        optimizer = self.optimizer._create_optimizer(global_step)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
      raise ValueError('Unknown mode')

    # Create the Estimator.

    return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    """This is called by make_estimator() to create tensors for the inputs.
    feature_columns and weight_column are the arguments passed to
    make_estimator().  features, labels, and mode are the arguments passed to
    the estimator's model function.  This method creates and returns a dict with
    one entry for every Feature, Label, or Weights layer in the graph.  The keys
    are the layers, and the values are the tensors that correspond to them.
    Any subclass that overrides default_generator() must also override this
    method.
    """
    if self.__class__.default_generator != TensorGraph.default_generator:
      raise ValueError(
          "Class overrides default_generator() but not create_estimator_inputs()"
      )
    tensors = {}
    for layer, column in zip(self.features, feature_columns):
      tensors[layer] = tf.feature_column.input_layer(features, [column])
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      tensors[self.labels[0]] = tf.cast(labels, self.labels[0].dtype)
    return tensors


def _ensure_value_shape(value, layer):
  """Ensure that a value has the right shape for an input layer."""
  # Add or remove dimensions of size 1 to match the shape of the layer.
  try:
    value_dims = len(value.shape)
    layer_dims = len(layer.shape)
    if value_dims < layer_dims:
      if all(i == 1 for i in layer.shape[value_dims:]):
        value = value.reshape(
            list(value.shape) + [1] * (layer_dims - value_dims))
    if value_dims > layer_dims:
      if all(i == 1 for i in value.shape[layer_dims:]):
        value = value.reshape(value.shape[:layer_dims])
  except:
    pass
  return value


def _enqueue_batch(tg, generator, graph, sess, n_enqueued, final_sample):
  """
  Function to load data into
  Parameters
  ----------
  tg
  dataset
  graph
  sess
  Returns
  -------
  """
  with graph.as_default():
    num_samples = 0
    for feed_dict in generator:
      enq = {}
      enq[tg._training_placeholder] = 1.0
      for layer in tg.features + tg.labels + tg.task_weights:
        if layer in feed_dict:
          value = feed_dict[layer]
          value = _ensure_value_shape(value, layer)
        else:
          value = np.zeros(
              [0] + list(layer.shape[1:]), dtype=layer.dtype.as_numpy_dtype)
        enq[tg.get_pre_q_input(layer).out_tensor] = value
      sess.run(tg.input_queue.out_tensor, feed_dict=enq)
      n_enqueued[0] += 1
    final_sample[0] = n_enqueued[0]


class TFWrapper(object):
  """This class exists as a workaround for Tensorflow objects not being picklable.
  The job of a TFWrapper is to create Tensorflow objects by passing defined arguments
  to a constructor.  There are cases where we really want to store Tensorflow objects
  of various sorts (optimizers, initializers, etc.), but we can't because they cannot
  be pickled.  So instead we store a TFWrapper that creates the object when needed.
  """

  def __init__(self, tf_class, **kwargs):
    """Create a TFWrapper for constructing a Tensorflow object.
    Parameters
    ----------
    tf_class: class
      the type of object to create
    kwargs:
      any other arguments will be passed on to the object's constructor
    """
    self.tf_class = tf_class
    self.kwargs = kwargs

  def __call__(self):
    return self.tf_class(**self.kwargs)


class Submodel(object):
  """An alternate objective for training one piece of a TensorGraph."""

  def __init__(self, graph, layers, loss, optimizer):
    """Create a submodel.
    In normal use, you should call create_submodel() on the TensorGraph instead
    of using this constructor directly."""
    self.graph = graph
    self.layers = layers
    self.loss = loss
    self.optimizer = optimizer
    self._train_op = None

  def get_train_op(self):
    """Get the Tensorflow operator to use for training."""
    if self._train_op is None:
      if self.layers is None:
        variables = None
      else:
        variables = []
        for layer in self.layers:
          variables += self.graph.get_layer_variables(layer)
      if self.loss is None:
        loss = self.graph.loss
      else:
        loss = self.loss
      tf_opt = self.create_optimizer()
      global_step = self.graph._get_tf('GlobalStep')
      self._train_op = tf_opt.minimize(loss.out_tensor, global_step, variables)
    return self._train_op

  def create_optimizer(self):
    """Create the Tensorflow optimizer to use for training."""
    if self.optimizer is None:
      optimizer = self.graph.optimizer
    else:
      optimizer = self.optimizer
    # Should we keep a separate global step count for each submodel?
    global_step = self.graph._get_tf('GlobalStep')
    return optimizer._create_optimizer(global_step)

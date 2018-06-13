# -*- coding: UTF-8 -*-
from __future__ import division
import random
import string
from collections import Sequence
from copy import deepcopy

import tensorflow as tf
import numpy as np

from deepchem.models.tensorgraph import model_ops, initializations, regularizers, activations
from deepchem.models.tensorgraph.model_ops import create_variable
import tensorflow.contrib.eager as tfe
import math

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class Layer(object):
  layer_number_dict = {}

  def __init__(self, in_layers=None, **kwargs):
    if "name" in kwargs:
      self.name = kwargs['name']
    else:
      self.name = None
    if in_layers is None:
      in_layers = list()
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    self.in_layers = in_layers
    self.op_type = "gpu"
    self.variable_values = None
    self.out_tensor = None
    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    self.tensorboard = False
    self.tb_input = None
    if tfe.in_eager_mode():
      self.variables = []
      self._built = False
      self._non_pickle_fields = ['variables', '_built']
    else:
      self.variable_scope = ''
      self._non_pickle_fields = [
          'out_tensor', 'rnn_initial_states', 'rnn_final_states',
          'rnn_zero_states'
      ]

  def _get_layer_number(self):
    class_name = self.__class__.__name__
    if class_name not in Layer.layer_number_dict:
      Layer.layer_number_dict[class_name] = 0
    Layer.layer_number_dict[class_name] += 1
    return "%s" % Layer.layer_number_dict[class_name]

  def none_tensors(self):
    saved_tensors = []
    for field in self._non_pickle_fields:
      value = self.__getattribute__(field)
      saved_tensors.append(value)
      if isinstance(value, list):
        self.__setattr__(field, [])
      else:
        self.__setattr__(field, None)
    return saved_tensors

  def set_tensors(self, tensors):
    for field, t in zip(self._non_pickle_fields, tensors):
      self.__setattr__(field, t)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    raise NotImplementedError("Subclasses must implement for themselves")

  def clone(self, in_layers):
    """Create a copy of this layer with different inputs."""
    saved_inputs = self.in_layers
    self.in_layers = []
    saved_tensors = self.none_tensors()
    copy = deepcopy(self)
    self.in_layers = saved_inputs
    self.set_tensors(saved_tensors)
    copy.in_layers = in_layers
    return copy

  def shared(self, in_layers):
    """
    Create a copy of this layer that shares variables with it.

    This is similar to clone(), but where clone() creates two independent layers,
    this causes the layers to share variables with each other.

    Parameters
    ----------
    in_layers: list tensor
    List in tensors for the shared layer

    Returns
    -------
    Layer
    """
    if tfe.in_eager_mode():
      raise ValueError('shared() is not supported in eager mode')
    if self.variable_scope == '':
      return self.clone(in_layers)
    raise ValueError('%s does not implement shared()' % self.__class__.__name__)

  def __call__(self, *inputs, **kwargs):
    """Execute the layer in eager mode to compute its output as a function of inputs.

    If the layer defines any variables, they are created the first time it is invoked.

    Arbitrary keyword arguments may be specified after the list of inputs.  Most
    layers do not expect or use any additional arguments, but there are a few
    significant cases.

    - Recurrent layers usually accept an argument `initial_state` which can be
      used to specify the initial state for the recurrent cell.  When this
      argument is omitted, they use a default initial state, usually all zeros.
    - A few layers behave differently during training than during inference,
      such as Dropout and CombineMeanStd.  You can specify a boolean value with
      the `training` argument to tell it which mode it is being called in.

    Parameters
    ----------
    inputs: tensors
      the inputs to pass to the layer.  The values may be tensors, numpy arrays,
      or anything else that can be converted to tensors of the correct shape.
    """
    return self.create_tensor(in_layers=inputs, set_tensors=False, **kwargs)

  @property
  def shape(self):
    """Get the shape of this Layer's output."""
    if '_shape' not in dir(self):
      raise NotImplementedError(
          "%s: shape is not known" % self.__class__.__name__)
    return self._shape

  def _get_input_tensors(self, in_layers, reshape=False):
    """Get the input tensors to his layer.

    Parameters
    ----------
    in_layers: list of Layers or tensors
      the inputs passed to create_tensor().  If None, this layer's inputs will
      be used instead.
    reshape: bool
      if True, try to reshape the inputs to all have the same shape
    """
    if in_layers is None:
      if tfe.in_eager_mode():
        raise ValueError('in_layers must be specified in eager mode')
      in_layers = self.in_layers
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    tensors = []
    for input in in_layers:
      tensors.append(tf.convert_to_tensor(input))
    if reshape and len(tensors) > 1 and all(
        '_shape' in dir(l) for l in in_layers):
      shapes = [t.get_shape() for t in tensors]
      if any(s != shapes[0] for s in shapes[1:]):
        # Reshape everything to match the input with the most dimensions.

        shape = shapes[0]
        for s in shapes:
          if len(s) > len(shape):
            shape = s
        shape = [-1 if x is None else x for x in shape.as_list()]
        for i in range(len(tensors)):
          tensors[i] = tf.reshape(tensors[i], shape)
    return tensors

  def _record_variable_scope(self, local_scope):
    """Record the scope name used for creating variables.

    This should be called from create_tensor().  It allows the list of variables
    belonging to this layer to be retrieved later."""
    parent_scope = tf.get_variable_scope().name
    if len(parent_scope) > 0:
      self.variable_scope = '%s/%s' % (parent_scope, local_scope)
    else:
      self.variable_scope = local_scope

  def set_variable_initial_values(self, values):
    """Set the initial values of all variables.

    This takes a list, which contains the initial values to use for all of
    this layer's values (in the same order retured by
    TensorGraph.get_layer_variables()).  When this layer is used in a
    TensorGraph, it will automatically initialize each variable to the value
    specified in the list.  Note that some layers also have separate mechanisms
    for specifying variable initializers; this method overrides them. The
    purpose of this method is to let a Layer object represent a pre-trained
    layer, complete with trained values for its variables."""
    self.variable_values = values

  def set_summary(self, summary_op, include_variables=True, summary_description=None, collections=None):
    """Annotates a tensor with a tf.summary operation

    This causes self.out_tensor to be logged to Tensorboard.

    Parameters
    ----------
    summary_op: str
      summary operation to annotate node
    include_variables: bool
      Optional bool to include layer variables to summary
    summary_description: object, optional
      Optional summary_pb2.SummaryDescription()
    collections: list of graph collections keys, optional
      New summary op is added to these collections. Defaults to [GraphKeys.SUMMARIES]
    """
    supported_ops = {'tensor_summary', 'scalar', 'histogram'}
    if summary_op not in supported_ops:
      raise ValueError(
          "Invalid summary_op arg. Only 'tensor_summary', 'scalar', 'histogram' supported"
      )
    self.summary_op = summary_op
    self.include_variables = include_variables
    self.summary_description = summary_description
    self.collections = collections
    self.tensorboard = True

  def add_summary_to_tg(self, layer_vars):
    """
    Create the summary operation for this layer, if set_summary() has been called on it.

    Can only be called after self.create_layer to guarantee that name is not None.

    Parameters
    ----------
    layer_vars: list of variables
      the list of variables to log to Tensorboard.
    """
    if self.tensorboard == False:
      return
    
    if self.summary_op == "tensor_summary":
      tf.summary.tensor_summary(self.name, self.out_tensor, self.summary_description,
                                self.collections)
      if self.include_variables:
        for var in layer_vars:
          tf.summary.tensor_summary(var.name, var, self.summary_description,
                                   self.collections)
          
    elif self.summary_op == 'scalar':
      tf.summary.scalar(self.name, self.out_tensor, self.collections)
      if self.include_variables:
        for var in layer_vars:
          tf.summary.scalar(var.name, var, self.collections)
          
    elif self.summary_op == 'histogram':
      tf.summary.histogram(self.name, self.out_tensor, self.collections)
      if self.include_variables:
        for var in layer_vars:
          tf.summary.histogram(var.name, var, self.collections)

  def copy(self, replacements={}, variables_graph=None, shared=False):
    """Duplicate this Layer and all its inputs.

    This is similar to clone(), but instead of only cloning one layer, it also
    recursively calls copy() on all of this layer's inputs to clone the entire
    hierarchy of layers.  In the process, you can optionally tell it to replace
    particular layers with specific existing ones.  For example, you can clone a
    stack of layers, while connecting the topmost ones to different inputs.

    For example, consider a stack of dense layers that depend on an input:

    >>> input = Feature(shape=(None, 100))
    >>> dense1 = Dense(100, in_layers=input)
    >>> dense2 = Dense(100, in_layers=dense1)
    >>> dense3 = Dense(100, in_layers=dense2)

    The following will clone all three dense layers, but not the input layer.
    Instead, the input to the first dense layer will be a different layer
    specified in the replacements map.

    >>> new_input = Feature(shape=(None, 100))
    >>> replacements = {input: new_input}
    >>> dense3_copy = dense3.copy(replacements)

    Parameters
    ----------
    replacements: map
      specifies existing layers, and the layers to replace them with (instead of
      cloning them).  This argument serves two purposes.  First, you can pass in
      a list of replacements to control which layers get cloned.  In addition,
      as each layer is cloned, it is added to this map.  On exit, it therefore
      contains a complete record of all layers that were copied, and a reference
      to the copy of each one.
    variables_graph: TensorGraph
      an optional TensorGraph from which to take variables.  If this is specified,
      the current value of each variable in each layer is recorded, and the copy
      has that value specified as its initial value.  This allows a piece of a
      pre-trained model to be copied to another model.
    shared: bool
      if True, create new layers by calling shared() on the input layers.
      This means the newly created layers will share variables with the original
      ones.
    """
    if self in replacements:
      return replacements[self]
    copied_inputs = [
        layer.copy(replacements, variables_graph, shared)
        for layer in self.in_layers
    ]
    if shared:
      copy = self.shared(copied_inputs)
    else:
      copy = self.clone(copied_inputs)
    if variables_graph is not None:
      if shared:
        raise ValueError('Cannot specify variables_graph when shared==True')
      variables = variables_graph.get_layer_variables(self)
      if len(variables) > 0:
        with variables_graph._get_tf("Graph").as_default():
          if tfe.in_eager_mode():
            values = [v.numpy() for v in variables]
          else:
            values = variables_graph.session.run(variables)
          copy.set_variable_initial_values(values)
    return copy

  def _as_graph_element(self):
    if '_as_graph_element' in dir(self.out_tensor):
      return self.out_tensor._as_graph_element()
    else:
      return self.out_tensor

  def __add__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Add([self, other])

  def __radd__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Add([other, self])

  def __sub__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Add([self, other], weights=[1.0, -1.0])

  def __rsub__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Add([other, self], weights=[1.0, -1.0])

  def __mul__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Multiply([self, other])

  def __rmul__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Multiply([other, self])

  def __neg__(self):
    return Multiply([self, Constant(-1.0)])

  def __div__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Divide([self, other])

  def __truediv__(self, other):
    if not isinstance(other, Layer):
      other = Constant(other)
    return Divide([self, other])


def _convert_layer_to_tensor(value, dtype=None, name=None, as_ref=False):
  return tf.convert_to_tensor(value.out_tensor, dtype=dtype, name=name)


tf.register_tensor_conversion_function(Layer, _convert_layer_to_tensor)


class TensorWrapper(Layer):
  """Used to wrap a tensorflow tensor."""

  def __init__(self, out_tensor, **kwargs):
    super(TensorWrapper, self).__init__(**kwargs)
    self.out_tensor = out_tensor
    self._shape = tuple(out_tensor.get_shape().as_list())

  def create_tensor(self, in_layers=None, **kwargs):
    """Take no actions."""
    return self.out_tensor


def convert_to_layers(in_layers):
  """Wrap all inputs into tensors if necessary."""
  layers = []
  for in_layer in in_layers:
    if isinstance(in_layer, Layer):
      layers.append(in_layer)
    elif isinstance(in_layer, tf.Tensor):
      layers.append(TensorWrapper(in_layer))
    else:
      raise ValueError("convert_to_layers must be invoked on layers or tensors")
  return layers


class SharedVariableScope(Layer):
  """A Layer that can share variables with another layer via name scope.

  This abstract class can be used as a parent for any layer that implements
  shared() by means of the variable name scope.  It exists to avoid duplicated
  code.
  """

  def __init__(self, **kwargs):
    super(SharedVariableScope, self).__init__(**kwargs)
    self._reuse = False
    self._shared_with = None

  def shared(self, in_layers):
    if tfe.in_eager_mode():
      raise ValueError('shared() is not supported in eager mode')
    copy = self.clone(in_layers)
    self._reuse = True
    copy._reuse = True
    copy._shared_with = self
    return copy

  def _get_scope_name(self):
    if tfe.in_eager_mode():
      return None
    if self._shared_with is None:
      return self.name
    else:
      return self._shared_with._get_scope_name()


class Conv1D(Layer):
  """A 1D convolution on the input.

  This layer expects its input to be a three dimensional tensor of shape (batch size, width, # channels).
  If there is only one channel, the third dimension may optionally be omitted.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               in_layers=None,
               **kwargs):
    """1D convolution layer (e.g. temporal convolution).

      This layer creates a convolution kernel that is convolved
      with the layer input over a single spatial (or temporal) dimension
      to produce a tensor of outputs.
      If `use_bias` is True, a bias vector is created and added to the outputs.
      Finally, if `activation` is not `None`,
      it is applied to the outputs as well.

      When using this layer as the first layer in a model,
      provide an `input_shape` argument
      (tuple of integers or `None`, e.g.
      `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
      or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

      TODO(LESWING): Calculate output shape at construction time
      Arguments:
          filters: Integer, the dimensionality of the output space
              (i.e. the number output of filters in the convolution).
          kernel_size: An integer or tuple/list of a single integer,
              specifying the length of the 1D convolution window.
          strides: An integer or tuple/list of a single integer,
              specifying the stride length of the convolution.
              Specifying any stride value != 1 is incompatible with specifying
              any `dilation_rate` value != 1.
          padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
              `"causal"` results in causal (dilated) convolutions, e.g. output[t]
              does not depend on input[t+1:]. Useful when modeling temporal data
              where the model should not violate the temporal order.
              See [WaveNet: A Generative Model for Raw Audio, section
                2.1](https://arxiv.org/abs/1609.03499).
          dilation_rate: an integer or tuple/list of a single integer, specifying
              the dilation rate to use for dilated convolution.
              Currently, specifying any `dilation_rate` value != 1 is
              incompatible with specifying any `strides` value != 1.
          activation: Activation function to use.
              If you don't specify anything, no activation is applied
              (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
              the `kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          activity_regularizer: Regularizer function applied to
              the output of the layer (its "activation")..
          kernel_constraint: Constraint function applied to the kernel matrix.
          bias_constraint: Constraint function applied to the bias vector.

      Input shape:
          3D tensor with shape: `(batch_size, steps, input_dim)`

      Output shape:
          3D tensor with shape: `(batch_size, new_steps, filters)`
          `steps` value might have changed due to padding or strides.
    """
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.dilation_rate = dilation_rate
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    super(Conv1D, self).__init__(in_layers, **kwargs)

  def _build_layer(self):
    return tf.keras.layers.Conv1D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        activation=self.activation,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        activity_regularizer=self.activity_regularizer,
        kernel_constraint=self.kernel_constraint,
        bias_constraint=self.bias_constraint)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Conv1D layer must have exactly one parent")
    parent = inputs[0]
    if len(parent.get_shape()) == 2:
      parent = tf.expand_dims(parent, 2)
    elif len(parent.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    if tfe.in_eager_mode():
      if not self._built:
        self._layer = self._build_layer()
        self._non_pickle_fields.append('_layer')
      layer = self._layer
    else:
      layer = self._build_layer()
    out_tensor = layer(parent)
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Dense(SharedVariableScope):

  def __init__(
      self,
      out_channels,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer,
      time_series=False,
      **kwargs):
    """Create a dense layer.

    The weight and bias initializers are specified by callable objects that construct
    and return a Tensorflow initializer when invoked with no arguments.  This will typically
    be either the initializer class itself (if the constructor does not require arguments),
    or a TFWrapper (if it does).

    Parameters
    ----------
    out_channels: int
      the number of output values
    activation_fn: object
      the Tensorflow activation function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    time_series: bool
      if True, the dense layer is applied to each element of a batch in sequence
    """
    super(Dense, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer
    self.time_series = time_series
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(parent_shape[:-1]) + (out_channels,)
    except:
      pass

  def _build_layer(self, reuse):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    return tf.layers.Dense(
        self.out_channels,
        activation=self.activation_fn,
        use_bias=biases_initializer is not None,
        kernel_initializer=self.weights_initializer(),
        bias_initializer=biases_initializer,
        _scope=self._get_scope_name(),
        _reuse=reuse)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Dense layer can only have one input")
    parent = inputs[0]
    for reuse in (self._reuse, False):
      if tfe.in_eager_mode():
        if not self._built:
          self._layer = self._build_layer(False)
          self._non_pickle_fields.append('_layer')
        layer = self._layer
      else:
        layer = self._build_layer(reuse)
      try:
        if self.time_series:
          out_tensor = tf.map_fn(layer, parent)
        else:
          out_tensor = layer(parent)
        break
      except ValueError:
        if reuse:
          # This probably means the variable hasn't been created yet, so try again
          # with reuse set to false.
          continue
        raise
    if set_tensors:
      self._record_variable_scope(self._get_scope_name())
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Highway(Layer):
  """ Create a highway layer. y = H(x) * T(x) + x * (1 - T(x))
  H(x) = activation_fn(matmul(W_H, x) + b_H) is the non-linear transformed output
  T(x) = sigmoid(matmul(W_T, x) + b_T) is the transform gate

  reference: https://arxiv.org/pdf/1505.00387.pdf

  This layer expects its input to be a two dimensional tensor of shape (batch size, # input features).
  Outputs will be in the same shape.
  """

  def __init__(
      self,
      activation_fn=tf.nn.relu,
      biases_initializer=tf.zeros_initializer,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer,
      **kwargs):
    """

    Parameters
    ----------
    activation_fn: object
      the Tensorflow activation function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    super(Highway, self).__init__(**kwargs)
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer
    try:
      self._shape = self.in_layers[0].shape
    except:
      pass

  def _build_layers(self, out_channels):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    dense_H = tf.layers.Dense(
        out_channels,
        activation=self.activation_fn,
        bias_initializer=biases_initializer,
        kernel_initializer=self.weights_initializer())
    dense_T = tf.layers.Dense(
        out_channels,
        activation=tf.nn.sigmoid,
        bias_initializer=tf.constant_initializer(-1),
        kernel_initializer=self.weights_initializer())
    return (dense_H, dense_T)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent = inputs[0]
    out_channels = parent.get_shape().as_list()[1]
    if tfe.in_eager_mode():
      if not self._built:
        self._layers = self._build_layers(out_channels)
        self._non_pickle_fields.append('_layers')
      layers = self._layers
    else:
      layers = self._build_layers(out_channels)
    dense_H = layers[0](parent)
    dense_T = layers[1](parent)
    out_tensor = tf.multiply(dense_H, dense_T) + tf.multiply(
        parent, 1 - dense_T)
    if set_tensors:
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layers[0].variables + self._layers[1].variables
    return out_tensor


class Flatten(Layer):
  """Flatten every dimension except the first"""

  def __init__(self, in_layers=None, **kwargs):
    super(Flatten, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      s = list(parent_shape[:2])
      for x in parent_shape[2:]:
        s[1] *= x
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Only One Parent to Flatten")
    parent = inputs[0]
    parent_shape = parent.get_shape()
    vector_size = 1
    for i in range(1, len(parent_shape)):
      vector_size *= parent_shape[i].value
    parent_tensor = parent
    out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Reshape(Layer):

  def __init__(self, shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self._new_shape = tuple(-1 if x is None else x for x in shape)
    try:
      parent_shape = self.in_layers[0].shape
      s = tuple(None if x == -1 else x for x in shape)
      if None in parent_shape or None not in s:
        self._shape = s
      else:
        # Calculate what the new shape will be.
        t = 1
        for x in parent_shape:
          t *= x
        for x in s:
          if x is not None:
            t //= x
        self._shape = tuple(t if x is None else x for x in s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.reshape(parent_tensor, self._new_shape)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Cast(Layer):
  """
  Wrapper around tf.cast.  Changes the dtype of a single layer
  """

  def __init__(self, in_layers=None, dtype=None, **kwargs):
    """
    Parameters
    ----------
    dtype: tf.DType
      the dtype to cast the in_layer to
      e.x. tf.int32
    """
    if dtype is None:
      raise ValueError("Must cast to a dtype")
    self.dtype = dtype
    super(Cast, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = parent_shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.cast(parent_tensor, self.dtype)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Squeeze(Layer):

  def __init__(self, in_layers=None, squeeze_dims=None, **kwargs):
    self.squeeze_dims = squeeze_dims
    super(Squeeze, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      if squeeze_dims is None:
        self._shape = [i for i in parent_shape if i != 1]
      else:
        self._shape = [
            parent_shape[i]
            for i in range(len(parent_shape))
            if i not in squeeze_dims
        ]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.squeeze(parent_tensor, squeeze_dims=self.squeeze_dims)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Transpose(Layer):

  def __init__(self, perm, **kwargs):
    super(Transpose, self).__init__(**kwargs)
    self.perm = perm
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(parent_shape[i] for i in perm)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Only One Parent to Transpose over")
    out_tensor = tf.transpose(inputs[0], self.perm)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class CombineMeanStd(Layer):
  """Generate Gaussian nose."""

  def __init__(self,
               in_layers=None,
               training_only=False,
               noise_epsilon=0.01,
               **kwargs):
    """Create a CombineMeanStd layer.

    This layer should have two inputs with the same shape, and its output also has the
    same shape.  Each element of the output is a Gaussian distributed random number
    whose mean is the corresponding element of the first input, and whose standard
    deviation is the corresponding element of the second input.

    Parameters
    ----------
    in_layers: list
      the input layers.  The first one specifies the mean, and the second one specifies
      the standard deviation.
    training_only: bool
      if True, noise is only generated during training.  During prediction, the output
      is simply equal to the first input (that is, the mean of the distribution used
      during training).
    noise_epsilon: float
      The standard deviation of the random noise
    """
    super(CombineMeanStd, self).__init__(in_layers, **kwargs)
    self.training_only = training_only
    self.noise_epsilon = noise_epsilon
    try:
      self._shape = self.in_layers[0].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 2:
      raise ValueError("Must have two in_layers")
    mean_parent, std_parent = inputs[0], inputs[1]
    sample_noise = tf.random_normal(
        mean_parent.get_shape(), 0, self.noise_epsilon, dtype=tf.float32)
    if self.training_only and 'training' in kwargs:
      sample_noise *= kwargs['training']
    out_tensor = mean_parent + tf.exp(std_parent * 0.5) * sample_noise
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Repeat(Layer):

  def __init__(self, n_times, **kwargs):
    self.n_times = n_times
    super(Repeat, self).__init__(**kwargs)
    try:
      s = list(self.in_layers[0].shape)
      s.insert(1, n_times)
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = inputs[0]
    t = tf.expand_dims(parent_tensor, 1)
    pattern = tf.stack([1, self.n_times, 1])
    out_tensor = tf.tile(t, pattern)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Gather(Layer):
  """Gather elements or slices from the input."""

  def __init__(self, in_layers=None, indices=None, **kwargs):
    """Create a Gather layer.

    The indices can be viewed as a list of element identifiers, where each
    identifier is itself a length N array specifying the indices of the
    first N dimensions of the input tensor.  Those elements (or slices, depending
    on the shape of the input) are then stacked together.  For example,
    indices=[[0],[2],[4]] will produce [input[0], input[2], input[4]], while
    indices=[[1,2]] will produce [input[1,2]].

    The indices may be specified in two ways.  If they are constants, you can pass
    them to this constructor as a list or array.  Alternatively, the indices can
    be calculated by another layer.  In that case, pass None for indices, and
    instead provide them as the second input layer.

    Parameters
    ----------
    in_layers: list
      the input layers.  If indices is not None, this should be of length 1.  If indices
      is None, this should be of length 2, with the first entry calculating the tensor
      from which to take slices, and the second entry calculating the slice indices.
    indices: array
      the slice indices (if they are constants) or None (if the indices are provided by
      an input)
    """
    self.indices = indices
    super(Gather, self).__init__(in_layers, **kwargs)
    try:
      s = tuple(self.in_layers[0].shape)
      if indices is None:
        s2 = self.in_layers[1].shape
        self._shape = (s2[0],) + s[s2[-1]:]
      else:
        self._shape = (len(indices),) + s[np.array(indices).shape[-1]:]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if self.indices is None:
      if len(inputs) != 2:
        raise ValueError("Must have two parents")
      indices = inputs[1]
    if self.indices is not None:
      if len(inputs) != 1:
        raise ValueError("Must have one parent")
      indices = self.indices
    parent_tensor = inputs[0]
    out_tensor = tf.gather_nd(parent_tensor, indices)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class GRU(Layer):
  """A Gated Recurrent Unit.

  This layer expects its input to be of shape (batch_size, sequence_length, ...).
  It consists of a set of independent sequences (one for each element in the batch),
  that are each propagated independently through the GRU.

  When this layer is called in eager execution mode, it behaves slightly differently.
  It returns two tensors: the output of the recurrent layer, and the final state
  of the recurrent cell.  In addition, you can specify the initial_state option
  to tell it what to use as the initial state of the recurrent cell.  If that
  option is omitted, it defaults to all zeros for the initial state:

  outputs, final_state = gru_layer(input, initial_state=state)
  """

  def __init__(self, n_hidden, batch_size, **kwargs):
    """Create a Gated Recurrent Unit.

    Parameters
    ----------
    n_hidden: int
      the size of the GRU's hidden state, which also determines the size of its output
    batch_size: int
      the batch size that will be used with this layer
    """
    self.n_hidden = n_hidden
    self.batch_size = batch_size
    super(GRU, self).__init__(**kwargs)
    if tfe.in_eager_mode():
      self._cell = tf.contrib.rnn.GRUCell(n_hidden)
      self._zero_state = self._cell.zero_state(batch_size, tf.float32)
      self._non_pickle_fields += ['_cell', '_zero_state']
    else:
      self._non_pickle_fields.append('out_tensors')
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (batch_size, parent_shape[1], n_hidden)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = inputs[0]
    if tfe.in_eager_mode():
      gru_cell = self._cell
      zero_state = self._zero_state
    else:
      gru_cell = tf.contrib.rnn.GRUCell(self.n_hidden)
      zero_state = gru_cell.zero_state(self.batch_size, tf.float32)
    if set_tensors:
      initial_state = tf.placeholder(tf.float32, zero_state.get_shape())
    elif 'initial_state' in kwargs:
      initial_state = kwargs['initial_state']
    else:
      initial_state = zero_state
    out_tensor, final_state = tf.nn.dynamic_rnn(
        gru_cell, parent_tensor, initial_state=initial_state, scope=self.name)
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
      self.rnn_initial_states.append(initial_state)
      self.rnn_final_states.append(final_state)
      self.rnn_zero_states.append(np.zeros(zero_state.get_shape(), np.float32))
      self.out_tensors = [
          self.out_tensor, initial_state, final_state, zero_state
      ]
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._cell.variables
    if tfe.in_eager_mode():
      return (out_tensor, final_state)
    else:
      return out_tensor


class LSTM(Layer):
  """A Long Short Term Memory.

  This layer expects its input to be of shape (batch_size, sequence_length, ...).
  It consists of a set of independent sequences (one for each element in the batch),
  that are each propagated independently through the LSTM.

  When this layer is called in eager execution mode, it behaves slightly differently.
  It returns two values: the output of the recurrent layer, and the final state
  of the recurrent cell.  The state is a tuple of two tensors.  In addition, you
  can specify the initial_state option to tell it what to use as the initial
  state of the recurrent cell.  If that option is omitted, it defaults to all
  zeros for the initial state:

  outputs, final_state = lstm_layer(input, initial_state=state)
  """

  def __init__(self, n_hidden, batch_size, **kwargs):
    """Create a Long Short Term Memory.

    Parameters
    ----------
    n_hidden: int
      the size of the LSTM's hidden state, which also determines the size of its output
    batch_size: int
      the batch size that will be used with this layer
    """
    self.n_hidden = n_hidden
    self.batch_size = batch_size
    super(LSTM, self).__init__(**kwargs)
    if tfe.in_eager_mode():
      self._cell = tf.contrib.rnn.LSTMCell(n_hidden)
      self._zero_state = self._cell.zero_state(batch_size, tf.float32)
      self._non_pickle_fields += ['_cell', '_zero_state']
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (batch_size, parent_shape[1], n_hidden)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = inputs[0]
    if tfe.in_eager_mode():
      lstm_cell = self._cell
      zero_state = self._zero_state
    else:
      lstm_cell = tf.contrib.rnn.LSTMCell(self.n_hidden)
      zero_state = lstm_cell.zero_state(self.batch_size, tf.float32)
    if set_tensors:
      initial_state = tf.contrib.rnn.LSTMStateTuple(
          tf.placeholder(tf.float32, zero_state.c.get_shape()),
          tf.placeholder(tf.float32, zero_state.h.get_shape()))
    elif 'initial_state' in kwargs:
      initial_state = kwargs['initial_state']
    else:
      initial_state = zero_state
    out_tensor, final_state = tf.nn.dynamic_rnn(
        lstm_cell, parent_tensor, initial_state=initial_state, scope=self.name)
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
      self.rnn_initial_states.append(initial_state.c)
      self.rnn_initial_states.append(initial_state.h)
      self.rnn_final_states.append(final_state.c)
      self.rnn_final_states.append(final_state.h)
      self.rnn_zero_states.append(
          np.zeros(zero_state.c.get_shape(), np.float32))
      self.rnn_zero_states.append(
          np.zeros(zero_state.h.get_shape(), np.float32))
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._cell.variables
    if tfe.in_eager_mode():
      return (out_tensor, final_state)
    else:
      return out_tensor


class TimeSeriesDense(Layer):

  def __init__(self, out_channels, **kwargs):
    self.out_channels = out_channels
    super(TimeSeriesDense, self).__init__(**kwargs)
    if tfe.in_eager_mode():
      self._layer = self._build_layer()

  def _build_layer(self):
    return tf.layers.Dense(self.out_channels, activation=tf.nn.sigmoid)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = inputs[0]
    if tfe.in_eager_mode():
      if not self._built:
        self._layer = self._build_layer()
        self._non_pickle_fields.append('_layer')
      layer = self._layer
    else:
      layer = self._build_layer()
    out_tensor = tf.map_fn(layer, parent_tensor)
    if set_tensors:
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Input(Layer):

  def __init__(self, shape, dtype=tf.float32, **kwargs):
    if shape is not None:
      self._shape = tuple(shape)
    self.dtype = dtype
    super(Input, self).__init__(**kwargs)
    self.op_type = "cpu"

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    try:
      shape = self.shape
    except NotImplementedError:
      shape = None
    if len(in_layers) > 0:
      queue = in_layers[0]
      placeholder = queue.out_tensors[self.get_pre_q_name()]
      self.out_tensor = tf.placeholder_with_default(placeholder, shape)
      return self.out_tensor
    out_tensor = tf.placeholder(dtype=self.dtype, shape=shape)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

  def create_pre_q(self):
    try:
      q_shape = (None,) + self.shape[1:]
    except NotImplementedError:
      q_shape = None
    return Input(shape=q_shape, name="%s_pre_q" % self.name, dtype=self.dtype)

  def get_pre_q_name(self):
    return "%s_pre_q" % self.name


class Feature(Input):

  def __init__(self, **kwargs):
    super(Feature, self).__init__(**kwargs)


class Label(Input):

  def __init__(self, **kwargs):
    super(Label, self).__init__(**kwargs)


class Weights(Input):

  def __init__(self, **kwargs):
    super(Weights, self).__init__(**kwargs)


class L1Loss(Layer):
  """Compute the mean absolute difference between the elements of the inputs.

  This layer should have two or three inputs.  If there is a third input, the
  difference between the first two inputs is multiplied by the third one to
  produce a weighted error.
  """

  def __init__(self, in_layers=None, **kwargs):
    super(L1Loss, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    guess, label = inputs[0], inputs[1]
    l1 = tf.abs(guess - label)
    if len(inputs) > 2:
      l1 *= inputs[2]
    out_tensor = tf.reduce_mean(l1, axis=list(range(1, len(label.shape))))
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class L2Loss(Layer):
  """Compute the mean squared difference between the elements of the inputs.

  This layer should have two or three inputs.  If there is a third input, the
  squared difference between the first two inputs is multiplied by the third one to
  produce a weighted error.
  """

  def __init__(self, in_layers=None, **kwargs):
    super(L2Loss, self).__init__(in_layers, **kwargs)
    try:
      shape1 = self.in_layers[0].shape
      shape2 = self.in_layers[1].shape
      if shape1[0] is None:
        self._shape = (shape2[0],)
      else:
        self._shape = (shape1[0],)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    guess, label = inputs[0], inputs[1]
    l2 = tf.square(guess - label)
    if len(inputs) > 2:
      l2 *= inputs[2]
    out_tensor = tf.reduce_mean(l2, axis=list(range(1, len(label.shape))))
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class SoftMax(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(SoftMax, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Softmax must have a single input layer.")
    parent = inputs[0]
    out_tensor = tf.contrib.layers.softmax(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Sigmoid(Layer):
  """ Compute the sigmoid of input: f(x) = sigmoid(x)
  Only one input is allowed, output will have the same shape as input
  """

  def __init__(self, in_layers=None, **kwargs):
    super(Sigmoid, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Sigmoid must have a single input layer.")
    parent = inputs[0]
    out_tensor = tf.nn.sigmoid(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ReLU(Layer):
  """ Compute the relu activation of input: f(x) = relu(x)
  Only one input is allowed, output will have the same shape as input
  """

  def __init__(self, in_layers=None, **kwargs):
    super(ReLU, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("ReLU must have a single input layer.")
    parent = inputs[0]
    out_tensor = tf.nn.relu(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Concat(Layer):

  def __init__(self, in_layers=None, axis=1, **kwargs):
    self.axis = axis
    super(Concat, self).__init__(in_layers, **kwargs)
    try:
      s = list(self.in_layers[0].shape)
      for parent in self.in_layers[1:]:
        if s[axis] is None or parent.shape[axis] is None:
          s[axis] = None
        else:
          s[axis] += parent.shape[axis]
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) == 1:
      self.out_tensor = inputs[0]
      return self.out_tensor

    out_tensor = tf.concat(inputs, axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Stack(Layer):

  def __init__(self, in_layers=None, axis=1, **kwargs):
    self.axis = axis
    super(Stack, self).__init__(in_layers, **kwargs)
    try:
      s = list(self.in_layers[0].shape)
      s.insert(axis, len(self.in_layers))
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    out_tensor = tf.stack(inputs, axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Constant(Layer):
  """Output a constant value."""

  def __init__(self, value, dtype=tf.float32, **kwargs):
    """Construct a constant layer.

    Parameters
    ----------
    value: array
      the value the layer should output
    dtype: tf.DType
      the data type of the output value.
    """
    if not isinstance(value, np.ndarray):
      value = np.array(value)
    self.value = value
    self.dtype = dtype
    self._shape = tuple(value.shape)
    super(Constant, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    out_tensor = tf.constant(self.value, dtype=self.dtype)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Variable(Layer):
  """Output a trainable value."""

  def __init__(self, initial_value, dtype=tf.float32, **kwargs):
    """Construct a variable layer.

    Parameters
    ----------
    initial_value: array
      the initial value the layer should output
    dtype: tf.DType
      the data type of the output value.
    """
    if not isinstance(initial_value, np.ndarray):
      initial_value = np.array(initial_value)
    self.initial_value = initial_value
    self.dtype = dtype
    self._shape = tuple(initial_value.shape)
    super(Variable, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if tfe.in_eager_mode():
      if not self._built:
        self.variables = [tfe.Variable(self.initial_value, dtype=self.dtype)]
        self._built = True
      out_tensor = self.variables[0]
    else:
      out_tensor = tf.Variable(self.initial_value, dtype=self.dtype)
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    return out_tensor


class StopGradient(Layer):
  """Block the flow of gradients.

  This layer copies its input directly to its output, but reports that all
  gradients of its output are zero.  This means, for example, that optimizers
  will not try to optimize anything "upstream" of this layer.

  For example, suppose you have pre-trained a stack of layers to perform a
  calculation.  You want to use the result of that calculation as the input to
  another layer, but because they are already pre-trained, you do not want the
  optimizer to modify them.  You can wrap the output in a StopGradient layer,
  then use that as the input to the next layer."""

  def __init__(self, in_layers=None, **kwargs):
    super(StopGradient, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) > 1:
      raise ValueError("Only one layer supported.")
    out_tensor = tf.stop_gradient(inputs[0])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


def _max_dimension(x, y):
  if x is None:
    return y
  if y is None:
    return x
  return max(x, y)


class Add(Layer):
  """Compute the (optionally weighted) sum of the input layers."""

  def __init__(self, in_layers=None, weights=None, **kwargs):
    """Create an Add layer.

    Parameters
    ----------
    weights: array
      an array of length equal to the number of input layers, giving the weight
      to multiply each input by.  If None, all weights are set to 1.
    """
    super(Add, self).__init__(in_layers, **kwargs)
    self.weights = weights
    try:
      shape1 = list(self.in_layers[0].shape)
      shape2 = list(self.in_layers[1].shape)
      if len(shape1) < len(shape2):
        shape2, shape1 = shape1, shape2
      offset = len(shape1) - len(shape2)
      for i in range(len(shape2)):
        shape1[i + offset] = _max_dimension(shape1[i + offset], shape2[i])
      self._shape = tuple(shape1)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    weights = self.weights
    if weights is None:
      weights = [1] * len(inputs)
    out_tensor = inputs[0]
    if weights[0] != 1:
      out_tensor *= weights[0]
    for layer, weight in zip(inputs[1:], weights[1:]):
      if weight == 1:
        out_tensor += layer
      else:
        out_tensor += weight * layer
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Multiply(Layer):
  """Compute the product of the input layers."""

  def __init__(self, in_layers=None, **kwargs):
    super(Multiply, self).__init__(in_layers, **kwargs)
    try:
      shape1 = list(self.in_layers[0].shape)
      shape2 = list(self.in_layers[1].shape)
      if len(shape1) < len(shape2):
        shape2, shape1 = shape1, shape2
      offset = len(shape1) - len(shape2)
      for i in range(len(shape2)):
        shape1[i + offset] = _max_dimension(shape1[i + offset], shape2[i])
      self._shape = tuple(shape1)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    out_tensor = inputs[0]
    for layer in inputs[1:]:
      out_tensor *= layer
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Divide(Layer):
  """Compute the ratio of the input layers."""

  def __init__(self, in_layers=None, **kwargs):
    super(Divide, self).__init__(in_layers, **kwargs)
    try:
      shape1 = list(self.in_layers[0].shape)
      shape2 = list(self.in_layers[1].shape)
      if len(shape1) < len(shape2):
        shape2, shape1 = shape1, shape2
      offset = len(shape1) - len(shape2)
      for i in range(len(shape2)):
        shape1[i + offset] = _max_dimension(shape1[i + offset], shape2[i])
      self._shape = tuple(shape1)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    out_tensor = inputs[0] / inputs[1]
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Log(Layer):
  """Compute the natural log of the input."""

  def __init__(self, in_layers=None, **kwargs):
    super(Log, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[0].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError('Log must have a single parent')
    out_tensor = tf.log(inputs[0])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Exp(Layer):
  """Compute the exponential of the input."""

  def __init__(self, in_layers=None, **kwargs):
    super(Exp, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[0].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError('Exp must have a single parent')
    out_tensor = tf.exp(inputs[0])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class InteratomicL2Distances(Layer):
  """Compute (squared) L2 Distances between atoms given neighbors."""

  def __init__(self, N_atoms, M_nbrs, ndim, **kwargs):
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    super(InteratomicL2Distances, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 2:
      raise ValueError("InteratomicDistances requires coords,nbr_list")
    coords, nbr_list = (inputs[0], inputs[1])
    N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
    # Shape (N_atoms, M_nbrs, ndim)
    nbr_coords = tf.gather(coords, nbr_list)
    # Shape (N_atoms, M_nbrs, ndim)
    tiled_coords = tf.tile(
        tf.reshape(coords, (N_atoms, 1, ndim)), (1, M_nbrs, 1))
    # Shape (N_atoms, M_nbrs)
    dists = tf.reduce_sum((tiled_coords - nbr_coords)**2, axis=2)
    out_tensor = dists
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class SparseSoftMaxCrossEntropy(Layer):
  """Computes Sparse softmax cross entropy between logits and labels.
  labels: Tensor of shape [d_0,d_1,....,d_{r-1}](where r is rank of logits) and must be of dtype int32 or int64.
  logits: Unscaled log probabilities of shape [d_0,....d{r-1},num_classes] and of dtype float32 or float64.
  Note: the rank of the logits should be 1 greater than that of labels.
  The output will be a tensor of same shape as labels and of same type as logits with the loss.
  """

  def __init__(self, in_layers=None, **kwargs):
    super(SparseSoftMaxCrossEntropy, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[1].shape[:-1]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, False)
    if len(inputs) != 2:
      raise ValueError()
    labels, logits = inputs[0], inputs[1]
    out_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class SoftMaxCrossEntropy(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(SoftMaxCrossEntropy, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[1].shape[:-1]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    if len(inputs) != 2:
      raise ValueError()
    labels, logits = inputs[0], inputs[1]
    out_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class SigmoidCrossEntropy(Layer):
  """ Compute the sigmoid cross entropy of inputs: [labels, logits]
  `labels` hold the binary labels(with no axis of n_classes),
  `logits` hold the log probabilities for positive class(label=1),
  `labels` and `logits` should have same shape and type.
  Output will have the same shape as `logits`
  """

  def __init__(self, in_layers=None, **kwargs):
    super(SigmoidCrossEntropy, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[1].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    if len(inputs) != 2:
      raise ValueError()
    labels, logits = inputs[0], inputs[1]
    out_tensor = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ReduceMean(Layer):

  def __init__(self, in_layers=None, axis=None, **kwargs):
    if axis is not None and not isinstance(axis, Sequence):
      axis = [axis]
    self.axis = axis
    super(ReduceMean, self).__init__(in_layers, **kwargs)
    if axis is None:
      self._shape = tuple()
    else:
      try:
        parent_shape = self.in_layers[0].shape
        self._shape = [
            parent_shape[i] for i in range(len(parent_shape)) if i not in axis
        ]
      except:
        pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) > 1:
      out_tensor = tf.stack(inputs)
    else:
      out_tensor = inputs[0]

    out_tensor = tf.reduce_mean(out_tensor, axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ReduceMax(Layer):

  def __init__(self, in_layers=None, axis=None, **kwargs):
    if axis is not None and not isinstance(axis, Sequence):
      axis = [axis]
    self.axis = axis
    super(ReduceMax, self).__init__(in_layers, **kwargs)
    if axis is None:
      self._shape = tuple()
    else:
      try:
        parent_shape = self.in_layers[0].shape
        self._shape = [
            parent_shape[i] for i in range(len(parent_shape)) if i not in axis
        ]
      except:
        pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) > 1:
      out_tensor = tf.stack(inputs)
    else:
      out_tensor = inputs[0]

    out_tensor = tf.reduce_max(out_tensor, axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ToFloat(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(ToFloat, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) > 1:
      raise ValueError("Only one layer supported.")
    out_tensor = tf.to_float(inputs[0])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ReduceSum(Layer):

  def __init__(self, in_layers=None, axis=None, **kwargs):
    if axis is not None and not isinstance(axis, Sequence):
      axis = [axis]
    self.axis = axis
    super(ReduceSum, self).__init__(in_layers, **kwargs)
    if axis is None:
      self._shape = tuple()
    else:
      try:
        parent_shape = self.in_layers[0].shape
        self._shape = [
            parent_shape[i] for i in range(len(parent_shape)) if i not in axis
        ]
      except:
        pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) > 1:
      out_tensor = tf.stack(inputs)
    else:
      out_tensor = inputs[0]

    out_tensor = tf.reduce_sum(out_tensor, axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ReduceSquareDifference(Layer):

  def __init__(self, in_layers=None, axis=None, **kwargs):
    if axis is not None and not isinstance(axis, Sequence):
      axis = [axis]
    self.axis = axis
    super(ReduceSquareDifference, self).__init__(in_layers, **kwargs)
    if axis is None:
      self._shape = tuple()
    else:
      try:
        parent_shape = self.in_layers[0].shape
        self._shape = [
            parent_shape[i] for i in range(len(parent_shape)) if i not in axis
        ]
      except:
        pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    a = inputs[0]
    b = inputs[1]
    out_tensor = tf.reduce_mean(tf.squared_difference(a, b), axis=self.axis)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Conv2D(SharedVariableScope):
  """A 2D convolution on the input.

  This layer expects its input to be a four dimensional tensor of shape (batch size, height, width, # channels).
  If there is only one channel, the fourth dimension may optionally be omitted.
  """

  def __init__(self,
               num_outputs,
               kernel_size=5,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               biases_initializer=tf.zeros_initializer,
               weights_initializer=tf.contrib.layers.xavier_initializer,
               scope_name=None,
               **kwargs):
    """Create a Conv2D layer.

    Parameters
    ----------
    num_outputs: int
      the number of outputs produced by the convolutional kernel
    kernel_size: int or tuple
      the width of the convolutional kernel.  This can be either a two element tuple, giving
      the kernel size along each dimension, or an integer to use the same size along both
      dimensions.
    stride: int or tuple
      the stride between applications of the convolutional kernel.  This can be either a two
      element tuple, giving the stride along each dimension, or an integer to use the same
      stride along both dimensions.
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    activation_fn: object
      the Tensorflow activation function to apply to the output
    normalizer_fn: object
      the Tensorflow normalizer function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.activation_fn = activation_fn
    self.normalizer_fn = normalizer_fn
    self.weights_initializer = weights_initializer
    self.biases_initializer = biases_initializer
    super(Conv2D, self).__init__(**kwargs)
    if scope_name is None:
      scope_name = self.name
    self.scope_name = scope_name
    try:
      parent_shape = self.in_layers[0].shape
      strides = stride
      if isinstance(stride, int):
        strides = (stride, stride)
      self._shape = (parent_shape[0], parent_shape[1] // strides[0],
                     parent_shape[2] // strides[1], num_outputs)
    except:
      pass

  def _build_layer(self, reuse):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    return tf.layers.Conv2D(
        self.num_outputs,
        self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        activation=self.activation_fn,
        use_bias=biases_initializer is not None,
        bias_initializer=self.biases_initializer(),
        kernel_initializer=self.weights_initializer(),
        _scope=self._get_scope_name(),
        _reuse=reuse)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    if len(parent_tensor.get_shape()) == 3:
      parent_tensor = tf.expand_dims(parent_tensor, 3)
    for reuse in (self._reuse, False):
      try:
        if tfe.in_eager_mode():
          if not self._built:
            self._layer = self._build_layer(False)
            self._non_pickle_fields.append('_layer')
          layer = self._layer
        else:
          layer = self._build_layer(reuse)
        out_tensor = layer(parent_tensor)
        if self.normalizer_fn is not None:
          out_tensor = self.normalizer_fn(out_tensor)
        break
      except ValueError:
        if reuse:
          # This probably means the variable hasn't been created yet, so try again
          # with reuse set to false.
          continue
        raise
    if set_tensors:
      self._record_variable_scope(self.scope_name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Conv3D(SharedVariableScope):
  """A 3D convolution on the input.

  This layer expects its input to be a five dimensional tensor of shape
  (batch size, height, width, depth, # channels).
  If there is only one channel, the fifth dimension may optionally be omitted.
  """

  def __init__(self,
               num_outputs,
               kernel_size=5,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               biases_initializer=tf.zeros_initializer,
               weights_initializer=tf.contrib.layers.xavier_initializer,
               scope_name=None,
               **kwargs):
    """Create a Conv3D layer.

    Parameters
    ----------
    num_outputs: int
      the number of outputs produced by the convolutional kernel
    kernel_size: int or tuple
      the width of the convolutional kernel.  This can be either a three element tuple, giving
      the kernel size along each dimension, or an integer to use the same size along both
      dimensions.
    stride: int or tuple
      the stride between applications of the convolutional kernel.  This can be either a three
      element tuple, giving the stride along each dimension, or an integer to use the same
      stride along both dimensions.
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    activation_fn: object
      the Tensorflow activation function to apply to the output
    normalizer_fn: object
      the Tensorflow normalizer function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.activation_fn = activation_fn
    self.normalizer_fn = normalizer_fn
    self.weights_initializer = weights_initializer
    self.biases_initializer = biases_initializer
    super(Conv3D, self).__init__(**kwargs)
    if scope_name is None:
      scope_name = self.name
    self.scope_name = scope_name
    try:
      parent_shape = self.in_layers[0].shape
      strides = stride
      if isinstance(stride, int):
        strides = (stride, stride, stride)
      self._shape = (parent_shape[0], parent_shape[1] // strides[0],
                     parent_shape[2] // strides[1],
                     parent_shape[3] // strides[2], num_outputs)
    except:
      pass

  def _build_layer(self, reuse):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    return tf.layers.Conv3D(
        self.num_outputs,
        self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        activation=self.activation_fn,
        use_bias=biases_initializer is not None,
        bias_initializer=self.biases_initializer(),
        kernel_initializer=self.weights_initializer(),
        _scope=self._get_scope_name(),
        _reuse=reuse)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    if len(parent_tensor.get_shape()) == 4:
      parent_tensor = tf.expand_dims(parent_tensor, 4)
    for reuse in (self._reuse, False):
      try:
        if tfe.in_eager_mode():
          if not self._built:
            self._layer = self._build_layer(False)
            self._non_pickle_fields.append('_layer')
          layer = self._layer
        else:
          layer = self._build_layer(reuse)
        out_tensor = layer(parent_tensor)
        if self.normalizer_fn is not None:
          out_tensor = self.normalizer_fn(out_tensor)
        break
      except ValueError:
        if reuse:
          # This probably means the variable hasn't been created yet, so try again
          # with reuse set to false.
          continue
        raise
    out_tensor = out_tensor
    if set_tensors:
      self._record_variable_scope(self.scope_name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Conv2DTranspose(SharedVariableScope):
  """A transposed 2D convolution on the input.

  This layer is typically used for upsampling in a deconvolutional network.  It
  expects its input to be a four dimensional tensor of shape (batch size, height, width, # channels).
  If there is only one channel, the fourth dimension may optionally be omitted.
  """

  def __init__(self,
               num_outputs,
               kernel_size=5,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               biases_initializer=tf.zeros_initializer,
               weights_initializer=tf.contrib.layers.xavier_initializer,
               scope_name=None,
               **kwargs):
    """Create a Conv2DTranspose layer.

    Parameters
    ----------
    num_outputs: int
      the number of outputs produced by the convolutional kernel
    kernel_size: int or tuple
      the width of the convolutional kernel.  This can be either a two element tuple, giving
      the kernel size along each dimension, or an integer to use the same size along both
      dimensions.
    stride: int or tuple
      the stride between applications of the convolutional kernel.  This can be either a two
      element tuple, giving the stride along each dimension, or an integer to use the same
      stride along both dimensions.
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    activation_fn: object
      the Tensorflow activation function to apply to the output
    normalizer_fn: object
      the Tensorflow normalizer function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.activation_fn = activation_fn
    self.normalizer_fn = normalizer_fn
    self.weights_initializer = weights_initializer
    self.biases_initializer = biases_initializer
    super(Conv2DTranspose, self).__init__(**kwargs)
    if scope_name is None:
      scope_name = self.name
    self.scope_name = scope_name
    try:
      parent_shape = self.in_layers[0].shape
      strides = stride
      if isinstance(stride, int):
        strides = (stride, stride)
      self._shape = (parent_shape[0], parent_shape[1] * strides[0],
                     parent_shape[2] * strides[1], num_outputs)
    except:
      pass

  def _build_layer(self, reuse):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    return tf.layers.Conv2DTranspose(
        self.num_outputs,
        self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        activation=self.activation_fn,
        use_bias=biases_initializer is not None,
        bias_initializer=self.biases_initializer(),
        kernel_initializer=self.weights_initializer(),
        _scope=self._get_scope_name(),
        _reuse=reuse)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    if len(parent_tensor.get_shape()) == 3:
      parent_tensor = tf.expand_dims(parent_tensor, 3)
    for reuse in (self._reuse, False):
      try:
        if tfe.in_eager_mode():
          if not self._built:
            self._layer = self._build_layer(False)
            self._non_pickle_fields.append('_layer')
          layer = self._layer
        else:
          layer = self._build_layer(reuse)
        out_tensor = layer(parent_tensor)
        if self.normalizer_fn is not None:
          out_tensor = self.normalizer_fn(out_tensor)
        break
      except ValueError:
        if reuse:
          # This probably means the variable hasn't been created yet, so try again
          # with reuse set to false.
          continue
        raise
    if set_tensors:
      self._record_variable_scope(self.scope_name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class Conv3DTranspose(SharedVariableScope):
  """A transposed 3D convolution on the input.

  This layer is typically used for upsampling in a deconvolutional network.  It
  expects its input to be a five dimensional tensor of shape (batch size, height, width, depth, # channels).
  If there is only one channel, the fifth dimension may optionally be omitted.
  """

  def __init__(self,
               num_outputs,
               kernel_size=5,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               biases_initializer=tf.zeros_initializer,
               weights_initializer=tf.contrib.layers.xavier_initializer,
               scope_name=None,
               **kwargs):
    """Create a Conv3DTranspose layer.

    Parameters
    ----------
    num_outputs: int
      the number of outputs produced by the convolutional kernel
    kernel_size: int or tuple
      the width of the convolutional kernel.  This can be either a three element tuple, giving
      the kernel size along each dimension, or an integer to use the same size along both
      dimensions.
    stride: int or tuple
      the stride between applications of the convolutional kernel.  This can be either a three
      element tuple, giving the stride along each dimension, or an integer to use the same
      stride along both dimensions.
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    activation_fn: object
      the Tensorflow activation function to apply to the output
    normalizer_fn: object
      the Tensorflow normalizer function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.activation_fn = activation_fn
    self.normalizer_fn = normalizer_fn
    self.weights_initializer = weights_initializer
    self.biases_initializer = biases_initializer
    super(Conv3DTranspose, self).__init__(**kwargs)
    if scope_name is None:
      scope_name = self.name
    self.scope_name = scope_name
    try:
      parent_shape = self.in_layers[0].shape
      strides = stride
      if isinstance(stride, int):
        strides = (stride, stride, stride)
      self._shape = (parent_shape[0], parent_shape[1] * strides[0],
                     parent_shape[2] * strides[1], parent_shape[3] * strides[2],
                     num_outputs)
    except:
      pass

  def _build_layer(self, reuse):
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    return tf.layers.Conv3DTranspose(
        self.num_outputs,
        self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        activation=self.activation_fn,
        use_bias=biases_initializer is not None,
        bias_initializer=self.biases_initializer(),
        kernel_initializer=self.weights_initializer(),
        _scope=self._get_scope_name(),
        _reuse=reuse)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    if len(parent_tensor.get_shape()) == 4:
      parent_tensor = tf.expand_dims(parent_tensor, 4)
    for reuse in (self._reuse, False):
      try:
        if tfe.in_eager_mode():
          if not self._built:
            self._layer = self._build_layer(False)
            self._non_pickle_fields.append('_layer')
          layer = self._layer
        else:
          layer = self._build_layer(reuse)
        out_tensor = layer(parent_tensor)
        if self.normalizer_fn is not None:
          out_tensor = self.normalizer_fn(out_tensor)
        break
      except ValueError:
        if reuse:
          # This probably means the variable hasn't been created yet, so try again
          # with reuse set to false.
          continue
        raise
    if set_tensors:
      self._record_variable_scope(self.scope_name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class MaxPool1D(Layer):
  """A 1D max pooling on the input.

  This layer expects its input to be a three dimensional tensor of shape
  (batch size, width, # channels).
  """

  def __init__(self, window_shape=2, strides=1, padding="SAME", **kwargs):
    """Create a MaxPool1D layer.

    Parameters
    ----------
    window_shape: int, optional
      size of the window(assuming input with only one dimension)
    strides: int, optional
      stride of the sliding window
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    """
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.pooling_type = "MAX"
    super(MaxPool1D, self).__init__(**kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (parent_shape[0], parent_shape[1] // strides,
                     parent_shape[2])
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    in_tensor = inputs[0]
    out_tensor = tf.nn.pool(
        in_tensor,
        window_shape=[self.window_shape],
        pooling_type=self.pooling_type,
        padding=self.padding,
        strides=[self.strides])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class MaxPool2D(Layer):

  def __init__(self,
               ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1],
               padding="SAME",
               **kwargs):
    self.ksize = ksize
    self.strides = strides
    self.padding = padding
    super(MaxPool2D, self).__init__(**kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(
          None if p is None else p // s for p, s in zip(parent_shape, strides))
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    in_tensor = inputs[0]
    out_tensor = tf.nn.max_pool(
        in_tensor, ksize=self.ksize, strides=self.strides, padding=self.padding)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class MaxPool3D(Layer):
  """A 3D max pooling on the input.

  This layer expects its input to be a five dimensional tensor of shape
  (batch size, height, width, depth, # channels).
  """

  def __init__(self,
               ksize=[1, 2, 2, 2, 1],
               strides=[1, 2, 2, 2, 1],
               padding='SAME',
               **kwargs):
    """Create a MaxPool3D layer.

    Parameters
    ----------
    ksize: list
      size of the window for each dimension of the input tensor. Must have
      length of 5 and ksize[0] = ksize[4] = 1.
    strides: list
      stride of the sliding window for each dimension of input. Must have
      length of 5 and strides[0] = strides[4] = 1.
    padding: str
      the padding method to use, either 'SAME' or 'VALID'
    """

    self.ksize = ksize
    self.strides = strides
    self.padding = padding
    super(MaxPool3D, self).__init__(**kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(
          None if p is None else p // s for p, s in zip(parent_shape, strides))
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    in_tensor = inputs[0]
    out_tensor = tf.nn.max_pool3d(
        in_tensor, ksize=self.ksize, strides=self.strides, padding=self.padding)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class InputFifoQueue(Layer):
  """
  This Queue Is used to allow asynchronous batching of inputs
  During the fitting process
  """

  def __init__(self, shapes, names, capacity=5, **kwargs):
    self.shapes = shapes
    self.names = names
    self.capacity = capacity
    super(InputFifoQueue, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    # TODO(rbharath): Note sure if this layer can be called with __call__
    # meaningfully, so not going to support that functionality for now.
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self.dtypes = [x.out_tensor.dtype for x in in_layers]
    self.queue = tf.FIFOQueue(self.capacity, self.dtypes, names=self.names)
    feed_dict = {x.name: x.out_tensor for x in in_layers}
    self.out_tensor = self.queue.enqueue(feed_dict)
    self.close_op = self.queue.close()
    self.out_tensors = self.queue.dequeue()
    self._non_pickle_fields += ['queue', 'out_tensors', 'close_op']


class GraphConv(Layer):

  def __init__(self,
               out_channel,
               min_deg=0,
               max_deg=10,
               activation_fn=None,
               **kwargs):
    self.out_channel = out_channel
    self.min_degree = min_deg
    self.max_degree = max_deg
    self.num_deg = 2 * max_deg + (1 - min_deg)
    self.activation_fn = activation_fn
    super(GraphConv, self).__init__(**kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (parent_shape[0], out_channel)
    except:
      pass

  def _create_variables(self, in_channels):
    # Generate the nb_affine weights and biases
    W_list = [
        initializations.glorot_uniform([in_channels, self.out_channel], name='kernel')
        for k in range(self.num_deg)
    ]
    b_list = [
        model_ops.zeros(shape=[
            self.out_channel,
        ], name='bias') for k in range(self.num_deg)
    ]
    return (W_list, b_list)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    # in_layers = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    in_channels = inputs[0].get_shape()[-1].value
    if tfe.in_eager_mode():
      if not self._built:
        W_list, b_list = self._create_variables(in_channels)
        self.variables = W_list + b_list
        self._built = True
      else:
        W_list = self.variables[:self.num_deg]
        b_list = self.variables[self.num_deg:]
    else:
      W_list, b_list = self._create_variables(in_channels)

    # Extract atom_features
    atom_features = inputs[0]

    # Extract graph topology
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    # Perform the mol conv
    # atom_features = graph_conv(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_deg, self.min_deg, W_list,
    #                            b_list)

    W = iter(W_list)
    b = iter(b_list)

    # Sum all neighbors using adjacency matrix
    deg_summed = self.sum_neigh(atom_features, deg_adj_lists)

    # Get collection of modified atom features
    new_rel_atoms_collection = (self.max_degree + 1 - self.min_degree) * [None]

    for deg in range(1, self.max_degree + 1):
      # Obtain relevant atoms for this degree
      rel_atoms = deg_summed[deg - 1]

      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Apply hidden affine to relevant atoms and append
      rel_out = tf.matmul(rel_atoms, next(W)) + next(b)
      self_out = tf.matmul(self_atoms, next(W)) + next(b)
      out = rel_out + self_out

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Determine the min_deg=0 case
    if self.min_degree == 0:
      deg = 0

      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Only use the self layer
      out = tf.matmul(self_atoms, next(W)) + next(b)

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Combine all atoms back into the list
    atom_features = tf.concat(axis=0, values=new_rel_atoms_collection)

    if self.activation_fn is not None:
      atom_features = self.activation_fn(atom_features)

    out_tensor = atom_features
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    return out_tensor

  def sum_neigh(self, atoms, deg_adj_lists):
    """Store the summed atoms by degree"""
    deg_summed = self.max_degree * [None]

    # Tensorflow correctly processes empty lists when using concat
    for deg in range(1, self.max_degree + 1):
      gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
      # Sum along neighbors as well as self, and store
      summed_atoms = tf.reduce_sum(gathered_atoms, 1)
      deg_summed[deg - 1] = summed_atoms

    return deg_summed


class GraphPool(Layer):

  def __init__(self, min_degree=0, max_degree=10, **kwargs):
    self.min_degree = min_degree
    self.max_degree = max_degree
    super(GraphPool, self).__init__(**kwargs)
    try:
      self._shape = self.in_layers[0].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    atom_features = inputs[0]
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    # Perform the mol gather
    # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_degree, self.min_degree)

    deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

    # Tensorflow correctly processes empty lists when using concat

    for deg in range(1, self.max_degree + 1):
      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Expand dims
      self_atoms = tf.expand_dims(self_atoms, 1)

      # always deg-1 for deg_adj_lists
      gathered_atoms = tf.gather(atom_features, deg_adj_lists[deg - 1])
      gathered_atoms = tf.concat(axis=1, values=[self_atoms, gathered_atoms])

      maxed_atoms = tf.reduce_max(gathered_atoms, 1)
      deg_maxed[deg - self.min_degree] = maxed_atoms

    if self.min_degree == 0:
      begin = tf.stack([deg_slice[0, 0], 0])
      size = tf.stack([deg_slice[0, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)
      deg_maxed[0] = self_atoms

    out_tensor = tf.concat(axis=0, values=deg_maxed)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class GraphGather(Layer):

  def __init__(self, batch_size, activation_fn=None, **kwargs):
    self.batch_size = batch_size
    self.activation_fn = activation_fn
    super(GraphGather, self).__init__(**kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (batch_size, 2 * parent_shape[1])
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)

    # x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    atom_features = inputs[0]

    # Extract graph topology
    membership = inputs[2]

    # Perform the mol gather

    assert self.batch_size > 1, "graph_gather requires batches larger than 1"

    # Obtain the partitions for each of the molecules
    activated_par = tf.dynamic_partition(atom_features, membership,
                                         self.batch_size)

    # Sum over atoms for each molecule
    sparse_reps = [
        tf.reduce_mean(activated, 0, keepdims=True)
        for activated in activated_par
    ]
    max_reps = [
        tf.reduce_max(activated, 0, keepdims=True)
        for activated in activated_par
    ]

    # Get the final sparse representations
    sparse_reps = tf.concat(axis=0, values=sparse_reps)
    max_reps = tf.concat(axis=0, values=max_reps)
    mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])

    if self.activation_fn is not None:
      mol_features = self.activation_fn(mol_features)
    out_tensor = mol_features
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class LSTMStep(Layer):
  """Layer that performs a single step LSTM update.

  This layer performs a single step LSTM update. Note that it is *not*
  a full LSTM recurrent network. The LSTMStep layer is useful as a
  primitive for designing layers such as the AttnLSTMEmbedding or the
  IterRefLSTMEmbedding below.
  """

  def __init__(self,
               output_dim,
               input_dim,
               init_fn=initializations.glorot_uniform,
               inner_init_fn=initializations.orthogonal,
               activation_fn=activations.tanh,
               inner_activation_fn=activations.hard_sigmoid,
               **kwargs):
    """
    Parameters
    ----------
    output_dim: int
      Dimensionality of output vectors.
    input_dim: int
      Dimensionality of input vectors.
    init_fn: object
      TensorFlow initialization to use for W.
    inner_init_fn: object
      TensorFlow initialization to use for U.
    activation_fn: object
      TensorFlow activation to use for output.
    inner_activation_fn: object
      TensorFlow activation to use for inner steps.
    """

    super(LSTMStep, self).__init__(**kwargs)

    self.init = init_fn
    self.inner_init = inner_init_fn
    self.output_dim = output_dim

    # No other forget biases supported right now.
    self.activation = activation_fn
    self.inner_activation = inner_activation_fn
    self.input_dim = input_dim

  def get_initial_states(self, input_shape):
    return [model_ops.zeros(input_shape), model_ops.zeros(input_shape)]

  def _create_variables(self):
    """Constructs learnable weights for this layer."""
    init = self.init
    inner_init = self.inner_init
    W = init((self.input_dim, 4 * self.output_dim))
    U = inner_init((self.output_dim, 4 * self.output_dim))

    b = create_variable(
        np.hstack((np.zeros(self.output_dim), np.ones(self.output_dim),
                   np.zeros(self.output_dim), np.zeros(self.output_dim))),
        dtype=tf.float32)
    return [W, U, b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    in_layers: list
      List of three tensors (x, h_tm1, c_tm1). h_tm1 means "h, t-1".

    Returns
    -------
    list
      Returns h, [h + c]
    """
    activation = self.activation
    inner_activation = self.inner_activation

    if tfe.in_eager_mode():
      if not self._built:
        self.variables = self._create_variables()
        self._built = True
      W, U, b = self.variables
    else:
      W, U, b = self._create_variables()
    inputs = self._get_input_tensors(in_layers)
    x, h_tm1, c_tm1 = inputs

    # Taken from Keras code [citation needed]
    z = model_ops.dot(x, W) + model_ops.dot(h_tm1, U) + b

    z0 = z[:, :self.output_dim]
    z1 = z[:, self.output_dim:2 * self.output_dim]
    z2 = z[:, 2 * self.output_dim:3 * self.output_dim]
    z3 = z[:, 3 * self.output_dim:]

    i = inner_activation(z0)
    f = inner_activation(z1)
    c = f * c_tm1 + i * activation(z2)
    o = inner_activation(z3)

    h = o * activation(c)

    if set_tensors:
      self.out_tensor = h
    return h, [h, c]


def _cosine_dist(x, y):
  """Computes the inner product (cosine distance) between two tensors.

  Parameters
  ----------
  x: tf.Tensor
    Input Tensor
  y: tf.Tensor
    Input Tensor
  """
  denom = (
      model_ops.sqrt(model_ops.sum(tf.square(x)) * model_ops.sum(tf.square(y)))
      + model_ops.epsilon())
  return model_ops.dot(x, tf.transpose(y)) / denom


class AttnLSTMEmbedding(Layer):
  """Implements AttnLSTM as in matching networks paper.

  The AttnLSTM embedding adjusts two sets of vectors, the "test" and
  "support" sets. The "support" consists of a set of evidence vectors.
  Think of these as the small training set for low-data machine
  learning.  The "test" consists of the queries we wish to answer with
  the small amounts ofavailable data. The AttnLSTMEmbdding allows us to
  modify the embedding of the "test" set depending on the contents of
  the "support".  The AttnLSTMEmbedding is thus a type of learnable
  metric that allows a network to modify its internal notion of
  distance.

  References:
  Matching Networks for One Shot Learning
  https://arxiv.org/pdf/1606.04080v1.pdf

  Order Matters: Sequence to sequence for sets
  https://arxiv.org/abs/1511.06391
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of features per atom
    max_depth: int
      Number of "processing steps" used by sequence-to-sequence for sets model.
    """
    super(AttnLSTMEmbedding, self).__init__(**kwargs)

    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def _create_variables(self):
    n_feat = self.n_feat
    lstm = LSTMStep(n_feat, 2 * n_feat)
    q_init = model_ops.zeros([self.n_test, n_feat])
    r_init = model_ops.zeros([self.n_test, n_feat])
    states_init = lstm.get_initial_states([self.n_test, n_feat])
    return (lstm, q_init, r_init, states_init)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    in_layers: list
      List of two tensors (X, Xp). X should be of shape (n_test,
      n_feat) and Xp should be of shape (n_support, n_feat) where
      n_test is the size of the test set, n_support that of the support
      set, and n_feat is the number of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output
      shape will be [(n_test, n_feat), (n_support, n_feat)]
    """
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 2:
      raise ValueError("AttnLSTMEmbedding layer must have exactly two parents")
    # x is test set, xp is support set.
    x, xp = inputs

    if tfe.in_eager_mode():
      if not self._built:
        self._lstm, self.q_init, self.r_init, self.states_init = self._create_variables(
        )
        self._non_pickle_fields += ['_lstm', 'q_init', 'r_init', 'states_init']
      lstm = self._lstm
      q_init = self.q_init
      r_init = self.r_init
      states_init = self.states_init
    else:
      lstm, q_init, r_init, states_init = self._create_variables()

    ### Performs computations

    # Get initializations
    q = q_init
    states = states_init

    for d in range(self.max_depth):
      # Process using attention
      # Eqn (4), appendix A.1 of Matching Networks paper
      e = _cosine_dist(x + q, xp)
      a = tf.nn.softmax(e)
      r = model_ops.dot(a, xp)

      # Generate new attention states
      y = model_ops.concatenate([q, r], axis=1)
      q, states = lstm(y, *states)

    if set_tensors:
      self.out_tensor = xp
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = lstm.variables + [q_init, r_init] + states_init
    return [x + q, xp]


class IterRefLSTMEmbedding(Layer):
  """Implements the Iterative Refinement LSTM.

  Much like AttnLSTMEmbedding, the IterRefLSTMEmbedding is another type
  of learnable metric which adjusts "test" and "support." Recall that
  "support" is the small amount of data available in a low data machine
  learning problem, and that "test" is the query. The AttnLSTMEmbedding
  only modifies the "test" based on the contents of the support.
  However, the IterRefLSTM modifies both the "support" and "test" based
  on each other. This allows the learnable metric to be more malleable
  than that from AttnLSTMEmbeding.
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Unlike the AttnLSTM model which only modifies the test vectors
    additively, this model allows for an additive update to be
    performed to both test and support using information from each
    other.

    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of input atom features
    max_depth: int
      Number of LSTM Embedding layers.
    """
    super(IterRefLSTMEmbedding, self).__init__(**kwargs)

    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def _create_variables(self):
    n_feat = self.n_feat

    # Support set lstm
    support_lstm = LSTMStep(n_feat, 2 * n_feat)
    q_init = model_ops.zeros([self.n_support, n_feat])
    support_states_init = support_lstm.get_initial_states(
        [self.n_support, n_feat])

    # Test lstm
    test_lstm = LSTMStep(n_feat, 2 * n_feat)
    p_init = model_ops.zeros([self.n_test, n_feat])
    test_states_init = test_lstm.get_initial_states([self.n_test, n_feat])
    return (support_lstm, q_init, support_states_init, test_lstm, p_init,
            test_states_init)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    in_layers: list
      List of two tensors (X, Xp). X should be of shape (n_test, n_feat) and
      Xp should be of shape (n_support, n_feat) where n_test is the size of
      the test set, n_support that of the support set, and n_feat is the number
      of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output shape will
      be [(n_test, n_feat), (n_support, n_feat)]
    """
    if tfe.in_eager_mode():
      if not self._built:
        self._support_lstm, self.q_init, self.support_states_init, self._test_lstm, self.p_init, self.test_states_init = self._create_variables(
        )
        self._non_pickle_fields += [
            '_support_lstm', 'q_init', 'support_states_init', '_test_lstm',
            'p_init', 'test_states_init'
        ]
      support_lstm = self._support_lstm
      q_init = self.q_init
      support_states_init = self.support_states_init
      test_lstm = self._test_lstm
      p_init = self.p_init
      test_states_init = self.test_states_init
    else:
      support_lstm, q_init, support_states_init, test_lstm, p_init, test_states_init = self._create_variables(
      )

    # self.build()
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 2:
      raise ValueError(
          "IterRefLSTMEmbedding layer must have exactly two parents")
    x, xp = inputs

    # Get initializations
    p = p_init
    q = q_init
    # Rename support
    z = xp
    states = support_states_init
    x_states = test_states_init

    for d in range(self.max_depth):
      # Process support xp using attention
      e = _cosine_dist(z + q, xp)
      a = tf.nn.softmax(e)
      # Get linear combination of support set
      r = model_ops.dot(a, xp)

      # Process test x using attention
      x_e = _cosine_dist(x + p, z)
      x_a = tf.nn.softmax(x_e)
      s = model_ops.dot(x_a, z)

      # Generate new support attention states
      qr = model_ops.concatenate([q, r], axis=1)
      q, states = support_lstm(qr, *states)

      # Generate new test attention states
      ps = model_ops.concatenate([p, s], axis=1)
      p, x_states = test_lstm(ps, *x_states)

      # Redefine
      z = r

    if set_tensors:
      self.out_tensor = xp
    if tfe.in_eager_mode() and not self._built:
      self.variables = support_lstm.variables + test_lstm.variables + [
          q_init, p_init
      ] + support_states_init + test_states_init
      self._built = True

    return [x + p, xp + q]


class BatchNorm(Layer):

  def __init__(self,
               in_layers=None,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               **kwargs):
    super(BatchNorm, self).__init__(in_layers, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def _build_layer(self):
    return tf.layers.BatchNormalization(
        axis=self.axis, momentum=self.momentum, epsilon=self.epsilon)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    if tfe.in_eager_mode():
      if not self._built:
        self._layer = self._build_layer()
        self._non_pickle_fields.append('_layer')
      layer = self._layer
    else:
      layer = self._build_layer()
    out_tensor = layer(parent_tensor)
    if set_tensors:
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
      self.variables = self._layer.variables
    return out_tensor


class BatchNormalization(Layer):

  def __init__(self,
               epsilon=1e-5,
               axis=-1,
               momentum=0.99,
               beta_init='zero',
               gamma_init='one',
               **kwargs):
    warnings.warn(
        'BatchNormalization is deprecated and will be removed in a future release.  Use BatchNorm instead.',
        DeprecationWarning)
    self.beta_init = initializations.get(beta_init)
    self.gamma_init = initializations.get(gamma_init)
    self.epsilon = epsilon
    self.axis = axis
    self.momentum = momentum
    super(BatchNormalization, self).__init__(**kwargs)

  def add_weight(self, shape, initializer, name=None):
    initializer = initializations.get(initializer)
    weight = initializer(shape, name=name)
    return weight

  def build(self, input_shape):
    shape = (input_shape[self.axis],)
    self.gamma = self.add_weight(
        shape, initializer=self.gamma_init, name='{}_gamma'.format(self.name))
    self.beta = self.add_weight(
        shape, initializer=self.beta_init, name='{}_beta'.format(self.name))
    self._non_pickle_fields += ['gamma', 'beta']

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    x = inputs[0]
    input_shape = model_ops.int_shape(x)
    self.build(input_shape)
    m = model_ops.mean(x, axis=-1, keepdims=True)
    std = model_ops.sqrt(
        model_ops.var(x, axis=-1, keepdims=True) + self.epsilon)
    x_normed = (x - m) / (std + self.epsilon)
    x_normed = self.gamma * x_normed + self.beta
    out_tensor = x_normed
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class WeightedError(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(WeightedError, self).__init__(in_layers, **kwargs)
    self._shape = tuple()

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    entropy, weights = inputs[0], inputs[1]
    out_tensor = tf.reduce_sum(entropy * weights)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class VinaFreeEnergy(Layer):
  """Computes free-energy as defined by Autodock Vina.

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self,
               N_atoms,
               M_nbrs,
               ndim,
               nbr_cutoff,
               start,
               stop,
               stddev=.3,
               Nrot=1,
               **kwargs):
    self.stddev = stddev
    # Number of rotatable bonds
    # TODO(rbharath): Vina actually sets this per-molecule. See if makes
    # a difference.
    self.Nrot = Nrot
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop
    super(VinaFreeEnergy, self).__init__(**kwargs)

  def _build_layers(self):
    weighted_combo = WeightedLinearCombo()
    w = create_variable(tf.random_normal((1,), stddev=self.stddev))
    return (weighted_combo, w)

  def cutoff(self, d, x):
    out_tensor = tf.where(d < 8, x, tf.zeros_like(x))
    return out_tensor

  def nonlinearity(self, c, w):
    """Computes non-linearity used in Vina."""
    out_tensor = c / (1 + w * self.Nrot)
    return w, out_tensor

  def repulsion(self, d):
    """Computes Autodock Vina's repulsion interaction term."""
    out_tensor = tf.where(d < 0, d**2, tf.zeros_like(d))
    return out_tensor

  def hydrophobic(self, d):
    """Computes Autodock Vina's hydrophobic interaction term."""
    out_tensor = tf.where(d < 0.5, tf.ones_like(d),
                          tf.where(d < 1.5, 1.5 - d, tf.zeros_like(d)))
    return out_tensor

  def hydrogen_bond(self, d):
    """Computes Autodock Vina's hydrogen bond interaction term."""
    out_tensor = tf.where(d < -0.7, tf.ones_like(d),
                          tf.where(d < 0, (1.0 / 0.7) * (0 - d),
                                   tf.zeros_like(d)))
    return out_tensor

  def gaussian_first(self, d):
    """Computes Autodock Vina's first Gaussian interaction term."""
    out_tensor = tf.exp(-(d / 0.5)**2)
    return out_tensor

  def gaussian_second(self, d):
    """Computes Autodock Vina's second Gaussian interaction term."""
    out_tensor = tf.exp(-((d - 3) / 2)**2)
    return out_tensor

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (N, d)
      Coordinates/features.
    Z: tf.Tensor of shape (N)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B)
      The free energy of each complex in batch
    """
    inputs = self._get_input_tensors(in_layers)
    X = inputs[0]
    Z = inputs[1]

    if tfe.in_eager_mode():
      if not self._built:
        self._weighted_combo, self._w = self._build_layers()
        self._non_pickle_fields += ['_weighted_combo', '_w']
      weighted_combo = self._weighted_combo
      w = self._w
    else:
      weighted_combo, w = self._build_layers()

    # TODO(rbharath): This layer shouldn't be neighbor-listing. Make
    # neighbors lists an argument instead of a part of this layer.
    nbr_list = NeighborList(self.N_atoms, self.M_nbrs, self.ndim,
                            self.nbr_cutoff, self.start, self.stop)(X)

    # Shape (N, M)
    dists = InteratomicL2Distances(self.N_atoms, self.M_nbrs,
                                   self.ndim)(X, nbr_list)

    repulsion = self.repulsion(dists)
    hydrophobic = self.hydrophobic(dists)
    hbond = self.hydrogen_bond(dists)
    gauss_1 = self.gaussian_first(dists)
    gauss_2 = self.gaussian_second(dists)

    # Shape (N, M)
    interactions = weighted_combo(repulsion, hydrophobic, hbond, gauss_1,
                                  gauss_2)

    # Shape (N, M)
    thresholded = self.cutoff(dists, interactions)

    weight, free_energies = self.nonlinearity(thresholded, w)
    free_energy = ReduceSum()(free_energies)

    out_tensor = free_energy
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self.variables = weighted_combo.variables + [w]
      self._built = True
    return out_tensor


class WeightedLinearCombo(Layer):
  """Computes a weighted linear combination of input layers, with the weights defined by trainable variables."""

  def __init__(self, in_layers=None, std=.3, **kwargs):
    self.std = std
    super(WeightedLinearCombo, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def _create_variables(self, inputs):
    return [
        create_variable(tf.random_normal([1], stddev=self.std))
        for i in range(len(inputs))
    ]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers, True)
    if tfe.in_eager_mode():
      if not self._built:
        self.variables = self._create_variables(inputs)
        self._built = True
      weights = self.variables
    else:
      weights = self._create_variables(inputs)
    out_tensor = None
    for in_tensor, w in zip(inputs, weights):
      if out_tensor is None:
        out_tensor = w * in_tensor
      else:
        out_tensor += w * in_tensor
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    return out_tensor


class NeighborList(Layer):
  """Computes a neighbor-list in Tensorflow.

  Neighbor-lists (also called Verlet Lists) are a tool for grouping atoms which
  are close to each other spatially

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self, N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop, **kwargs):
    """
    Parameters
    ----------
    N_atoms: int
      Maximum number of atoms this layer will neighbor-list.
    M_nbrs: int
      Maximum number of spatial neighbors possible for atom.
    ndim: int
      Dimensionality of space atoms live in. (Typically 3D, but sometimes will
      want to use higher dimensional descriptors for atoms).
    nbr_cutoff: float
      Length in Angstroms (?) at which atom boxes are gridded.
    """
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    # Number of grid cells
    n_cells = int(((stop - start) / nbr_cutoff)**ndim)
    self.n_cells = n_cells
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop
    super(NeighborList, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """Creates tensors associated with neighbor-listing."""
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("NeighborList can only have one input")
    parent = inputs[0]
    if len(parent.get_shape()) != 2:
      # TODO(rbharath): Support batching
      raise ValueError("Parent tensor must be (num_atoms, ndum)")
    coords = parent
    out_tensor = self.compute_nbr_list(coords)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

  def compute_nbr_list(self, coords):
    """Get closest neighbors for atoms.

    Needs to handle padding for atoms with no neighbors.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)

    Returns
    -------
    nbr_list: tf.Tensor
      Shape (N_atoms, M_nbrs) of atom indices
    """
    # Shape (n_cells, ndim)
    cells = self.get_cells()

    # List of length N_atoms, each element of different length uniques_i
    nbrs = self.get_atoms_in_nbrs(coords, cells)
    padding = tf.fill((self.M_nbrs,), -1)
    padded_nbrs = [tf.concat([unique_nbrs, padding], 0) for unique_nbrs in nbrs]

    # List of length N_atoms, each element of different length uniques_i
    # List of length N_atoms, each a tensor of shape
    # (uniques_i, ndim)
    nbr_coords = [tf.gather(coords, atom_nbrs) for atom_nbrs in nbrs]

    # Add phantom atoms that exist far outside the box
    coord_padding = tf.to_float(
        tf.fill((self.M_nbrs, self.ndim), 2 * self.stop))
    padded_nbr_coords = [
        tf.concat([nbr_coord, coord_padding], 0) for nbr_coord in nbr_coords
    ]

    # List of length N_atoms, each of shape (1, ndim)
    atom_coords = tf.split(coords, self.N_atoms)
    # TODO(rbharath): How does distance need to be modified here to
    # account for periodic boundary conditions?
    # List of length N_atoms each of shape (M_nbrs)
    padded_dists = [
        tf.reduce_sum((atom_coord - padded_nbr_coord)**2, axis=1)
        for (atom_coord,
             padded_nbr_coord) in zip(atom_coords, padded_nbr_coords)
    ]

    padded_closest_nbrs = [
        tf.nn.top_k(-padded_dist, k=self.M_nbrs)[1]
        for padded_dist in padded_dists
    ]

    # N_atoms elts of size (M_nbrs,) each
    padded_neighbor_list = [
        tf.gather(padded_atom_nbrs, padded_closest_nbr)
        for (padded_atom_nbrs,
             padded_closest_nbr) in zip(padded_nbrs, padded_closest_nbrs)
    ]

    neighbor_list = tf.stack(padded_neighbor_list)

    return neighbor_list

  def get_atoms_in_nbrs(self, coords, cells):
    """Get the atoms in neighboring cells for each cells.

    Returns
    -------
    atoms_in_nbrs = (N_atoms, n_nbr_cells, M_nbrs)
    """
    # Shape (N_atoms, 1)
    cells_for_atoms = self.get_cells_for_atoms(coords, cells)

    # Find M_nbrs atoms closest to each cell
    # Shape (n_cells, M_nbrs)
    closest_atoms = self.get_closest_atoms(coords, cells)

    # Associate each cell with its neighbor cells. Assumes periodic boundary
    # conditions, so does wrapround. O(constant)
    # Shape (n_cells, n_nbr_cells)
    neighbor_cells = self.get_neighbor_cells(cells)

    # Shape (N_atoms, n_nbr_cells)
    neighbor_cells = tf.squeeze(tf.gather(neighbor_cells, cells_for_atoms))

    # Shape (N_atoms, n_nbr_cells, M_nbrs)
    atoms_in_nbrs = tf.gather(closest_atoms, neighbor_cells)

    # Shape (N_atoms, n_nbr_cells*M_nbrs)
    atoms_in_nbrs = tf.reshape(atoms_in_nbrs, [self.N_atoms, -1])

    # List of length N_atoms, each element length uniques_i
    nbrs_per_atom = tf.split(atoms_in_nbrs, self.N_atoms)
    uniques = [
        tf.unique(tf.squeeze(atom_nbrs))[0] for atom_nbrs in nbrs_per_atom
    ]

    # TODO(rbharath): FRAGILE! Uses fact that identity seems to be the first
    # element removed to remove self from list of neighbors. Need to verify
    # this holds more broadly or come up with robust alternative.
    uniques = [unique[1:] for unique in uniques]

    return uniques

  def get_closest_atoms(self, coords, cells):
    """For each cell, find M_nbrs closest atoms.

    Let N_atoms be the number of atoms.

    Parameters
    ----------
    coords: tf.Tensor
      (N_atoms, ndim) shape.
    cells: tf.Tensor
      (n_cells, ndim) shape.

    Returns
    -------
    closest_inds: tf.Tensor
      Of shape (n_cells, M_nbrs)
    """
    N_atoms, n_cells, ndim, M_nbrs = (self.N_atoms, self.n_cells, self.ndim,
                                      self.M_nbrs)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.reshape(
        tf.tile(cells, (1, N_atoms)), (N_atoms * n_cells, ndim))

    # Shape (N_atoms*n_cells, ndim) after tile
    tiled_coords = tf.tile(coords, (n_cells, 1))

    # Shape (N_atoms*n_cells)
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    # Shape (n_cells, N_atoms)
    coords_norm = tf.reshape(coords_vec, (n_cells, N_atoms))

    # Find k atoms closest to this cell. Notice negative sign since
    # tf.nn.top_k returns *largest* not smallest.
    # Tensor of shape (n_cells, M_nbrs)
    closest_inds = tf.nn.top_k(-coords_norm, k=M_nbrs)[1]

    return closest_inds

  def get_cells_for_atoms(self, coords, cells):
    """Compute the cells each atom belongs to.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    cells_for_atoms: tf.Tensor
      Shape (N_atoms, 1)
    """
    N_atoms, n_cells, ndim = self.N_atoms, self.n_cells, self.ndim
    n_cells = int(n_cells)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.tile(cells, (N_atoms, 1))

    # Shape (N_atoms*n_cells, 1) after tile
    tiled_coords = tf.reshape(
        tf.tile(coords, (1, n_cells)), (n_cells * N_atoms, ndim))
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (N_atoms, n_cells))

    closest_inds = tf.nn.top_k(-coords_norm, k=1)[1]
    return closest_inds

  def _get_num_nbrs(self):
    """Get number of neighbors in current dimensionality space."""
    ndim = self.ndim
    if ndim == 1:
      n_nbr_cells = 3
    elif ndim == 2:
      # 9 neighbors in 2-space
      n_nbr_cells = 9
    # TODO(rbharath): Shoddy handling of higher dimensions...
    elif ndim >= 3:
      # Number of cells for cube in 3-space is
      n_nbr_cells = 27  # (26 faces on Rubik's cube for example)
    return n_nbr_cells

  def get_neighbor_cells(self, cells):
    """Compute neighbors of cells in grid.

    # TODO(rbharath): Do we need to handle periodic boundary conditions
    properly here?
    # TODO(rbharath): This doesn't handle boundaries well. We hard-code
    # looking for n_nbr_cells neighbors, which isn't right for boundary cells in
    # the cube.

    Parameters
    ----------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    nbr_cells: tf.Tensor
      (n_cells, n_nbr_cells)
    """
    ndim, n_cells = self.ndim, self.n_cells
    n_nbr_cells = self._get_num_nbrs()
    # Tile cells to form arrays of size (n_cells*n_cells, ndim)
    # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
    # Tile (a, a, a, b, b, b, etc.)
    tiled_centers = tf.reshape(
        tf.tile(cells, (1, n_cells)), (n_cells * n_cells, ndim))
    # Tile (a, b, c, a, b, c, ...)
    tiled_cells = tf.tile(cells, (n_cells, 1))

    coords_vec = tf.reduce_sum((tiled_centers - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (n_cells, n_cells))
    closest_inds = tf.nn.top_k(-coords_norm, k=n_nbr_cells)[1]

    return closest_inds

  def get_cells(self):
    """Returns the locations of all grid points in box.

    Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
    Then would return a list of length 20^3 whose entries would be
    [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

    Returns
    -------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    """
    start, stop, nbr_cutoff = self.start, self.stop, self.nbr_cutoff
    mesh_args = [tf.range(start, stop, nbr_cutoff) for _ in range(self.ndim)]
    return tf.to_float(
        tf.reshape(
            tf.transpose(tf.stack(tf.meshgrid(*mesh_args))),
            (self.n_cells, self.ndim)))


class Dropout(Layer):

  def __init__(self, dropout_prob, **kwargs):
    self.dropout_prob = dropout_prob
    super(Dropout, self).__init__(**kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    training = kwargs['training'] if 'training' in kwargs else 1.0
    keep_prob = 1.0 - self.dropout_prob * training
    out_tensor = tf.nn.dropout(parent_tensor, keep_prob)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class WeightDecay(Layer):
  """Apply a weight decay penalty.

  The input should be the loss value.  This layer adds a weight decay penalty to it
  and outputs the sum.
  """

  def __init__(self, penalty, penalty_type, **kwargs):
    """Create a weight decay penalty layer.

    Parameters
    ----------
    penalty: float
      magnitude of the penalty term
    penalty_type: str
      type of penalty to compute, either 'l1' or 'l2'
    """
    self.penalty = penalty
    self.penalty_type = penalty_type
    super(WeightDecay, self).__init__(**kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = parent_tensor + model_ops.weight_decay(self.penalty_type,
                                                        self.penalty)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class AtomicConvolution(Layer):

  def __init__(self,
               atom_types=None,
               radial_params=list(),
               boxsize=None,
               **kwargs):
    """Atomic convoluation layer

    N = max_num_atoms, M = max_num_neighbors, B = batch_size, d = num_features
    l = num_radial_filters * num_atom_types

    Parameters
    ----------

    atom_types: list or None
      Of length a, where a is number of atom types for filtering.
    radial_params: list
      Of length l, where l is number of radial filters learned.
    boxsize: float or None
      Simulation box length [Angstrom].

    """
    self.boxsize = boxsize
    self.radial_params = radial_params
    self.atom_types = atom_types
    super(AtomicConvolution, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list.
    Nbrs_Z: tf.Tensor of shape (B, N, M)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B, N, l)
      A new tensor representing the output of the atomic conv layer
    """
    inputs = self._get_input_tensors(in_layers)
    X = inputs[0]
    Nbrs = tf.to_int32(inputs[1])
    Nbrs_Z = inputs[2]

    # N: Maximum number of atoms
    # M: Maximum number of neighbors
    # d: Number of coordinates/features/filters
    # B: Batch Size
    N = X.get_shape()[-2].value
    d = X.get_shape()[-1].value
    M = Nbrs.get_shape()[-1].value
    B = X.get_shape()[0].value

    D = self.distance_tensor(X, Nbrs, self.boxsize, B, N, M, d)
    R = self.distance_matrix(D)
    sym = []
    rsf_zeros = tf.zeros((B, N, M))
    for i, param in enumerate(self.radial_params):

      if tfe.in_eager_mode():
        if not self._built:
          self.variables += self._create_radial_variables(*param)
        param_variables = self.variables[3 * i:3 * i + 3]
      else:
        param_variables = self._create_radial_variables(*param)

      # We apply the radial pooling filter before atom type conv
      # to reduce computation
      rsf = self.radial_symmetry_function(R, *param_variables)

      if not self.atom_types:
        cond = tf.not_equal(Nbrs_Z, 0.0)
        sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))
      else:
        for j in range(len(self.atom_types)):
          cond = tf.equal(Nbrs_Z, self.atom_types[j])
          sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))

    layer = tf.stack(sym)
    layer = tf.transpose(layer, [1, 2, 0])  # (l, B, N) -> (B, N, l)
    m, v = tf.nn.moments(layer, axes=[0])
    out_tensor = tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)
    if set_tensors:
      self._record_variable_scope(self.name)
      self.out_tensor = out_tensor
    if tfe.in_eager_mode() and not self._built:
      self._built = True
    return out_tensor

  def _create_radial_variables(self, rc, rs, e):
    with tf.name_scope(None, "NbrRadialSymmetryFunction", [rc, rs, e]):
      rc = create_variable(rc)
      rs = create_variable(rs)
      e = create_variable(e)
    return (rc, rs, e)

  def radial_symmetry_function(self, R, rc, rs, e):
    """Calculates radial symmetry function.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_filters

    Parameters
    ----------
    R: tf.Tensor of shape (B, N, M)
      Distance matrix.
    rc: float
      Interaction cutoff [Angstrom].
    rs: float
      Gaussian distance matrix mean.
    e: float
      Gaussian distance matrix width.

    Returns
    -------
    retval: tf.Tensor of shape (B, N, M)
      Radial symmetry function (before summation)

    """

    K = self.gaussian_distance_matrix(R, rs, e)
    FC = self.radial_cutoff(R, rc)
    return tf.multiply(K, FC)

  def radial_cutoff(self, R, rc):
    """Calculates radial cutoff matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
      R [B, N, M]: tf.Tensor
        Distance matrix.
      rc: tf.Variable
        Interaction cutoff [Angstrom].

    Returns
    -------
      FC [B, N, M]: tf.Tensor
        Radial cutoff matrix.

    """

    T = 0.5 * (tf.cos(np.pi * R / (rc)) + 1)
    E = tf.zeros_like(T)
    cond = tf.less_equal(R, rc)
    FC = tf.where(cond, T, E)
    return FC

  def gaussian_distance_matrix(self, R, rs, e):
    """Calculates gaussian distance matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
      R [B, N, M]: tf.Tensor
        Distance matrix.
      rs: tf.Variable
        Gaussian distance matrix mean.
      e: tf.Variable
        Gaussian distance matrix width (e = .5/std**2).

    Returns
    -------
      retval [B, N, M]: tf.Tensor
        Gaussian distance matrix.

    """

    return tf.exp(-e * (R - rs)**2)

  def distance_tensor(self, X, Nbrs, boxsize, B, N, M, d):
    """Calculates distance tensor for batch of molecules.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features tensor.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list tensor.
    boxsize: float or None
      Simulation box length [Angstrom].

    Returns
    -------
    D: tf.Tensor of shape (B, N, M, d)
      Coordinates/features distance tensor.

    """
    atom_tensors = tf.unstack(X, axis=1)
    nbr_tensors = tf.unstack(Nbrs, axis=1)
    D = []
    if boxsize is not None:
      for atom, atom_tensor in enumerate(atom_tensors):
        nbrs = self.gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
        nbrs_tensors = tf.unstack(nbrs, axis=1)
        for nbr, nbr_tensor in enumerate(nbrs_tensors):
          _D = tf.subtract(nbr_tensor, atom_tensor)
          _D = tf.subtract(_D, boxsize * tf.round(tf.div(_D, boxsize)))
          D.append(_D)
    else:
      for atom, atom_tensor in enumerate(atom_tensors):
        nbrs = self.gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
        nbrs_tensors = tf.unstack(nbrs, axis=1)
        for nbr, nbr_tensor in enumerate(nbrs_tensors):
          _D = tf.subtract(nbr_tensor, atom_tensor)
          D.append(_D)
    D = tf.stack(D)
    D = tf.transpose(D, perm=[1, 0, 2])
    D = tf.reshape(D, [B, N, M, d])
    return D

  def gather_neighbors(self, X, nbr_indices, B, N, M, d):
    """Gathers the neighbor subsets of the atoms in X.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features tensor.
    atom_indices: tf.Tensor of shape (B, M)
      Neighbor list for single atom.

    Returns
    -------
    neighbors: tf.Tensor of shape (B, M, d)
      Neighbor coordinates/features tensor for single atom.

    """

    example_tensors = tf.unstack(X, axis=0)
    example_nbrs = tf.unstack(nbr_indices, axis=0)
    all_nbr_coords = []
    for example, (example_tensor, example_nbr) in enumerate(
        zip(example_tensors, example_nbrs)):
      nbr_coords = tf.gather(example_tensor, example_nbr)
      all_nbr_coords.append(nbr_coords)
    neighbors = tf.stack(all_nbr_coords)
    return neighbors

  def distance_matrix(self, D):
    """Calcuates the distance matrix from the distance tensor

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    D: tf.Tensor of shape (B, N, M, d)
      Distance tensor.

    Returns
    -------
    R: tf.Tensor of shape (B, N, M)
       Distance matrix.

    """

    R = tf.reduce_sum(tf.multiply(D, D), 3)
    R = tf.sqrt(R)
    return R


def AlphaShare(in_layers=None, **kwargs):
  """
  This method should be used when constructing AlphaShare layers from Sluice Networks

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  output_layers: list of Layers or tensors with same size as in_layers
    Distance matrix.

  References:
  Sluice networks: Learning what to share between loosely related tasks
  https://arxiv.org/abs/1705.08142
  """
  output_layers = []
  alpha_share = AlphaShareLayer(in_layers=in_layers, **kwargs)
  num_outputs = len(in_layers)
  return [LayerSplitter(x, in_layers=alpha_share) for x in range(num_outputs)]


class AlphaShareLayer(Layer):
  """
  Part of a sluice network. Adds alpha parameters to control
  sharing between the main and auxillary tasks

  Factory method AlphaShare should be used for construction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  out_tensor: a tensor with shape [len(in_layers), x, y] where x, y were the original layer dimensions
  Distance matrix.
  """

  def __init__(self, **kwargs):
    super(AlphaShareLayer, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    # check that there isnt just one or zero inputs
    if len(inputs) <= 1:
      raise ValueError("AlphaShare must have more than one input")
    self.num_outputs = len(inputs)
    # create subspaces
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1].value)
    subspace_size = int(original_cols / 2)
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor[:, :subspace_size], [-1]))
      subspaces.append(tf.reshape(input_tensor[:, subspace_size:], [-1]))
    n_alphas = len(subspaces)
    subspaces = tf.reshape(tf.stack(subspaces), [n_alphas, -1])

    # create the alpha learnable parameters
    if tfe.in_eager_mode():
      if not self._built:
        self.variables = [
            create_variable(
                tf.random_normal([n_alphas, n_alphas]), name='alphas')
        ]
        self._built = True
      alphas = self.variables[0]
    else:
      alphas = create_variable(
          tf.random_normal([n_alphas, n_alphas]), name='alphas')

    subspaces = tf.matmul(alphas, subspaces)

    # concatenate subspaces, reshape to size of original input, then stack
    # such that out_tensor has shape (2,?,original_cols)
    count = 0
    out_tensors = []
    tmp_tensor = []
    for row in range(n_alphas):
      tmp_tensor.append(tf.reshape(subspaces[row,], [-1, subspace_size]))
      count += 1
      if (count == 2):
        out_tensors.append(tf.concat(tmp_tensor, 1))
        tmp_tensor = []
        count = 0

    if set_tensors:
      self.out_tensor = out_tensors[0]
      self.out_tensors = out_tensors
      self.alphas = alphas
      self._non_pickle_fields += ['out_tensors', 'alphas']
    return out_tensors


class SluiceLoss(Layer):
  """
  Calculates the loss in a Sluice Network
  Every input into an AlphaShare should be used in SluiceLoss
  """

  def __init__(self, **kwargs):
    super(SluiceLoss, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    temp = []
    subspaces = []
    # creates subspaces the same way it was done in AlphaShare
    for input_tensor in inputs:
      subspace_size = int(input_tensor.get_shape()[-1].value / 2)
      subspaces.append(input_tensor[:, :subspace_size])
      subspaces.append(input_tensor[:, subspace_size:])
      product = tf.matmul(tf.transpose(subspaces[0]), subspaces[1])
      subspaces = []
      # calculate squared Frobenius norm
      temp.append(tf.reduce_sum(tf.pow(product, 2)))
    out_tensor = tf.reduce_sum(temp)
    self.out_tensor = out_tensor
    return out_tensor


class BetaShare(Layer):
  """
  Part of a sluice network. Adds beta params to control which layer
  outputs are used for prediction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  output_layers: list of Layers or tensors with same size as in_layers
    Distance matrix.
  """

  def __init__(self, **kwargs):
    super(BetaShare, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """
        Size of input layers must all be the same
        """
    inputs = self._get_input_tensors(in_layers)
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1].value)
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor, [-1]))
    n_betas = len(inputs)
    subspaces = tf.reshape(tf.stack(subspaces), [n_betas, -1])

    if tfe.in_eager_mode():
      if not self._built:
        self.variables = [
            create_variable(tf.random_normal([1, n_betas]), name='betas')
        ]
        self._built = True
      betas = self.variables[0]
    else:
      betas = create_variable(tf.random_normal([1, n_betas]), name='betas')
    out_tensor = tf.matmul(betas, subspaces)
    out_tensor = tf.reshape(out_tensor, [-1, original_cols])
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class ANIFeat(Layer):
  """Performs transform from 3D coordinates to ANI symmetry functions
  """

  def __init__(self,
               in_layers=None,
               max_atoms=23,
               radial_cutoff=4.6,
               angular_cutoff=3.1,
               radial_length=32,
               angular_length=8,
               atom_cases=[1, 6, 7, 8, 16],
               atomic_number_differentiated=True,
               coordinates_in_bohr=True,
               **kwargs):
    """
    Only X can be transformed
    """
    self.max_atoms = max_atoms
    self.radial_cutoff = radial_cutoff
    self.angular_cutoff = angular_cutoff
    self.radial_length = radial_length
    self.angular_length = angular_length
    self.atom_cases = atom_cases
    self.atomic_number_differentiated = atomic_number_differentiated
    self.coordinates_in_bohr = coordinates_in_bohr
    super(ANIFeat, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """
    In layers should be of shape dtype tf.float32, (None, self.max_atoms, 4)

    """
    inputs = self._get_input_tensors(in_layers)[0]
    atom_numbers = tf.cast(inputs[:, :, 0], tf.int32)
    flags = tf.sign(atom_numbers)
    flags = tf.to_float(tf.expand_dims(flags, 1) * tf.expand_dims(flags, 2))
    coordinates = inputs[:, :, 1:]
    if self.coordinates_in_bohr:
      coordinates = coordinates * 0.52917721092

    d = self.distance_matrix(coordinates, flags)

    d_radial_cutoff = self.distance_cutoff(d, self.radial_cutoff, flags)
    d_angular_cutoff = self.distance_cutoff(d, self.angular_cutoff, flags)

    radial_sym = self.radial_symmetry(d_radial_cutoff, d, atom_numbers)
    angular_sym = self.angular_symmetry(d_angular_cutoff, d, atom_numbers,
                                        coordinates)

    out_tensor = tf.concat(
        [tf.to_float(tf.expand_dims(atom_numbers, 2)), radial_sym, angular_sym],
        axis=2)

    if set_tensors:
      self.out_tensor = out_tensor

    return out_tensor

  def distance_matrix(self, coordinates, flags):
    """ Generate distance matrix """
    # (TODO YTZ:) faster, less memory intensive way
    # r = tf.reduce_sum(tf.square(coordinates), 2)
    # r = tf.expand_dims(r, -1)
    # inner = 2*tf.matmul(coordinates, tf.transpose(coordinates, perm=[0,2,1]))
    # # inner = 2*tf.matmul(coordinates, coordinates, transpose_b=True)

    # d = r - inner + tf.transpose(r, perm=[0,2,1])
    # d = tf.nn.relu(d) # fix numerical instabilities about diagonal
    # d = tf.sqrt(d) # does this have negative elements? may be unstable for diagonals

    max_atoms = self.max_atoms
    tensor1 = tf.stack([coordinates] * max_atoms, axis=1)
    tensor2 = tf.stack([coordinates] * max_atoms, axis=2)

    # Calculate pairwise distance
    d = tf.sqrt(
        tf.reduce_sum(tf.squared_difference(tensor1, tensor2), axis=3) + 1e-7)

    d = d * flags
    return d

  def distance_cutoff(self, d, cutoff, flags):
    """ Generate distance matrix with trainable cutoff """
    # Cutoff with threshold Rc
    d_flag = flags * tf.sign(cutoff - d)
    d_flag = tf.nn.relu(d_flag)
    d_flag = d_flag * tf.expand_dims((1 - tf.eye(self.max_atoms)), 0)
    d = 0.5 * (tf.cos(np.pi * d / cutoff) + 1)
    return d * d_flag
    # return d

  def radial_symmetry(self, d_cutoff, d, atom_numbers):
    """ Radial Symmetry Function """
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.radial_cutoff, self.radial_length)
    ita = np.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
    Rs = tf.to_float(np.reshape(Rs, (1, 1, 1, -1)))
    ita = tf.to_float(np.reshape(ita, (1, 1, 1, -1)))
    length = ita.get_shape().as_list()[-1]

    d_cutoff = tf.stack([d_cutoff] * length, axis=3)
    d = tf.stack([d] * length, axis=3)

    out = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type in self.atom_cases:
        selected_atoms = tf.expand_dims(
            tf.expand_dims(atom_numbers_embedded[:, :, atom_type], axis=1),
            axis=3)
        out_tensors.append(tf.reduce_sum(out * selected_atoms, axis=2))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out, axis=2)

  def angular_symmetry(self, d_cutoff, d, atom_numbers, coordinates):
    """ Angular Symmetry Function """

    max_atoms = self.max_atoms
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.angular_cutoff, self.angular_length)
    ita = 3 / (Rs[1] - Rs[0])**2
    thetas = np.linspace(0., np.pi, self.angular_length)
    zeta = float(self.angular_length**2)

    ita, zeta, Rs, thetas = np.meshgrid(ita, zeta, Rs, thetas)
    zeta = tf.to_float(np.reshape(zeta, (1, 1, 1, 1, -1)))
    ita = tf.to_float(np.reshape(ita, (1, 1, 1, 1, -1)))
    Rs = tf.to_float(np.reshape(Rs, (1, 1, 1, 1, -1)))
    thetas = tf.to_float(np.reshape(thetas, (1, 1, 1, 1, -1)))
    length = zeta.get_shape().as_list()[-1]

    # tf.stack issues again...
    vector_distances = tf.stack([coordinates] * max_atoms, 1) - tf.stack(
        [coordinates] * max_atoms, 2)
    R_ij = tf.stack([d] * max_atoms, axis=3)
    R_ik = tf.stack([d] * max_atoms, axis=2)
    f_R_ij = tf.stack([d_cutoff] * max_atoms, axis=3)
    f_R_ik = tf.stack([d_cutoff] * max_atoms, axis=2)

    # Define angle theta = arccos(R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance))
    vector_mul = tf.reduce_sum(tf.stack([vector_distances] * max_atoms, axis=3) * \
                               tf.stack([vector_distances] * max_atoms, axis=2), axis=4)
    vector_mul = vector_mul * tf.sign(f_R_ij) * tf.sign(f_R_ik)
    theta = tf.acos(tf.div(vector_mul, R_ij * R_ik + 1e-5))

    R_ij = tf.stack([R_ij] * length, axis=4)
    R_ik = tf.stack([R_ik] * length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * length, axis=4)
    theta = tf.stack([theta] * length, axis=4)

    out_tensor = tf.pow((1. + tf.cos(theta - thetas)) / 2., zeta) * \
                 tf.exp(-ita * tf.square((R_ij + R_ik) / 2. - Rs)) * f_R_ij * f_R_ik * 2

    if self.atomic_number_differentiated:
      out_tensors = []
      for id_j, atom_type_j in enumerate(self.atom_cases):
        for atom_type_k in self.atom_cases[id_j:]:
          selected_atoms = tf.stack([atom_numbers_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
                           tf.stack([atom_numbers_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
          selected_atoms = tf.expand_dims(
              tf.expand_dims(selected_atoms, axis=1), axis=4)
          out_tensors.append(
              tf.reduce_sum(out_tensor * selected_atoms, axis=(2, 3)))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out_tensor, axis=(2, 3))

  def get_num_feats(self):
    n_feat = self.outputs.get_shape().as_list()[-1]
    return n_feat


class LayerSplitter(Layer):
  """
  Layer which takes a tensor from in_tensor[0].out_tensors at an index
  Only layers which need to output multiple layers set and use the variable
  self.out_tensors.
  This is a utility for those special layers which set self.out_tensors
  to return a layer wrapping a specific tensor in in_layers[0].out_tensors
  """

  def __init__(self, output_num, **kwargs):
    """
    Parameters
    ----------
    output_num: int
      The index which to use as this layers out_tensor from in_layers[0]
    kwargs
    """
    self.output_num = output_num
    super(LayerSplitter, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    out_tensor = self.in_layers[0].out_tensors[self.output_num]
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class GraphEmbedPoolLayer(Layer):
  """
  GraphCNNPool Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  This is a learnable pool operation
  It constructs a new adjacency matrix for a graph of specified number of nodes.

  This differs from our other pool opertions which set vertices to a function value
  without altering the adjacency matrix.

  $V_{emb} = SpatialGraphCNN({V_{in}})$\\
  $V_{out} = \sigma(V_{emb})^{T} * V_{in}$
  $A_{out} = V_{emb}^{T} * A_{in} * V_{emb}$

  """

  def __init__(self, num_vertices, **kwargs):
    self.num_vertices = num_vertices
    super(GraphEmbedPoolLayer, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """

    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output

    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns: tf.tensor
    Returns a tf.tensor with a graph convolution applied
    The shape will be (batch, vertex, self.num_filters)
    """
    in_tensors = self._get_input_tensors(in_layers)
    if len(in_tensors) == 3:
      V, A, mask = in_tensors
    else:
      V, A = in_tensors
      mask = None
    factors = self.embedding_factors(
        V, self.num_vertices, name='%s_Factors' % self.name)

    if mask is not None:
      factors = tf.multiply(factors, mask)
    factors = self.softmax_factors(factors)

    result = tf.matmul(factors, V, transpose_a=True)

    result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
    result_A = tf.matmul(result_A, factors)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
    result_A = tf.matmul(factors, result_A, transpose_a=True)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], self.num_vertices,
                                     A.get_shape()[2].value, self.num_vertices))
    # We do not need the mask because every graph has self.num_vertices vertices now
    if set_tensors:
      self.out_tensor = result[0]
      self.out_tensors = [result, result_A]
      self._non_pickle_fields.append('out_tensors')
    return result, result_A

  def _create_variables(self, no_features, no_filters, name):
    W = create_variable(
        tf.truncated_normal(
            [no_features, no_filters], stddev=1.0 / math.sqrt(no_features)),
        name='%s_weights' % name,
        dtype=tf.float32)
    b = create_variable(
        tf.constant(0.1), name='%s_bias' % self.name, dtype=tf.float32)
    return [W, b]

  def embedding_factors(self, V, no_filters, name="default"):
    no_features = V.get_shape()[-1].value
    if tfe.in_eager_mode():
      if not self._built:
        self.variables = self._create_variables(no_features, no_filters, name)
        self._built = True
      W, b = self.variables
    else:
      W, b = self._create_variables(no_features, no_filters, name)
    V_reshape = tf.reshape(V, (-1, no_features))
    s = tf.slice(tf.shape(V), [0], [len(V.get_shape()) - 1])
    s = tf.concat([s, tf.stack([no_filters])], 0)
    result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
    return result

  def softmax_factors(self, V, axis=1, name=None):
    max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
    exp = tf.exp(tf.subtract(V, max_value))
    prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
    return prob

  def none_tensors(self):
    out_tensors, out_tensor = self.out_tensors, self.out_tensor
    self.out_tensors = None
    self.out_tensor = None
    return out_tensors, out_tensor

  def set_tensors(self, tensor):
    self.out_tensors, self.out_tensor = tensor


def GraphCNNPool(num_vertices, **kwargs):
  gcnnpool_layer = GraphEmbedPoolLayer(num_vertices, **kwargs)
  return [LayerSplitter(x, in_layers=gcnnpool_layer) for x in range(2)]


class GraphCNN(Layer):
  """
  GraphCNN Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  Spatial-domain convolutions can be defined as
  H = h_0I + h_1A + h_2A^2 + ... + hkAk, H ∈ R**(N×N)

  We approximate it by
  H ≈ h_0I + h_1A

  We can define a convolution as applying multiple these linear filters
  over edges of different types (think up, down, left, right, diagonal in images)
  Where each edge type has its own adjacency matrix
  H ≈ h_0I + h_1A_1 + h_2A_2 + . . . h_(L−1)A_(L−1)

  V_out = \sum_{c=1}^{C} H^{c} V^{c} + b
  """

  def __init__(self, num_filters, **kwargs):
    """

    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output

    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns: tf.tensor
    Returns a tf.tensor with a graph convolution applied
    The shape will be (batch, vertex, self.num_filters)
    """
    self.num_filters = num_filters
    super(GraphCNN, self).__init__(**kwargs)

  def _create_variables(self, no_features, no_A):
    W = create_variable(
        tf.truncated_normal(
            [no_features * no_A, self.num_filters],
            stddev=math.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='%s_weights' % self.name,
        dtype=tf.float32)
    W_I = create_variable(
        tf.truncated_normal(
            [no_features, self.num_filters],
            stddev=math.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='%s_weights_I' % self.name,
        dtype=tf.float32)
    b = create_variable(
        tf.constant(0.1), name='%s_bias' % self.name, dtype=tf.float32)
    return [W, W_I, b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) == 3:
      V, A, mask = inputs
    else:
      V, A = inputs
    no_A = A.get_shape()[2].value
    no_features = V.get_shape()[2].value
    if tfe.in_eager_mode():
      if not self._built:
        self.variables = self._create_variables(no_features, no_A)
        self._built = True
      W, W_I, b = self.variables
    else:
      W, W_I, b = self._create_variables(no_features, no_A)

    n = self.graphConvolution(V, A)
    A_shape = tf.shape(A)
    n = tf.reshape(n, [-1, A_shape[1], no_A * no_features])
    result = self.batch_mat_mult(n, W) + self.batch_mat_mult(V, W_I) + b
    if set_tensors:
      self.out_tensor = result
    return result

  def graphConvolution(self, V, A):
    no_A = A.get_shape()[2].value
    no_features = V.get_shape()[2].value

    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1] * no_A, A_shape[1]]))
    n = tf.matmul(A_reshape, V)
    return tf.reshape(n, [-1, A_shape[1], no_A, no_features])

  def batch_mat_mult(self, A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])

    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
      axis_2 = -1
    else:
      axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
    return result


class HingeLoss(Layer):
  """This layer computes the hinge loss on inputs:[labels,logits]
  labels: The values of this tensor is expected to be 1.0 or 0.0. The shape should be the same as logits.
  logits: Holds the log probabilities for labels, a float tensor.
  The output is a weighted loss tensor of same shape as labels.
  """

  def __init__(self, in_layers=None, separation=1.0, **kwargs):
    """
    Parameters
    ----------
    separation: float
      The absolute minimum value of logits to not incur a sample loss
    kwargs
    """
    self.separation = separation
    super(HingeLoss, self).__init__(in_layers, **kwargs)
    try:
      self._shape = self.in_layers[1].shape
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 2:
      raise ValueError()
    labels, logits = inputs[0], inputs[1]

    all_ones = array_ops.ones_like(labels)
    labels = tf.cast(math_ops.subtract(2 * labels, all_ones), dtype=tf.float32)
    seperation = tf.multiply(
        tf.cast(all_ones, dtype=tf.float32), self.separation)
    out_tensor = nn_ops.relu(
        math_ops.subtract(seperation, math_ops.multiply(labels, logits)))
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

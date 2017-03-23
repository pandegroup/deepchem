import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import MultiTaskTensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, Concat, SoftMax, SoftMaxCrossEntropy, Layer


class WeightedError(Layer):
  def __call__(self, *parents):
    entropy, weights = parents[0], parents[1]
    self.out_tensor = tf.reduce_sum(entropy.out_tensor * weights.out_tensor)
    return self.out_tensor


def tensorGraphMultitaskClassifier(n_tasks, n_features,
                                   layer_sizes=[500],
                                   bypass_layer_sizes=[100],
                                   model_dir=None):
  g = MultiTaskTensorGraph(model_dir=model_dir)
  in_layer = Input(t_shape=(None, n_features))
  g.add_layer(in_layer)
  g.add_feature(in_layer)

  # Shared Dense Layers
  prev_layer = in_layer
  dense_layers = []
  for i in range(len(layer_sizes)):
    dense = Dense(out_channels=layer_sizes[i], activation_fn=tf.nn.relu)
    g.add_layer(dense, parents=[prev_layer])
    dense_layers.append(dense)
    prev_layer = dense

  # Individual Bypass Layers
  costs = []
  for task in range(n_tasks):
    prev_layer = in_layer
    for i in range(len(bypass_layer_sizes)):
      dense = Dense(out_channels=bypass_layer_sizes[i], activation_fn=tf.nn.relu)
      g.add_layer(dense, parents=[prev_layer])
      prev_layer = dense
    joined_layer = Concat()
    g.add_layer(joined_layer, parents=[dense_layers[-1], prev_layer])

    classification = Dense(out_channels=2)
    g.add_layer(classification, parents=[joined_layer])
    g.add_output(classification)

    label = Input(t_shape=(None, 2))
    g.add_layer(label)
    g.add_label(label)

    cost = SoftMaxCrossEntropy()
    g.add_layer(cost, parents=[label, classification])
    costs.append(cost)

  entropy = Concat()
  g.add_layer(entropy, parents=costs)

  task_weights = Input(t_shape=(None, n_tasks))
  g.add_layer(task_weights)
  g.set_task_weights(task_weights)

  loss = WeightedError()
  g.add_layer(loss, parents=[entropy, task_weights])
  g.set_loss(loss)

  return g

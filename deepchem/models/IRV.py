import numpy as np
import tensorflow as tf

from deepchem.models import KerasModel, layers
from deepchem.models.losses import SigmoidCrossEntropy
from tensorflow.keras.layers import Input, Layer, Activation, Concatenate, Lambda

import torch
import torch.nn as nn
import torch.nn.functional as F

class IRVLayer(nn.Module):
    """ Core layer of IRV classifier, architecture described in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2750043/
    
    The paper introduces the Influence Relevance Voter (IRV), a novel machine learning model 
    designed for virtual high-throughput screening (vHTS).vHTS predicts the biological activity 
    of chemical compounds using computational methods, reducing the need for expensive experimental
    screening.
    
    The IRV model extends the k-Nearest Neighbors (kNN) algorithm by improving how neighbors 
    influence predictions. Instead of treating all neighbors equally, IRV assigns each neighbor
    a relevance score based on its similarity to the query compound.This similarity is calculated
    using molecular fingerprint comparisons between the query compound and its neighbors,allowing
    more relevant neighbors to have a greater impact on the prediction.
    """
    
    def __init__(self, n_tasks, K, penalty, **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)
        """
        self.n_tasks = n_tasks
        self.K = K
        self.penalty = penalty
        super(IRVLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """ 
        Initializes the trainable parameters for the voting mechanism 
        """
        self.V = nn.Parameter(torch.tensor([0.01, 1.], dtype=torch.float32), requires_grad=True)
        self.W = nn.Parameter(torch.tensor([1., 1.], dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)
        self.b2 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)

    def call(self, inputs):
        """
        Processes the input data and computes the relevance and influence of neighbors
        """

        K = self.K
        outputs = []

        for count in range(self.n_tasks):
            
            similarity = inputs[:, 2 * K * count:(2 * K * count + K)]
            
            ys = inputs[:, (2 * K * count + K):2 * K * (count + 1)].long()

            R = self.b + self.W[0] * similarity + self.W[1] * torch.arange(1, K + 1, dtype=torch.float32, device=inputs.device)
            R = torch.sigmoid(R)

            z = torch.sum(R * self.V[ys], dim=1) + self.b2
            outputs.append(z.view(-1, 1))
            
        loss = self.penalty * (torch.norm(self.W, 2) + torch.norm(self.V, 2) + torch.norm(self.b, 2) + torch.norm(self.b2, 2))
        self.add_loss(loss)
        output = torch.cat(outputs, dim=1)
        return output

class Slice(Layer):
    """ Choose a slice of input on the last axis given order,
    Suppose input x has two dimensions,
    output f(x) = x[:, slice_num:slice_num+1]
    """

    def __init__(self, slice_num, axis=1, **kwargs):
        """
        Parameters
        ----------
        slice_num: int
            index of slice number
        axis: int
            axis id
        """
        self.slice_num = slice_num
        self.axis = axis
        super(Slice, self).__init__(**kwargs)

    def call(self, inputs):
        slice_num = self.slice_num
        axis = self.axis
        return tf.slice(inputs, [0] * axis + [slice_num], [-1] * axis + [1])


class MultitaskIRVClassifier(KerasModel):

    def __init__(self,
                 n_tasks,
                 K=10,
                 penalty=0.0,
                 mode="classification",
                 **kwargs):
        """Initialize MultitaskIRVClassifier

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        K: int
            Number of nearest neighbours used in classification
        penalty: float
            Amount of penalty (l2 or l1 applied)

        """
        self.n_tasks = n_tasks
        self.K = K
        self.n_features = 2 * self.K * self.n_tasks
        self.penalty = penalty
        mol_features = Input(shape=(self.n_features,))
        predictions = IRVLayer(self.n_tasks, self.K, self.penalty)(mol_features)
        logits = []
        outputs = []
        for task in range(self.n_tasks):
            task_output = Slice(task, 1)(predictions)
            sigmoid = Activation(tf.sigmoid)(task_output)
            logits.append(task_output)
            outputs.append(sigmoid)
        outputs = layers.Stack(axis=1)(outputs)
        outputs2 = Lambda(lambda x: 1 - x)(outputs)
        outputs = [
            Concatenate(axis=2)([outputs2, outputs]),
            logits[0] if len(logits) == 1 else Concatenate(axis=1)(logits)
        ]
        model = tf.keras.Model(inputs=[mol_features], outputs=outputs)
        super(MultitaskIRVClassifier,
              self).__init__(model,
                             SigmoidCrossEntropy(),
                             output_types=['prediction', 'loss'],
                             **kwargs)


import warnings  # noqa: E402


class TensorflowMultitaskIRVClassifier(MultitaskIRVClassifier):

    def __init__(self, *args, **kwargs):

        warnings.warn(
            "TensorflowMultitaskIRVClassifier is deprecated and has been renamed to MultitaskIRVClassifier",
            FutureWarning)

        super(TensorflowMultitaskIRVClassifier, self).__init__(*args, **kwargs)

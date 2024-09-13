"""Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes
# License: BSD 3-clause


from typing import Callable

import numpy as np
import sklearn.metrics as skm

from mlrose_ky.neural import activation as act
from mlrose_ky.neural.utils import unflatten_weights


class NetworkWeights:
    """
    Fitness function for neural network weights optimization problem.

    Parameters
    ----------
    X : np.ndarray
        Numpy array containing feature dataset with each row representing a
        single observation.
    y : np.ndarray
        Numpy array containing true values of data labels.
        Length must be the same as the length of X.
    node_list : list[int]
        Number of nodes in each layer, including the input and output layers.
    activation : Callabe
        Activation function for each of the hidden layers with the signature
        `activation(x, deriv)`, where setting deriv is a boolean that
        determines whether to return the activation function or its derivative.
    bias : bool, default=True
        Whether a bias term is included in the network.
    is_classifier : bool, default=True
        Whether the network is for classification or regression. Set True for
        classification and False for regression.
    learning_rate : float, default=0.1
        The learning rate for gradient descent updates.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node_list: list[int],
        activation: Callable,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
    ):
        if not callable(activation):
            raise TypeError("Activation function must be callable.")
        try:
            activation(np.array([0.1]), deriv=False)
        except TypeError:
            raise TypeError("Activation function must accept two arguments: 'x' and 'deriv'.")

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty.")

        y = np.array(y)

        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError(f"The length of X ({np.shape(X)[0]}) and y ({np.shape(y)[0]}) must be equal.")

        if len(node_list) < 2:
            raise ValueError("node_list must contain at least 2 elements.")

        if np.shape(X)[1] != (node_list[0] - bias):
            raise ValueError(f"The number of columns in X must equal {node_list[0] - bias}.")

        if np.shape(y)[1] != node_list[-1]:
            raise ValueError(f"The number of columns in y must equal {node_list[-1]}.")

        if not isinstance(bias, bool):
            raise ValueError("bias must be True or False.")

        if not isinstance(is_classifier, bool):
            raise ValueError("is_classifier must be True or False.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        self.X = X
        self.y_true = y
        self.node_list = node_list
        self.activation = activation
        self.bias = bias
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate

        if self.is_classifier:
            self.loss = skm.log_loss
            self.output_activation = act.sigmoid if np.shape(self.y_true)[1] == 1 else act.softmax
        else:
            self.loss = skm.mean_squared_error
            self.output_activation = act.identity

        self.inputs_list = []
        self.y_pred = y
        self.weights = []
        self.prob_type = "continuous"

        self.nodes = sum(node_list[i] * node_list[i + 1] for i in range(len(node_list) - 1))

    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate the fitness of a state.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        fitness : float
            Value of fitness function.
        """
        if len(state) != self.nodes:
            raise ValueError(f"state must have length {self.nodes}, got {len(state)}.")

        self.inputs_list = []
        self.weights = list(unflatten_weights(state, self.node_list))

        if self.bias:
            ones = np.ones([np.shape(self.X)[0], 1])
            inputs = np.hstack((self.X, ones))
        else:
            inputs = self.X

        for i in range(len(self.weights)):
            outputs = np.dot(inputs, self.weights[i])
            self.inputs_list.append(inputs)

            inputs = self.activation(outputs) if i < len(self.weights) - 1 else self.output_activation(outputs)

        self.y_pred = inputs
        return self.loss(self.y_true, self.y_pred)

    def get_output_activation(self) -> Callable:
        """
        Return the activation function for the output layer.

        Returns
        -------
        Callable
            Activation function for the output layer.
        """
        return self.output_activation

    def get_prob_type(self) -> str:
        """
        Return the problem type.

        Returns
        -------
        str
            Problem type as 'discrete', 'continuous', 'tsp', or 'either'.
        """
        return self.prob_type

    def calculate_updates(self) -> list[np.ndarray]:
        """
        Calculate gradient descent updates.

        Returns
        -------
        list of np.ndarray
            List of back propagation weight updates.
        """
        delta_list: list = []
        updates_list: list = []

        for i in range(len(self.inputs_list) - 1, -1, -1):
            if i == len(self.inputs_list) - 1:
                delta = self.y_pred - self.y_true
            else:
                dot = np.dot(delta_list[-1], np.transpose(self.weights[i + 1]))
                activation = self.activation(self.inputs_list[i + 1], deriv=True)
                delta = dot * activation

            delta_list.append(delta)
            updates = -self.learning_rate * np.dot(np.transpose(self.inputs_list[i]), delta)
            updates_list.append(updates)

        return updates_list[::-1]

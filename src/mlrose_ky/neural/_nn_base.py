"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from mlrose_ky.neural.fitness.network_weights import NetworkWeights
from mlrose_ky.neural.utils import unflatten_weights
from mlrose_ky.opt_probs import ContinuousOpt


class _NNBase(BaseEstimator, ABC):
    """
    Abstract base class for neural network models.

    Defines the necessary methods and utility functions for training and
    predicting with neural network models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None, init_weights: np.ndarray = None):
        """Fit the neural network to the data."""
        raise NotImplementedError("Subclasses must implement fit method")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the model to predict data labels for a given feature array."""
        raise NotImplementedError("Subclasses must implement predict method")

    @staticmethod
    def _calculate_state_size(node_list: list[int]) -> int:
        """
        Calculate the total number of nodes in the network layers.

        Parameters
        ----------
        node_list : list of int
            List of the number of nodes in each layer, including input
            and output layers.

        Returns
        -------
        int
            The total number of nodes in the network.
        """
        return sum(node_list[i] * node_list[i + 1] for i in range(len(node_list) - 1))

    @staticmethod
    def _build_node_list(X: np.ndarray, y: np.ndarray, hidden_nodes: list[int], bias: bool = False) -> list[int]:
        """
        Build a list of nodes in each layer of the network.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset with each row representing a single observation.
        y : np.ndarray
            Array of data labels.
        hidden_nodes : list of int
            List of the number of nodes in the hidden layers.
        bias : bool, optional, default=False
            Whether to include a bias term in the network.

        Returns
        -------
        list of int
            A list containing the number of nodes in each layer of the network.
        """
        input_nodes = np.shape(X)[1] + bias
        output_nodes = np.shape(y)[1]

        return [input_nodes] + hidden_nodes + [output_nodes]

    @staticmethod
    def _format_x_y_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure that the X and y data are correctly formatted.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset with each row representing a single observation.
        y : np.ndarray
            Data labels.

        Returns
        -------
        X : np.ndarray
            Formatted feature dataset.
        y : np.ndarray
            Formatted data labels.

        Raises
        ------
        ValueError
            If the lengths of X and y do not match.
        """
        y = np.array(y)

        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError(f"The length of X ({np.shape(X)[0]}) and y ({np.shape(y)[0]}) must be equal.")

        return X, y

    @staticmethod
    def _build_problem_and_fitness_function(
        X: np.ndarray,
        y: np.ndarray,
        node_list: list[int],
        activation: Callable,
        learning_rate: float,
        clip_max: float,
        bias: bool = False,
        is_classifier: bool = True,
    ) -> tuple[NetworkWeights, ContinuousOpt]:
        """
        Initialize the optimization problem and fitness function.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset.
        y : np.ndarray
            Data labels.
        node_list : list of int
            List of the number of nodes in each layer.
        activation : Callable
            Activation function for the network.
        learning_rate : float
            Learning rate for weight updates.
        clip_max : float
            Maximum value for weights.
        bias : bool, optional, default=False
            Whether to include a bias term in the network.
        is_classifier : bool, optional, default=True
            Whether the network is a classifier.

        Returns
        -------
        NetworkWeights
            The fitness function for optimizing the network weights.
        ContinuousOpt
            The continuous optimization problem for gradient descent.
        """
        fitness = NetworkWeights(X, y, node_list, activation, bias, is_classifier, learning_rate=learning_rate)
        num_nodes = _NNBase._calculate_state_size(node_list)

        problem = ContinuousOpt(
            length=num_nodes, fitness_fn=fitness, maximize=False, min_val=-clip_max, max_val=clip_max, step=learning_rate
        )

        return fitness, problem

    @staticmethod
    def _predict(
        X: np.ndarray,
        fitted_weights: np.ndarray,
        node_list: list[int],
        input_activation: Callable,
        output_activation: Callable,
        bias: bool = False,
        is_classifier: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predict data labels based on the fitted weights of the network.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset.
        fitted_weights : np.ndarray
            Flattened weight array for the network.
        node_list : list of int
            List of the number of nodes in each layer.
        input_activation : Callable
            Activation function for the hidden layers.
        output_activation : Callable
            Activation function for the output layer.
        bias : bool, optional, default=False
            Whether to include a bias term in the network.
        is_classifier : bool, optional, default=True
            Whether the network is a classifier.

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels for the input dataset.
        predicted_probs : np.ndarray or None
            Predicted probabilities for the input dataset, if the network is a classifier.
        """
        if not node_list:
            raise ValueError("node_list cannot be empty.")

        weights = list(unflatten_weights(fitted_weights, node_list))

        if bias:
            ones = np.ones([np.shape(X)[0], 1])
            inputs = np.hstack((X, ones))
        else:
            inputs = X

        y_pred = np.empty(0)  # Initialize y_pred to prevent uninitialized warning
        predicted_probs = None  # Initialize predicted_probs

        for i in range(len(weights)):
            outputs = np.dot(inputs, weights[i])

            if i < len(weights) - 1:
                inputs = input_activation(outputs)
            else:
                y_pred = output_activation(outputs)

        if is_classifier:
            predicted_probs = y_pred

            if node_list[-1] == 1:
                y_pred = np.round(y_pred).astype(int)
            else:
                zeros = np.zeros_like(y_pred)
                zeros[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
                y_pred = zeros.astype(int)

        return y_pred, predicted_probs

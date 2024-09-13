"""Unit tests for neural/neural_network.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, learning_curve

from tests.globals import SEED, sample_data
from mlrose_ky import softmax
from mlrose_ky.neural.neural_network import NeuralNetwork

# noinspection PyProtectedMember
from mlrose_ky.neural._nn_base import _NNBase


class TestNeuralNetwork:
    """Test cases for the NeuralNetwork class."""

    def test_fit_random_hill_climb(self, sample_data):
        """Test fitting the network using random hill climb."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        network = NeuralNetwork(hidden_nodes=hidden_nodes, activation="identity", bias=bias, learning_rate=1, clip_max=1)

        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        num_weights = _NNBase._calculate_state_size(node_list)

        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) < num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_simulated_annealing(self, sample_data):
        """Test fitting the network using simulated annealing."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        network = NeuralNetwork(
            hidden_nodes=hidden_nodes, activation="identity", algorithm="simulated_annealing", bias=bias, learning_rate=1, clip_max=1
        )

        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        num_weights = _NNBase._calculate_state_size(node_list)

        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) < num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_genetic_alg(self, sample_data):
        """Test fitting the network using genetic algorithm."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        network = NeuralNetwork(
            hidden_nodes=hidden_nodes,
            activation="identity",
            algorithm="genetic_alg",
            bias=bias,
            learning_rate=1,
            clip_max=1,
            max_attempts=1,
        )

        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        num_weights = _NNBase._calculate_state_size(node_list)

        network.fit(X, y_classifier)
        fitted = network.fitted_weights

        assert sum(fitted) < num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_gradient_descent(self, sample_data):
        """Test fitting the network using gradient descent."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        network = NeuralNetwork(
            hidden_nodes=hidden_nodes, activation="identity", algorithm="gradient_descent", bias=bias, learning_rate=1, clip_max=1
        )

        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        num_weights = _NNBase._calculate_state_size(node_list)

        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) < num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_predict_no_bias(self, sample_data):
        """Test prediction without bias."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        network = NeuralNetwork(hidden_nodes=hidden_nodes, activation="identity", bias=bias, learning_rate=1, clip_max=1)

        node_list = [X.shape[1], *hidden_nodes, 2]  # Unsure why this last number needs to be 2 even though bias is False
        network.fitted_weights = np.array([0.2, 0.5, 0.3, 0.4, 0.4, 0.3, 0.5, 0.2, -1, 1, 1, -1])
        network.node_list = node_list
        network.output_activation = softmax

        labels = np.array([[0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]])
        probs = np.array([[0.40131, 0.59869], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.31003, 0.68997], [0.64566, 0.35434]])
        assert np.array_equal(network.predict(X), labels) and np.allclose(network.predicted_probs, probs, atol=0.0001)

    def test_predict_bias(self, sample_data):
        """Test prediction with bias."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        network = NeuralNetwork(hidden_nodes=hidden_nodes, activation="identity", learning_rate=1, clip_max=1)

        node_list = [5, *hidden_nodes, 2]  # Unsure why this first number needs to be 5 even though X.shape[1] is 6
        network.fitted_weights = np.array([0.2, 0.5, 0.3, 0.4, 0.4, 0.3, 0.5, 0.2, 1, -1, -0.1, 0.1, 0.1, -0.1])
        network.node_list = node_list
        network.output_activation = softmax

        labels = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        probs = np.array(
            [[0.39174, 0.60826], [0.40131, 0.59869], [0.40131, 0.59869], [0.40131, 0.59869], [0.38225, 0.61775], [0.41571, 0.58419]]
        )
        assert np.array_equal(network.predict(X), labels) and np.allclose(network.predicted_probs, probs, atol=0.0001)

    def test_learning_curve(self):
        """Test scikit-learn learning curve method."""
        network = NeuralNetwork(
            hidden_nodes=[2],
            activation="identity",
            algorithm="simulated_annealing",
            curve=True,
            learning_rate=1,
            clip_max=1,
            max_attempts=100,
        )

        X = np.array(
            [
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 1],
            ]
        )
        y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])

        train_sizes = [0.5, 1.0]
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_sizes, train_scores, test_scores = learning_curve(network, X, y, train_sizes=train_sizes, cv=cv, scoring="accuracy")

        assert not np.isnan(train_scores).any() and not np.isnan(test_scores).any()

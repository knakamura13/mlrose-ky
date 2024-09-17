"""Unit tests for neural/fitness/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky.neural.activation import sigmoid, identity
from mlrose_ky.neural.fitness import NetworkWeights
from tests.globals import sample_data


class TestNeuralFitness:
    """Test cases for the neural.fitness module."""

    def test_initialization_valid(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)

        assert nw.X.shape == X.shape
        assert nw.y_true.shape == y.shape
        assert nw.node_list == node_list
        assert nw.activation == activation

    def test_initialization_invalid_shapes(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(Exception):
            NetworkWeights(X, y, node_list, activation)

    def test_evaluate(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_evaluate_invalid_state_length(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5])

        nw = NetworkWeights(X, y, node_list, activation)

        with pytest.raises(Exception):
            nw.evaluate(state)

    def test_get_output_activation(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)
        output_activation = nw.get_output_activation()

        assert output_activation == sigmoid

    def test_get_prob_type(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)
        prob_type = nw.get_prob_type()

        assert prob_type == "continuous"

    def test_no_hidden_layers(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = identity
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_multiple_hidden_layers(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 3, 2, 1]
        activation = sigmoid
        state = np.array([0.5] * 14)

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        updates = nw.calculate_updates()
        assert isinstance(updates, list)
        assert len(updates) == len(node_list) - 1

    def test_multiclass_classification(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1, 0, 0], [0, 1, 0]])
        node_list = [2, 3]
        activation = sigmoid
        state = np.array([0.5] * 6)

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        assert nw.output_activation.__name__ == "softmax"

    def test_regression(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1.0], [0.5]])
        node_list = [2, 1]
        activation = identity
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation, is_classifier=False)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        assert nw.output_activation.__name__ == "identity"
        assert nw.loss.__name__ == "mean_squared_error"

    def test_invalid_activation_function(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]  # Adjusted to have 2 input nodes

        def invalid_activation(x):
            """An invalid activation function that doesn't have the required signature"""
            return x

        with pytest.raises(TypeError):
            NetworkWeights(X, y, node_list, invalid_activation)

    def test_extreme_values(self):
        X = np.array([[1e10], [1e-10]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_empty_dataset(self):
        X = np.array([]).reshape(0, 1)
        y = np.array([]).reshape(0, 1)
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="X and y cannot be empty"):
            NetworkWeights(X, y, node_list, activation)

    def test_evaluate_no_bias_classifier(self, sample_data):
        """Test evaluation of network weights without bias for classification."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7393

    def test_evaluate_no_bias_multi(self, sample_data):
        """Test evaluation of network weights without bias for multiclass classification."""
        X, _, y_multiclass, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2]  # Unsure why last layer needs 2 nodes even though bias is False
        fitness = NetworkWeights(X, y_multiclass, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(4) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7183

    def test_evaluate_no_bias_regressor(self, sample_data):
        """Test evaluation of network weights without bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_regressor, node_list, activation=identity, bias=bias, is_classifier=False)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.5542

    def test_evaluate_bias_regressor(self, sample_data):
        """Test evaluation of network weights with bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = True
        node_list = [5, *hidden_nodes, 1]  # Unsure why this first number needs to be 5 even though X.shape[1] is 6
        fitness = NetworkWeights(X, y_regressor, node_list, bias=bias, activation=identity, is_classifier=False)

        a = list(np.arange(10) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.4363

    def test_calculate_updates(self, sample_data):
        """Test calculation of weight updates for the network."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias, is_classifier=False, learning_rate=1)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b
        fitness.evaluate(np.asarray(weights))

        updates = list(fitness.calculate_updates())
        update1 = np.array([[-0.0017, -0.0034], [-0.0046, -0.0092], [-0.0052, -0.0104], [0.0014, 0.0028]])
        update2 = np.array([[-3.17], [-4.18]])

        assert np.allclose(updates[0], update1, atol=0.001) and np.allclose(updates[1], update2, atol=0.001)

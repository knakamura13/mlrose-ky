"""Unit tests for neural/_nn_base.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import NetworkWeights, ContinuousOpt

# noinspection PyProtectedMember
from mlrose_ky.neural._nn_base import _NNBase


class TestNNBase:
    """Test cases for the neural network base class _NNBase."""

    def test_nn_base_instantiation_raises(self):
        """Test that instantiating _NNBase raises TypeError due to abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class _NNBase with abstract methods fit, predict"):
            _NNBase()

    def test_nn_base_abstract_methods(self):
        """Test that calling abstract methods raises NotImplementedError."""

        class TestNN(_NNBase):

            def fit(self, X, y=None, init_weights=None):
                super().fit(X, y, init_weights)

            def predict(self, X):
                super().predict(X)

        nn = TestNN()
        _X = np.array([[0]])
        _y = np.array([0])

        with pytest.raises(NotImplementedError, match="Subclasses must implement fit method"):
            nn.fit(_X, _y)

        with pytest.raises(NotImplementedError, match="Subclasses must implement predict method"):
            nn.predict(_X)

    def test_calculate_state_size(self):
        """Test _calculate_state_size static method."""
        node_list = [2, 3, 1]
        expected_size = 2 * 3 + 3 * 1  # 6 + 3 = 9
        size = _NNBase._calculate_state_size(node_list)
        assert size == expected_size

        node_list = [4]
        size = _NNBase._calculate_state_size(node_list)
        assert size == 0

        node_list = []
        size = _NNBase._calculate_state_size(node_list)
        assert size == 0

    def test_build_node_list(self):
        """Test _build_node_list static method."""
        X = np.zeros((10, 5))
        y = np.zeros((10, 2))
        hidden_nodes = [4, 3]
        bias = False
        node_list = _NNBase._build_node_list(X, y, hidden_nodes, bias)
        expected_node_list = [5, 4, 3, 2]
        assert node_list == expected_node_list

        bias = True
        node_list = _NNBase._build_node_list(X, y, hidden_nodes, bias)
        expected_node_list = [6, 4, 3, 2]
        assert node_list == expected_node_list

        hidden_nodes = []
        node_list = _NNBase._build_node_list(X, y, hidden_nodes)
        expected_node_list = [5, 2]
        assert node_list == expected_node_list

    def test_format_x_y_data(self):
        """Test _format_x_y_data static method."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 0])
        X_formatted, y_formatted = _NNBase._format_x_y_data(X, y)
        assert np.array_equal(X_formatted, X)
        assert y_formatted.shape == (2, 1)
        assert np.array_equal(y_formatted, np.array([[1], [0]]))

        y = np.array([[1], [0]])
        X_formatted, y_formatted = _NNBase._format_x_y_data(X, y)
        assert np.array_equal(y_formatted, y)

        y = np.array([1])
        with pytest.raises(ValueError, match="The length of X \\(2\\) and y \\(1\\) must be equal."):
            _NNBase._format_x_y_data(X, y)

    def test_build_problem_and_fitness_function(self):
        """Test _build_problem_and_fitness_function static method."""
        X = np.array([[0, 1], [1, 0]])
        y = np.array([[1], [0]])
        node_list = [2, 2, 1]

        # noinspection PyMissingOrEmptyDocstring
        def activation(x, deriv=False):
            if deriv:
                return np.ones_like(x)
            return np.tanh(x)

        learning_rate = 0.1
        clip_max = 5.0
        bias = False
        is_classifier = True

        fitness, problem = _NNBase._build_problem_and_fitness_function(
            X, y, node_list, activation, learning_rate, clip_max, bias, is_classifier
        )
        assert isinstance(fitness, NetworkWeights), "Fitness function is not of type NetworkWeights."
        assert isinstance(problem, ContinuousOpt), "Problem is not of type ContinuousOpt."
        assert problem.length == _NNBase._calculate_state_size(node_list), "Incorrect problem length."
        assert problem.maximize == -1.0, "Problem should be a minimization problem."
        assert problem.min_val == -clip_max, "Incorrect min_val in problem."
        assert problem.max_val == clip_max, "Incorrect max_val in problem."
        assert problem.step == learning_rate, "Incorrect step size in problem."

    def test_predict(self):
        """Test _predict static method."""

        # noinspection PyMissingOrEmptyDocstring
        def input_activation(x):
            return x

        # noinspection PyMissingOrEmptyDocstring
        def output_activation(x):
            return x

        X = np.array([[1, 2], [3, 4]])
        node_list = [2, 2, 1]
        bias = False
        is_classifier = True
        total_weights = _NNBase._calculate_state_size(node_list)
        fitted_weights = np.ones(total_weights)

        y_pred, predicted_probs = _NNBase._predict(X, fitted_weights, node_list, input_activation, output_activation, bias, is_classifier)
        assert y_pred.shape == (2, 1)
        assert predicted_probs.shape == (2, 1)

        # Test with bias
        bias = True
        node_list = [3, 2, 1]
        total_weights = _NNBase._calculate_state_size(node_list)
        fitted_weights = np.ones(total_weights)
        y_pred, predicted_probs = _NNBase._predict(X, fitted_weights, node_list, input_activation, output_activation, bias, is_classifier)
        assert y_pred.shape == (2, 1)
        assert predicted_probs.shape == (2, 1)

        # Test for regression
        is_classifier = False
        y_pred, predicted_probs = _NNBase._predict(X, fitted_weights, node_list, input_activation, output_activation, bias, is_classifier)
        assert y_pred.shape == (2, 1)
        assert predicted_probs is None

        # Edge case: Empty node_list
        node_list = []
        fitted_weights = np.array([])
        with pytest.raises(ValueError, match="node_list cannot be empty."):
            _NNBase._predict(X, fitted_weights, node_list, input_activation, output_activation, bias, is_classifier)

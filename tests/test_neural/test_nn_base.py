"""Unit tests for neural/_nn_base.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from tests.globals import sample_data
from mlrose_ky import flatten_weights, unflatten_weights, identity
from mlrose_ky.neural.fitness.network_weights import NetworkWeights
from mlrose_ky.opt_probs import ContinuousOpt
from mlrose_ky.algorithms.gd import gradient_descent

# noinspection PyProtectedMember
from mlrose_ky.neural._nn_base import _NNBase

# TODO: test ._build_node_list(), ._format_x_y_data(), ._build_problem_and_fitness_function(), ._predict()


class TestNNBase:
    """Test cases for neural network-related utilities."""

    def test_flatten_weights(self):
        """Test flatten_weights function."""
        x = np.arange(12)
        y = np.arange(6)
        z = np.arange(16)

        a = np.reshape(x, (4, 3))
        b = np.reshape(y, (3, 2))
        c = np.reshape(z, (2, 8))

        weights = [a, b, c]
        flat = list(x) + list(y) + list(z)

        assert np.array_equal(np.array(flatten_weights(weights)), np.array(flat))

    def test_unflatten_weights(self):
        """Test unflatten_weights function."""
        x = np.arange(12)
        y = np.arange(6)
        z = np.arange(16)

        a = np.reshape(x, (4, 3))
        b = np.reshape(y, (3, 2))
        c = np.reshape(z, (2, 8))

        flat = list(x) + list(y) + list(z)
        nodes = [4, 3, 2, 8]
        weights = list(unflatten_weights(np.asarray(flat), nodes))

        assert np.array_equal(weights[0], a) and np.array_equal(weights[1], b) and np.array_equal(weights[2], c)

    def test_gradient_descent(self, sample_data):
        """Test gradient descent algorithm on sample data."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias, is_classifier=False)

        num_weights = _NNBase._calculate_state_size(node_list)
        test_weights = np.ones(num_weights)

        problem = ContinuousOpt(num_weights, fitness, maximize=False, min_val=-1)
        test_fitness = -1 * problem.eval_fitness(test_weights)
        best_state, best_fitness, _ = gradient_descent(problem)

        assert len(best_state) == num_weights and min(best_state) >= -1 and max(best_state) <= 1 and best_fitness < test_fitness

    def test_gradient_descent_iter1(self, sample_data):
        """Test gradient descent with one iteration."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias, is_classifier=False)

        num_weights = _NNBase._calculate_state_size(node_list)
        problem = ContinuousOpt(num_weights, fitness, maximize=False, min_val=-1)
        init_weights = np.ones(num_weights)
        best_state, best_fitness, _ = gradient_descent(problem, max_iters=1, init_state=init_weights)

        x = np.array([-0.7, -0.7, -0.9, -0.9, -0.9, -0.9, -1, -1, -1, -1])
        assert np.allclose(best_state, x, atol=0.001) and round(best_fitness, 2) == 19.14

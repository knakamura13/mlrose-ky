"""Unit tests for neural/utils/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest
from mlrose_ky.neural.utils import flatten_weights, unflatten_weights, gradient_descent_original


# noinspection PyMissingOrEmptyDocstring
class MockProblem:
    """A mock problem class to simulate the optimization problem."""

    def __init__(self, state, maximize=True):
        self.state = np.array(state)
        self.maximize = 1 if maximize else -1
        self.fitness = self.get_fitness()

    @staticmethod
    def eval_fitness(state):
        return np.sum(state)

    def get_length(self):
        return len(self.state)

    def get_maximize(self):
        return self.maximize

    def get_fitness(self):
        return np.sum(self.state)

    def get_adjusted_fitness(self):
        return self.get_fitness()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.array(state)
        self.fitness = self.get_fitness()

    def reset(self):
        self.state = np.zeros_like(self.state)
        self.fitness = self.get_fitness()

    def calculate_updates(self):
        return [self.state * -0.1]

    def update_state(self, updates):
        return self.state + np.array(updates)


@pytest.fixture
def mock_problem():
    """Fixture to create a MockProblem instance with a default state."""
    return MockProblem([1, 2, 3, 4])


class TestNeuralOptimizationFunctions:
    """Test cases for neural network weight optimization functions."""

    def test_flatten_weights(self):
        weights = [np.array([[1, 2], [3, 4]]), np.array([[5], [6]])]
        expected_output = np.array([1, 2, 3, 4, 5, 6])
        flat_weights = flatten_weights(weights)

        assert np.array_equal(flat_weights, expected_output)

    def test_flatten_weights_empty_list(self):
        weights = []
        flat_weights = flatten_weights(weights)

        assert flat_weights.size == 0

    def test_unflatten_weights(self):
        flat_weights = np.array([1, 2, 3, 4, 5, 6])
        node_list = [2, 2, 1]
        expected_output = [np.array([[1, 2], [3, 4]]), np.array([[5], [6]])]
        weights = unflatten_weights(flat_weights, node_list)

        for w, ew in zip(weights, expected_output):
            assert np.array_equal(w, ew)

    def test_unflatten_weights_empty_array(self):
        flat_weights = np.array([])
        node_list = [2, 2, 1]

        with pytest.raises(Exception):
            unflatten_weights(flat_weights, node_list)

    def test_unflatten_weights_invalid_length(self):
        flat_weights = np.array([1, 2, 3])
        node_list = [2, 2, 1]

        with pytest.raises(Exception):
            unflatten_weights(flat_weights, node_list)

    def test_gradient_descent_original(self, mock_problem):
        best_state, best_fitness, _ = gradient_descent_original(mock_problem, max_iters=100)

        assert np.array_equal(best_state, np.zeros(4))
        assert best_fitness == 0

    def test_gradient_descent_original_with_curve(self, mock_problem):
        best_state, best_fitness, fitness_curve = gradient_descent_original(mock_problem, max_iters=100, curve=True)

        assert np.array_equal(best_state, np.zeros(4))
        assert best_fitness == 0
        assert isinstance(fitness_curve, np.ndarray)

    def test_gradient_descent_original_with_init_state(self, mock_problem):
        init_state = np.array([0.5, 0.5, 0.5, 0.5])
        best_state, best_fitness, _ = gradient_descent_original(mock_problem, max_iters=100, init_state=init_state)

        assert np.array_equal(best_state, init_state)
        assert best_fitness == np.sum(init_state)

    def test_gradient_descent_original_invalid_max_attempts(self, mock_problem):
        with pytest.raises(Exception):
            gradient_descent_original(mock_problem, max_attempts=-1)

    def test_gradient_descent_original_invalid_max_iters(self, mock_problem):
        with pytest.raises(Exception):
            gradient_descent_original(mock_problem, max_iters=-1)

    def test_gradient_descent_original_invalid_init_state(self, mock_problem):
        init_state = np.array([1, 2])
        with pytest.raises(Exception):
            gradient_descent_original(mock_problem, init_state=init_state)

    def test_gradient_descent_original_with_random_state(self, mock_problem):
        best_state_1, best_fitness_1, _ = gradient_descent_original(mock_problem, max_iters=100, random_state=42)

        mock_problem.reset()
        best_state_2, best_fitness_2, _ = gradient_descent_original(mock_problem, max_iters=100)

        assert np.array_equal(best_state_1, best_state_2)
        assert best_fitness_1 == best_fitness_2

    def test_gradient_descent_original_single_iteration(self, mock_problem):
        best_state, best_fitness, _ = gradient_descent_original(mock_problem, max_iters=1)

        assert best_fitness == 0
        assert np.array_equal(best_state, np.zeros(4))

    def test_gradient_descent_original_single_attempt(self, mock_problem):
        best_state, best_fitness, _ = gradient_descent_original(mock_problem, max_attempts=1, max_iters=100)

        assert best_fitness == 0
        assert np.array_equal(best_state, np.zeros(4))

    def test_gradient_descent_original_infinite_iterations(self, mock_problem):
        best_state, best_fitness, _ = gradient_descent_original(mock_problem)

        assert best_fitness == 0
        assert np.array_equal(best_state, np.zeros(4))

    def test_gradient_descent_original_large_init_state(self):
        problem = MockProblem([1e10, 1e10, 1e10, 1e10])
        best_state, best_fitness, _ = gradient_descent_original(problem, max_iters=100)

        assert np.allclose(best_state, np.zeros(4))
        assert best_fitness == 0

    def test_gradient_descent_original_random_init(self, mock_problem):
        best_state_1, best_fitness_1, _ = gradient_descent_original(mock_problem, max_iters=100)

        mock_problem.reset()
        best_state_2, best_fitness_2, _ = gradient_descent_original(mock_problem, max_iters=100)

        assert np.array_equal(best_state_1, best_state_2)
        assert best_fitness_1 == best_fitness_2

    def test_gradient_descent_original_no_improvement(self):
        class NoImprovementProblem(MockProblem):
            # noinspection PyMissingOrEmptyDocstring
            def calculate_updates(self):
                return [np.zeros_like(self.state)]

        problem = NoImprovementProblem([1, 2, 3, 4])
        best_state, best_fitness, _ = gradient_descent_original(problem, max_iters=100)

        assert best_fitness == 0
        assert np.array_equal(best_state, np.zeros(4))

    def test_gradient_descent_original_small_updates(self):
        class SmallUpdatesProblem(MockProblem):
            # noinspection PyMissingOrEmptyDocstring
            def calculate_updates(self):
                return [self.state * 1e-10]

        problem = SmallUpdatesProblem([1, 2, 3, 4])
        best_state, best_fitness, _ = gradient_descent_original(problem, max_iters=100)

        assert np.array_equal(best_state, np.zeros(4))
        assert best_fitness == 0

    def test_gradient_descent_original_zero_iterations(self, mock_problem):
        problem = MockProblem([1, 2, 3, 4])
        best_state, best_fitness, _ = gradient_descent_original(problem, max_iters=0)

        assert np.array_equal(best_state, np.zeros(4))
        assert best_fitness == 0

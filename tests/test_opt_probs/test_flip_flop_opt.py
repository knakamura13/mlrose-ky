"""Unit tests for opt_probs/test_flip_flop_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky.opt_probs import FlipFlopOpt


class TestFlipFlopOpt:
    """Tests for FlipFlopOpt class."""

    def test_set_state(self):
        """Test set_state method"""
        problem = FlipFlopOpt(5)
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        assert np.array_equal(problem.get_state(), x)

    def test_set_population(self):
        """Test set_population method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        assert np.array_equal(problem.get_population(), pop)

    def test_best_child(self):
        """Test best_child method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        x = problem.best_child()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_best_neighbor(self):
        """Test best_neighbor method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.neighbors = pop
        x = problem.best_neighbor()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_evaluate_population_fitness(self):
        """Test evaluate_population_fitness method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        problem.evaluate_population_fitness()
        expected_fitness = np.array([1, 4, 1, 2, 0, 0])
        assert np.array_equal(problem.get_pop_fitness(), expected_fitness)

    def test_random_pop(self):
        """Test random_pop method"""
        problem = FlipFlopOpt(5)
        problem.random_pop(10)
        pop = problem.get_population()
        assert pop.shape == (10, 5) and np.all((pop == 0) | (pop == 1))

    def test_random_pop_edge_cases(self):
        """Test random_pop method with edge cases"""
        problem = FlipFlopOpt(5)
        # Test with population size 0
        try:
            problem.random_pop(0)
            assert False, "Expected an exception for population size 0"
        except Exception as e:
            assert str(e) == "pop_size must be a positive integer."

        # Test with a large population size
        problem.random_pop(10000)
        pop = problem.get_population()
        assert pop.shape == (10000, 5) and np.all((pop == 0) | (pop == 1))

    def test_set_state_boundary_conditions(self):
        """Test set_state method with boundary conditions"""
        problem = FlipFlopOpt(5)
        # Test with minimum state vector
        min_state = np.array([0, 0, 0, 0, 0])
        problem.set_state(min_state)
        assert np.array_equal(problem.get_state(), min_state)

        # Test with maximum state vector
        max_state = np.array([1, 1, 1, 1, 1])
        problem.set_state(max_state)
        assert np.array_equal(problem.get_state(), max_state)

    def test_invalid_inputs(self):
        """Test methods with invalid inputs"""
        problem_size, invalid_size = 5, 3
        problem = FlipFlopOpt(problem_size)
        invalid_state = np.ones((invalid_size,))
        try:
            problem.set_state(invalid_state)
            assert False, "Expected a ValueError exception for invalid state length"
        except ValueError as e:
            assert str(e) == f"new_state length {invalid_size} must match problem length {problem_size}"

        # Test random_pop with negative size
        try:
            problem.random_pop(-1)
            assert False, "Expected an exception for negative population size"
        except Exception as e:
            assert str(e) == "pop_size must be a positive integer."

    def test_can_stop_with_sub_optimal_state(self):
        """Test can_stop method given a sub-optimal state"""
        problem = FlipFlopOpt(5)
        problem.set_state(np.array([1, 1, 1, 1, 1]))
        assert not problem.can_stop()

    def test_can_stop_with_optimal_state(self):
        problem = FlipFlopOpt(5)
        problem.set_state(np.array([1, 0, 1, 0, 1]))
        assert problem.can_stop()

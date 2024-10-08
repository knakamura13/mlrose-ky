"""Unit tests for opt_probs/_opt_prob.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import re

import numpy as np
import pytest

from mlrose_ky.fitness import OneMax

# noinspection PyProtectedMember
from mlrose_ky.opt_probs._opt_prob import _OptProb


class TestOptProb:
    """Tests for _OptProb class."""

    def test_opt_prob_invalid_params(self):
        """Test initialization with invalid parameters"""
        with pytest.raises(ValueError, match=re.escape("Expected both length and fitness_fn to be defined.")):
            # noinspection PyTypeChecker
            _ = _OptProb(length=None, fitness_fn=None)
        with pytest.raises(ValueError, match=re.escape(f"fitness_fn must be an object with an evaluate attribute, got dict instead.")):
            # noinspection PyTypeChecker
            _ = _OptProb(length=1, fitness_fn={})
        with pytest.raises(TypeError, match="maximize must be a bool, got int instead."):
            # noinspection PyTypeChecker
            _ = _OptProb(length=1, fitness_fn=OneMax(), maximize=-1)

    def test_opt_prob_eval_fitness_invalid_state(self):
        """Test evaluate method with invalid state parameter."""
        state = [0, 1, 2, 3, 5]
        with pytest.raises(TypeError, match="Expected state to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _ = _OptProb(length=4, fitness_fn=OneMax()).eval_fitness(state)
        with pytest.raises(ValueError, match=f"State length 5 must match problem length 4."):
            # noinspection PyTypeChecker
            _ = _OptProb(length=4, fitness_fn=OneMax()).eval_fitness(np.asarray(state))

    def test_opt_prob_set_population_invalid_pop(self):
        """Test set_population method with invalid population parameter."""
        with pytest.raises(TypeError, match="Expected new_population to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _OptProb(length=4, fitness_fn=OneMax()).set_population([None, None, None, None])
        with pytest.raises(ValueError, match=f"new_population must contain at least one individual."):
            _OptProb(length=4, fitness_fn=OneMax()).set_population(np.array([]))

    def test_set_state_max(self):
        """Test set_state method for a maximization problem"""
        problem = _OptProb(5, OneMax())
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        assert np.array_equal(problem.get_state(), x) and problem.get_fitness() == 10

    def test_set_state_min(self):
        """Test set_state method for a minimization problem"""
        problem = _OptProb(5, OneMax(), maximize=False)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        assert np.array_equal(problem.get_state(), x) and problem.get_fitness() == -10

    def test_set_population_max(self):
        """Test set_population method for a maximization problem"""
        problem = _OptProb(5, OneMax())
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        pop_fit = np.array([1, 3, 4, 2, 100, 0, 5, -50])
        problem.set_population(pop)
        assert np.array_equal(problem.get_population(), pop) and np.array_equal(problem.get_pop_fitness(), pop_fit)

    def test_set_population_min(self):
        """Test set_population method for a minimization problem"""
        problem = _OptProb(5, OneMax(), maximize=False)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        pop_fit = -np.array([1, 3, 4, 2, 100, 0, 5, -50])
        problem.set_population(pop)
        assert np.array_equal(problem.get_population(), pop) and np.array_equal(problem.get_pop_fitness(), pop_fit)

    def test_best_child_max(self):
        """Test best_child method for a maximization problem"""
        problem = _OptProb(5, OneMax())
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        problem.set_population(pop)
        x = problem.best_child()
        assert np.array_equal(x, np.array([100, 0, 0, 0, 0]))

    def test_best_child_min(self):
        """Test best_child method for a minimization problem"""
        problem = _OptProb(5, OneMax(), maximize=False)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        problem.set_population(pop)
        x = problem.best_child()
        assert np.array_equal(x, np.array([0, 0, 0, 0, -50]))

    def test_best_neighbor_max(self):
        """Test best_neighbor method for a maximization problem"""
        problem = _OptProb(5, OneMax())
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        problem.neighbors = pop
        x = problem.best_neighbor()
        assert np.array_equal(x, np.array([100, 0, 0, 0, 0]))

    def test_best_neighbor_min(self):
        """Test best_neighbor method for a minimization problem"""
        problem = _OptProb(5, OneMax(), maximize=False)
        pop = np.array(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [100, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, -50],
            ]
        )
        problem.neighbors = pop
        x = problem.best_neighbor()
        assert np.array_equal(x, np.array([0, 0, 0, 0, -50]))

    def test_eval_fitness_max(self):
        """Test eval_fitness method for a maximization problem"""
        problem = _OptProb(5, OneMax())
        x = np.array([0, 1, 2, 3, 4])
        fitness = problem.eval_fitness(x)
        assert fitness == 10

    def test_eval_fitness_min(self):
        """Test eval_fitness method for a minimization problem"""
        problem = _OptProb(5, OneMax(), maximize=False)
        x = np.array([0, 1, 2, 3, 4])
        fitness = problem.eval_fitness(x)
        assert fitness == -10

    def test_eval_mate_probs(self):
        """Test eval_mate_probs method"""
        problem = _OptProb(5, OneMax())
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        problem.eval_mate_probs()
        probs = np.array([0.06667, 0.2, 0.26667, 0.13333, 0, 0.33333])
        assert np.allclose(problem.get_mate_probs(), probs, atol=0.00001)

    def test_eval_mate_probs_maximize_false(self):
        """Test eval_mate_probs method"""
        problem = _OptProb(5, OneMax(), maximize=False)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        problem.eval_mate_probs()
        probs = np.array([0.26667, 0.13333, 0.06667, 0.2, 0.33333, 0])
        assert np.allclose(problem.get_mate_probs(), probs, atol=0.00001)

    def test_eval_mate_probs_all_zero(self):
        """Test eval_mate_probs method when all states have zero fitness"""
        problem = _OptProb(5, OneMax())
        pop = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        problem.set_population(pop)
        problem.eval_mate_probs()
        probs = np.array([0.16667, 0.16667, 0.16667, 0.16667, 0.16667, 0.16667])
        assert np.allclose(problem.get_mate_probs(), probs, atol=0.00001)

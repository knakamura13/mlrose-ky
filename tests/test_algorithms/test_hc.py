"""Unit tests for algorithms/hc.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, OneMax, ContinuousOpt
from mlrose_ky.algorithms import hill_climb
from tests.globals import SEED


class TestHillClimb:
    """Unit tests for hill_climb."""

    def test_hill_climb_discrete_max(self):
        """Test hill_climb function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_hill_climb_continuous_max(self):
        """Test hill_climb function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_hill_climb_discrete_min(self):
        """Test hill_climb function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_hill_climb_continuous_min(self):
        """Test hill_climb function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_hill_climb_max_iters(self):
        """Test hill_climb function with max_iters less than infinite"""
        problem = DiscreteOpt(5, OneMax())
        x = np.zeros(5)
        best_state, best_fitness, _ = hill_climb(problem, max_iters=1, init_state=x, random_state=SEED)
        assert best_fitness == 1

"""Unit tests for algorithms/sa.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, OneMax, ContinuousOpt
from mlrose_ky.algorithms import simulated_annealing
from tests.globals import SEED


class TestSimulatedAnnealing:
    """Unit tests for simulated_annealing."""

    def test_simulated_annealing_discrete_max(self):
        """Test simulated_annealing function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_simulated_annealing_continuous_max(self):
        """Test simulated_annealing function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=20, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_simulated_annealing_discrete_min(self):
        """Test simulated_annealing function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_simulated_annealing_continuous_min(self):
        """Test simulated_annealing function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_simulated_annealing_max_iters(self):
        """Test simulated_annealing function with low max_iters"""
        problem = DiscreteOpt(5, OneMax())
        x = np.zeros(5)
        best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=1, max_iters=1, init_state=x, random_state=SEED)
        assert best_fitness == 1

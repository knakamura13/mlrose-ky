"""Unit tests for algorithms/ga.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, OneMax, ContinuousOpt
from mlrose_ky.algorithms import genetic_alg
from tests.globals import SEED


class TestGeneticAlg:
    """Unit tests for genetic_alg."""

    def test_genetic_alg_discrete_max(self):
        """Test genetic_alg function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_genetic_alg_continuous_max(self):
        """Test genetic_alg function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.ones(5)
        assert np.allclose(best_state, x, atol=0.5) and best_fitness > 4

    def test_genetic_alg_discrete_min(self):
        """Test genetic_alg function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_genetic_alg_continuous_min(self):
        """Test genetic_alg function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.allclose(best_state, x, atol=0.5) and best_fitness < 1

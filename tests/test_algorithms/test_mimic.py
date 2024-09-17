"""Unit tests for algorithms/mimic.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, OneMax
from mlrose_ky.algorithms import mimic
from tests.globals import SEED


class TestMimic:
    """Unit tests for mimic."""

    def test_mimic_discrete_max(self):
        """Test mimic function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = mimic(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_mimic_discrete_min(self):
        """Test mimic function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = mimic(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

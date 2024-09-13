"""Unit tests for algorithms/"""

# Authors: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np
from tests.globals import SEED

from mlrose_ky import OneMax, DiscreteOpt, ContinuousOpt, hill_climb, random_hill_climb, simulated_annealing, genetic_alg, mimic


def test_mimic_discrete_max():
    """Test mimic function for a discrete maximization problem"""
    problem = DiscreteOpt(5, OneMax())
    best_state, best_fitness, _ = mimic(problem, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_mimic_discrete_min():
    """Test mimic function for a discrete minimization problem"""
    problem = DiscreteOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = mimic(problem, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_hill_climb_discrete_max():
    """Test hill_climb function for a discrete maximization problem"""
    problem = DiscreteOpt(5, OneMax())
    best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_hill_climb_continuous_max():
    """Test hill_climb function for a continuous maximization problem"""
    problem = ContinuousOpt(5, OneMax())
    best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_hill_climb_discrete_min():
    """Test hill_climb function for a discrete minimization problem"""
    problem = DiscreteOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_hill_climb_continuous_min():
    """Test hill_climb function for a continuous minimization problem"""
    problem = ContinuousOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = hill_climb(problem, restarts=10, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_hill_climb_max_iters():
    """Test hill_climb function with max_iters less than infinite"""
    problem = DiscreteOpt(5, OneMax())
    x = np.zeros(5)
    best_state, best_fitness, _ = hill_climb(problem, max_iters=1, init_state=x, random_state=SEED)
    assert best_fitness == 1


def test_random_hill_climb_discrete_max():
    """Test random_hill_climb function for a discrete maximization problem"""
    problem = DiscreteOpt(5, OneMax())
    best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_random_hill_climb_continuous_max():
    """Test random_hill_climb function for a continuous maximization problem"""
    problem = ContinuousOpt(5, OneMax())
    best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_random_hill_climb_discrete_min():
    """Test random_hill_climb function for a discrete minimization problem"""
    problem = DiscreteOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_random_hill_climb_continuous_min():
    """Test random_hill_climb function for a continuous minimization problem"""
    problem = ContinuousOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_random_hill_climb_max_iters():
    """Test random_hill_climb function with low max_iters"""
    problem = DiscreteOpt(5, OneMax())
    x = np.zeros(5)
    best_state, best_fitness, _ = random_hill_climb(problem, max_attempts=1, max_iters=1, init_state=x, random_state=SEED)
    assert best_fitness == 1


def test_simulated_annealing_discrete_max():
    """Test simulated_annealing function for a discrete maximization problem"""
    problem = DiscreteOpt(5, OneMax())
    best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_simulated_annealing_continuous_max():
    """Test simulated_annealing function for a continuous maximization problem"""
    problem = ContinuousOpt(5, OneMax())
    best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=20, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_simulated_annealing_discrete_min():
    """Test simulated_annealing function for a discrete minimization problem"""
    problem = DiscreteOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_simulated_annealing_continuous_min():
    """Test simulated_annealing function for a continuous minimization problem"""
    problem = ContinuousOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_simulated_annealing_max_iters():
    """Test simulated_annealing function with low max_iters"""
    problem = DiscreteOpt(5, OneMax())
    x = np.zeros(5)
    best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=1, max_iters=1, init_state=x, random_state=SEED)
    assert best_fitness == 1


def test_genetic_alg_discrete_max():
    """Test genetic_alg function for a discrete maximization problem"""
    problem = DiscreteOpt(5, OneMax())
    best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
    x = np.ones(5)
    assert np.array_equal(best_state, x) and best_fitness == 5


def test_genetic_alg_continuous_max():
    """Test genetic_alg function for a continuous maximization problem"""
    problem = ContinuousOpt(5, OneMax())
    best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
    x = np.ones(5)
    assert np.allclose(best_state, x, atol=0.5) and best_fitness > 4


def test_genetic_alg_discrete_min():
    """Test genetic_alg function for a discrete minimization problem"""
    problem = DiscreteOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
    x = np.zeros(5)
    assert np.array_equal(best_state, x) and best_fitness == 0


def test_genetic_alg_continuous_min():
    """Test genetic_alg function for a continuous minimization problem"""
    problem = ContinuousOpt(5, OneMax(), maximize=False)
    best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
    x = np.zeros(5)
    assert np.allclose(best_state, x, atol=0.5) and best_fitness < 1

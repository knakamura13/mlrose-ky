"""Unit tests for opt_probs/continuous_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky.opt_probs import ContinuousOpt
from mlrose_ky.fitness import OneMax
from mlrose_ky.neural import NetworkWeights
from mlrose_ky.neural.activation import identity


class TestContinuousOpt:
    """Tests for ContinuousOpt class."""

    def test_calculate_updates(self):
        """Test calculate_updates method"""
        X = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])
        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])
        nodes = [4, 2, 1]
        fitness = NetworkWeights(X, y, nodes, activation=identity, bias=False, is_classifier=False, learning_rate=1)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b
        fitness.evaluate(np.asarray(weights))

        problem = ContinuousOpt(10, fitness, maximize=False)

        updates = list(problem.calculate_updates())
        update1 = np.array([[-0.0017, -0.0034], [-0.0046, -0.0092], [-0.0052, -0.0104], [0.0014, 0.0028]])
        update2 = np.array([[-3.17], [-4.18]])

        assert np.allclose(updates[0], update1, atol=0.001) and np.allclose(updates[1], update2, atol=0.001)

    def test_find_neighbors_range_eq_step(self):
        """Test find_neighbors method when range equals step size"""
        problem = ContinuousOpt(5, OneMax(), step=1)
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array([[1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1]])
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_find_neighbors_range_gt_step(self):
        """Test find_neighbors method when range greater than step size"""
        problem = ContinuousOpt(5, OneMax(), max_val=2, step=1)
        x = np.array([0, 1, 2, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array(
            [[1, 1, 2, 1, 0], [0, 0, 2, 1, 0], [0, 2, 2, 1, 0], [0, 1, 1, 1, 0], [0, 1, 2, 0, 0], [0, 1, 2, 2, 0], [0, 1, 2, 1, 1]]
        )
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_random(self):
        """Test random method"""
        problem = ContinuousOpt(5, OneMax(), max_val=4)
        rand = problem.random()
        assert len(rand) == 5 and max(rand) >= 0 and min(rand) <= 4

    def test_random_neighbor_range_eq_step(self):
        """Test random_neighbor method when range equals step size"""
        problem = ContinuousOpt(5, OneMax(), step=1)
        x = np.array([0, 0, 1, 1, 1])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        sum_diff = np.sum(np.abs(x - neigh))
        assert len(neigh) == 5 and sum_diff == 1

    def test_random_neighbor_range_gt_step(self):
        """Test random_neighbor method when range greater than step size"""
        problem = ContinuousOpt(5, OneMax(), max_val=2, step=1)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1
        sum_diff = np.sum(abs_diff)
        assert len(neigh) == 5 and sum_diff == 1

    def test_random_pop(self):
        """Test random_pop method"""
        problem = ContinuousOpt(5, OneMax(), step=1)
        problem.random_pop(100)
        pop = problem.get_population()
        pop_fitness = problem.get_pop_fitness()
        assert np.shape(pop)[0] == 100 and np.shape(pop)[1] == 5 and 0 < np.sum(pop) < 500 and len(pop_fitness) == 100

    def test_reproduce_mut0(self):
        """Test reproduce method when mutation_prob is 0"""
        problem = ContinuousOpt(5, OneMax(), step=1)
        father = np.zeros(5)
        mother = np.ones(5)
        child = problem.reproduce(father, mother, mutation_prob=0)
        assert len(child) == 5 and 0 < sum(child) < 5

    def test_reproduce_mut1_range_eq_step(self):
        """Test reproduce method when mutation_prob is 1 and range equals step size"""
        problem = ContinuousOpt(5, OneMax(), step=1)
        father = np.zeros(5)
        mother = np.ones(5)
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and 0 < sum(child) < 5

    def test_reproduce_mut1_range_gt_step(self):
        """Test reproduce method when mutation_prob is 1 and range is greater than step size"""
        problem = ContinuousOpt(5, OneMax(), max_val=2, step=1)
        father = np.zeros(5)
        mother = np.array([2, 2, 2, 2, 2])
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and 0 < sum(child) < 10

    def test_update_state_in_range(self):
        """Test update_state method where all updated values are within the tolerated range"""
        problem = ContinuousOpt(5, OneMax(), max_val=20, step=1)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        y = np.array([2, 4, 6, 8, 10])
        updated = problem.update_state(y)
        assert np.array_equal(updated, (x + y))

    def test_update_state_outside_range(self):
        """Test update_state method where some updated values are outside the tolerated range"""
        problem = ContinuousOpt(5, OneMax(), max_val=5, step=1)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        y = np.array([2, -4, 6, -8, 10])
        updated = problem.update_state(y)
        z = np.array([2, 0, 5, 0, 5])
        assert np.array_equal(updated, z)

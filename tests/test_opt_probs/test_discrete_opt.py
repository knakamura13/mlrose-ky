"""Unit tests for opt_probs/discrete_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

import mlrose_ky
from mlrose_ky.opt_probs import DiscreteOpt
from mlrose_ky.fitness import OneMax
from mlrose_ky.algorithms import OnePointCrossover


class TestDiscreteOpt:
    """Tests for DiscreteOpt class."""

    def test_eval_node_probs(self):
        """Test eval_node_probs method"""
        problem = DiscreteOpt(5, OneMax())
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.keep_sample = pop
        problem.eval_node_probs()
        parent = np.array([2, 0, 1, 0])
        probs = np.array(
            [
                [[0.33333, 0.66667], [0.33333, 0.66667]],
                [[1.0, 0.0], [0.33333, 0.66667]],
                [[1.0, 0.0], [0.25, 0.75]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.5], [0.25, 0.75]],
            ]
        )
        assert np.allclose(problem.node_probs, probs, atol=0.00001) and np.array_equal(problem.parent_nodes, parent)

    def test_find_neighbors_max2(self):
        """Test find_neighbors method when max_val is equal to 2"""
        problem = DiscreteOpt(5, OneMax())
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array([[1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1]])
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_find_neighbors_max_gt2(self):
        """Test find_neighbors method when max_val is greater than 2"""
        problem = DiscreteOpt(5, OneMax(), max_val=3)
        x = np.array([0, 1, 2, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array(
            [
                [1, 1, 2, 1, 0],
                [2, 1, 2, 1, 0],
                [0, 0, 2, 1, 0],
                [0, 2, 2, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 0, 0],
                [0, 1, 2, 2, 0],
                [0, 1, 2, 1, 1],
                [0, 1, 2, 1, 2],
            ]
        )
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_find_sample_order(self):
        """Test find_sample_order method"""
        problem = DiscreteOpt(5, OneMax())
        problem.parent_nodes = np.array([2, 0, 1, 0])
        order = np.array([0, 2, 4, 1, 3])
        problem.find_sample_order()
        assert np.array_equal(np.array(problem.sample_order), order)

    def test_find_top_pct_max(self):
        """Test find_top_pct method for a maximization problem"""
        problem = DiscreteOpt(5, OneMax())
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
        problem.find_top_pct(keep_pct=0.25)
        x = np.array([[100, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        assert np.array_equal(problem.get_keep_sample(), x)

    def test_find_top_pct_min(self):
        """Test find_top_pct method for a minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
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
        problem.find_top_pct(keep_pct=0.25)
        x = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, -50]])
        assert np.array_equal(problem.get_keep_sample(), x)

    def test_random(self):
        """Test random method"""
        problem = DiscreteOpt(5, OneMax(), max_val=5)
        rand = problem.random()
        assert len(rand) == 5 and max(rand) >= 0 and min(rand) <= 4

    def test_random_neighbor_max2(self):
        """Test random_neighbor method when max_val is equal to 2"""
        problem = DiscreteOpt(5, OneMax())
        x = np.array([0, 0, 1, 1, 1])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        sum_diff = np.sum(np.abs(x - neigh))
        assert len(neigh) == 5 and sum_diff == 1

    def test_random_neighbor_max_gt2(self):
        """Test random_neighbor method when max_val is greater than 2"""
        problem = DiscreteOpt(5, OneMax(), max_val=5)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1
        sum_diff = np.sum(abs_diff)
        assert len(neigh) == 5 and sum_diff == 1

    def test_random_pop(self):
        """Test random_pop method"""
        problem = DiscreteOpt(5, OneMax())
        problem.random_pop(100)
        pop = problem.get_population()
        pop_fitness = problem.get_pop_fitness()
        assert np.shape(pop)[0] == 100 and np.shape(pop)[1] == 5 and 0 < np.sum(pop) < 500 and len(pop_fitness) == 100

    def test_reproduce_mut0(self):
        """Test reproduce method when mutation_prob is 0"""
        problem = DiscreteOpt(5, OneMax())
        father = np.zeros(5)
        mother = np.ones(5)
        child = problem.reproduce(father, mother, mutation_prob=0)
        assert len(child) == 5 and 0 <= sum(child) <= 5

    def test_reproduce_mut1_max2(self):
        """Test reproduce method when mutation_prob is 1 and max_val is 2"""
        problem = DiscreteOpt(5, OneMax())
        father = np.zeros(5)
        mother = np.ones(5)
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and 0 <= sum(child) <= 5

    def test_reproduce_mut1_max_gt2(self):
        """Test reproduce method when mutation_prob is 1 and max_val is greater than 2"""
        problem = DiscreteOpt(5, OneMax(), max_val=3)
        problem._crossover = OnePointCrossover(problem)
        father = np.zeros(5)
        mother = np.ones(5) * 2
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and 0 < sum(child) < 10

    def test_sample_pop(self):
        """Test sample_pop method"""
        problem = DiscreteOpt(5, OneMax())
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.keep_sample = pop
        problem.eval_node_probs()
        sample = problem.sample_pop(100)
        assert np.shape(sample)[0] == 100 and np.shape(sample)[1] == 5 and 0 < np.sum(sample) < 500

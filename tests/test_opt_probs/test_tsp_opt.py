"""Unit tests for opt_probs/tsp_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky.opt_probs import TSPOpt


class TestTSPOpt:
    """Tests for TSPOpt class."""

    def test_adjust_probs_all_zero(self):
        """Test adjust_probs method when all elements in input vector sum to zero."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        probs = np.zeros(5)
        assert np.array_equal(problem.adjust_probs(probs), np.zeros(5))

    def test_adjust_probs_non_zero(self):
        """Test adjust_probs method when all elements in input vector sum to some value other than zero."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        probs = np.array([0.1, 0.2, 0, 0, 0.5])
        x = np.array([0.125, 0.25, 0, 0, 0.625])
        assert np.array_equal(problem.adjust_probs(probs), x)

    def test_find_neighbors(self):
        """Test find_neighbors method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array(
            [
                [1, 0, 2, 3, 4],
                [2, 1, 0, 3, 4],
                [3, 1, 2, 0, 4],
                [4, 1, 2, 3, 0],
                [0, 2, 1, 3, 4],
                [0, 3, 2, 1, 4],
                [0, 4, 2, 3, 1],
                [0, 1, 3, 2, 4],
                [0, 1, 4, 3, 2],
                [0, 1, 2, 4, 3],
            ]
        )
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_random(self):
        """Test random method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        rand = problem.random()
        assert len(rand) == 5 and len(set(rand)) == 5

    def test_random_mimic(self):
        """Test random_mimic method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        pop = np.array([[1, 0, 3, 2, 4], [0, 2, 1, 3, 4], [0, 2, 4, 3, 1], [4, 1, 3, 2, 0], [3, 4, 0, 2, 1], [2, 4, 0, 3, 1]])
        problem = TSPOpt(5, distances=dists)
        problem.keep_sample = pop
        problem.eval_node_probs()
        problem.find_sample_order()
        rand = problem.random_mimic()
        assert len(rand) == 5 and len(set(rand)) == 5

    def test_random_neighbor(self):
        """Test random_neighbor method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1
        sum_diff = np.sum(abs_diff)
        assert len(neigh) == 5 and sum_diff == 2 and len(set(neigh)) == 5

    def test_reproduce_mut0(self):
        """Test reproduce method when mutation_prob is 0"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        father = np.array([0, 1, 2, 3, 4])
        mother = np.array([0, 4, 3, 2, 1])
        child = problem.reproduce(father, mother, mutation_prob=0)
        assert len(child) == 5 and len(set(child)) == 5

    def test_reproduce_mut1(self):
        """Test reproduce method when mutation_prob is 1"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        father = np.array([0, 1, 2, 3, 4])
        mother = np.array([4, 3, 2, 1, 0])
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and len(set(child)) == 5

    def test_sample_pop(self):
        """Test sample_pop method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        pop = np.array([[1, 0, 3, 2, 4], [0, 2, 1, 3, 4], [0, 2, 4, 3, 1], [4, 1, 3, 2, 0], [3, 4, 0, 2, 1], [2, 4, 0, 3, 1]])
        problem = TSPOpt(5, distances=dists)
        problem.keep_sample = pop
        problem.eval_node_probs()
        sample = problem.sample_pop(100)
        row_sums = np.sum(sample, axis=1)
        assert np.shape(sample)[0] == 100 and np.shape(sample)[1] == 5 and max(row_sums) == 10 and min(row_sums) == 10

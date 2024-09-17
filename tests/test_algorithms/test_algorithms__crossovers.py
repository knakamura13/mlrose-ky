"""Unit tests for algorithms/crossovers/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

from unittest.mock import patch

import numpy as np
import pytest

from mlrose_ky.algorithms.crossovers import OnePointCrossOver, TSPCrossOver, UniformCrossOver

# noinspection PyProtectedMember
from mlrose_ky.algorithms.crossovers._crossover_base import _CrossOverBase


class MockOptProb:
    """Mock optimization problem class for `mock_opt_prob`."""

    def __init__(self, length):
        self.length = length


@pytest.fixture
def mock_opt_prob():
    """Fixture for creating a MockOptProb instance with default length."""

    def _create_problem(length):
        return MockOptProb(length=length)

    return _create_problem


class TestAlgorithmsCrossovers:
    """Test cases for the algorithms.crossovers module."""

    def test_crossover_base_mate_not_implemented(self, mock_opt_prob):
        """Test that calling mate on _CrossOverBase raises NotImplementedError."""

        class TestCrossOver(_CrossOverBase):
            def mate(self, p1, p2):
                return super().mate(p1, p2)

        opt_prob = mock_opt_prob(length=5)
        crossover = TestCrossOver(opt_prob)
        parent_1 = [1, 2, 3]
        parent_2 = [4, 5, 6]
        with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
            crossover.mate(parent_1, parent_2)

    def test_one_point_crossover(self, mock_opt_prob):
        """Test OnePointCrossOver with length > 1."""
        opt_prob = mock_opt_prob(length=5)
        crossover = OnePointCrossOver(opt_prob)
        p1 = [1, 2, 3, 4, 5]
        p2 = [5, 4, 3, 2, 1]
        with patch("numpy.random.randint", return_value=2):
            child = crossover.mate(p1, p2)
            crossover_point = 1 + 2
            expected_child = np.array(p1[:crossover_point] + p2[crossover_point:])
            assert np.array_equal(child, expected_child), "OnePointCrossOver failed."

    def test_one_point_crossover_length_eq1_raises(self, mock_opt_prob):
        """Test OnePointCrossOver with length == 1 raises ValueError."""
        opt_prob = mock_opt_prob(length=1)
        crossover = OnePointCrossOver(opt_prob)
        p1 = [1]
        p2 = [2]
        with pytest.raises(ValueError):
            crossover.mate(p1, p2)

    def test_uniform_crossover(self, mock_opt_prob):
        """Test UniformCrossOver with length > 1."""
        opt_prob = mock_opt_prob(length=5)
        crossover = UniformCrossOver(opt_prob)
        p1 = np.array([1, 2, 3, 4, 5])
        p2 = np.array([5, 4, 3, 2, 1])
        gene_selector = np.array([0, 1, 0, 1, 0])

        def mock_randint(low, high=None, size=None):
            if low == 2 and size == opt_prob.length:
                return gene_selector
            else:
                return np.random.randint(low, high=high, size=size)

        with patch("numpy.random.randint", side_effect=mock_randint):
            child = crossover.mate(p1, p2)
            expected_child = np.where(gene_selector == 0, p1, p2)
            assert np.array_equal(child, expected_child), "UniformCrossOver failed."

    def test_uniform_crossover_length_eq1(self, mock_opt_prob):
        """Test UniformCrossOver with length == 1."""
        opt_prob = mock_opt_prob(length=1)
        crossover = UniformCrossOver(opt_prob)
        p1 = np.array([1])
        p2 = np.array([2])
        gene_selector = np.array([1])

        def mock_randint(low, high=None, size=None):
            if low == 2 and size == opt_prob.length:
                return gene_selector
            else:
                return np.random.randint(low, high=high, size=size)

        with patch("numpy.random.randint", side_effect=mock_randint):
            child = crossover.mate(p1, p2)
            expected_child = np.where(gene_selector == 0, p1, p2)
            assert np.array_equal(child, expected_child), "UniformCrossOver with length 1 failed."

    def test_tsp_crossover_length_gt1(self, mock_opt_prob):
        """Test TSPCrossOver with length > 1."""
        opt_prob = mock_opt_prob(length=5)
        crossover = TSPCrossOver(opt_prob)
        p1 = [1, 2, 3, 4, 5]
        p2 = [5, 4, 3, 2, 1]
        with patch("numpy.random.randint", return_value=2):
            child = crossover.mate(p1, p2)
            n = 1 + 2
            unvisited = [city for city in p2 if city not in p1[:n]]
            expected_child = np.array(p1[:n] + unvisited)
            assert np.array_equal(child, expected_child), "TSPCrossOver failed."

    def test_tsp_crossover_length_eq1_random0(self, mock_opt_prob):
        """Test TSPCrossOver with length == 1 and random choice 0."""
        opt_prob = mock_opt_prob(length=1)
        crossover = TSPCrossOver(opt_prob)
        p1 = [1]
        p2 = [2]
        with patch("numpy.random.randint", return_value=0):
            child = crossover.mate(p1, p2)
            expected_child = np.array(p1)
            assert np.array_equal(child, expected_child), "TSPCrossOver with length 1 and random choice 0 failed."

    def test_tsp_crossover_length_eq1_random1(self, mock_opt_prob):
        """Test TSPCrossOver with length == 1 and random choice 1."""
        opt_prob = mock_opt_prob(length=1)
        crossover = TSPCrossOver(opt_prob)
        p1 = [1]
        p2 = [2]
        with patch("numpy.random.randint", return_value=1):
            child = crossover.mate(p1, p2)
            expected_child = np.array(p2)
            assert np.array_equal(child, expected_child), "TSPCrossOver with length 1 and random choice 1 failed."

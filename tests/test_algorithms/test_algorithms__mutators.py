"""Unit tests for algorithms/mutators/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

from unittest.mock import patch

import numpy as np
import pytest

from mlrose_ky.algorithms.mutators import ChangeOneMutator, DiscreteMutator, ShiftOneMutator, SwapMutator

# noinspection PyProtectedMember
from mlrose_ky.algorithms.mutators._mutator_base import _MutatorBase


class MockOptProb:
    """Mock optimization problem class for testing."""

    def __init__(self, length, max_val=2):
        self.length = length
        self.max_val = max_val


@pytest.fixture
def mock_opt_prob():
    """Fixture for creating a MockOptProb instance."""

    def _create_problem(length, max_val=2):
        return MockOptProb(length=length, max_val=max_val)

    return _create_problem


class TestAlgorithmsMutators:
    """Test cases for the algorithms.mutators module."""

    def test_mutator_base_mutate_not_implemented(self, mock_opt_prob):
        """Test that calling mutate on _MutatorBase raises NotImplementedError."""

        class TestMutator(_MutatorBase):
            def mutate(self, child, mutation_probability):
                return super().mutate(child, mutation_probability)

        opt_prob = mock_opt_prob(length=5)
        mutator = TestMutator(opt_prob)
        child = np.array([1, 2, 3])
        mutation_probability = 0.5
        with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
            mutator.mutate(child, mutation_probability)

    def test_change_one_mutator_mutate(self, mock_opt_prob):
        """Test ChangeOneMutator when mutation occurs."""
        opt_prob = mock_opt_prob(length=5, max_val=10)
        mutator = ChangeOneMutator(opt_prob)
        child = np.array([1, 2, 3, 4, 5])
        mutation_probability = 1.0  # Ensure mutation occurs

        with patch("numpy.random.rand", return_value=0.0), patch("numpy.random.randint", side_effect=[2, 7]):
            # First randint returns mutation_index=2, second returns new_value=7
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = child.copy()
            expected_child[2] = 7
            assert np.array_equal(mutated_child, expected_child), "ChangeOneMutator failed to mutate correctly."

    def test_change_one_mutator_no_mutation(self, mock_opt_prob):
        """Test ChangeOneMutator when no mutation occurs."""
        opt_prob = mock_opt_prob(length=5, max_val=10)
        mutator = ChangeOneMutator(opt_prob)
        child = np.array([1, 2, 3, 4, 5])
        mutation_probability = 0.0  # Ensure mutation does not occur

        with patch("numpy.random.rand", return_value=1.0):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            assert np.array_equal(mutated_child, child), "ChangeOneMutator should not have mutated the child."

    def test_change_one_mutator_invalid_probability(self, mock_opt_prob):
        """Test ChangeOneMutator with invalid mutation probability."""
        opt_prob = mock_opt_prob(length=5, max_val=10)
        mutator = ChangeOneMutator(opt_prob)
        child = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            mutator.mutate(child.copy(), -0.1)

        with pytest.raises(ValueError):
            mutator.mutate(child.copy(), 1.1)

    def test_discrete_mutator_binary(self, mock_opt_prob):
        """Test DiscreteMutator with binary genes."""
        opt_prob = mock_opt_prob(length=5, max_val=2)
        mutator = DiscreteMutator(opt_prob)
        child = np.array([0, 1, 0, 1, 0])
        mutation_probability = 1.0  # Ensure all genes are considered for mutation

        with patch("numpy.random.uniform", return_value=np.array([0.0, 0.1, 0.2, 0.3, 0.4])):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = 1 - child  # Binary inversion
            assert np.array_equal(mutated_child, expected_child), "DiscreteMutator failed for binary genes."

    def test_discrete_mutator_non_binary(self, mock_opt_prob):
        """Test DiscreteMutator with non-binary genes."""
        opt_prob = mock_opt_prob(length=5, max_val=4)
        mutator = DiscreteMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 0])
        mutation_probability = 1.0  # Ensure all genes are considered for mutation

        with (
            patch("numpy.random.uniform", return_value=np.array([0.0, 0.1, 0.2, 0.3, 0.4])),
            patch("numpy.random.choice", side_effect=[1, 2, 3, 0, 2]),
        ):
            # Mocking choices for each gene mutation
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = np.array([1, 2, 3, 0, 2])
            assert np.array_equal(mutated_child, expected_child), "DiscreteMutator failed for non-binary genes."

    def test_discrete_mutator_no_mutation(self, mock_opt_prob):
        """Test DiscreteMutator when no mutation occurs."""
        opt_prob = mock_opt_prob(length=5, max_val=4)
        mutator = DiscreteMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 0])
        mutation_probability = 0.0  # Ensure no mutation

        with patch("numpy.random.uniform", return_value=np.array([0.9, 0.9, 0.9, 0.9, 0.9])):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            assert np.array_equal(mutated_child, child), "DiscreteMutator should not have mutated the child."

    def test_shift_one_mutator_mutate_up(self, mock_opt_prob):
        """Test ShiftOneMutator when shifting up."""
        opt_prob = mock_opt_prob(length=5, max_val=5)
        mutator = ShiftOneMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 4])
        mutation_probability = 1.0  # Ensure mutation occurs

        with patch("numpy.random.rand", return_value=0.0), patch("numpy.random.randint", side_effect=[2, 0]):
            # mutation_index=2, shift_direction=1 (since randint returns 0)
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = child.copy()
            expected_child[2] = (expected_child[2] + 1) % opt_prob.max_val  # Shift up
            assert np.array_equal(mutated_child, expected_child), "ShiftOneMutator failed to shift up correctly."

    def test_shift_one_mutator_mutate_down(self, mock_opt_prob):
        """Test ShiftOneMutator when shifting down."""
        opt_prob = mock_opt_prob(length=5, max_val=5)
        mutator = ShiftOneMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 4])
        mutation_probability = 1.0  # Ensure mutation occurs

        with patch("numpy.random.rand", return_value=0.0), patch("numpy.random.randint", side_effect=[2, 1]):
            # mutation_index=2, shift_direction=-1 (since randint returns 1)
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = child.copy()
            expected_child[2] = (expected_child[2] - 1) % opt_prob.max_val  # Shift down
            assert np.array_equal(mutated_child, expected_child), "ShiftOneMutator failed to shift down correctly."

    def test_shift_one_mutator_no_mutation(self, mock_opt_prob):
        """Test ShiftOneMutator when no mutation occurs."""
        opt_prob = mock_opt_prob(length=5, max_val=5)
        mutator = ShiftOneMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 4])
        mutation_probability = 0.0  # Ensure no mutation occurs

        with patch("numpy.random.rand", return_value=1.0):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            assert np.array_equal(mutated_child, child), "ShiftOneMutator should not have mutated the child."

    def test_swap_mutator_mutate(self, mock_opt_prob):
        """Test SwapMutator when mutation occurs."""
        opt_prob = mock_opt_prob(length=5)
        mutator = SwapMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 4])
        mutation_probability = 1.0  # Ensure mutation occurs

        with patch("numpy.random.rand", return_value=0.0), patch("numpy.random.choice", return_value=np.array([1, 3])):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            expected_child = child.copy()
            # Swap positions 1 and 3
            expected_child[1], expected_child[3] = expected_child[3], expected_child[1]
            assert np.array_equal(mutated_child, expected_child), "SwapMutator failed to swap genes correctly."

    def test_swap_mutator_no_mutation(self, mock_opt_prob):
        """Test SwapMutator when no mutation occurs."""
        opt_prob = mock_opt_prob(length=5)
        mutator = SwapMutator(opt_prob)
        child = np.array([0, 1, 2, 3, 4])
        mutation_probability = 0.0  # Ensure no mutation occurs

        with patch("numpy.random.rand", return_value=1.0):
            mutated_child = mutator.mutate(child.copy(), mutation_probability)
            assert np.array_equal(mutated_child, child), "SwapMutator should not have mutated the child."

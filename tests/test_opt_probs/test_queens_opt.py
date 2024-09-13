"""Unit tests for opt_probs/queens_opt.py"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

import mlrose_ky
from mlrose_ky.opt_probs import QueensOpt


class TestQueensOpt:
    """Tests for QueensOpt class."""

    def test_initialization(self):
        """Test that the QueensOpt class is initialized correctly."""
        length = 8
        queens_opt = QueensOpt(length=length)

        assert queens_opt.length == length
        assert isinstance(queens_opt.fitness_fn, mlrose_ky.Queens)
        assert queens_opt.max_val == length
        assert queens_opt.stop_fitness == 0

    def test_state_initialization(self):
        """Test that the initial state is correctly set."""
        length = 8
        queens_opt = QueensOpt(length=length)
        state = queens_opt.get_state()

        assert len(state) == length
        assert np.all((0 <= state) & (state < length))

    def test_set_state(self):
        """Test that setting a new state updates fitness correctly."""
        queens_opt = QueensOpt(length=8)
        new_state = np.array([0, 6, 4, 7, 1, 3, 5, 2])
        queens_opt.set_state(new_state)

        assert np.array_equal(queens_opt.get_state(), new_state)
        assert queens_opt.get_fitness() == 0

    def test_can_stop(self):
        """Test the can_stop method works correctly."""
        queens_opt = QueensOpt(length=8)

        queens_opt.set_state(np.array([0, 0, 4, 7, 1, 3, 5, 2]))
        assert not queens_opt.can_stop()

        queens_opt.set_state(np.array([0, 6, 4, 7, 1, 3, 5, 2]))
        assert queens_opt.can_stop()

    def test_random_state_generation(self):
        """Test that random state generation produces valid states."""
        queens_opt = QueensOpt(length=8)
        random_state = queens_opt.random()

        assert len(random_state) == 8
        assert np.all((0 <= random_state) & (random_state < 8))

    def test_fitness_evaluation_suboptimal_state(self):
        """Test fitness evaluation for a sub-optimal state."""
        queens_opt = QueensOpt(length=8)
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        queens_opt.set_state(state)

        assert queens_opt.get_fitness() == -7

    def test_fitness_evaluation_optimal_state(self):
        """Test fitness evaluation for an optimal state."""
        queens_opt = QueensOpt(length=8)
        state = np.array([0, 6, 4, 7, 1, 3, 5, 2])
        queens_opt.set_state(state)

        assert queens_opt.get_fitness() == 0

    def test_minimization_mode(self):
        """Test QueensOpt in minimization mode."""
        queens_opt = QueensOpt(length=8)

        assert queens_opt.get_fitness() <= 0

    def test_maximization_mode(self):
        """Test QueensOpt in maximization mode."""
        queens_opt = QueensOpt(length=8, maximize=True)

        assert queens_opt.get_fitness() >= 0

    def test_edge_case_small_board(self):
        """Test edge case with the smallest possible board."""
        queens_opt = QueensOpt(length=1)
        queens_opt.set_state(np.array([0]))

        assert queens_opt.get_fitness() == 0

    def test_edge_case_two_queens(self):
        """Test edge case with a 2x2 board."""
        queens_opt = QueensOpt(length=2)

        queens_opt.set_state(np.array([0, 0]))
        assert queens_opt.get_fitness() == -1

        queens_opt.set_state(np.array([0, 1]))
        assert queens_opt.get_fitness() == -1

        queens_opt.set_state(np.array([1, 0]))
        assert queens_opt.get_fitness() == -1

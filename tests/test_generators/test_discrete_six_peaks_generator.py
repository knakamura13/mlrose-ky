"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

import mlrose_ky
from mlrose_ky import DiscreteOpt, SixPeaks
from mlrose_ky.generators import SixPeaksGenerator


# noinspection PyTypeChecker
class TestSixPeaksGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        with pytest.raises(ValueError) as excinfo:
            SixPeaksGenerator.generate(seed=SEED, size=0)
        assert str(excinfo.value) == "Size must be a positive integer. Got 0."

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        with pytest.raises(ValueError) as excinfo:
            SixPeaksGenerator.generate(seed=SEED, size=-5)
        assert str(excinfo.value) == "Size must be a positive integer. Got -5."

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        with pytest.raises(TypeError):
            SixPeaksGenerator.generate(seed=SEED, size="ten")

    def test_generate_negative_t_pct(self):
        """Test generate method raises ValueError when t_pct is less than 0."""
        with pytest.raises(ValueError) as excinfo:
            SixPeaksGenerator.generate(seed=SEED, size=10, t_pct=-0.1)
        assert str(excinfo.value) == "Threshold percentage must be between 0 and 1. Got -0.1."

    def test_generate_t_pct_greater_than_one(self):
        """Test generate method raises ValueError when t_pct is greater than 1."""
        with pytest.raises(ValueError) as excinfo:
            SixPeaksGenerator.generate(seed=SEED, size=10, t_pct=1.1)
        assert str(excinfo.value) == "Threshold percentage must be between 0 and 1. Got 1.1."

    def test_generate_non_float_t_pct(self):
        """Test generate method raises ValueError when t_pct is a non-float value."""
        with pytest.raises(TypeError):
            SixPeaksGenerator.generate(seed=SEED, size=10, t_pct="high")

    def test_generate_default_size(self):
        """Test generate method with default size."""
        problem = SixPeaksGenerator.generate(seed=SEED)

        assert problem.length == 20

    def test_generate_default_t_pct(self):
        """Test generate method with default t_pct."""
        size = 20
        problem = SixPeaksGenerator.generate(seed=SEED, size=size)

        assert problem.length == size
        assert problem.fitness_fn.t_pct == 0.1

    def test_generate_with_seed(self):
        """Test generate method with a specified SEED."""
        problem = SixPeaksGenerator.generate(seed=SEED)
        np.random.seed(seed=SEED)
        expected_fitness = SixPeaks()
        expected_problem = DiscreteOpt(length=20, fitness_fn=expected_fitness)

        assert problem.length == expected_problem.length
        assert problem.fitness_fn.__class__ == expected_problem.fitness_fn.__class__
        assert problem.fitness_fn.t_pct == expected_problem.fitness_fn.t_pct

    def test_generate_custom_size_and_threshold(self):
        """Test generate method with custom size and t_pct."""
        size = 30
        t_pct = 0.2
        problem = SixPeaksGenerator.generate(seed=SEED, size=size, t_pct=t_pct)

        assert problem.length == size
        assert problem.fitness_fn.t_pct == t_pct

"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

from mlrose_ky import DiscreteOpt, OneMax
from mlrose_ky.generators import OneMaxGenerator


# noinspection PyTypeChecker
class TestOneMaxGenerator:

    def test_generate_size_zero(self):
        """Test generate method raises ValueError when size is zero."""
        with pytest.raises(ValueError) as excinfo:
            OneMaxGenerator.generate(seed=SEED, size=0)
        assert str(excinfo.value) == "Size must be a positive integer. Got 0"

    def test_generate_negative_size(self):
        """Test generate method raises ValueError when size is negative."""
        with pytest.raises(ValueError) as excinfo:
            OneMaxGenerator.generate(seed=SEED, size=-5)
        assert str(excinfo.value) == "Size must be a positive integer. Got -5"

    def test_generate_non_integer_size(self):
        """Test generate method raises ValueError when size is a non-integer value."""
        with pytest.raises(ValueError) as excinfo:
            OneMaxGenerator.generate(seed=SEED, size="ten")
        assert str(excinfo.value) == "Size must be a positive integer. Got ten"

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when seed is not an integer."""
        with pytest.raises(ValueError) as excinfo:
            OneMaxGenerator.generate(seed="not_an_int")
        assert str(excinfo.value) == "Seed must be an integer. Got not_an_int"

    def test_generate_default_size(self):
        """Test generate method with default size."""
        problem = OneMaxGenerator.generate(seed=SEED)

        assert problem.length == 20

    def test_generate_with_seed(self):
        """Test generate method with a specified seed."""
        problem = OneMaxGenerator.generate(seed=SEED)
        np.random.seed(seed=SEED)
        expected_problem = DiscreteOpt(length=20, fitness_fn=OneMax())

        assert problem.length == expected_problem.length
        assert problem.__class__ == expected_problem.__class__

    def test_generate_custom_size(self):
        """Test generate method with custom size."""
        size = 30
        problem = OneMaxGenerator.generate(seed=SEED, size=size)

        assert problem.length == size

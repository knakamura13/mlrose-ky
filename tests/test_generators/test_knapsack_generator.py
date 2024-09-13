"""Unit tests for generators/"""

import pytest

from tests.globals import SEED

import mlrose_ky
from mlrose_ky.generators import KnapsackGenerator


# noinspection PyTypeChecker
class TestKnapsackGenerator:

    def test_generate_valid_case(self):
        """Test generate method with valid parameters."""
        number_of_item_types = 10
        max_item_count = 5
        max_weight_per_item = 25
        max_value_per_item = 10
        max_weight_percentage = 0.6
        multiply_by_max_item_count = True

        problem = KnapsackGenerator.generate(
            seed=SEED,
            number_of_item_types=number_of_item_types,
            max_item_count=max_item_count,
            max_weight_per_item=max_weight_per_item,
            max_value_per_item=max_value_per_item,
            max_weight_pct=max_weight_percentage,
            multiply_by_max_item_count=multiply_by_max_item_count,
        )

        assert problem.length == number_of_item_types
        assert problem.max_val == max_item_count

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when SEED is not an integer."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed="not_an_int")
        assert str(excinfo.value) == "Seed must be an integer. Got not_an_int"

    def test_generate_invalid_number_of_item_types(self):
        """Test generate method raises ValueError when number_of_item_types is invalid."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, number_of_item_types=0)
        assert str(excinfo.value) == "Number of item types must be a positive integer. Got 0"

    def test_generate_invalid_max_item_count(self):
        """Test generate method raises ValueError when max_item_count is invalid."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, max_item_count=-1)
        assert str(excinfo.value) == "Max item count must be a positive integer. Got -1"

    def test_generate_invalid_max_weight_per_item(self):
        """Test generate method raises ValueError when max_weight_per_item is invalid."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, max_weight_per_item=0)
        assert str(excinfo.value) == "Max weight per item must be a positive integer. Got 0"

    def test_generate_invalid_max_value_per_item(self):
        """Test generate method raises ValueError when max_value_per_item is invalid."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, max_value_per_item=-10)
        assert str(excinfo.value) == "Max value per item must be a positive integer. Got -10"

    def test_generate_default_parameters(self):
        """Test generate method with default parameters."""
        problem = KnapsackGenerator.generate(seed=SEED)

        assert problem.length == 10
        assert problem.max_val == 5

    def test_generate_invalid_max_weight_percentage(self):
        """Test generate method raises ValueError when max_weight_percentage is invalid."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, max_weight_pct=1.5)
        assert str(excinfo.value) == "Max weight percentage must be a float between 0 and 1. Got 1.5"

    def test_generate_invalid_multiply_by_max_item_count(self):
        """Test generate method raises ValueError when multiply_by_max_item_count is not a boolean."""
        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, multiply_by_max_item_count="yes")
        assert str(excinfo.value) == "multiply_by_max_item_count must be a boolean. Got yes"

    def test_generate_max_weight_percentage_zero(self):
        """Test generate method with max_weight_percentage set to 0"""
        max_weight_percentage = 0.0

        with pytest.raises(ValueError) as excinfo:
            KnapsackGenerator.generate(seed=SEED, max_weight_pct=max_weight_percentage)
        assert str(excinfo.value) == "max_weight_pct must be between 0 and 1."

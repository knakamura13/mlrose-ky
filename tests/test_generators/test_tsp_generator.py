"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

try:
    import mlrose_ky
except ImportError:
    import sys

    sys.path.append("../..")
    import mlrose_ky

from mlrose_ky.generators import TSPGenerator


# noinspection PyTypeChecker
class TestTSPGenerator:

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when seed is not an integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(seed="not_an_int", number_of_cities=5)
        assert str(excinfo.value) == "Seed must be an integer. Got not_an_int"

    def test_generate_invalid_number_of_cities(self):
        """Test generate method raises ValueError when number_of_cities is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(SEED, number_of_cities=0)
        assert str(excinfo.value) == "Number of cities must be a positive integer. Got 0"

    def test_generate_invalid_area_width(self):
        """Test generate method raises ValueError when area_width is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(SEED, number_of_cities=5, area_width=0)
        assert str(excinfo.value) == "Area width must be a positive integer. Got 0"

    def test_generate_invalid_area_height(self):
        """Test generate method raises ValueError when area_height is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(SEED, number_of_cities=5, area_height=0)
        assert str(excinfo.value) == "Area height must be a positive integer. Got 0"

    def test_generate_default_parameters(self):
        """Test generate method with default parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(seed=SEED, number_of_cities=num_cities)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_custom_parameters(self):
        """Test generate method with custom parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(seed=SEED, number_of_cities=num_cities, area_width=100, area_height=100)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_no_duplicate_coordinates(self):
        """Test generate method ensures no duplicate coordinates."""
        problem = TSPGenerator.generate(seed=SEED, number_of_cities=5)
        coords = problem.coords

        assert len(coords) == len(set(coords))

    def test_generate_distances(self):
        """Test generate method calculates distances correctly."""
        problem = TSPGenerator.generate(seed=SEED, number_of_cities=5)
        distances = problem.distances
        for u, v, d in distances:
            assert d == np.linalg.norm(np.subtract(problem.coords[u], problem.coords[v]))

    def test_generate_graph(self):
        """Test generate method creates a valid graph."""
        problem = TSPGenerator.generate(seed=SEED, number_of_cities=5)
        graph = problem.source_graph

        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == len(problem.distances)
"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

from mlrose_ky.generators import TSPGenerator


# noinspection PyTypeChecker
class TestTSPGenerator:

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when seed is not an integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(5, seed="not_an_int")

        assert str(excinfo.value) == "Seed must be an integer. Got not_an_int"

    def test_generate_invalid_number_of_cities(self):
        """Test generate method raises ValueError when number_of_cities is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(0, seed=SEED)

        assert str(excinfo.value) == "Number of cities must be a positive integer. Got 0"

    def test_generate_invalid_area_width(self):
        """Test generate method raises ValueError when area_width is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(5, area_width=0, seed=SEED)

        assert str(excinfo.value) == "Area width must be a positive integer. Got 0"

    def test_generate_invalid_area_height(self):
        """Test generate method raises ValueError when area_height is not a positive integer."""
        with pytest.raises(ValueError) as excinfo:
            TSPGenerator.generate(5, area_height=0, seed=SEED)

        assert str(excinfo.value) == "Area height must be a positive integer. Got 0"

    def test_generate_default_parameters(self):
        """Test generate method with default parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, seed=SEED)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_custom_parameters(self):
        """Test generate method with custom parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, area_width=100, area_height=100, seed=SEED)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_no_duplicate_coordinates(self):
        """Test generate method ensures no duplicate coordinates."""
        problem = TSPGenerator.generate(5, seed=SEED)
        coords = problem.coords

        assert len(coords) == len(set(coords))

    def test_generate_distances(self):
        """Test generate method calculates distances correctly."""
        problem = TSPGenerator.generate(5, seed=SEED)
        distances = problem.distances

        for u, v, d in distances:
            assert d == np.linalg.norm(np.subtract(problem.coords[u], problem.coords[v]))

    def test_generate_graph(self):
        """Test generate method creates a valid graph."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, seed=SEED)
        graph = problem.source_graph

        assert graph.number_of_nodes() == num_cities
        assert graph.number_of_edges() == len(problem.distances)

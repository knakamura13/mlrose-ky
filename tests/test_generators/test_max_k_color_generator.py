"""Unit tests for generators/"""

import pytest

from tests.globals import SEED

import mlrose_ky
from mlrose_ky.generators import MaxKColorGenerator


# noinspection PyTypeChecker
class TestMaxKColorGenerator:

    def test_generate_negative_max_colors(self):
        """Test generate method raises ValueError when max_colors is a negative integer."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_colors=-3)
        assert str(excinfo.value) == "Max colors must be a positive integer or None. Got -3"

    def test_generate_non_integer_max_colors(self):
        """Test generate method raises ValueError when max_colors is a non-integer value."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=3, max_colors="five")
        assert str(excinfo.value) == "Max colors must be a positive integer or None. Got five"

    def test_generate_seed_float(self):
        """Test generate method raises ValueError when SEED is a float."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=1.5)
        assert str(excinfo.value) == "Seed must be an integer. Got 1.5"

    def test_generate_float_number_of_nodes(self):
        """Test generate method raises ValueError when number_of_nodes is a float."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10.5)
        assert str(excinfo.value) == "Number of nodes must be a positive integer. Got 10.5"

    def test_generate_max_connections_per_node_float(self):
        """Test generate method raises ValueError when max_connections_per_node is a float."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=4.5)
        assert str(excinfo.value) == "Max connections per node must be a positive integer. Got 4.5"

    def test_generate_maximize_string(self):
        """Test generate method raises ValueError when maximize is a string."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, maximize="true")
        assert str(excinfo.value) == "Maximize must be a boolean. Got true"

    def test_generate_single_node_one_connection(self):
        """Test generate method with one node and up to one connection."""
        number_of_nodes = 1
        max_connections_per_node = 1
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 0

    def test_generate_single_node_two_connections(self):
        """Test generate method with one node and up to two connections."""
        number_of_nodes = 1
        max_connections_per_node = 2
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 0

    def test_generate_two_nodes_one_connection(self):
        """Test generate method with two nodes and up to one connection."""
        number_of_nodes = 2
        max_connections_per_node = 1
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 1

    def test_generate_two_nodes_two_connections(self):
        """Test generate method with two nodes and up to two connections."""
        number_of_nodes = 2
        max_connections_per_node = 2
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 1

    def test_generate_large_graph(self):
        """Test generate method with a large graph."""
        number_of_nodes = 100
        max_connections_per_node = 10
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() > 0

    def test_generate_no_edges(self):
        """Test generate method with no possible connections."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=0)
        assert str(excinfo.value) == "Max connections per node must be a positive integer. Got 0"

    def test_generate_max_colors_none(self):
        """Test generate method with max_colors set to None."""
        number_of_nodes = 5
        max_connections_per_node = 3
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.max_val > 1

    def test_generate_zero_nodes(self):
        """Test generate method raises ValueError when number_of_nodes is zero."""
        with pytest.raises(ValueError) as excinfo:
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=0)
        assert str(excinfo.value) == "Number of nodes must be a positive integer. Got 0"

    def test_generate_large_max_colors(self):
        """Test generate method with a large max_colors value."""
        number_of_nodes = 10
        max_connections_per_node = 3
        max_colors = 100
        problem = MaxKColorGenerator.generate(
            SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node, max_colors=max_colors
        )

        assert problem.length == number_of_nodes
        assert problem.max_val == max_colors

    def test_generate_large_max_connections(self):
        """Test generate method with large max_connections_per_node value."""
        number_of_nodes = 10
        max_connections_per_node = 9
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() > 0

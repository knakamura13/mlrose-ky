"""Class defining a Max-K Color optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import networkx as nx
import numpy as np

from mlrose_ky.opt_probs import MaxKColorOpt


class MaxKColorGenerator:
    """A class to generate Max-K Color optimization problems."""

    @staticmethod
    def generate(
        seed: int = 42, number_of_nodes: int = 20, max_connections_per_node: int = 4, max_colors: int = None, maximize: bool = False
    ) -> MaxKColorOpt:
        """
        Generate a Max-K Color optimization problem instance.

        Parameters
        ----------
        seed : int, optional, default=42
            Seed for the random number generator.
        number_of_nodes : int, optional, default=20
            The number of nodes in the graph.
        max_connections_per_node : int, optional, default=4
            The maximum number of connections (edges) per node.
        max_colors : int or None, optional, default=None
            The maximum number of colors available.
        maximize : bool, optional, default=False
            Whether the optimization problem should be maximized.

        Returns
        -------
        MaxKColorOpt
            An instance of MaxKColorOpt configured with the specified parameters.

        Raises
        ------
        ValueError
            If any parameter is not of the expected type or value.

        Examples
        --------
        >>> import mlrose_ky
        >>> test_edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_ky.MaxKColor(test_edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        3.0
        """
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer. Got {seed}")
        if not isinstance(number_of_nodes, int) or number_of_nodes <= 0:
            raise ValueError(f"Number of nodes must be a positive integer. Got {number_of_nodes}")
        if not isinstance(max_connections_per_node, int) or max_connections_per_node <= 0:
            raise ValueError(f"Max connections per node must be a positive integer. Got {max_connections_per_node}")
        if max_colors is not None and (not isinstance(max_colors, int) or max_colors <= 0):
            raise ValueError(f"Max colors must be a positive integer or None. Got {max_colors}")
        if not isinstance(maximize, bool):
            raise ValueError(f"Maximize must be a boolean. Got {maximize}")

        # Handle single-node edge case
        if number_of_nodes == 1:
            return MaxKColorOpt(edges=[], length=1, maximize=maximize, max_colors=max_colors)

        np.random.seed(seed)

        # Generate random connection counts for each node
        node_connection_counts = 1 + np.random.choice(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for node in nodes:
            valid_other_nodes = [
                other for other in nodes if (other != node and (other not in node_connections or node not in node_connections[other]))
            ]
            count = min(node_connection_counts[node], len(valid_other_nodes))
            connected_nodes = sorted(np.random.choice(valid_other_nodes, count, replace=False))
            node_connections[node] = [(node, other) for other in connected_nodes]

        # Ensure graph connectivity
        graph = nx.Graph()
        graph.add_edges_from([edge for edges in node_connections.values() for edge in edges])

        for node in nodes:
            unreachable = [
                (node, other) if node < other else (other, node) for other in nodes if other not in nx.bfs_tree(graph, node).nodes()
            ]
            for start, end in unreachable:
                graph.add_edge(start, end)
                remaining_unreachable = len(
                    [(node, other) if node < other else (other, node) for other in nodes if other not in nx.bfs_tree(graph, node).nodes()]
                )
                if remaining_unreachable == 0:
                    break

        return MaxKColorOpt(edges=list(graph.edges()), length=number_of_nodes, maximize=maximize, max_colors=max_colors, source_graph=graph)

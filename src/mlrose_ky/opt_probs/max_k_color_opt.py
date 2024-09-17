"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import networkx as nx
import numpy as np

from mlrose_ky.algorithms.crossovers import UniformCrossOver
from mlrose_ky.algorithms.mutators import ChangeOneMutator
from mlrose_ky.fitness import MaxKColor
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class MaxKColorOpt(DiscreteOpt):
    """Class for defining Max-K Color optimization problems.

    Parameters
    ----------
    edges : list[tuple[int, int]], default=None
        List of edges in the graph. Each edge is represented as a tuple of two node indices.

    length : int, default=None
        Number of elements in the state vector. If not specified, it is inferred from the edges or fitness function.

    fitness_fn : MaxKColor, default=None
        Object to implement the fitness function for optimization. If not provided, a new `MaxKColor` function will be created.

    maximize : bool, default=False
        Whether to maximize the fitness function. Set :code:`False` for minimization problems.

    max_colors : int, default=None
        Maximum number of colors to use for coloring the graph.

    crossover : UniformCrossOver, default=None
        Crossover operation used for reproduction. If None, defaults to `UniformCrossOver`.

    mutator : ChangeOneMutator, default=None
        Mutation operation used for reproduction. If None, defaults to `ChangeOneMutator`.

    source_graph : networkx.Graph, default=None
        Graph object representing the graph to be colored. If not provided, a new graph is created based on the edges.

    Attributes
    ----------
    length : int
        Number of elements in the state vector.

    fitness_fn : MaxKColor
        Fitness function for the optimization problem.

    max_val : int
        Maximum number of colors used in the state vector.

    source_graph : nx.Graph
        Graph representing the nodes and edges for the coloring problem.

    stop_fitness : int
        The fitness value at which the optimization process can stop.
    """

    def __init__(
        self,
        edges: list[tuple[int, int]] = None,
        length: int = None,
        fitness_fn: Any = None,
        maximize: bool = False,
        max_colors: int = None,
        crossover: "UniformCrossOver" = None,
        mutator: "ChangeOneMutator" = None,
        source_graph: nx.Graph = None,
    ):
        # Ensure that either fitness_fn or edges are provided
        if fitness_fn is None and edges is None:
            raise ValueError("Either fitness_fn or edges must be specified.")

        # If length is not provided, infer it from the edges or the fitness function
        if length is None:
            if fitness_fn is None:
                length = len(edges)  # Infer from the number of edges
            else:
                length = len(fitness_fn.weights)  # Infer from the fitness function

        self.length: int = length

        # If fitness function is not provided, create a MaxKColor fitness function
        if fitness_fn is None:
            fitness_fn = MaxKColor(edges, maximize)

        # Special handling for a single-node graph
        if length == 1:
            self.max_val: int = 1
            self.source_graph: nx.Graph = nx.Graph()  # Create an empty graph
            fitness_fn.set_graph(self.source_graph)
            self.stop_fitness: int = 0 if maximize else 0
            super().__init__(length, fitness_fn, maximize, 1, crossover, mutator)
            self.set_state(np.array([0]))
            return

        # Create or update the graph based on the provided edges or source_graph
        if source_graph is None:
            g = nx.Graph()
            g.add_edges_from(edges)
            # Ensure all nodes up to `length` are added to the graph
            for i in range(length):
                if i not in g:
                    g.add_node(i)
            self.source_graph = g
        else:
            self.source_graph = source_graph

        # Set stop fitness based on whether the problem is maximization or minimization
        self.stop_fitness: int = self.source_graph.number_of_edges() if maximize else 0

        fitness_fn.set_graph(self.source_graph)

        # Automatically calculate max_colors if not provided
        if max_colors is None:
            total_neighbor_count = [len(list(self.source_graph.neighbors(n))) for n in range(length)]
            if total_neighbor_count:
                max_colors = 1 + max(total_neighbor_count)  # Set to 1 + max neighbors
            else:
                max_colors = 1  # Default to 1 when there are no edges
        self.max_val: int = max_colors

        # Use default crossover and mutator if none are provided
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_colors, crossover, mutator)

        # Initialize the state with a shuffled random assignment of colors
        state = np.random.randint(max_colors, size=self.length)
        np.random.shuffle(state)
        self.set_state(state)

    def can_stop(self) -> bool:
        """Determine if the optimization process can stop.

        Returns
        -------
        bool
            True if the fitness value matches the stopping criterion, otherwise False.
        """
        return int(self.get_fitness()) == self.stop_fitness

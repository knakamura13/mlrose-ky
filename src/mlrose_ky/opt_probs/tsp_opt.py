"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np

from mlrose_ky.algorithms.crossovers import TSPCrossOver
from mlrose_ky.algorithms.mutators import SwapMutator
from mlrose_ky.fitness import TravellingSales
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class TSPOpt(DiscreteOpt):
    """Class for defining travelling salesperson optimization problems.

    Parameters
    ----------
    length : int, default=None
        Number of elements in state vector. Must equal the number of nodes in the tour.

    fitness_fn : TravellingSalesperson, default=None
        Object to implement the fitness function for optimization.
        If None, a `TravellingSalesperson` fitness function is created using `coords` or `distances`.

    maximize : bool, default=False
        Whether to maximize the fitness function. Set :code:`False` for minimization problems.

    coords : list[tuple], default=None
        Ordered list of the (x, y) coordinates of all nodes. Used to calculate distances if no `fitness_fn` is provided.

    distances : list[tuple[int, int, float]], default=None
        List giving the distances between pairs of nodes for which travel is possible.
        Each list item is in the form (u, v, d), representing nodes u and v, and the distance d.

    source_graph : networkx.Graph, default=None
        Optional graph object for representing the TSP problem structure.
    """

    def __init__(
        self,
        length: int = None,
        fitness_fn: Any = None,
        maximize: bool = False,
        coords: list[tuple] = None,
        distances: list[tuple] = None,
        source_graph: Any = None,
    ):
        # Ensure that at least one of fitness_fn, coords, or distances is provided
        if fitness_fn is None and coords is None and distances is None:
            raise ValueError("At least one of fitness_fn, coords, or distances must be specified.")

        # If fitness_fn is not provided, create a TravellingSalesperson fitness function
        if fitness_fn is None:
            fitness_fn = TravellingSales(coords=coords, distances=distances)

        self.distances: list[tuple] | None = distances
        self.coords: list[tuple] | None = coords

        # If length is not provided, infer it from coords or distances
        if length is None:
            if coords is not None:
                length = len(coords)
            elif distances is not None:
                # Calculate length from the set of nodes in the distances list
                length = len(set([x for (x, _, _) in distances] + [x for (_, x, _) in distances]))

        self.length: int = length

        # Initialize the parent class with the TSPCrossOver and SwapMutator
        super().__init__(length, fitness_fn, maximize, max_val=length, crossover=TSPCrossOver(self), mutator=SwapMutator(self))

        # Ensure that the fitness function type is 'tsp'
        if self.fitness_fn.get_prob_type() != "tsp":
            raise ValueError("fitness_fn must have problem type 'tsp'.")

        self.source_graph = source_graph
        self.prob_type = "tsp"

    @staticmethod
    def adjust_probs(probs: np.ndarray) -> np.ndarray:
        """Normalize a vector of probabilities so that the vector sums to 1.

        Parameters
        ----------
        probs : np.ndarray
            Vector of probabilities that may or may not sum to 1.

        Returns
        -------
        adj_probs : np.ndarray
            Vector of probabilities that sums to 1. Returns a zero vector if sum(probs) = 0.
        """
        sp = np.sum(probs)
        return np.zeros(np.shape(probs)) if sp == 0 else probs / sp

    def find_neighbors(self):
        """Find all neighbors of the current state."""
        self.neighbors = []

        for node1 in range(self.length - 1):
            for node2 in range(node1 + 1, self.length):
                neighbor = np.copy(self.state)

                # Swap the positions of node1 and node2
                neighbor[node1] = self.state[node2]
                neighbor[node2] = self.state[node1]
                self.neighbors.append(neighbor)

    def random(self) -> np.ndarray:
        """Return a random state vector.

        Returns
        -------
        np.ndarray
            Randomly generated state vector (a random permutation of nodes).
        """
        return np.random.permutation(self.length)

    def random_mimic(self) -> np.ndarray:
        """Generate single MIMIC sample from probability density.

        Returns
        -------
        np.ndarray
            State vector of MIMIC random sample.
        """
        remaining = list(np.arange(self.length))
        state = np.zeros(self.length, dtype=np.int8)
        node_probs = np.copy(self.node_probs)

        # Get value of the first element in the new sample
        state[0] = np.random.choice(self.length, p=node_probs[0, 0])
        remaining.remove(state[0])
        node_probs[:, :, state[0]] = 0

        # Get the sample order
        self.find_sample_order()
        sample_order = self.sample_order[1:]

        # Set values of remaining elements of state
        for i in sample_order:
            par_ind = self.parent_nodes[i - 1]
            par_value = state[par_ind]
            probs = node_probs[i, par_value]

            if np.sum(probs) == 0:
                next_node = np.random.choice(remaining)
            else:
                adj_probs = self.adjust_probs(probs)
                next_node = np.random.choice(self.length, p=adj_probs)

            state[i] = next_node
            remaining.remove(next_node)
            node_probs[:, :, next_node] = 0

        return state

    def random_neighbor(self) -> np.ndarray:
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor : np.ndarray
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        node1, node2 = np.random.choice(np.arange(self.length), size=2, replace=False)

        # Swap the positions of node1 and node2
        neighbor[node1] = self.state[node2]
        neighbor[node2] = self.state[node1]

        return neighbor

    def sample_pop(self, sample_size: int) -> np.ndarray:
        """Generate a new sample from the probability density.

        Parameters
        ----------
        sample_size : int
            Size of the sample to be generated.

        Returns
        -------
        new_sample : np.ndarray
            Numpy array containing the new sample.
        """
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer.")
        elif not isinstance(sample_size, int):
            if sample_size.is_integer():
                sample_size = int(sample_size)
            else:
                raise ValueError("sample_size must be a positive integer.")

        self.find_sample_order()
        new_sample = []

        # Generate MIMIC samples
        for _ in range(sample_size):
            state = self.random_mimic()
            new_sample.append(state)

        return np.array(new_sample)

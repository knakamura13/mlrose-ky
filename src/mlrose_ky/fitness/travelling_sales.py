"""Class defining the Travelling Salesperson fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable

import numpy as np


class TravellingSales:
    """
    Fitness function for the Travelling Salesperson optimization problem.

    Evaluates the fitness of a tour of n nodes, represented by state vector x, giving the order in which
    the nodes are visited, as the total distance travelled on the tour (including the distance travelled
    between the final node in the state vector and the first node in the state vector during the return
    leg of the tour). Each node must be visited exactly once for a tour to be considered valid.

    Parameters
    ----------
    coords : list[tuple[float, float]] | None, optional
        Ordered list of the (x, y) coordinates of all nodes (where element i gives the coordinates of node i).
        This assumes that travel between all pairs of nodes is possible. If this is not the case, then use
        distances instead.

    distances : list[tuple[int, int, float]] | None, optional
        List giving the distances, d, between all pairs of nodes, u and v, for which travel is possible, with each
        list item in the form (u, v, d). Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is assumed that travel between the two nodes
        is not possible. This argument is ignored if coords is not None.

    Examples
    --------
    >>> coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
    >>> dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
    >>> fitness_coords = TravellingSales(coords=coords)
    >>> state = np.array([0, 1, 4, 3, 2])
    >>> fitness_coords.evaluate(state)
    13.861384090800865

    >>> fitness_dists = TravellingSales(distances=dists)
    >>> fitness_dists.evaluate(state)
    29.0

    Note
    ----
    1. The TravellingSalesperson fitness function is suitable for use in travelling salesperson (tsp) optimization
       problems only.
    2. It is necessary to specify at least one of coords and distances in initializing a TravellingSalesperson
       fitness function object.
    """

    def __init__(self, coords: list[tuple] = None, distances: list[tuple] = None):
        # Ensure that at least one of coords or distances is provided
        if coords is None and distances is None:
            raise ValueError("At least one of coords and distances must be specified.")

        self.prob_type: str = "tsp"
        self.coords: list = coords
        self.distances: list = distances
        self.is_coords: bool = coords is not None

        # Determine which fitness calculation method to use
        self.calculate_fitness: Callable = self.__calculate_fitness_by_coords if self.is_coords else self.__calculate_fitness_by_distance

        if self.is_coords:
            # Precompute the coordinates array for faster access
            self.coords_array: np.ndarray = np.array(coords)
        else:
            # Remove duplicates and sort distances
            self.distances = list({tuple(sorted((u, v)) + [d]) for u, v, d in distances})

            # Unpack node lists and distances
            node1_list, node2_list, self.dist_list = zip(*self.distances)

            # Validation checks on distances
            if min(self.dist_list) <= 0:
                raise ValueError("The distance between each pair of nodes must be greater than 0.")
            if min(node1_list + node2_list) < 0:
                raise ValueError("The minimum node value must be 0.")
            if not max(node1_list + node2_list) == (len(set(node1_list + node2_list)) - 1):
                raise ValueError("All nodes must appear at least once in distances.")

            # Create a distance matrix for quick lookup of distances between nodes
            num_nodes = max(max(node1_list), max(node2_list)) + 1
            self.distance_matrix = np.full((num_nodes, num_nodes), np.inf)
            for u, v, d in self.distances:
                self.distance_matrix[u, v] = d
                self.distance_matrix[v, u] = d

    def __calculate_fitness_by_coords(self, state: np.ndarray) -> float:
        """
        Calculate fitness based on coordinates.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        fitness : float
            Calculated fitness value.
        """
        # Map state indices to coordinates
        nodes = self.coords_array[state]

        # Calculate total journey distance using Euclidean distance
        fitness = np.linalg.norm(nodes[1:] - nodes[:-1], axis=1).sum()
        fitness += np.linalg.norm(nodes[0] - nodes[-1])

        return float(fitness)

    def __calculate_fitness_by_distance(self, state: np.ndarray) -> float:
        """
        Calculate fitness based on distances.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        fitness : float
            Calculated fitness value. Returns np.inf if any segment of the tour is not possible.
        """
        fitness = 0.0
        num_nodes = len(state)

        # Iterate over each node in the state
        for i in range(num_nodes):
            start = state[i]
            end = state[(i + 1) % num_nodes]
            distance = self.distance_matrix[start, end]

            # Check if the segment is possible
            if np.isinf(distance):
                return np.inf

            fitness += distance

        return float(fitness)

    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation. Each integer between 0 and (len(state) - 1), inclusive must appear exactly once in the array.

        Returns
        -------
        fitness : float
            Value of fitness function. Returns np.inf if travel between two consecutive nodes on the tour is not possible.
        """
        # Validation checks on the state array
        if self.is_coords and len(state) != len(self.coords):
            raise ValueError("state must have the same length as coords.")
        if not len(state) == len(set(state)):
            raise ValueError("Each node must appear exactly once in state.")
        if min(state) < 0:
            raise ValueError("All elements of state must be non-negative integers.")
        if max(state) >= len(state):
            raise ValueError("All elements of state must be less than len(state).")

        # Calculate and return the fitness of the state
        return float(self.calculate_fitness(state))

    def get_prob_type(self) -> str:
        """
        Return the problem type.

        Returns
        -------
        prob_type : str
            Specifies problem type as 'tsp'.
        """
        return self.prob_type

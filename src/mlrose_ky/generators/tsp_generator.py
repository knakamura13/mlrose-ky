"""Class defining a Traveling Salesman Problem (TSP) optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import itertools as it
from collections import defaultdict

import networkx as nx
import numpy as np

from mlrose_ky import TSPOpt


class TSPGenerator:
    """A class to generate Traveling Salesman Problem (TSP) optimization problems."""

    @staticmethod
    def generate(number_of_cities: int, area_width: int = 250, area_height: int = 250, seed: int = 42) -> TSPOpt:
        """
        Generate a TSP optimization problem instance.

        Parameters
        ----------
        number_of_cities : int
            The number of cities (nodes) in the TSP.
        area_width : int, optional, default=250
            The width of the area in which cities are placed.
        area_height : int, optional, default=250
            The height of the area in which cities are placed.
        seed : int, optional, default=42
            Seed for the random number generator.

        Returns
        -------
        TSPOpt
            An instance of TSPOpt configured with the specified parameters.

        Raises
        ------
        ValueError
            If any of the parameters are not of the expected type or value.
        """
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer. Got {seed}")
        if not isinstance(number_of_cities, int) or number_of_cities <= 0:
            raise ValueError(f"Number of cities must be a positive integer. Got {number_of_cities}")
        if not isinstance(area_width, int) or area_width <= 0:
            raise ValueError(f"Area width must be a positive integer. Got {area_width}")
        if not isinstance(area_height, int) or area_height <= 0:
            raise ValueError(f"Area height must be a positive integer. Got {area_height}")

        np.random.seed(seed)

        # Generate random coordinates for cities
        x_coords = np.random.randint(area_width, size=number_of_cities)
        y_coords = np.random.randint(area_height, size=number_of_cities)

        coords = list(tuple(zip(x_coords, y_coords)))
        duplicates = TSPGenerator.list_duplicates_(coords)

        # Ensure no duplicate coordinates
        while len(duplicates) > 0:
            for d in duplicates:
                x_coords = np.random.randint(area_width, size=len(d))
                y_coords = np.random.randint(area_height, size=len(d))
                for i in range(len(d)):
                    coords[d[i]] = (x_coords[i], y_coords[i])
            duplicates = TSPGenerator.list_duplicates_(coords)

        # Calculate distances between all pairs of cities
        distances = TSPGenerator.get_distances(coords, truncate=False)

        # Create a graph with the calculated distances
        graph = nx.Graph()
        for a, b, distance in distances:
            graph.add_edge(a, b, length=int(round(distance)))

        return TSPOpt(coords=coords, distances=distances, source_graph=graph)

    @staticmethod
    def get_distances(coords: list[tuple], truncate: bool = True) -> list[tuple]:
        """
        Calculate the distances between all pairs of coordinates.

        Parameters
        ----------
        coords : list[tuple]
            A list of (x, y) coordinates representing the cities.
        truncate : bool, optional, default=True
            If True, truncate the distances to integers.

        Returns
        -------
        distances : list[tuple]
            A list of tuples representing the distance between each pair of cities.
        """
        # Calculate Euclidean distances between all pairs of coordinates
        distances = [
            (c1, c2, np.linalg.norm(np.subtract(coords[c1], coords[c2])))
            for c1, c2 in it.product(range(len(coords)), range(len(coords)))
            if c1 != c2 and c2 > c1
        ]

        if truncate:
            distances = [(c1, c2, int(d)) for c1, c2, d in distances]

        return distances

    @staticmethod
    def list_duplicates_(seq: list[tuple]) -> list[list[int]]:
        """
        Identify duplicate entries in the sequence.

        Parameters
        ----------
        seq : list[tuple]
            A list of (x, y) coordinates.

        Returns
        -------
        list of list of int
            A list containing lists of indices for duplicate coordinates.
        """
        # Create a tally of all coordinates
        tally = defaultdict(list)

        for i, item in enumerate(seq):
            tally[item].append(i)

        # Return indices of duplicate coordinates
        return [indices[1:] for _, indices in tally.items() if len(indices) > 1]

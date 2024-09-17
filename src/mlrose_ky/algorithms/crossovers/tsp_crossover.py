"""TSP Crossover implementation for Genetic Algorithms (GA).

This module defines a TSP-specific crossover operation used in genetic algorithms,
which handles the mating of parent solutions to produce offspring that respect the TSP
constraints.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Sequence

import numpy as np

from mlrose_ky.algorithms.crossovers._crossover_base import _CrossOverBase


class TSPCrossOver(_CrossOverBase):
    """
    Crossover operation tailored for the Travelling Salesperson Problem (TSP) in genetic algorithms.

    Implements specific crossover techniques that ensure valid TSP routes in the offspring.
    The crossover handles distinct city sequences without repetitions and uses specialized
    logic to combine parental genes.

    Inherits from:
    _CrossOverBase : Abstract base class for crossover operations.
    """

    def __init__(self, opt_prob: Any):
        """
        Initialize the TSPCrossOver with the given optimization problem.

        Parameters
        ----------
        opt_prob : Any
            An instance of the optimization problem related to the genetic algorithm.
        """
        super().__init__(opt_prob)

    def mate(self, p1: Sequence[int], p2: Sequence[int]) -> np.ndarray:
        """
        Perform the crossover (mating) between two parent sequences to produce offspring.

        Chooses between two internal methods to generate offspring based on TSP-specific
        constraints and optimizations.

        Parameters
        ----------
        p1 : Sequence[int]
            The first parent representing a TSP route.
        p2 : Sequence[int]
            The second parent representing a TSP route.

        Returns
        -------
        np.ndarray
            The offspring representing a new TSP route.
        """
        return self._mate_fill(p1, p2)

    def _mate_fill(self, p1: Sequence[int], p2: Sequence[int]) -> np.ndarray:
        """
        Perform a fill-based crossover using a segment of the first parent and filling
        the rest with non-repeated cities from the second parent.

        Parameters
        ----------
        p1 : Sequence[int]
            The first parent representing a TSP route.
        p2 : Sequence[int]
            The second parent representing a TSP route.

        Returns
        -------
        np.ndarray
            The offspring TSP route.
        """
        if self._length > 1:
            n = 1 + np.random.randint(self._length - 1)
            child = np.array([0] * self._length)
            child[:n] = p1[:n]
            unvisited = [city for city in p2 if city not in p1[:n]]
            child[n:] = unvisited
        else:
            child = np.copy(p1 if np.random.randint(2) == 0 else p2)

        return child

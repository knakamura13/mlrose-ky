"""One Point Crossover implementation for Genetic Algorithms (GA).

This module defines a one-point crossover operation used in genetic algorithms,
where a single crossover point is chosen randomly to combine two parent solutions.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Sequence

import numpy as np

from mlrose_ky.algorithms.crossovers._crossover_base import _CrossOverBase


class OnePointCrossOver(_CrossOverBase):
    """
    One-point crossover for genetic algorithms.

    This class implements a simple one-point crossover, where a single crossover
    point on the parent chromosomes is chosen randomly, and the genetic information
    is exchanged to create a new offspring.

    Inherits from:
    _CrossOverBase : Abstract base class for crossover operations.
    """

    def __init__(self, opt_prob: Any):
        """
        Initialize the OnePointCrossOver with the given optimization problem.

        Parameters
        ----------
        opt_prob : Any
            An instance of the optimization problem related to the genetic algorithm.
        """
        super().__init__(opt_prob)

    def mate(self, p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
        """
        Perform the one-point crossover between two parent sequences to produce offspring.

        A single crossover point is selected randomly from the chromosome. All genes before
        that point are copied from the first parent and the rest from the second parent.

        Parameters
        ----------
        p1 : Sequence[float]
            The first parent chromosome sequence.
        p2 : Sequence[float]
            The second parent chromosome sequence.

        Returns
        -------
        np.ndarray
            The offspring chromosome resulting from the crossover.
        """
        crossover_point = 1 + np.random.randint(self._length - 1)
        return np.array([*p1[:crossover_point], *p2[crossover_point:]])

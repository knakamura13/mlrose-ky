"""Uniform Crossover implementation for Genetic Algorithms (GA).

This module defines a uniform crossover operation used in genetic algorithms,
where each gene in the offspring is chosen randomly from one of the corresponding
genes of the parents.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Sequence

import numpy as np

from mlrose_ky.algorithms.crossovers._crossover_base import _CrossOverBase


class UniformCrossOver(_CrossOverBase):
    """
    Uniform crossover for genetic algorithms.

    This class implements a uniform crossover operation where each gene of the offspring
    is independently chosen from one of the two parents with equal probability. This
    method is often used when no prior knowledge about the problem structure is known.

    Inherits from:
    _CrossOverBase : Abstract base class for crossover operations.
    """

    def __init__(self, opt_prob: Any):
        """
        Initialize the UniformCrossOver with the given optimization problem.

        Parameters
        ----------
        opt_prob : Any
            An instance of the optimization problem related to the genetic algorithm.
        """
        super().__init__(opt_prob)

    def mate(self, p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
        """
        Perform the uniform crossover between two parent sequences to produce offspring.

        Each gene of the offspring is randomly chosen from one of the two parents'
        corresponding genes with a 50% chance for each.

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
        gene_selector = np.random.randint(2, size=self._length)
        stacked_parents = np.vstack((p1, p2))
        return stacked_parents[gene_selector, np.arange(self._length)]

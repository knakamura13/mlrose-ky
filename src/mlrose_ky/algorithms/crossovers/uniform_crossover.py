"""Uniform Crossover implementation for Genetic Algorithms (GA).

This module defines a uniform crossover operation used in genetic algorithms,
where each gene in the offspring is chosen randomly from one of the corresponding
genes of the parents.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Sequence

import numpy as np

from mlrose_ky.algorithms.crossovers._crossover_base import _CrossoverBase


class UniformCrossover(_CrossoverBase):
    """
    Uniform crossover for genetic algorithms.

    This class implements a uniform crossover operation where each gene of the offspring
    is independently chosen from one of the two parents with equal probability. This
    method is often used when no prior knowledge about the problem structure is known.

    Inherits from:
    _CrossoverBase : Abstract base class for crossover operations.
    """

    def __init__(self, optimization_problem: Any):
        """
        Initialize the UniformCrossover with the given optimization problem.

        Parameters
        ----------
        optimization_problem : Any
            An instance of the optimization problem related to the genetic algorithm.
        """
        super().__init__(optimization_problem)

    def mate(self, parent1: Sequence[float], parent2: Sequence[float]) -> np.ndarray:
        """
        Perform the uniform crossover between two parent sequences to produce offspring.

        Each gene of the offspring is randomly chosen from one of the two parents'
        corresponding genes with a 50% chance for each.

        Parameters
        ----------
        parent1 : Sequence[float]
            The first parent chromosome sequence.
        parent2 : Sequence[float]
            The second parent chromosome sequence.

        Returns
        -------
        np.ndarray
            The offspring chromosome resulting from the crossover.
        """
        gene_selector = np.random.randint(2, size=self._length)
        stacked_parents = np.vstack((parent1, parent2))
        return stacked_parents[gene_selector, np.arange(self._length)]

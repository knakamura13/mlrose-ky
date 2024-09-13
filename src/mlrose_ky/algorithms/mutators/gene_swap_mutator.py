"""Gene Swap Mutator for Genetic Algorithms (GA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np

from mlrose_ky.algorithms.mutators._mutator_base import _MutatorBase


class SwapMutator(_MutatorBase):
    """
    A mutator class that implements the 'Gene Swap' mutation strategy in genetic algorithms.

    This mutator randomly selects two genes in a chromosome and swaps their positions,
    thereby altering the genetic makeup of the offspring without introducing external genetic material.

    Attributes
    ----------
    _opt_prob : Any
        The optimization problem instance associated with the mutation operations.

    Parameters
    ----------
    opt_prob : OptimizationProblem
        An instance of an optimization problem that the mutator will operate on.
    """

    def __init__(self, opt_prob: Any):
        super().__init__(opt_prob)

    def mutate(self, child: np.ndarray, mutation_probability: float) -> np.ndarray:
        """
        Perform a gene swap mutation on the given chromosome if the mutation probability condition is met.

        Two genes are randomly selected and their positions are swapped. This mutation method helps maintain
        genetic diversity within the population.

        Parameters
        ----------
        child : np.ndarray
            The chromosome of a child individual to be mutated.
        mutation_probability : float
            The probability of the child undergoing a mutation.

        Returns
        -------
        np.ndarray
            The chromosome after mutation.
        """
        if np.random.rand() < mutation_probability:
            index_one, index_two = np.random.choice(len(child), size=2, replace=False)
            child[index_one], child[index_two] = child[index_two], child[index_one]

        return child

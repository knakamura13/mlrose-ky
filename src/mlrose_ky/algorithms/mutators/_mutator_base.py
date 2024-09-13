"""Mutation implementations for Genetic Algorithms (GA).

This module defines a base class for mutation operations used in genetic algorithms,
detailing how individual solutions (chromosomes) can be altered to introduce genetic diversity.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class _MutatorBase(ABC):
    """
    Base class for mutation operations in a genetic algorithm.

    This class provides a structured way to define mutation operations that
    can be applied to genetic algorithm populations.

    Attributes
    ----------
    _opt_prob : Any
        The optimization problem instance associated with the mutation operations.
    _length : int
        The length of the chromosome in the genetic algorithm, derived from the optimization problem.

    Parameters
    ----------
    opt_prob : Any
        An instance of an optimization problem that the mutator will operate on.
    """

    def __init__(self, opt_prob: Any):
        super().__init__()

        self._opt_prob: Any = opt_prob
        self._length: int = opt_prob.length

    @abstractmethod
    def mutate(self, child: np.ndarray, mutation_probability: float) -> Any:
        """
        Apply mutation operation to a given child chromosome based on a mutation probability.

        Parameters
        ----------
        child : np.ndarray
            The chromosome of a child individual to be mutated.
        mutation_probability : float
            The probability of each gene being mutated.

        Returns
        -------
        Any
            The mutated chromosome.

        Raises
        ------
        ValueError
            If the mutation_probability is not within the range [0, 1].
        """
        pass

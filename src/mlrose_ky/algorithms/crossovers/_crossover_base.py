"""Crossover implementations for Genetic Algorithms (GA).

This module defines a base class for crossover operations used in genetic algorithms,
detailing how two parent solutions can be combined to create offspring.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class _CrossOverBase(ABC):
    """
    Base class for crossover operations in a genetic algorithm.

    Provides a structured way to define crossover behavior in genetic algorithms.
    It should be subclassed to implement specific crossover strategies.

    Parameters
    ----------
    opt_prob : Any
        An instance of the optimization problem related to the genetic algorithm.
        This problem instance should provide necessary properties like 'length'
        that might be needed for the crossover operation.

    Attributes
    ----------
    _opt_prob : Any
        The optimization problem instance.
    _length : int
        Length of the chromosome, typically derived from the optimization problem's
        'length' property.
    """

    def __init__(self, opt_prob: Any):
        """
        Initialize the _CrossOverBase with the given optimization problem.

        Parameters
        ----------
        opt_prob : Any
            An instance of the optimization problem related to the GA.
        """
        super().__init__()
        self._opt_prob: Any = opt_prob
        self._length: int = opt_prob.length

    @abstractmethod
    def mate(self, p1: Sequence[int | float], p2: Sequence[int | float]) -> np.ndarray:
        """
        Perform the crossover (mating) between two parents to produce offspring.

        This method must be implemented by subclasses to define specific crossover
        behavior based on the genetics of the parents.
        """
        raise NotImplementedError("Subclasses must implement this method")

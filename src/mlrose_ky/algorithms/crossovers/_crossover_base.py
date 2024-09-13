"""Crossover implementations for Genetic Algorithms (GA).

This module defines a base class for crossover operations used in genetic algorithms,
detailing how two parent solutions can be combined to create offspring.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from abc import ABC, abstractmethod
from typing import Any


class _CrossoverBase(ABC):
    """
    Base class for crossover operations in a genetic algorithm.

    Provides a structured way to define crossover behavior in genetic algorithms.
    It should be subclassed to implement specific crossover strategies.

    Parameters
    ----------
    optimization_problem : Any
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

    def __init__(self, optimization_problem: Any):
        """
        Initialize the CrossoverBase with the given optimization problem.

        Parameters
        ----------
        optimization_problem : Any
            An instance of the optimization problem related to the GA.
        """
        super().__init__()
        self._opt_prob: Any = optimization_problem
        self._length: int = optimization_problem.length

    @abstractmethod
    def mate(self, parent1: Any, parent2: Any) -> Any:
        """
        Perform the crossover (mating) between two parents to produce offspring.

        This method must be implemented by subclasses to define specific crossover
        behavior based on the genetics of the parents.

        Parameters
        ----------
        parent1 : Any
            The first parent participating in the crossover.
        parent2 : Any
            The second parent participating in the crossover.

        Returns
        -------
        Any
            The offspring resulting from the crossover. The type of this result
            can vary depending on the specific GA implementation.
        """
        pass

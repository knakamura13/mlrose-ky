"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np

from mlrose_ky.algorithms.crossovers import OnePointCrossOver
from mlrose_ky.algorithms.mutators import ChangeOneMutator
from mlrose_ky.fitness import FlipFlop
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class FlipFlopOpt(DiscreteOpt):
    """Class for defining FlipFlop optimization problems.

    Parameters
    ----------
    length : int, default=None
        Number of elements in the state vector. If not specified, it is inferred from the fitness function.

    fitness_fn : Any, default=None
        Object to implement the fitness function for optimization. If not specified, defaults to the `FlipFlop` fitness function.

    maximize : bool, default=True
        Whether to maximize the fitness function. Set :code:`False` for minimization problems.

    crossover : OnePointCrossOver, default=None
        Crossover operation used for reproduction. If None, defaults to `OnePointCrossOver`.

    mutator : ChangeOneMutator, default=None
        Mutation operation used for reproduction. If None, defaults to `ChangeOneMutator`.

    Attributes
    ----------
    length : int
        Number of elements in the state vector.

    fitness_fn : FlipFlop
        Fitness function for the optimization problem.

    population : np.ndarray
        Array containing the current population.

    pop_fitness : np.ndarray
        Array containing the fitness values for the current population.

    max_val : int
        Number of unique values that each element in the state vector can take (always 2 for FlipFlopOpt).
    """

    def __init__(
        self,
        length: int = None,
        fitness_fn: Any = None,
        maximize: bool = True,
        crossover: "OnePointCrossOver" = None,
        mutator: "ChangeOneMutator" = None,
    ):
        if (fitness_fn is None) and (length is None):
            raise ValueError("fitness_fn or length must be specified.")

        if length is None:
            length = len(fitness_fn.weights)

        self.length: int = length

        if fitness_fn is None:
            fitness_fn = FlipFlop()

        self.max_val: int = 2
        crossover = OnePointCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator

        super().__init__(length, fitness_fn, maximize, crossover=crossover, mutator=mutator)

        # Set initial state
        state = np.random.randint(2, size=self.length)
        self.set_state(state)

    def evaluate_population_fitness(self):
        """Calculate fitness for the current population."""
        self.pop_fitness = self.fitness_fn.evaluate_many(self.population)

    def random_pop(self, pop_size: int):
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size : int
            Size of population to be created.

        Raises
        ------
        ValueError
            If pop_size is not a positive integer.
        """
        if pop_size <= 0:
            raise ValueError("pop_size must be a positive integer.")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise ValueError("pop_size must be a positive integer.")

        # Generate random population
        population = np.random.rand(pop_size, self.length)
        population[population < 0.5] = 0
        population[population >= 0.5] = 1
        self.population = population

        # Evaluate fitness for the population
        self.evaluate_population_fitness()

    def can_stop(self) -> bool:
        """Determine if the optimization process can stop.

        Returns
        -------
        bool
            True if the fitness equals the length of the state vector minus 1, otherwise False.
        """
        return int(self.get_fitness()) == int(self.length - 1)

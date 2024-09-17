"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np

from mlrose_ky.algorithms.crossovers import UniformCrossOver
from mlrose_ky.algorithms.mutators import ChangeOneMutator
from mlrose_ky.fitness.queens import Queens
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class QueensOpt(DiscreteOpt):
    """Class for defining the N-Queens optimization problem.

    Parameters
    ----------
    length : int, default=None
        Number of queens (and the size of the board). If not provided, it is inferred from the fitness function.

    fitness_fn : Queens, default=None
        Object to implement the fitness function for optimization. If not provided, a new `Queens` function will be created.

    maximize : bool, default=False
        Whether to maximize the fitness function. Set :code:`False` for minimization problems.

    crossover : UniformCrossOver, default=None
        Crossover operation used for reproduction. If None, defaults to `UniformCrossOver`.

    mutator : ChangeOneMutator, default=None
        Mutation operation used for reproduction. If None, defaults to `ChangeOneMutator`.

    Attributes
    ----------
    length : int
        Number of queens and size of the board.

    fitness_fn : Queens
        Fitness function for the optimization problem.

    max_val : int
        Maximum value for each element in the state vector (equals to the length).

    stop_fitness : int
        The fitness value at which the optimization process can stop.
    """

    def __init__(
        self,
        length: int = None,
        fitness_fn: Any = None,
        maximize: bool = False,
        crossover: "UniformCrossOver" = None,
        mutator: "ChangeOneMutator" = None,
    ):
        # Ensure that either fitness_fn or length is provided
        if fitness_fn is None and length is None:
            raise ValueError("Either fitness_fn or length must be specified.")

        # Infer length from fitness_fn if not provided
        if length is None:
            length = len(fitness_fn.weights)

        self.length: int = length

        # If fitness_fn is not provided, create a new Queens fitness function
        if fitness_fn is None:
            fitness_fn = Queens(maximize=maximize)

        # Set the stopping fitness value based on whether we're maximizing or minimizing
        self.stop_fitness: int = Queens.get_max_size(length) if maximize else 0

        # Set max_val to length, as it represents the board size
        self.max_val: int = length

        # Use default crossover and mutator if none are provided
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, length, crossover, mutator)

        # Initialize the state with a random, shuffled assignment of queens
        state = np.random.randint(self.length, size=self.length)
        np.random.shuffle(state)
        self.set_state(state)

    def can_stop(self) -> bool:
        """Determine if the optimization process can stop.

        Returns
        -------
        bool
            True if the fitness value matches the stopping criterion, otherwise False.
        """
        return int(self.get_fitness()) == self.stop_fitness

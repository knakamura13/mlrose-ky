"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

from mlrose_ky.algorithms.crossovers import UniformCrossOver
from mlrose_ky.algorithms.mutators import ChangeOneMutator
from mlrose_ky.fitness.knapsack import Knapsack
from mlrose_ky.opt_probs.discrete_opt import DiscreteOpt


class KnapsackOpt(DiscreteOpt):
    """Class for defining Knapsack optimization problems.

    Parameters
    ----------
    length : int, default=None
        Number of elements in the state vector. If not specified, it is inferred from the weights, values, or fitness function.

    fitness_fn : Knapsack, default=None
        Object to implement the fitness function for optimization. If not specified, a new `Knapsack` fitness function will be created.

    maximize : bool, default=True
        Whether to maximize the fitness function. Set :code:`False` for minimization problems.

    max_val : int, default=2
        Maximum number of items that can be selected for each element in the state vector.

    weights : list[float] | None, default=None
        List of weights for each item. Required if `fitness_fn` is not provided.

    values : list[float] | None, default=None
        List of values for each item. Required if `fitness_fn` is not provided.

    max_weight_pct : float, default=0.35
        Maximum allowable weight as a percentage of the total weight.

    crossover : UniformCrossOver, default=None
        Crossover operation used for reproduction. If None, defaults to `UniformCrossOver`.

    mutator : ChangeOneMutator, default=None
        Mutation operation used for reproduction. If None, defaults to `ChangeOneMutator`.

    multiply_by_max_item_count : bool, default=False
        Whether to multiply the values by the maximum item count allowed.

    Attributes
    ----------
    length : int
        Number of elements in the state vector.

    fitness_fn : Knapsack
        Fitness function for the optimization problem.

    max_val : int
        Maximum number of items that can be selected for each element in the state vector.
    """

    def __init__(
        self,
        length: int = None,
        fitness_fn: Any = None,
        maximize: bool = True,
        max_val: int = 2,
        weights: list[float] = None,
        values: list[float] = None,
        max_weight_pct: float = 0.35,
        crossover: "UniformCrossOver" = None,
        mutator: "ChangeOneMutator" = None,
        multiply_by_max_item_count: bool = False,
    ):
        if fitness_fn is None and (weights is None and values is None):
            raise ValueError("Either fitness_fn or both weights and values must be specified.")

        if length is None:
            if weights is not None:
                length = len(weights)
            elif values is not None:
                length = len(values)
            elif fitness_fn is not None:
                length = len(fitness_fn.weights)

        self.length: int = length
        self.max_val: int = max_val

        if fitness_fn is None:
            fitness_fn = Knapsack(
                weights=weights,
                values=values,
                max_weight_pct=max_weight_pct,
                max_item_count=max_val,
                multiply_by_max_item_count=multiply_by_max_item_count,
            )

        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator

        super().__init__(length, fitness_fn, maximize, max_val, crossover, mutator)

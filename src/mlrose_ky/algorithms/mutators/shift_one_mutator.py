"""Single Shift (Shift One) Mutator for Genetic Algorithms (GA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np

from mlrose_ky.algorithms.mutators._mutator_base import _MutatorBase


class ShiftOneMutator(_MutatorBase):
    """
    A mutator class that implements the 'Shift One' mutation strategy in a genetic algorithm.

    This mutator adjusts a single gene in the chromosome either up or down by one,
    wrapping around the maximum value constraint to ensure all values remain valid.

    Attributes
    ----------
    _opt_prob : Any
        The optimization problem instance associated with the mutation operations.
    _max_val : int
        The maximum allowable value for any gene in the chromosome, used to enforce wrap-around.

    Parameters
    ----------
    opt_prob : Any
        An instance of an optimization problem that the mutator will operate on.
    """

    def __init__(self, opt_prob: Any):
        super().__init__(opt_prob)

        self._max_val: int = opt_prob.max_val

    def mutate(self, child: np.ndarray, mutation_probability: float) -> np.ndarray:
        """
        Apply a 'Shift One' mutation to a randomly selected gene in the chromosome if the mutation
        probability condition is met. The selected gene's value is incremented or decremented by one,
        with wrap-around handling to stay within permissible value ranges.

        Parameters
        ----------
        child : np.ndarray
            The chromosome of a child individual to be mutated.
        mutation_probability : float
            The probability of the child undergoing a mutation.

        Returns
        -------
        np.ndarray
            The mutated chromosome.
        """
        if np.random.rand() < mutation_probability:
            mutation_index = np.random.randint(len(child))
            shift_direction = 1 if np.random.randint(2) == 0 else -1
            new_value = (child[mutation_index] + shift_direction) % self._max_val
            child[mutation_index] = new_value

        return child

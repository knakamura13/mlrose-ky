"""Class defining a One Max optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, OneMax


class OneMaxGenerator:
    """A class to generate One Max optimization problems."""

    @staticmethod
    def generate(seed: int = 42, size: int = 20) -> DiscreteOpt:
        """
        Generate a One Max optimization problem instance.

        Parameters
        ----------
        seed : int, optional, default=42
            Seed for the random number generator.
        size : int, optional, default=20
            The size of the optimization problem (number of bits).

        Returns
        -------
        DiscreteOpt
            An instance of DiscreteOpt configured for the One Max problem.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        """
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer. Got {seed}")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Size must be a positive integer. Got {size}")

        np.random.seed(seed)

        return DiscreteOpt(length=size, fitness_fn=OneMax())

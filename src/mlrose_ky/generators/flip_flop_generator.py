"""Class defining a FlipFlop optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky import FlipFlopOpt


class FlipFlopGenerator:
    """A class to generate FlipFlop optimization problem instances."""

    @staticmethod
    def generate(seed: int = 42, size: int = 20) -> FlipFlopOpt:
        """
        Generate a FlipFlop optimization problem with a given seed and size.

        Parameters
        ----------
        seed : int, optional, default=42
            The seed for the random number generator.
        size : int, optional, default=20
            The size of the problem.

        Returns
        -------
        FlipFlopOpt
            An instance of FlipFlopOpt configured with the specified seed and size.

        Raises
        ------
        ValueError
            If the `size` is not a positive integer.
        """
        if size <= 0:
            raise ValueError(f"Size must be a positive integer. Got {size}.")

        np.random.seed(seed)

        return FlipFlopOpt(length=size)

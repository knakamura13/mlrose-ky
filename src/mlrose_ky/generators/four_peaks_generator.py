"""Class defining a Four Peaks optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky import DiscreteOpt, FourPeaks


class FourPeaksGenerator:
    """A class to generate Four Peaks optimization problem instances."""

    @staticmethod
    def generate(seed: int = 42, size: int = 20, t_pct: float = 0.1) -> DiscreteOpt:
        """
        Generate a Four Peaks optimization problem with a given seed, size, and threshold percentage.

        Parameters
        ----------
        seed : int, optional, default=42
            The seed for the random number generator.
        size : int, optional, default=20
            The size of the problem.
        t_pct : float, optional, default=0.1
            The threshold percentage for the Four Peaks problem.

        Returns
        -------
        DiscreteOpt
            An instance of DiscreteOpt configured with the specified parameters.

        Raises
        ------
        ValueError
            If the size is not a positive integer or if the `t_pct` is not between 0 and 1.
        """
        if size <= 0:
            raise ValueError(f"Size must be a positive integer. Got {size}.")
        if not (0 <= t_pct <= 1):
            raise ValueError(f"Threshold percentage must be between 0 and 1. Got {t_pct}.")

        np.random.seed(seed)

        fitness = FourPeaks(t_pct=t_pct)
        return DiscreteOpt(length=size, fitness_fn=fitness)

"""Class defining an N-Queens optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky import QueensOpt


class QueensGenerator:
    """A class to generate N-Queens optimization problems."""

    @staticmethod
    def generate(seed: int = 42, size: int = 20, maximize: bool = False) -> QueensOpt:
        """
        Generate an N-Queens optimization problem instance.

        Parameters
        ----------
        seed : int, optional, default=42
            Seed for the random number generator.
        size : int, optional, default=20
            The size of the board (number of queens).
        maximize : bool, optional, default=False
            Whether the optimization problem should be maximized.

        Returns
        -------
        QueensOpt
            An instance of QueensOpt configured with the specified parameters.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        """
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer. Got {seed}")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Size must be a positive integer. Got {size}")
        if not isinstance(maximize, bool):
            raise ValueError(f"Maximize must be a boolean. Got {maximize}")

        np.random.seed(seed)

        return QueensOpt(length=size, maximize=maximize)

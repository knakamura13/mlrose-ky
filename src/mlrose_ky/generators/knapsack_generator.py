"""Class defining a Knapsack optimization problem generator."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky import KnapsackOpt


class KnapsackGenerator:
    """A class to generate Knapsack optimization problems."""

    @staticmethod
    def generate(
        seed: int = 42,
        number_of_item_types: int = 10,
        max_item_count: int = 5,
        max_weight_per_item: int = 25,
        max_value_per_item: int = 10,
        max_weight_pct: float = 0.6,
        multiply_by_max_item_count: bool = True,
    ) -> KnapsackOpt:
        """
        Generate a Knapsack optimization problem instance.

        Parameters
        ----------
        seed : int, optional, default=42
            Seed for the random number generator.
        number_of_item_types : int, optional, default=10
            Number of different item types.
        max_item_count : int, optional, default=5
            Maximum count for each item type.
        max_weight_per_item : int, optional, default=25
            Maximum weight for each item type.
        max_value_per_item : int, optional, default=10
            Maximum value for each item type.
        max_weight_pct : float, optional, default=0.6
            Maximum weight percentage of the knapsack.
        multiply_by_max_item_count : bool, optional, default=True
            If True, multiply weights and values by max_item_count.

        Returns
        -------
        KnapsackOpt
            An instance of KnapsackOpt configured with the specified parameters.

        Raises
        ------
        ValueError
            If any parameter is not of the expected type or value.
        """
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer. Got {seed}")
        if not isinstance(number_of_item_types, int) or number_of_item_types <= 0:
            raise ValueError(f"Number of item types must be a positive integer. Got {number_of_item_types}")
        if not isinstance(max_item_count, int) or max_item_count <= 0:
            raise ValueError(f"Max item count must be a positive integer. Got {max_item_count}")
        if not isinstance(max_weight_per_item, int) or max_weight_per_item <= 0:
            raise ValueError(f"Max weight per item must be a positive integer. Got {max_weight_per_item}")
        if not isinstance(max_value_per_item, int) or max_value_per_item <= 0:
            raise ValueError(f"Max value per item must be a positive integer. Got {max_value_per_item}")
        if not isinstance(max_weight_pct, float) or not (0 <= max_weight_pct <= 1):
            raise ValueError(f"Max weight percentage must be a float between 0 and 1. Got {max_weight_pct}")
        if not isinstance(multiply_by_max_item_count, bool):
            raise ValueError(f"multiply_by_max_item_count must be a boolean. Got {multiply_by_max_item_count}")

        np.random.seed(seed)

        weights = np.random.randint(1, max_weight_per_item + 1, size=number_of_item_types).tolist()
        values = np.random.randint(1, max_value_per_item + 1, size=number_of_item_types).tolist()

        return KnapsackOpt(
            length=number_of_item_types,
            max_val=max_item_count,
            weights=weights,
            values=values,
            max_weight_pct=max_weight_pct,
            multiply_by_max_item_count=multiply_by_max_item_count,
        )

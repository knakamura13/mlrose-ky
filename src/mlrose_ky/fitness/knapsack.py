"""Class defining the Knapsack fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class Knapsack:
    """Fitness function for Knapsack optimization problem.

    Given a set of n items, where item i has known weight :math:`w_{i}` and known value
    :math:`v_{i}`; and maximum knapsack capacity, :math:`W`, the Knapsack fitness function
    evaluates the fitness of a state vector :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:

    .. math::

        Fitness(x) = \\sum_{i = 0}^{n-1}v_{i}x_{i}, \\text{ if}
        \\sum_{i = 0}^{n-1}w_{i}x_{i} \\leq W, \\text{ and 0, otherwise,}

    where :math:`x_{i}` denotes the number of copies of item i included in the knapsack.

    Parameters
    ----------
    weights : list[float]
        List of weights for each of the n items.

    values : list[float]
        List of values for each of the n items.

    max_weight_pct : float, default=0.35
        Parameter used to set maximum capacity of knapsack (W) as a percentage of
        the total of the weights list (:math:`W =` max_weight_pct :math:`\times` total_weight).

    max_item_count : int, default=1
        Maximum number of copies of each item that can be included in the knapsack.

    multiply_by_max_item_count : bool, default=False
        Whether to multiply the maximum weight by the maximum item count.

    Examples
    --------
    >>> weights = [10, 5, 2, 8, 15]
    >>> values = [1, 2, 3, 4, 5]
    >>> max_weight_pct = 0.6
    >>> fitness = Knapsack(weights, values, max_weight_pct)
    >>> state_vector = np.array([1, 0, 2, 1, 0])
    >>> fitness.evaluate(state_vector)
    11.0

    Note
    ----
    The Knapsack fitness function is suitable for use in discrete-state optimization problems *only*.
    """

    def __init__(
        self,
        weights: list[float],
        values: list[float],
        max_weight_pct: float = 0.35,
        max_item_count: int = 1,
        multiply_by_max_item_count: bool = False,
    ):
        self.prob_type: str = "discrete"
        self.weights: list[float] = weights
        self.values: list[float] = values

        count_multiplier = max_item_count if multiply_by_max_item_count else 1.0
        self._w = np.ceil(np.sum(self.weights) * max_weight_pct * count_multiplier)

        if len(self.weights) != len(self.values):
            raise ValueError("The weights and values lists must be the same size.")
        if len(self.weights) and min(self.weights) <= 0:
            raise ValueError("All weights must be greater than 0.")
        if len(self.values) and min(self.values) <= 0:
            raise ValueError("All values must be greater than 0.")
        if max_item_count <= 0:
            raise ValueError("max_item_count must be greater than 0.")
        if max_weight_pct <= 0 or max_weight_pct > 1.0:
            raise ValueError("max_weight_pct must be between 0 and 1.")

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation. Must be the same length as the weights
            and values arrays.

        Returns
        -------
        float
            Value of fitness function.

        Raises
        ------
        ValueError
            If `state` is not the same size as the weights and values arrays.
        TypeError
            If `state` is not an instance of `np.ndarray`.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state_vector to be np.ndarray, got {type(state).__name__} instead.")
        if len(state) != len(self.weights):
            raise ValueError("The state_vector must be the same size as the weights and values arrays.")

        total_weight = np.sum(state * self.weights)
        total_value = np.sum(state * self.values)

        if total_weight <= self._w:
            return float(total_value)

        return 0.0

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

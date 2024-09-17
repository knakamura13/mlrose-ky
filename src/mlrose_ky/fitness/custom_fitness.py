"""Class defining a customizable fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np


class CustomFitness:
    """Class for generating your own fitness function.

    Parameters
    ----------
    fitness_fn : Callable
        Function for calculating fitness of a state with the signature
        `fitness_fn(state, **kwargs)`.

    problem_type : str, default: 'either'
        Specifies problem type as 'discrete', 'continuous', 'tsp' or 'either'
        (denoting either discrete or continuous).

    kwargs : additional arguments
        Additional parameters to be passed to the fitness function.

    Examples
    --------
    >>> def custom_fn(state, c): return c * np.sum(state)
    >>> kwargs = {'c': 10}
    >>> fitness = CustomFitness(custom_fn, **kwargs)
    >>> state_vector = np.array([1, 2, 3, 4, 5])
    >>> fitness.evaluate(state_vector)
    150.0
    """

    def __init__(self, fitness_fn: Callable[..., float], problem_type: str = "either", **kwargs: Any):
        """
        Initialize the CustomFitness class.

        Parameters
        ----------
        fitness_fn : Callable
            Function for calculating fitness of a state with the signature
            `fitness_fn(state, **kwargs)`.

        problem_type : str, optional, default='either'
            Specifies problem type as 'discrete', 'continuous', 'tsp',
            or 'either' (denoting either discrete or continuous).

        kwargs : additional arguments
            Additional parameters to be passed to the fitness function.

        Raises
        ------
        ValueError
            If `problem_type` is not one of ['discrete', 'continuous', 'tsp', 'either'].
        """
        if problem_type not in ["discrete", "continuous", "tsp", "either"]:
            raise ValueError(f"Invalid problem_type: {problem_type}. Must be one of ['discrete', 'continuous', 'tsp', 'either'].")

        self.problem_type: str = problem_type
        self.fitness_fn: Callable[..., float] = fitness_fn
        self.kwargs: Any = kwargs

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        float
            Value of fitness function.

        Raises
        ------
        TypeError
            If `state_vector` is not an instance of `np.ndarray`.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state_vector to be np.ndarray, got {type(state).__name__} instead.")

        return float(self.fitness_fn(state, **self.kwargs))

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete', 'continuous', 'tsp' or 'either'.
        """
        return self.problem_type

"""Class defining the Flip Flop fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class FlipFlop:
    """Fitness function for Flip Flop optimization problem.

    Evaluates the fitness of a state vector :math:`x` as the total number of pairs of consecutive
    elements of :math:`x`, (:math:`x_{i}` and :math:`x_{i+1}`) where :math:`x_{i} \\neq x_{i+1}`.

    Examples
    --------
    >>> fitness = FlipFlop()
    >>> state_vector = np.array([0, 1, 0, 1, 1, 1, 1])
    >>> fitness.evaluate(state_vector)
    3.0

    Note
    ----
    The Flip Flop fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self):
        """Initialize the Flip Flop fitness function."""
        self.prob_type: str = "discrete"

    @staticmethod
    def evaluate(state: np.ndarray) -> float:
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

        differences = np.diff(state) != 0
        return float(np.sum(differences))

    @staticmethod
    def evaluate_many(states: np.ndarray) -> np.ndarray:
        """Evaluate the fitness of an ndarray of state vectors.

        Parameters
        ----------
        states : np.ndarray
            States array for evaluation.

        Returns
        -------
        np.ndarray
            Array of fitness values.

        Raises
        ------
        TypeError
            If `states` is not an instance of `np.ndarray`.
        """
        if not isinstance(states, np.ndarray):
            raise TypeError(f"Expected state_matrix to be np.ndarray, got {type(states).__name__} instead.")

        differences = np.diff(states, axis=1) != 0
        return np.array(np.sum(differences, axis=1))

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

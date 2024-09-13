"""Class defining the N-Queens fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class Queens:
    """Fitness function for N-Queens optimization problem.

    Evaluates the fitness of an n-dimensional state vector `x`, where `x_i`
    represents the row position (between 0 and n-1, inclusive) of the 'queen'
    in column i, as the number of pairs of attacking queens.

    Examples
    -------
    >>> fitness = Queens()
    >>> state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
    >>> fitness.evaluate(state)
    6.0

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*,
    3rd edition. Prentice Hall, New Jersey, USA.

    Note
    ----
    The Queens fitness function is suitable for use in discrete-state optimization problems *only*.
    """

    def __init__(self, maximize: bool = False):
        """
        Initialize the Queens fitness function.

        Parameters
        ----------
        maximize : bool, optional, default=False
            Whether to maximize or minimize the fitness function.
        """
        self.prob_type: str = "discrete"
        self.maximize: bool = maximize

    @staticmethod
    def shift(a: np.ndarray, num: int, fill_value: float | int = np.nan) -> np.ndarray:
        """Shift elements of an array by a given number of places.

        Parameters
        ----------
        a : np.ndarray
            Input array to be shifted.
        num : int
            Number of places to shift the elements of the array.
        fill_value : float or int, optional, default=np.nan
            Value to fill the empty positions after the shift.

        Returns
        -------
        np.ndarray
            Shifted array.
        """
        result = np.empty(a.shape)

        if num > 0:
            result[:num] = fill_value  # Fill the beginning with the fill_value
            result[num:] = a[:-num]  # Shift the rest of the array to the right
        elif num < 0:
            result[num:] = fill_value  # Fill the end with the fill_value
            result[:num] = a[-num:]  # Shift the rest of the array to the left
        else:
            result[:] = a  # No shift needed

        return result

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
            If `state` is not an instance of `np.ndarray`.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state_vector to be np.ndarray, got {type(state).__name__} instead.")

        # Check for horizontal conflicts (queens in the same row)
        horizontal_conflicts = np.sum(np.unique(state, return_counts=True)[1] - 1)

        size = state.size

        # Generate state shifts for diagonal conflict checks
        state_shifts = np.array(
            [self.shift(state, i) + i for i in range(1 - size, size) if i != 0]
            + [self.shift(state, -i) + i for i in range(1 - size, size) if i != 0]
        )

        # Check for diagonal conflicts (queens on the same diagonal)
        diagonal_conflicts = np.sum(state_shifts == state) // 2  # Each diagonal conflict is counted twice

        # Calculate total fitness value
        fitness_value = horizontal_conflicts + diagonal_conflicts

        if self.maximize:
            # In maximization mode, we invert the fitness value
            fitness_value = self.get_max_size(size) - fitness_value

        return float(fitness_value)

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

    @staticmethod
    def get_max_size(problem_size: int) -> int:
        """Get the maximum possible number of conflicts for a given problem size.

        Parameters
        ----------
        problem_size : int
            Size of the problem (number of queens).

        Returns
        -------
        int
            Maximum possible number of conflicts.
        """
        if problem_size <= 1:
            return 0

        if problem_size == 2:
            return 1

        return 3 * (problem_size - 2)

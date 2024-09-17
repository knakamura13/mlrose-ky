"""Class defining the Continuous Peaks fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class ContinuousPeaks:
    """
    Fitness function for Continuous Peaks optimization problem. Evaluates the fitness
    of an n-dimensional state vector `x`, given parameter T.

    Parameters
    ----------
    t_pct : float, default=0.1
        Threshold parameter (T) for Continuous Peaks fitness function, expressed as a
        percentage of the state space dimension, n (i.e., `T = t_pct * n`).

    Attributes
    ----------
    t_pct : float
        The threshold percentage for the fitness function.
    prob_type : str
        Specifies problem type as 'discrete'.

    Examples
    --------
    >>> fitness = ContinuousPeaks(t_pct=0.15)
    >>> state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    >>> fitness.evaluate(state)
    17.0

    Note
    ----
    The Continuous Peaks fitness function is suitable for use in bit-string (discrete-state
    with `max_val = 2`) optimization problems only.
    """

    def __init__(self, t_pct: float = 0.1):
        self.prob_type: str = "discrete"
        self.t_pct: float = t_pct

        if not (0 <= self.t_pct <= 1):
            raise ValueError(f"t_pct must be between 0 and 1, got {self.t_pct} instead.")

    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        float
            Value of the fitness function.
        """
        num_elements = len(state)
        threshold = int(np.ceil(self.t_pct * num_elements))

        max_zeros = self.max_run(0, state)
        max_ones = self.max_run(1, state)

        reward = num_elements if max_zeros > threshold and max_ones > threshold else 0

        return float(max(max_zeros, max_ones) + reward)

    def get_prob_type(self) -> str:
        """
        Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

    @staticmethod
    def max_run(_b: int, _x: np.ndarray) -> int:
        """
        Determine the length of the maximum run of a given value in a vector.

        Parameters
        ----------
        _b : int
            Value to count.
        _x : np.ndarray
            Vector of integers.

        Returns
        -------
        int
            Length of the maximum run of the given value.
        """
        # Create a boolean array where each element is True if it equals the given value
        is_value = np.array(_x == _b)

        # If the value does not exist in the vector, return 0
        if not np.any(is_value):
            return 0

        # Calculate the differences between consecutive elements in the boolean array
        diffs = np.diff(is_value.astype(int))

        # Find the indices where the value starts and ends
        run_starts = np.where(diffs == 1)[0] + 1
        run_ends = np.where(diffs == -1)[0] + 1

        # If the run starts at the beginning of the vector, include the first index
        if is_value[0]:
            run_starts = np.insert(run_starts, 0, 0)

        # If the run ends at the end of the vector, include the last index
        if is_value[-1]:
            run_ends = np.append(run_ends, len(_x))

        # Ensure that run_ends has the same length as run_starts
        if len(run_starts) > len(run_ends):
            run_ends = np.append(run_ends, len(_x))

        # Calculate the lengths of the runs
        run_lengths = run_ends - run_starts

        # Return the maximum run length, or 0 if no runs are found
        return run_lengths.max() if run_lengths.size > 0 else 0

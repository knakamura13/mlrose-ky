"""Class defining the Six Peaks fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky.fitness._discrete_peaks_base import _DiscretePeaksBase


class SixPeaks(_DiscretePeaksBase):
    """Fitness function for Six Peaks optimization problem. Evaluates the
    fitness of an n-dimensional state vector `x`, given parameter T, as:

    .. math::

        Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)

    where:

    * `tail(b, x)` is the number of trailing b's in `x`;
    * `head(b, x)` is the number of leading b's in `x`;
    * `R(x, T) = n`, if (`tail(0, x) > T` and
      `head(1, x) > T`) or (`tail(1, x) > T` and
      `head(0, x) > T`); and
    * `R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct : float, optional, default=0.1
        Threshold parameter (T) for Six Peaks fitness function, expressed as a percentage
        of the state space dimension, n (i.e. `T = threshold_pct \\times n`).

    Examples
    --------
    >>> fitness = SixPeaks(t_pct=0.15)
    >>> state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
    >>> fitness.evaluate(state)
    12.0

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    The Six Peaks fitness function is suitable for use in bit-string
    (discrete-state with `max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct: float = 0.1):
        """
        Initialize the Six Peaks fitness function.

        Parameters
        ----------
        t_pct : float, optional, default=0.1
            Threshold parameter (T) for Six Peaks fitness function.
        """
        self.prob_type: str = "discrete"
        self.t_pct: float = t_pct

        if not (0 <= self.t_pct <= 1):
            raise ValueError(f"threshold_pct must be between 0 and 1, got {self.t_pct}.")

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

        vector_length = len(state)
        threshold = np.ceil(self.t_pct * vector_length)

        # Calculate head and tail values
        leading_zeros = self.head(0, state)
        trailing_zeros = self.tail(0, state)
        leading_ones = self.head(1, state)
        trailing_ones = self.tail(1, state)

        # Calculate max(tail(0, x), head(1, x))
        max_score = max(trailing_zeros, leading_ones)

        # Calculate R(x, T)
        reward = (
            vector_length
            if (trailing_zeros > threshold and leading_ones > threshold) or (trailing_ones > threshold and leading_zeros > threshold)
            else 0
        )

        # Evaluate function
        return float(max_score + reward)

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

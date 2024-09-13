"""Class for defining an exponential decay schedule for Simulated Annealing (SA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class ExpDecay:
    """
    Defines an exponential decay schedule for the temperature parameter T in simulated annealing,
    using the formula:

    .. math::

        T(t) = \\max(T_{0} e^{-rt}, T_{min})

    where :math:`T_{0}` is the initial temperature at time `t = 0`, `r` is the rate of exponential decay,
    and :math:`T_{min}` is the minimum allowable temperature.

    Parameters
    ----------
    init_temp : float, default=1.0
        The initial value of the temperature parameter T. Must be greater than 0.
    exp_const : float, default=0.005
        The rate of exponential decay. Must be greater than 0.
    min_temp : float, default=0.001
        The minimum allowable temperature. Must be greater than 0 and less than `init_temp`.

    Attributes
    ----------
    init_temp : float
        Stores the initial temperature.
    exp_const : float
        Stores the rate of exponential decay.
    min_temp : float
        Stores the minimum temperature.

    Examples
    --------
    >>> schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
    >>> print(schedule.evaluate(5))
    7.788007830714049
    """

    def __init__(self, init_temp: float = 1.0, exp_const: float = 0.005, min_temp: float = 0.001):
        self.init_temp: float = init_temp
        self.exp_const: float = exp_const
        self.min_temp: float = min_temp

        if self.init_temp <= 0:
            raise ValueError("Initial temperature must be greater than 0.")
        if self.exp_const <= 0:
            raise ValueError("Decay rate must be greater than 0 and positive.")
        if not (0 < self.min_temp < self.init_temp):
            raise ValueError("Minimum temperature must be greater than 0 and less than initial temperature.")

    def __str__(self) -> str:
        return f"ExpDecay(init_temp={self.init_temp}, exp_const={self.exp_const}, min_temp={self.min_temp})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpDecay):
            return False
        return self.init_temp == other.init_temp and self.exp_const == other.exp_const and self.min_temp == other.min_temp

    def evaluate(self, t: int) -> float:
        """
        Evaluate the temperature parameter at the specified time using exponential decay.

        Parameters
        ----------
        t : int
            The time at which the temperature parameter T is evaluated.

        Returns
        -------
        float
            The temperature parameter at the given time, respecting the minimum temperature.
        """
        return float(max(self.init_temp * np.exp(-self.exp_const * t), self.min_temp))

    def get_info__(self, t: int = None, prefix: str = "") -> dict:
        """
        Retrieve a dictionary containing the configuration and optionally the current value of the decay schedule.

        Parameters
        ----------
        t : int | None, optional
            If provided, include the current temperature value at the given time.
        prefix : str, optional
            A prefix to append to each dictionary key, enhancing integration with other data structures.

        Returns
        -------
        dict
            A dictionary detailing the decay schedule settings and optionally the current temperature.
        """
        info_prefix = f"{prefix}schedule_" if prefix else "schedule_"

        info = {
            f"{info_prefix}type": "exponential",
            f"{info_prefix}init_temp": self.init_temp,
            f"{info_prefix}exp_const": self.exp_const,
            f"{info_prefix}min_temp": self.min_temp,
        }

        if t is not None:
            info[f"{info_prefix}current_value"] = self.evaluate(t)

        return info

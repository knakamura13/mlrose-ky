"""Class for defining an arithmetic decay schedule for Simulated Annealing (SA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause


class ArithDecay:
    """
    Schedule for arithmetically decaying the temperature parameter T in a
    simulated annealing process, calculated using the formula:

    .. math::

        T(t) = \\max(T_{0} - rt, T_{min})

    where :math:`T_{0}` is the initial temperature at time `t = 0`, `r` is the decay rate, and :math:`T_{min}`
    is the minimum temperature.

    Parameters
    ----------
    init_temp : float
        Initial value of the temperature parameter T. Must be greater than 0.
    decay : float
        Temperature decay parameter. Must be a positive value and less than or equal to 1.
    min_temp : float
        Minimum allowable value of the temperature parameter. Must be positive and less than `init_temp`.

    Attributes
    ----------
    init_temp : float
        Stores the initial temperature.
    decay : float
        Stores the rate of temperature decay.
    min_temp : float
        Stores the minimum temperature.

    Examples
    --------
    >>> schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
    >>> schedule.evaluate(5)
    5.25
    """

    def __init__(self, init_temp: float = 1.0, decay: float = 0.0001, min_temp: float = 0.001):
        self.init_temp: float = init_temp
        self.decay: float = decay
        self.min_temp: float = min_temp

        if self.init_temp <= 0:
            raise ValueError("Initial temperature must be greater than 0.")
        if not (0 < self.decay <= 1):
            raise ValueError("Decay rate must be greater than 0 and less than or equal to 1.")
        if not (0 < self.min_temp < self.init_temp):
            raise ValueError("Minimum temperature must be greater than 0 and less than initial temperature.")

    def __str__(self) -> str:
        return f"ArithDecay(init_temp={self.init_temp}, decay={self.decay}, min_temp={self.min_temp})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArithDecay):
            return False
        return self.init_temp == other.init_temp and self.decay == other.decay and self.min_temp == other.min_temp

    def evaluate(self, t: int) -> float:
        """
        Calculate and return the temperature parameter at the given time.

        Parameters
        ----------
        t : int
            The time at which to evaluate the temperature parameter.

        Returns
        -------
        float
            The temperature parameter at the given time, respecting the minimum temperature.
        """
        return max(self.init_temp - (self.decay * t), self.min_temp)

    def get_info__(self, t: int = None, prefix: str = "") -> dict:
        """
        Generate a dictionary containing the decay schedule's settings and optionally its current value.

        Parameters
        ----------
        t : int, optional
            The time at which to evaluate the current temperature value.
            If provided, the current value is included in the returned dictionary.
        prefix : str, optional
            A prefix to prepend to each key in the dictionary, useful for integrating this info into larger structured data.

        Returns
        -------
        dict
            A dictionary with keys reflecting the decay schedule's parameters and optionally the current temperature.
        """
        info_prefix = f"{prefix}schedule_" if prefix else "schedule_"

        info = {
            f"{info_prefix}type": "arithmetic",
            f"{info_prefix}init_temp": self.init_temp,
            f"{info_prefix}decay": self.decay,
            f"{info_prefix}min_temp": self.min_temp,
        }

        if t is not None:
            info[f"{info_prefix}current_value"] = self.evaluate(t)

        return info

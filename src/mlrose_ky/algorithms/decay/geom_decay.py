"""Class for defining a geometric decay schedule for Simulated Annealing (SA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause


class GeomDecay:
    """
    Defines a geometric decay schedule for the temperature parameter T in simulated annealing,
    using the formula:

    .. math::

        T(t) = \\max(T_{0} \\times r^{t}, T_{min})

    where :math:`T_{0}` is the initial temperature at time `t = 0`, `r` is the rate of geometric decay,
    and :math:`T_{min}` is the minimum allowable temperature.

    Parameters
    ----------
    init_temp : float, default=1.0
        The initial value of the temperature parameter T. Must be greater than 0.
    decay : float, default=0.99
        The rate of geometric decay. Must be between 0 (exclusive) and 1 (inclusive).
    min_temp : float, default=0.001
        The minimum allowable temperature. Must be greater than 0 and less than `init_temp`.

    Attributes
    ----------
    init_temp : float
        Stores the initial temperature.
    decay : float
        Stores the rate of geometric decay.
    min_temp : float
        Stores the minimum temperature.

    Examples
    --------
    >>> schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
    >>> print(schedule.evaluate(5))
    7.737809374999998
    """

    def __init__(self, init_temp: float = 1.0, decay: float = 0.99, min_temp: float = 0.001):
        self.init_temp: float = init_temp
        self.decay: float = decay
        self.min_temp: float = min_temp

        if self.init_temp <= 0:
            raise ValueError("Initial temperature must be greater than 0.")
        if not (0 < self.decay <= 1):
            raise ValueError("Decay rate must be between 0 and 1, exclusive of 0.")
        if not (0 < self.min_temp < self.init_temp):
            raise ValueError("Minimum temperature must be greater than 0 and less than initial temperature.")

    def __str__(self):
        return f"GeomDecay(init_temp={self.init_temp}, decay={self.decay}, min_temp={self.min_temp})"

    def __repr__(self):
        return f"{self.__class__.__name__}(init_temp={self.init_temp}, decay={self.decay}, min_temp={self.min_temp})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeomDecay):
            return False
        return self.init_temp == other.init_temp and self.decay == other.decay and self.min_temp == other.min_temp

    def evaluate(self, t: int) -> float:
        """
        Evaluate the temperature parameter at the specified time using geometric decay.

        Parameters
        ----------
        t : int
            The time at which the temperature parameter T is evaluated.

        Returns
        -------
        float
            The temperature parameter at the given time, respecting the minimum temperature.
        """
        return float(max(self.init_temp * (self.decay**t), self.min_temp))

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
        full_prefix = f"{prefix}__schedule_" if len(prefix) else "schedule_"

        info = {
            f"{full_prefix}type": "geometric",
            f"{full_prefix}init_temp": self.init_temp,
            f"{full_prefix}decay": self.decay,
            f"{full_prefix}min_temp": self.min_temp,
        }

        if t is not None:
            info[f"{full_prefix}current_value"] = self.evaluate(t)

        return info

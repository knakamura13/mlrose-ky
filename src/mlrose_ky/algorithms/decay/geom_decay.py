"""Class for defining a geometric decay schedule for Simulated Annealing (SA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import warnings


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
    initial_temperature : float, default=1.0
        The initial value of the temperature parameter T. Must be greater than 0.
    decay_rate : float, default=0.99
        The rate of geometric decay. Must be between 0 (exclusive) and 1 (inclusive).
    minimum_temperature : float, default=0.001
        The minimum allowable temperature. Must be greater than 0 and less than `initial_temperature`.

    Attributes
    ----------
    initial_temperature : float
        Stores the initial temperature.
    decay_rate : float
        Stores the rate of geometric decay.
    minimum_temperature : float
        Stores the minimum temperature.

    Examples
    --------
    >>> schedule = GeomDecay(initial_temperature=10, decay_rate=0.95, minimum_temperature=1)
    >>> print(schedule.evaluate(5))
    7.737809374999998
    """

    def __init__(self, initial_temperature: float = 1.0, decay_rate: float = 0.99, minimum_temperature: float = 0.001) -> None:
        self.initial_temperature: float = initial_temperature
        self.decay_rate: float = decay_rate
        self.minimum_temperature: float = minimum_temperature

        if self.initial_temperature <= 0:
            raise ValueError("Initial temperature must be greater than 0.")
        if not (0 < self.decay_rate <= 1):
            raise ValueError("Decay rate must be between 0 and 1, exclusive of 0.")
        if not (0 < self.minimum_temperature < self.initial_temperature):
            raise ValueError("Minimum temperature must be greater than 0 and less than initial temperature.")

    def __str__(self) -> str:
        return (
            f"GeometricDecay(initial_temperature={self.initial_temperature}, "
            f"decay_rate={self.decay_rate}, "
            f"minimum_temperature={self.minimum_temperature})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeomDecay):
            return False
        return (self.initial_temperature == other.initial_temperature
                and self.decay_rate == other.decay_rate
                and self.minimum_temperature == other.minimum_temperature)

    def evaluate(self, time: int) -> float:
        """
        Evaluate the temperature parameter at the specified time using geometric decay.

        Parameters
        ----------
        time : int
            The time at which the temperature parameter T is evaluated.

        Returns
        -------
        float
            The temperature parameter at the given time, respecting the minimum temperature.
        """
        return max(self.initial_temperature * (self.decay_rate ** time), self.minimum_temperature)

    def get_info(self, time: int | None = None, prefix: str = "") -> dict:
        """
        Retrieve a dictionary containing the configuration and optionally the current value of the decay schedule.

        Parameters
        ----------
        time : int | None, optional
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
            f"{info_prefix}type": "geometric",
            f"{info_prefix}initial_temperature": self.initial_temperature,
            f"{info_prefix}decay_rate": self.decay_rate,
            f"{info_prefix}minimum_temperature": self.minimum_temperature,
        }

        if time is not None:
            info[f"{info_prefix}current_value"] = self.evaluate(time)

        return info



"""Class for defining a custom decay schedule for Simulated Annealing (SA)."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import inspect
from typing import Callable


class CustomSchedule:
    """
    Class for generating a customizable temperature schedule for simulated annealing.

    This class allows the user to define a custom decay function that calculates the temperature
    based on the time and potentially other parameters.

    Parameters
    ----------
    schedule : Callable[..., float]
        A function with the signature `schedule(t: int, **kwargs)` that calculates the temperature at time t.
    **kwargs : dict
        Additional keyword arguments to be passed to the decay function.

    Examples
    -------
    >>> def custom_decay_function(time: int, offset: int) -> float: return time + offset
    >>> kwargs = {'offset': 10}
    >>> schedule = CustomSchedule(custom_decay_function, **kwargs)
    >>> schedule.evaluate(5)
    15
    """

    def __init__(self, schedule: Callable[..., float], **kwargs):
        if not inspect.isfunction(schedule):
            raise TypeError(f"'schedule' must be a function, got {type(schedule).__name__}")
        self.schedule: Callable[..., float] = schedule
        self.kwargs: dict = kwargs

    def __str__(self) -> str:
        return f"CustomSchedule(schedule={self.schedule.__name__}, kwargs={self.kwargs})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CustomSchedule):
            return False
        return self.schedule == other.schedule and self.kwargs == other.kwargs

    def evaluate(self, t: int) -> float:
        """
        Evaluate the temperature parameter at the specified time using the custom decay function.

        Parameters
        ----------
        t : int
            The time at which to evaluate the temperature parameter.

        Returns
        -------
        float
            The calculated temperature at the specified time.
        """
        return self.schedule(t, **self.kwargs)

    def get_info__(self, t: int = None, prefix: str = "") -> dict:
        """
        Retrieve a dictionary containing the configuration of the decay schedule and optionally the current value.

        Parameters
        ----------
        t : int | None, optional
            If provided, include the current temperature value at the given time.
        prefix : str, optional
            A prefix to append to each dictionary key, enhancing integration with other data structures.

        Returns
        -------
        dict
            A dictionary detailing the decay schedule's settings and optionally the current temperature.
        """
        info_prefix = f"{prefix}schedule_" if prefix else "schedule_"

        info = {
            f"{info_prefix}type": "custom",
            f"{info_prefix}schedule": self.schedule.__name__,
            **{f"{info_prefix}param_{key}": value for key, value in self.kwargs.items()},
        }

        if t is not None:
            info[f"{info_prefix}current_value"] = self.evaluate(t)

        return info

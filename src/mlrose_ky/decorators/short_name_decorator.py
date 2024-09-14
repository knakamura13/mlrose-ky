"""
Provides decorators to assign and retrieve short names for functions and methods.

This file contains decorators that allow for the easy assignment of a 'short name'
to functions and methods, which can be particularly useful for assigning an algorithm's
common or shortened name, such as 'GA' for 'Genetic Algorithms'.

It also includes a function to retrieve the assigned short name of a function.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any


def short_name(expr: str) -> Callable:
    """
    Decorator to assign a short name to a function.

    Parameters
    ----------
    expr : str
        The short name to be assigned to the function.

    Returns
    -------
    Callable
        A decorator that assigns the provided short name to a function and returns the function.
    """

    def short_name_func_applicator(func: Callable) -> Callable:
        """Assign a short name to the given function."""
        func.__short_name__ = expr
        return func

    return short_name_func_applicator


def get_short_name(v: Any) -> str:
    """
    Retrieve the short name of a variable, or its default name if a short name isn't assigned.

    Parameters
    ----------
    v : Any
        The variable from which the short name is retrieved.

    Returns
    -------
    str
        The short name of the variable, if assigned; otherwise, returns the full variable name or the variable itself as a fallback.
    """
    if hasattr(v, "__short_name__"):
        return v.__short_name__
    elif hasattr(v, "__name__"):
        return v.__name__
    return v

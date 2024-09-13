"""Neural network 'Tanh' activation function."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import warnings

import numpy as np

from mlrose_ky.decorators import short_name

warnings.filterwarnings("ignore")


@short_name("tanh")
def tanh(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Hyperbolic tan activation function.

    Parameters
    ----------
    x: np.ndarray
        Array containing input data.

    deriv: bool, optional, default=False
        Whether to return the function (when False) or its derivative (when True).

    Returns
    -------
    fx: np.ndarray
        Value of activation function at x.
    """
    fx = np.tanh(x)

    if deriv:
        fx = 1 - (fx**2)

    return fx

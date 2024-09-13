"""Neural network 'Leaky ReLu' activation function."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# Contributor : Ankit Grover
# License: BSD 3-clause

import warnings

import numpy as np

from mlrose_ky.decorators import short_name

warnings.filterwarnings("ignore")


@short_name("leaky_relu")
def leaky_relu(x: np.ndarray, alpha: float = 0.3, deriv: bool = False) -> np.ndarray:
    """Leaky ReLU activation function.

    Parameters
    ----------
    x: np.ndarray
        Array containing input data.
    alpha: int, optional, default=0.3
        Value to be set for applying small negative gradient.
    deriv: bool, option, default=False
        Whether to return the function (when False) or its derivative (when True).

    Returns
    -------
    fx: np.ndarray
        Value of activation function at x.
    """
    fx = np.copy(x)
    fx = np.where(fx < 0, fx * alpha, fx)

    if deriv:
        fx[np.where(fx > 0)] = 1
        fx[np.where(fx < 0)] = alpha

    return fx

"""Neural network 'ReLu' activation function."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import warnings

import numpy as np

from mlrose_ky.decorators import short_name

warnings.filterwarnings("ignore")


@short_name("relu")
def relu(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """ReLU activation function.

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
    fx = np.copy(x)
    fx[np.where(fx < 0)] = 0

    if deriv:
        fx[np.where(fx > 0)] = 1

    return fx

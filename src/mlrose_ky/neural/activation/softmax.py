"""Neural network 'Softmax' activation function."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import warnings

import numpy as np

from mlrose_ky.decorators import short_name

warnings.filterwarnings("ignore")


@short_name("softmax")
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function.

    Parameters
    ----------
    x: np.ndarray
        Array containing input data.

    Returns
    -------
    fx: np.ndarray
        Value of activation function at x.
    """
    max_prob = np.max(x, axis=1).reshape((-1, 1))
    fx = np.exp(x - max_prob)

    sum_prob = np.sum(fx, axis=1).reshape((-1, 1))
    fx = np.divide(fx, sum_prob)

    return fx

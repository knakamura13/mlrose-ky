"""Global variables and constants for unit tests."""

import pytest
import numpy as np

SEED = 12


@pytest.fixture
def sample_data():
    """Return sample data for testing."""
    X = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])  # X.shape = (6, 4)

    y_classifier = np.reshape(np.array([1, 1, 0, 0, 1, 1]), (X.shape[0], 1))
    y_multiclass = np.array([[1, 1], [1, 0], [0, 0], [0, 0], [1, 0], [1, 1]])
    y_regressor = y_classifier.copy()

    return X, y_classifier, y_multiclass, y_regressor

"""Unit tests for generators/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

from unittest.mock import patch
import matplotlib.pyplot as plt
import pytest
from sklearn.linear_model import LogisticRegression

from tests.globals import SEED
from mlrose_ky.samples import SyntheticData, plot_synthetic_dataset


# noinspection PyMissingOrEmptyDocstring
@pytest.fixture
def generator():
    return SyntheticData(seed=SEED)


class TestSyntheticDataGenerator:
    def test_get_synthetic_features_and_classes_default(self, generator):
        """Test default synthetic features and classes."""
        features, classes = generator.get_synthetic_features_and_classes()

        assert features == ["(1) A", "(2) B"]
        assert classes == ["RED", "BLUE"]

    def test_get_synthetic_features_and_classes_with_redundant(self, generator):
        """Test synthetic features and classes with redundant column."""
        features, classes = generator.get_synthetic_features_and_classes(with_redundant_column=True)

        assert features == ["(1) A", "(2) B", "(3) R"]
        assert classes == ["RED", "BLUE"]

    def test_get_synthetic_data(self, generator):
        """Test getting synthetic data."""
        data, features, classes, output_directory = generator.get_synthetic_data()

        assert data.shape == (400, 3)
        assert features == ["(1) A", "(2) B"]
        assert classes == ["RED", "BLUE"]
        assert output_directory is None

    def test_setup_synthetic_data_test_train(self, generator):
        """Test setting up synthetic data for train and test."""
        data, _, _, _ = generator.get_synthetic_data()
        x, y, x_train, x_test, y_train, y_test = generator.setup_synthetic_data_test_train(data)

        assert x_train.shape == (280, 2)
        assert x_test.shape == (120, 2)
        assert y_train.shape == (280,)
        assert y_test.shape == (120,)

    @patch("matplotlib.pyplot.show")  # Mocking plt.show
    def test_plot_synthetic_dataset(self, mock_show, generator):
        """Test plotting synthetic dataset."""
        data, _, _, _ = generator.get_synthetic_data()
        x, y, x_train, x_test, y_train, y_test = generator.setup_synthetic_data_test_train(data)
        classifier = LogisticRegression().fit(x_train, y_train)
        plt.figure()
        plot_synthetic_dataset(x_train, x_test, y_train, y_test, classifier=classifier)
        plt.close()

        assert mock_show.called

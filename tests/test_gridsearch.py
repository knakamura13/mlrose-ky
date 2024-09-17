"""Unit tests for gridsearch/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import pytest
import inspect
import numpy as np
from typing import Any
import sklearn.model_selection as skms
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tests.globals import SEED

from mlrose_ky.gridsearch import GridSearchMixin


@pytest.fixture
def sample_data():
    """Fixture for generating a simple dataset."""
    X, y = make_classification(random_state=SEED)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


@pytest.fixture
def dummy_classifier():
    """Fixture for creating a simple DummyClassifier."""
    return DummyClassifier(strategy="most_frequent", random_state=SEED)


@pytest.fixture
def grid_search_mixin():
    """Fixture for initializing a GridSearchMixin."""
    return GridSearchMixin()


@pytest.fixture
def grid_search_parameters():
    """Fixture for defining grid search parameters."""
    return {"strategy": ["most_frequent", "stratified"]}


class TestGridSearchMixin:

    def test_initialize_with_custom_scorer(self, grid_search_mixin):
        """Test initializing GridSearchMixin with a custom scoring method"""

        # noinspection PyMissingOrEmptyDocstring, PyShadowingNames
        def custom_scorer(y_true, y_pred):
            return np.mean(y_true == y_pred)

        grid_search_mixin._scorer_method = custom_scorer
        grid_search_mixin._params = inspect.signature(custom_scorer)  # Force update to match custom scorer

        assert grid_search_mixin._scorer_method == custom_scorer
        assert grid_search_mixin._params == inspect.signature(custom_scorer)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "param_grid, expected_best_param",
        [({"strategy": ["most_frequent", "stratified"]}, "most_frequent"), ({"strategy": ["stratified"]}, "stratified")],
    )
    def test_perform_grid_search(self, grid_search_mixin, sample_data, dummy_classifier, param_grid, expected_best_param):
        """Should perform grid search with a classifier and return GridSearchCV object"""
        X_train, X_test, y_train, y_test = sample_data

        search_results = grid_search_mixin._perform_grid_search(
            classifier=dummy_classifier, x_train=X_train, y_train=y_train, cv=3, parameters=param_grid
        )

        assert isinstance(search_results, skms.GridSearchCV)
        assert hasattr(search_results, "best_params_")
        assert hasattr(search_results, "best_score_")
        assert hasattr(search_results, "cv_results_")
        assert search_results.best_params_["strategy"] == expected_best_param

    def test_handle_empty_input(self, grid_search_mixin):
        """Should handle empty input data gracefully"""
        with pytest.raises(ValueError):
            grid_search_mixin._perform_grid_search(
                classifier=DummyClassifier(), x_train=np.array([]), y_train=np.array([]), cv=3, parameters={"strategy": ["most_frequent"]}
            )

    def test_grid_search_score_intercept_raises_value_error(self, grid_search_mixin):
        """Should raise ValueError when scoring method raises TypeError"""

        # noinspection PyMissingOrEmptyDocstring, PyShadowingNames
        def faulty_scorer(_, __):
            raise TypeError("Intentional TypeError for testing")

        grid_search_mixin._scorer_method = faulty_scorer
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="Error during scoring with method faulty_scorer"):
            grid_search_mixin._grid_search_score_intercept(y_pred=y_pred, y_true=y_true)

    def test_handle_multi_class_predictions_without_argmax(self, grid_search_mixin):
        """Test handling of multi-class predictions without applying argmax"""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        # noinspection PyMissingOrEmptyDocstring, PyShadowingNames
        def custom_scorer(y_true, y_pred):
            return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

        grid_search_mixin._scorer_method = custom_scorer
        grid_search_mixin._params = inspect.signature(custom_scorer)
        score = grid_search_mixin._grid_search_score_intercept(y_pred=y_pred, y_true=y_true)

        assert score == 1.0

    def test_apply_argmax_to_multiclass_predictions(self, grid_search_mixin):
        """Should apply argmax to multi-class predictions when get_y_argmax is True"""
        grid_search_mixin._get_y_argmax = True

        y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])

        score = grid_search_mixin._grid_search_score_intercept(y_pred=y_pred, y_true=y_true)
        expected_score = grid_search_mixin._scorer_method(y_pred=y_pred.argmax(axis=1), y_true=y_true.argmax(axis=1))

        assert score == expected_score

    def test_make_scorer(self, grid_search_mixin):
        """Should create a custom scorer function using make_scorer method"""
        scorer: Any = grid_search_mixin.make_scorer()

        assert scorer._score_func == grid_search_mixin._grid_search_score_intercept
        assert callable(scorer)

    def test_score_with_additional_arguments(self, grid_search_mixin):
        """Should calculate score using the score method with additional arguments"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        # noinspection PyMissingOrEmptyDocstring, PyShadowingNames
        def custom_scorer(y_true, y_pred, weight=1.0):
            return np.mean(y_true == y_pred) * weight

        grid_search_mixin._scorer_method = custom_scorer
        grid_search_mixin._params = inspect.signature(custom_scorer)

        score = grid_search_mixin.score(y_pred=y_pred, y_true=y_true, weight=2.0)

        assert score == 1.0  # Expected score is 0.5 * 2.0

    def test_grid_search_score_intercept_with_additional_arguments(self, grid_search_mixin):
        """Should handle additional arguments in _grid_search_score_intercept method"""

        # noinspection PyMissingOrEmptyDocstring, PyShadowingNames
        def custom_scorer(y_true, y_pred, sample_weight=None):
            if sample_weight is not None:
                return np.average(y_true == y_pred, weights=sample_weight)
            return np.mean(y_true == y_pred)

        grid_search_mixin._scorer_method = custom_scorer
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])
        sample_weight = np.array([0.5, 0.5, 1.0, 1.0])

        score = grid_search_mixin._grid_search_score_intercept(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)
        expected_score = custom_scorer(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        assert score == expected_score

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_perform_grid_search_with_verbose(self, grid_search_mixin, sample_data, dummy_classifier, grid_search_parameters):
        """Should perform grid search with verbose output enabled"""
        X_train, X_test, y_train, y_test = sample_data

        # Perform grid search with verbose output enabled
        search_results = grid_search_mixin._perform_grid_search(
            dummy_classifier, X_train, y_train, cv=3, parameters=grid_search_parameters, verbose=True
        )

        # Check that the result is a GridSearchCV object and has the expected attributes
        assert isinstance(search_results, skms.GridSearchCV)
        assert hasattr(search_results, "best_params_")
        assert hasattr(search_results, "best_score_")
        assert hasattr(search_results, "cv_results_")

    def test_perform_grid_search_with_multiple_jobs(self, grid_search_mixin, sample_data, dummy_classifier, grid_search_parameters):
        """Should perform grid search with n_jobs set to more than one"""
        X_train, X_test, y_train, y_test = sample_data

        # Perform grid search with n_jobs set to 2
        search_results = grid_search_mixin._perform_grid_search(
            dummy_classifier, X_train, y_train, cv=3, parameters=grid_search_parameters, n_jobs=2
        )

        # Check that the result is a GridSearchCV object and has the expected attributes
        assert isinstance(search_results, skms.GridSearchCV)
        assert hasattr(search_results, "best_params_")
        assert hasattr(search_results, "best_score_")
        assert hasattr(search_results, "cv_results_")

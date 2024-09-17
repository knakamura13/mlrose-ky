"""Unit tests for runners/skmlp_runner.py"""

import pytest
import warnings
from unittest.mock import patch
import sklearn.metrics as skmt

from tests.globals import SEED
from mlrose_ky import SKMLPRunner
from mlrose_ky.neural import activation


class TestSKMLPRunner:
    """Tests for SKMLPRunner."""

    @pytest.fixture
    def data(self):
        """Fixture to provide dummy training and test data."""
        x_train = [[0, 0], [1, 1], [1, 0], [0, 1]]
        y_train = [0, 1, 1, 0]
        x_test = [[0, 0], [1, 1]]
        y_test = [0, 1]
        return x_train, y_train, x_test, y_test

    @pytest.fixture
    def runner_kwargs(self, data):
        """Fixture to provide common kwargs for SKMLPRunner initialization."""
        x_train, y_train, x_test, y_test = data
        grid_search_parameters = {
            "max_iters": [100, 200],
            "activation": [activation.relu, activation.sigmoid],
            "learning_rate_init": [0.001, 0.01],
        }

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 10, 50],
            "grid_search_parameters": grid_search_parameters,
            "grid_search_scorer_method": skmt.balanced_accuracy_score,
            "early_stopping": True,
            "max_attempts": 100,
            "n_jobs": 1,
            "cv": 2,
            "generate_curves": True,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize an SKMLPRunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return SKMLPRunner(**runner_kwargs)

    def test_initialize_with_default_grid_search_params(self, runner):
        """Test initialization with default grid search parameters."""
        assert runner.grid_search_parameters["max_iter"] == [100, 200]
        assert runner.grid_search_parameters["activation"] == ["relu", "logistic"]

    def test_skmlp_runner_initialization_sets_classifier(self, runner_kwargs):
        """Test SKMLPRunner initialization sets the classifier."""
        runner = SKMLPRunner(**runner_kwargs)
        assert isinstance(runner.classifier, runner._MLPClassifier)

    def test_run_with_grid_search_parameters(self, runner_kwargs):
        """Test run with grid search parameters."""
        with (
            patch.object(SKMLPRunner._MLPClassifier, "fit", autospec=True) as mock_fit,
            patch.object(SKMLPRunner._MLPClassifier, "predict", autospec=True) as mock_predict,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", category=UserWarning)
            runner = SKMLPRunner(**runner_kwargs)
            runner.run()

        mock_fit.assert_called()
        mock_predict.assert_called()

    def test_dynamic_runner_name(self, runner_kwargs):
        """Test that the runner name is set dynamically based on the algorithm."""
        runner = SKMLPRunner(**runner_kwargs)
        expected_name = "skmlp"
        assert runner.runner_name() == expected_name

    def test_grid_search_scorer_method(self, runner):
        """Test that the grid search scorer method is set correctly."""
        assert runner._scorer_method == skmt.balanced_accuracy_score

    def test_max_attempts_respected_during_initialization(self, runner):
        """Test max attempts respected during initialization."""
        assert runner.classifier.mlp.n_iter_no_change == 100

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_skmlp_runner_initialization_with_additional_kwargs(self, runner_kwargs):
        """Test SKMLPRunner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = SKMLPRunner(**runner_kwargs, **additional_kwargs)

        assert runner.runner_name() == "skmlp"
        assert runner.classifier.mlp.early_stopping == runner_kwargs["early_stopping"]
        assert runner.classifier.mlp.random_state == SEED

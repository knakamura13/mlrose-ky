"""Unit tests for runners/nngs_runner.py"""

import pytest
from unittest.mock import patch

import sklearn.metrics as skmt

import mlrose_ky
from mlrose_ky import NNGSRunner
from mlrose_ky.decorators import get_short_name


class TestNNGSRunner:
    """Tests for NNGSRunner."""

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
        """Fixture to provide common kwargs for NNGSRunner initialization."""
        x_train, y_train, x_test, y_test = data
        grid_search_parameters = {
            "max_iters": [1, 2],
            "learning_rate": [0.001, 0.002],
            "hidden_layer_sizes": [[2], [2, 2]],
            "activation": [mlrose_ky.relu, mlrose_ky.sigmoid],
        }

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "experiment_name": "test_experiment",
            "seed": 42,
            "iteration_list": [1, 10],
            "algorithm": mlrose_ky.algorithms.sa.simulated_annealing,
            "grid_search_parameters": grid_search_parameters,
            "grid_search_scorer_method": skmt.balanced_accuracy_score,
            "bias": True,
            "early_stopping": False,
            "clip_max": 1e10,
            "max_attempts": 500,
            "generate_curves": True,
            "n_jobs": 1,
            "cv": 2,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize an NNGSRunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return NNGSRunner(**runner_kwargs)

    def test_initialize_with_default_grid_search_params(self, runner):
        """Test initialization with default grid search parameters."""
        assert runner.grid_search_parameters["max_iters"] == [1, 2]
        assert runner.grid_search_parameters["learning_rate"] == [0.001, 0.002]

    def test_nngs_runner_initialization_sets_algorithm(self, runner_kwargs):
        """Test NNGS runner initialization sets the algorithm."""
        runner = NNGSRunner(**runner_kwargs)
        assert runner.classifier.algorithm == runner_kwargs["algorithm"]

    def test_run_with_grid_search_parameters(self, runner_kwargs):
        """Test run with grid search parameters."""
        with patch("mlrose_ky.runners._NNRunnerBase.run") as mock_run:
            runner = NNGSRunner(**runner_kwargs)
            runner.run()
            mock_run.assert_called()

    def test_dynamic_runner_name(self, runner_kwargs):
        """Test that the runner name is set dynamically based on the algorithm."""
        runner = NNGSRunner(**runner_kwargs)
        expected_name = f"nngs_{get_short_name(runner_kwargs['algorithm'])}"
        assert runner._dynamic_short_name == expected_name

    def test_grid_search_scorer_method(self, runner):
        """Test that the grid search scorer method is set correctly."""
        assert runner._scorer_method == skmt.balanced_accuracy_score

    def test_max_attempts_respected_during_initialization(self, runner):
        """Test max attempts respected during initialization."""
        assert runner.classifier.max_attempts == 500

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_nngsrunner_initialization_with_additional_kwargs(self, runner_kwargs):
        """Test NNGS runner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = NNGSRunner(**runner_kwargs, **additional_kwargs)

        assert runner.classifier.bias == runner_kwargs["bias"]
        assert runner.classifier.early_stopping == runner_kwargs["early_stopping"]
        assert runner.classifier.clip_max == runner_kwargs["clip_max"]

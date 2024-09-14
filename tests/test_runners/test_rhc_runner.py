"""Unit tests for runners/rhc_runner.py"""

import pytest
from unittest.mock import patch

from tests.globals import SEED

from mlrose_ky import RHCRunner, FlipFlopGenerator


class TestRHCRunner:
    """Tests for RHCRunner."""

    @pytest.fixture
    def problem(self):
        """Fixture to create an optimization problem instance for testing."""
        generator = FlipFlopGenerator()
        return generator.generate(SEED, 5)

    @pytest.fixture
    def runner_kwargs(self, problem):
        """Fixture to provide common kwargs for RHCRunner initialization."""
        return {
            "problem": problem,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 2, 3],
            "restart_list": [25, 50, 75],
            "max_attempts": 500,
            "generate_curves": True,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize an RHCRunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return RHCRunner(**runner_kwargs)

    def test_initialize_with_default_restart_list(self, runner):
        """Test initialization with default restart list."""
        assert runner.restart_list == [25, 50, 75]

    def test_rhc_runner_initialization_sets_restart_list(self, runner_kwargs):
        """Test RHC runner initialization sets the restart list."""
        runner = RHCRunner(**runner_kwargs)
        assert runner.restart_list == runner_kwargs["restart_list"]

    def test_run_with_restart_list(self, runner_kwargs):
        """Test run with restart list."""
        with patch("mlrose_ky.random_hill_climb") as mock_rhc:
            runner = RHCRunner(**runner_kwargs)
            runner.run()
            mock_rhc.assert_called()
            assert dict(mock_rhc.call_args[1]["callback_user_info"])["restarts"] in runner_kwargs["restart_list"]

    def test_max_attempts_respected_during_initialization(self, runner_kwargs):
        """Test max attempts respected during initialization."""
        runner_kwargs["max_attempts"] = 1000
        runner = RHCRunner(**runner_kwargs)
        assert runner.max_attempts == 1000

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_rhc_runner_initialization_with_additional_kwargs(self, problem, runner_kwargs):
        """Test RHC runner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = RHCRunner(**runner_kwargs, **additional_kwargs)

        assert runner.problem == problem
        assert runner.runner_name() == "rhc"
        assert runner._experiment_name == runner_kwargs["experiment_name"]
        assert runner.seed == runner_kwargs["seed"]
        assert runner.iteration_list == runner_kwargs["iteration_list"]
        assert runner.restart_list == runner_kwargs["restart_list"]

"""Unit tests for runners/mimic_runner.py"""

import pytest
from unittest.mock import patch

from tests.globals import SEED

from mlrose_ky import MIMICRunner, FlipFlopGenerator


class TestMIMICRunner:
    """Tests for MIMICRunner."""

    @pytest.fixture
    def problem(self):
        """Fixture to create an optimization problem instance for testing."""
        generator = FlipFlopGenerator()
        return generator.generate(SEED, 5)

    @pytest.fixture
    def runner_kwargs(self, problem):
        """Fixture to provide common kwargs for MIMICRunner initialization."""
        return {
            "problem": problem,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 2, 3],
            "population_sizes": [100, 200],
            "keep_percent_list": [0.25, 0.5],
            "max_attempts": 500,
            "generate_curves": True,
            "use_fast_mimic": True,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize a MIMICRunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return MIMICRunner(**runner_kwargs)

    def test_initialize_with_default_keep_percent_list(self, runner):
        """Test initialization with default keep percent list."""
        assert runner.keep_percent_list == [0.25, 0.5]

    def test_mimic_runner_initialization_sets_population_sizes(self, runner_kwargs):
        """Test MIMIC runner initialization sets population sizes."""
        runner = MIMICRunner(**runner_kwargs)
        assert runner.population_sizes == runner_kwargs["population_sizes"]

    def test_run_with_population_sizes_and_keep_percent_list(self, runner_kwargs):
        """Test run with population sizes and keep percent list."""
        with patch("mlrose_ky.mimic") as mock_mimic:
            runner = MIMICRunner(**runner_kwargs)
            runner.run()
            mock_mimic.assert_called()
            assert dict(mock_mimic.call_args[1]["callback_user_info"])["pop_size"] in runner_kwargs["population_sizes"]
            assert dict(mock_mimic.call_args[1]["callback_user_info"])["keep_pct"] in runner_kwargs["keep_percent_list"]

    def test_use_fast_mimic_flag(self, runner_kwargs):
        """Test the use_fast_mimic flag is set correctly."""
        problem = runner_kwargs["problem"]

        # Mock the set_mimic_fast_mode method if it exists
        if hasattr(problem, "set_mimic_fast_mode"):
            with patch.object(problem, "set_mimic_fast_mode") as mock_set_mimic_fast_mode:
                runner = MIMICRunner(**runner_kwargs)
                mock_set_mimic_fast_mode.assert_called_with(runner_kwargs["use_fast_mimic"])
                assert runner._use_fast_mimic == runner_kwargs["use_fast_mimic"]
        else:
            # If the method doesn't exist, proceed with the test as normal
            runner = MIMICRunner(**runner_kwargs)
            assert runner._use_fast_mimic is None

    def test_handle_none_keep_percent_list(self, runner_kwargs):
        """Test handling of None keep percent list."""
        runner_kwargs["keep_percent_list"] = None
        runner = MIMICRunner(**runner_kwargs)
        try:
            runner.run()
            assert True
        except Exception as e:
            pytest.fail(f"Runner raised an exception: {e}")

    def test_max_attempts_respected_during_initialization(self, runner_kwargs):
        """Test max attempts respected during initialization."""
        runner_kwargs["max_attempts"] = 1000
        runner = MIMICRunner(**runner_kwargs)
        assert runner.max_attempts == 1000

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_mimicrunner_initialization_with_additional_kwargs(self, problem, runner_kwargs):
        """Test MIMIC runner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = MIMICRunner(**runner_kwargs, **additional_kwargs)

        assert runner.problem == problem
        assert runner.runner_name() == "mimic"
        assert runner._experiment_name == runner_kwargs["experiment_name"]
        assert runner.seed == runner_kwargs["seed"]
        assert runner.iteration_list == runner_kwargs["iteration_list"]
        assert runner.population_sizes == runner_kwargs["population_sizes"]
        assert runner.keep_percent_list == runner_kwargs["keep_percent_list"]

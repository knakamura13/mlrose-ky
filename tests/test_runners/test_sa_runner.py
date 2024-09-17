"""Unit tests for runners/sa_runner.py"""

import pytest
from unittest.mock import patch

from tests.globals import SEED

import mlrose_ky
from mlrose_ky import SARunner, FlipFlopGenerator


class TestSARunner:
    """Tests for SARunner."""

    @pytest.fixture
    def problem(self):
        """Fixture to create an optimization problem instance for testing."""
        generator = FlipFlopGenerator()
        return generator.generate(SEED, 5)

    @pytest.fixture
    def runner_kwargs(self, problem):
        """Fixture to provide common kwargs for SARunner initialization."""
        return {
            "problem": problem,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 2, 3],
            "temperature_list": [1, 10, 50, 100],
            "decay_list": [mlrose_ky.GeomDecay],
            "max_attempts": 500,
            "generate_curves": True,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize an SARunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return SARunner(**runner_kwargs)

    def test_initialize_with_default_temperature_list(self, runner):
        """Test initialization with default temperature list."""
        assert runner.temperature_list == [1, 10, 50, 100]

    def test_sarunner_initialization_sets_decay_list(self, runner_kwargs):
        """Test SARunner initialization sets the decay list."""
        runner = SARunner(**runner_kwargs)
        assert runner.decay_list == [mlrose_ky.GeomDecay]

    def test_run_with_temperature_and_decay_list(self, runner_kwargs):
        """Test run with temperature and decay list."""
        with patch("mlrose_ky.simulated_annealing") as mock_sa:
            runner = SARunner(**runner_kwargs)
            runner.run()
            mock_sa.assert_called()

            # Ensure that temperatures were processed correctly
            expected_temperatures = [
                mlrose_ky.GeomDecay(init_temp=1),
                mlrose_ky.GeomDecay(init_temp=10),
                mlrose_ky.GeomDecay(init_temp=50),
                mlrose_ky.GeomDecay(init_temp=100),
            ]

            # Check that the schedules used in the call match the expected temperatures
            for call, expected_schedule in zip(mock_sa.call_args_list, expected_temperatures):
                schedule = dict(call[1]["callback_user_info"])["schedule"]
                assert isinstance(schedule, mlrose_ky.GeomDecay)
                assert schedule.__getattribute__("init_temp") == expected_schedule.__getattribute__("init_temp")

    def test_max_attempts_respected_during_initialization(self, runner_kwargs):
        """Test max attempts respected during initialization."""
        runner_kwargs["max_attempts"] = 1000
        runner = SARunner(**runner_kwargs)
        assert runner.max_attempts == 1000

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_sarunner_initialization_with_additional_kwargs(self, problem, runner_kwargs):
        """Test SARunner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = SARunner(**runner_kwargs, **additional_kwargs)

        assert runner.problem == problem
        assert runner.runner_name() == "sa"
        assert runner._experiment_name == runner_kwargs["experiment_name"]
        assert runner.seed == runner_kwargs["seed"]
        assert runner.iteration_list == runner_kwargs["iteration_list"]
        assert runner.temperature_list == runner_kwargs["temperature_list"]

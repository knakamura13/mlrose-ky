"""Unit tests for runners/ga_runner.py"""

import pytest
from unittest.mock import patch

from tests.globals import SEED

from mlrose_ky import GARunner, FlipFlopGenerator


class TestGARunner:
    """Tests for GARunner."""

    @pytest.fixture
    def problem(self):
        """Fixture to create an optimization problem instance for testing."""
        generator = FlipFlopGenerator()
        return generator.generate(SEED, 5)

    @pytest.fixture
    def runner_kwargs(self, problem):
        """Fixture to provide common kwargs for GARunner initialization."""
        return {
            "problem": problem,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 2, 3],
            "population_sizes": [100, 200],
            "mutation_rates": [0.1, 0.2],
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize a GARunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return GARunner(**runner_kwargs)

    def test_initialize_with_default_hamming_factors_and_decays(self, runner):
        """Test initialization with default hamming factors and decays."""
        assert runner.hamming_factors is None
        assert runner.hamming_factor_decays is None

    def test_ga_runner_initialization_sets_population_sizes(self, runner_kwargs):
        """Test GA runner initialization sets population sizes."""
        runner = GARunner(**runner_kwargs)
        assert runner.population_sizes == runner_kwargs["population_sizes"]

    def test_initialization_sets_mutation_rates(self, runner_kwargs):
        """Test initialization sets mutation rates."""
        runner = GARunner(**runner_kwargs)
        assert runner.mutation_rates == runner_kwargs["mutation_rates"]

    def test_run_with_population_sizes(self, runner_kwargs):
        """Test run with population sizes."""
        with patch("mlrose_ky.genetic_alg") as mock_genetic_alg:
            runner = GARunner(**runner_kwargs)
            runner.run()
            mock_genetic_alg.assert_called()
            assert dict(mock_genetic_alg.call_args[1]["callback_user_info"])["pop_size"] in runner_kwargs["population_sizes"]

    def test_run_with_mutation_rates(self, runner_kwargs):
        """Test run with mutation rates."""
        with patch("mlrose_ky.genetic_alg") as mock_genetic_alg:
            runner = GARunner(**runner_kwargs)
            runner.run()
            mock_genetic_alg.assert_called()
            assert dict(mock_genetic_alg.call_args[1]["callback_user_info"])["mutation_prob"] in runner_kwargs["mutation_rates"]

    def test_handle_none_hamming_factors(self, runner):
        """Test handling of None hamming factors."""
        try:
            runner.run()
            assert True
        except Exception as e:
            pytest.fail(f"Runner raised an exception: {e}")

    def test_handle_none_hamming_factor_decays(self, runner_kwargs):
        """Test handling of None hamming factor decays."""
        runner_kwargs["hamming_factors"] = None
        runner_kwargs["hamming_factor_decays"] = None

        runner = GARunner(**runner_kwargs)
        try:
            runner.run()
            assert True
        except Exception as e:
            pytest.fail(f"Runner raised an exception: {e}")

    def test_max_attempts_respected_during_initialization(self, runner_kwargs):
        """Test max attempts respected during initialization."""
        runner_kwargs["max_attempts"] = 1000
        runner = GARunner(**runner_kwargs)
        assert runner.max_attempts == 1000

    def test_generate_curves_true(self, runner):
        """Test generate curves is set to True."""
        assert runner.generate_curves is True

    def test_garunner_initialization_with_additional_kwargs(self, problem, runner_kwargs):
        """Test GA runner initialization with additional kwargs."""
        additional_kwargs = {"custom_arg": "custom_value"}
        runner = GARunner(**runner_kwargs, **additional_kwargs)

        assert runner.problem == problem
        assert runner.get_runner_name() == "ga"
        assert runner._experiment_name == runner_kwargs["experiment_name"]
        assert runner.seed == runner_kwargs["seed"]
        assert runner.iteration_list == runner_kwargs["iteration_list"]
        assert runner.population_sizes == runner_kwargs["population_sizes"]
        assert runner.mutation_rates == runner_kwargs["mutation_rates"]

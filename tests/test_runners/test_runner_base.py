"""Unit tests for runners/_runner_base.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import signal
from unittest.mock import patch

from tests.globals import SEED

# noinspection PyProtectedMember
from mlrose_ky.runners._runner_base import _RunnerBase


class TestRunnerBase:
    @pytest.fixture
    def _test_runner_fixture(self):
        """Fixture to create a TestRunner instance for testing."""

        # noinspection PyMissingOrEmptyDocstring
        class TestRunner(_RunnerBase):
            def run(self):
                pass

        def _create_runner(**kwargs):
            default_kwargs = {
                "problem": None,
                "experiment_name": "test_experiment",
                "seed": 1,
                "iteration_list": [1, 2, 3],
                "output_directory": "test_output",
                "override_ctrl_c_handler": False,
            }
            # Update default_kwargs with any provided kwargs
            default_kwargs.update(kwargs)
            return TestRunner(**default_kwargs)

        return _create_runner

    def test_increment_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            initial_count = runner._get_spawn_count()
            runner._increment_spawn_count()

            assert runner._get_spawn_count() == initial_count + 1

    def test_decrement_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            runner._increment_spawn_count()
            initial_count = runner._get_spawn_count()
            runner._decrement_spawn_count()

            assert runner._get_spawn_count() == initial_count - 1

    def test_get_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            initial_spawn_count = runner._get_spawn_count()
            runner._increment_spawn_count()
            incremented_spawn_count = runner._get_spawn_count()
            assert incremented_spawn_count == initial_spawn_count + 1

            runner._decrement_spawn_count()
            decremented_spawn_count = runner._get_spawn_count()
            assert decremented_spawn_count == initial_spawn_count

    def test_abort_sets_abort_flag(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            runner.abort()

            assert runner.has_aborted() is True

    def test_has_aborted_after_abort_called(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(seed=SEED, iteration_list=[0])
            runner.abort()

            assert runner.has_aborted() is True

    def test_set_replay_mode(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            assert not runner.replay_mode()

            runner.set_replay_mode()
            assert runner.replay_mode()

            runner.set_replay_mode(False)
            assert not runner.replay_mode()

    def test_replay_mode(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(replay=True)
            assert runner.replay_mode() is True

            runner.set_replay_mode(False)
            assert runner.replay_mode() is False

    def test_setup_method(self, _test_runner_fixture):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=False):
            runner = _test_runner_fixture(problem="dummy_problem", seed=SEED, iteration_list=[0, 1, 2], output_directory="test_output")
            runner._setup()

            assert runner._raw_run_stats == []
            assert runner._fitness_curves == []
            assert runner._curve_base == 0
            assert runner._iteration_times == []
            assert runner._copy_zero_curve_fitness_from_first == runner._copy_zero_curve_fitness_from_first_original
            assert runner._current_logged_algorithm_args == {}
            mock_makedirs.assert_called_once_with("test_output")

    def test_tear_down_restores_original_sigint_handler(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            original_handler = signal.getsignal(signal.SIGINT)
            runner = _test_runner_fixture()
            runner._tear_down()
            restored_handler = signal.getsignal(signal.SIGINT)

            assert restored_handler == original_handler

    def test_log_current_argument(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(seed=SEED, iteration_list=[0, 1, 2])
            arg_name = "test_arg"
            arg_value = "test_value"
            runner._log_current_argument(arg_name, arg_value)

            assert runner._current_logged_algorithm_args[arg_name] == arg_value

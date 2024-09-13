"""Unit tests for runners/_nn_runner_base.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pandas as pd
import numpy as np
import sklearn.metrics as skmt
from unittest.mock import patch, MagicMock, mock_open

from tests.globals import SEED

# noinspection PyProtectedMember
from mlrose_ky.runners._nn_runner_base import _NNRunnerBase


class TestNNRunnerBase:
    """Tests for _NNRunnerBase."""

    def test_nn_runner_base_initialization(self):
        """Test _NNRunnerBase initialization with default parameters"""
        x_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=100)
        x_test = np.random.rand(20, 10)
        y_test = np.random.randint(2, size=20)
        experiment_name = "test_experiment"
        iteration_list = [1, 2, 3]
        grid_search_parameters = {"param1": [0.1, 0.2], "param2": [1, 2]}
        grid_search_scorer_method = skmt.accuracy_score

        runner = _NNRunnerBase(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_name=experiment_name,
            seed=SEED,
            iteration_list=iteration_list,
            grid_search_parameters=grid_search_parameters,
            grid_search_scorer_method=grid_search_scorer_method,
        )

        assert np.array_equal(runner.x_train, x_train)
        assert np.array_equal(runner.y_train, y_train)
        assert np.array_equal(runner.x_test, x_test)
        assert np.array_equal(runner.y_test, y_test)
        assert runner._experiment_name == experiment_name
        assert runner.seed == SEED
        assert runner.iteration_list == iteration_list
        assert runner.grid_search_parameters == runner.build_grid_search_parameters(grid_search_parameters)
        assert runner._scorer_method == grid_search_scorer_method
        assert runner.cv == 5
        assert runner.generate_curves is True
        assert runner._output_directory is None
        assert runner.verbose_grid_search is True
        assert runner.override_ctrl_c_handler is True
        assert runner.n_jobs == 1
        assert runner._replay_mode.value is False
        assert runner.cv_results_df is None
        assert runner.best_params is None

    def test_nn_runner_base_run_method(self):
        """Test _NNRunnerBase run method execution with mock data"""
        x_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=100)
        x_test = np.random.rand(20, 10)
        y_test = np.random.randint(2, size=20)
        experiment_name = "test_experiment"
        iteration_list = [1, 2, 3]
        grid_search_parameters = {"param1": [0.1, 0.2], "param2": [1, 2]}
        grid_search_scorer_method = skmt.accuracy_score

        runner = _NNRunnerBase(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_name=experiment_name,
            seed=SEED,
            iteration_list=iteration_list,
            grid_search_parameters=grid_search_parameters,
            grid_search_scorer_method=grid_search_scorer_method,
        )

        # Create a mock GridSearchCV object
        mock_grid_search_result = MagicMock()
        mock_best_estimator = MagicMock()
        mock_best_estimator.runner = MagicMock()

        # Mock the predict method
        mock_best_estimator.predict.return_value = np.random.randint(2, size=y_test.shape)
        mock_grid_search_result.best_estimator_ = mock_best_estimator

        # Mock score method to avoid exceptions
        runner.score = MagicMock(return_value=0.8)

        with (
            patch.object(runner, "_setup", return_value=None) as mock_setup,
            patch.object(runner, "perform_grid_search", return_value=mock_grid_search_result) as mock_grid_search,
            patch.object(runner, "_tear_down", return_value=None) as mock_tear_down,
            patch.object(runner, "_print_banner", return_value=None) as mock_print_banner,
        ):
            runner.run()

            # Verify calls
            mock_setup.assert_called_once()
            mock_grid_search.assert_called_once()
            mock_tear_down.assert_called_once()
            mock_print_banner.assert_called()

        # Additional check to ensure prediction is made
        mock_best_estimator.predict.assert_called_once_with(runner.x_test)

    def test_nn_runner_base_teardown_removes_files(self):
        """Test _NNRunnerBase _tear_down method to ensure suboptimal files are removed"""
        with patch("os.makedirs"):
            runner = _NNRunnerBase(
                x_train=np.random.rand(100, 10),
                y_train=np.random.randint(2, size=100),
                x_test=np.random.rand(20, 10),
                y_test=np.random.randint(2, size=20),
                experiment_name="test_experiment",
                seed=SEED,
                iteration_list=[1, 2, 3],
                grid_search_parameters={"param1": [0.1, 0.2], "param2": [1, 2]},
                grid_search_scorer_method=skmt.accuracy_score,
                output_directory="test_output",
            )

            runner.get_runner_name = MagicMock(return_value="TestRunner")
            runner.best_params = {"param1": 0.1, "param2": 1}
            runner._output_directory = "test_output"
            runner.replay_mode = MagicMock(return_value=False)

            # Mock the list of filenames
            mock_file = mock_open(read_data=b"mocked binary data")  # Note the `b` prefix for binary data
            with (
                patch("os.path.isdir", return_value=True),
                patch(
                    "os.listdir", return_value=["testrunner__test_experiment__df_.p", "testrunner__test_experiment__df_1.p"]
                ) as mock_listdir,
                patch("os.remove") as mock_remove,
                patch("pandas.DataFrame", return_value=None) as mock_dataframe,
                patch.object(runner, "_check_match", return_value=False) as mock_check_match,
                patch("builtins.open", mock_file),
                patch("pickle.load", return_value=MagicMock()),
            ):
                runner._tear_down()

                # Validate os.listdir was called the correct number of times and with the correct arguments
                assert mock_listdir.call_count == 3
                mock_listdir.assert_any_call("test_output/test_experiment")
                mock_check_match.assert_called()
                mock_remove.assert_called()
                mock_file.assert_called()

    def test_nn_runner_base_get_pickle_filename_root(self):
        """Test _NNRunnerBase _get_pickle_filename_root method to ensure correct filename root generation"""
        with patch("os.makedirs"):
            runner = _NNRunnerBase(
                x_train=np.random.rand(100, 10),
                y_train=np.random.randint(2, size=100),
                x_test=np.random.rand(20, 10),
                y_test=np.random.randint(2, size=20),
                experiment_name="test_experiment",
                seed=SEED,
                iteration_list=[1, 2, 3],
                grid_search_parameters={"param1": [0.1, 0.2], "param2": [1, 2]},
                grid_search_scorer_method=skmt.accuracy_score,
                output_directory="test_output",
            )

            with patch.object(runner, "_sanitize_value", return_value="sanitized_value"):
                filename_root = runner._get_pickle_filename_root("test_file")
                assert filename_root.startswith("test_output/test_experiment/_nnrunnerbase__test_experiment__test_file")

    def test_nn_runner_base_check_match(self):
        """Test _NNRunnerBase _check_match static method to ensure correct match checking"""
        df_ref = pd.DataFrame({"col1": [1], "col2": [2]})
        df_to_check = pd.DataFrame({"col1": [1, 1], "col2": [2, 3]})

        match_found = _NNRunnerBase._check_match(df_ref, df_to_check)
        assert match_found is True

        df_to_check = pd.DataFrame({"col1": [3, 4], "col2": [5, 6]})
        match_found = _NNRunnerBase._check_match(df_ref, df_to_check)
        assert match_found is False

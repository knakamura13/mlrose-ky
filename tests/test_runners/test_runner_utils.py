"""Unit tests for runners/utils.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import os

from unittest.mock import patch

from mlrose_ky.runners import build_data_filename


class TestRunnerUtils:
    def test_build_data_filename_default(self):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=True):
            output_directory = "test_output"
            runner_name = "TestRunner"
            experiment_name = "experiment"
            df_name = "results"

            expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results")
            result = build_data_filename(output_directory, runner_name, experiment_name, df_name)

            assert result == expected_filename
            mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    def test_build_data_filename_with_params(self):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=True):
            output_directory = "test_output"
            runner_name = "TestRunner"
            experiment_name = "experiment"
            df_name = "results"
            x_param = "x_value"
            y_param = "y_value"

            expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results_x_value__y_value")
            result = build_data_filename(output_directory, runner_name, experiment_name, df_name, x_param, y_param)

            assert result == expected_filename
            mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    def test_build_data_filename_with_extension(self):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=True):
            output_directory = "test_output"
            runner_name = "TestRunner"
            experiment_name = "experiment"
            df_name = "results"
            ext = "csv"

            expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results.csv")
            result = build_data_filename(output_directory, runner_name, experiment_name, df_name, ext=ext)

            assert result == expected_filename
            mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)

    def test_build_data_filename_creates_directory(self):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=False):
            output_directory = "test_output"
            runner_name = "TestRunner"
            experiment_name = "experiment"
            df_name = "results"

            expected_filename = os.path.join(output_directory, experiment_name, "testrunner__experiment__results")
            result = build_data_filename(output_directory, runner_name, experiment_name, df_name)

            mock_makedirs.assert_called_once_with(os.path.join(output_directory, experiment_name), exist_ok=True)
            assert result == expected_filename

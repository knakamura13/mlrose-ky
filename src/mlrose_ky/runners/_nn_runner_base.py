"""Base class for running neural network weights optimization experiments, including grid search functionality."""

# Authors: Andrew Rollings (modified by Kyle Nakamura)
# License: BSD 3-clause

import hashlib
import logging
import os
import pickle as pk
import time
from abc import ABC
from typing import Callable, Any

import numpy as np
import pandas as pd

from mlrose_ky.gridsearch import GridSearchMixin
from mlrose_ky.runners._runner_base import _RunnerBase


class _NNRunnerBase(_RunnerBase, GridSearchMixin, ABC):
    """
    A base class for running neural network experiments with grid search.
    It extends functionality from _RunnerBase and GridSearchMixin.

    This class provides methods for setting up and executing grid search over neural network hyperparameters,
    handling cross-validation, saving results, and managing file operations.

    Attributes
    ----------
    x_train : np.ndarray
        Features for the training data.
    y_train : np.ndarray
        Labels for the training data.
    x_test : np.ndarray
        Features for the testing data.
    y_test : np.ndarray
        Labels for the testing data.
    _experiment_name : str
        Name of the experiment.
    seed : int
        Random seed to ensure reproducibility.
    iteration_list : list[int]
        List of iteration counts to perform.
    grid_search_parameters : dict
        Hyperparameters for grid search.
    cv : int
        Number of cross-validation folds.
    generate_curves : bool
        Whether to generate learning curves during the experiment.
    _output_directory : str | None
        Directory where outputs will be saved.
    verbose_grid_search : bool
        Whether to output detailed grid search information.
    override_ctrl_c_handler : bool
        Whether to override the default CTRL+C handler.
    n_jobs : int
        Number of parallel jobs for grid search.
    cv_results_df : pd.DataFrame | None
        DataFrame to store cross-validation results.
    best_params : dict | None
        Dictionary storing the best parameters found during grid search.
    """

    _interrupted_results: list = []

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
        iteration_list: list[int],
        grid_search_parameters: dict,
        grid_search_scorer_method: Callable,
        cv: int = 5,
        generate_curves: bool = True,
        output_directory: str = None,
        verbose_grid_search: bool = True,
        override_ctrl_c_handler: bool = True,
        n_jobs: int = 1,
        replay: bool = False,
        **kwargs,
    ):
        """
        Initializes the _NNRunnerBase class.

        Parameters
        ----------
        x_train : np.ndarray
            Features for the training data.
        y_train : np.ndarray
            Labels for the training data.
        x_test : np.ndarray
            Features for the testing data.
        y_test : np.ndarray
            Labels for the testing data.
        experiment_name : str
            Name of the experiment.
        seed : int
            Random seed to ensure reproducibility.
        iteration_list : list[int]
            List of iteration counts to perform.
        grid_search_parameters : dict
            Hyperparameters for grid search.
        grid_search_scorer_method : Callable
            Scorer method for evaluating the grid search.
        cv : int, optional, default=5
            Number of cross-validation folds.
        generate_curves : bool, optional, default=True
            Whether to generate learning curves.
        output_directory : str | None, optional, default=None
            Directory where outputs will be saved.
        verbose_grid_search : bool, optional, default=True
            Whether to output detailed grid search information.
        override_ctrl_c_handler : bool, optional, default=True
            Whether to override the default CTRL+C handler.
        n_jobs : int, optional, default=1
            Number of parallel jobs for grid search.
        replay : bool, optional, default=False
            Whether to replay previous results.
        **kwargs :
            Additional hyperparameters for grid search.
        """
        super().__init__(
            problem=None,
            experiment_name=experiment_name,
            seed=seed,
            iteration_list=iteration_list,
            generate_curves=generate_curves,
            output_directory=output_directory,
            replay=replay,
            override_ctrl_c_handler=override_ctrl_c_handler,
            copy_zero_curve_fitness_from_first=True,
        )

        GridSearchMixin.__init__(self, scorer_method=grid_search_scorer_method)

        self.classifier: Any = None
        self.grid_search_parameters: dict = self.build_grid_search_parameters(grid_search_parameters, **kwargs)
        self.x_train: np.ndarray = x_train
        self.y_train: np.ndarray = y_train
        self.x_test: np.ndarray = x_test
        self.y_test: np.ndarray = y_test
        self.cv: int = cv
        self.n_jobs: int = n_jobs
        self.verbose_grid_search: bool = verbose_grid_search
        self.cv_results_df: pd.DataFrame | None = None
        self.best_params: dict | None = None

    def run(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Any | None]:
        """
        Executes the runner, performing grid search and handling the results.

        Returns
        -------
        tuple
            Contains the run statistics DataFrame, curves DataFrame, CV results DataFrame, and the grid search result.
        """
        try:
            self._setup()

            logging.info(f"Running experiment: {self._experiment_name}")

            # Replay mode allows reusing previous grid search results, avoiding re-execution
            if self.replay_mode():
                gsr_name = f"{super()._get_pickle_filename_root('grid_search_results')}.p"
                with open(gsr_name, "rb") as pickle_file:
                    search_results = pk.load(pickle_file)
            else:
                run_start = time.perf_counter()
                search_results = self._perform_grid_search(
                    classifier=self.classifier,
                    parameters=self.grid_search_parameters,
                    x_train=self.x_train,
                    y_train=self.y_train,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose_grid_search,
                )
                run_end = time.perf_counter()
                logging.info(f"Run time: {run_end - run_start}")

            # Updates internal attributes with the best estimator found by grid search
            self.__dict__.update(search_results.best_estimator_.runner.__dict__)
            self.best_params = search_results.best_params_

            self.cv_results_df = self._make_cv_results_data_frame(search_results.cv_results_)
            extra_data_frames = {"cv_results_df": self.cv_results_df}

            self._create_and_save_run_data_frames(extra_data_frames=extra_data_frames, final_save=True)

            # Save grid search results to disk
            try:
                self._dump_pickle_to_disk(search_results, "grid_search_results", final_save=True)
            except (OSError, IOError, pk.PickleError):
                pass

            # Perform predictions and score using the best estimator found in the search
            try:
                y_pred = search_results.best_estimator_.predict(self.x_test)
                score = self.score(y_pred=y_pred, y_true=self.y_train)
                self._print_banner(f"Score: {score}")
            except Exception as e:
                logging.error(f"An error occurred while predicting: {e}")

            return self.run_stats_df, self.curves_df, self.cv_results_df, search_results
        except KeyboardInterrupt:
            # Handling graceful termination in case of manual interruption
            return None, None, None, None
        finally:
            self._tear_down()

    def _get_pickle_filename_root(self, name: str) -> str:
        """
        Generates a root filename for pickle files with a hash based on the current algorithm arguments.

        Parameters
        ----------
        name : str
            The base name for the pickle file.

        Returns
        -------
        str
            The root filename with a hash appended.
        """
        filename_root = super()._get_pickle_filename_root(name)

        # Create a unique hash based on argument values (excluding state-related keys)
        arg_text = "".join([f"{k}_{self._sanitize_value(v)}_" for k, v in self._current_logged_algorithm_args.items() if "state" not in k])
        arg_hash = f"__{hashlib.md5(arg_text.encode()).hexdigest()}".upper() if len(arg_text) > 0 else ""

        return filename_root + arg_hash

    @staticmethod
    def _check_match(df_reference: pd.DataFrame, df_to_check: pd.DataFrame) -> bool:
        """
        Checks if two DataFrames have matching rows based on their columns.

        Parameters
        ----------
        df_reference : pd.DataFrame
            The reference DataFrame to check against.
        df_to_check : pd.DataFrame
            The DataFrame to check for matching rows.

        Returns
        -------
        bool
            True if a matching row is found, False otherwise.
        """
        cols = [col for col in df_reference.columns]

        for _, row in df_to_check.iterrows():
            # Returns True as soon as a match is found
            if all(df_reference[col][0] == row[col] for col in cols):
                return True

        return False

    def _tear_down(self, filename: str | None = None):
        """
        Finalizes the runner, ensuring that the proper files are saved or cleaned up.

        Parameters
        ----------
        filename : str | None, optional, default=None
            Filename to clean up.
        """
        if self.best_params is None or self.replay_mode() is None or self._output_directory is None:
            super()._tear_down()
            return

        filename_root = super()._get_pickle_filename_root("")
        path = os.path.join(*filename_root.split(os.sep)[:-1])
        filename_part = filename_root.split(os.sep)[-1]

        if not os.path.isdir(path) and path[0] != os.sep:
            path = f"{os.sep}{path}"

        filenames = [fn for fn in os.listdir(str(path)) if (filename_part in fn and fn.endswith(".p") and "_df_" in fn)]

        if not filenames:
            raise FileNotFoundError(f"No matching filenames found in path: {path}")

        df_best_params = pd.DataFrame([{k: self._sanitize_value(v) for k, v in self.best_params.items()}])

        correct_files = []
        incorrect_files = []
        for fn in filenames:
            filename = os.path.join(str(path), fn)
            with open(filename, "rb") as pickle_file:
                try:
                    df = pk.load(pickle_file)
                    found = self._check_match(df_best_params, df)
                    (correct_files if found else incorrect_files).append(filename)
                except (EOFError, pk.PickleError):
                    pass

        # Extracts md5 hashes from correct and incorrect files for renaming or deletion
        correct_md5s = {p.split("_")[-1][:-2] for p in correct_files}
        incorrect_md5s = {p.split("_")[-1][:-2] for p in incorrect_files}

        # Remove files corresponding to incorrect md5 hashes
        all_incorrect_files = [
            os.path.join(str(path), fn) for incorrect_md5 in incorrect_md5s for fn in os.listdir(str(path)) if incorrect_md5 in fn
        ]
        for _filename in all_incorrect_files:
            os.remove(_filename)

        # Rename correct files by removing md5 hash from the filename
        all_correct_files = [
            (os.path.join(str(path), fn), f"__{_correct_md5}")
            for _correct_md5 in correct_md5s
            for fn in os.listdir(str(path))
            if _correct_md5 in fn
        ]
        for _filename, _correct_md5 in all_correct_files:
            correct_filename = _filename.replace(_correct_md5, "")
            if os.path.exists(correct_filename):
                os.rename(correct_filename, f"{correct_filename}.bak")
            os.rename(_filename, correct_filename)

        super()._tear_down()

    @staticmethod
    def _make_cv_results_data_frame(cv_results: dict) -> pd.DataFrame:
        """
        Creates a DataFrame from cross-validation results.

        Parameters
        ----------
        cv_results : dict
            Cross-validation results.

        Returns
        -------
        pd.DataFrame
            The cross-validation results as a DataFrame.
        """
        cv_results = cv_results.copy()
        param_prefix = "param_"
        param_labels = [k for k in cv_results if param_prefix in k]

        new_param_values = {p: [] for p in param_labels}
        for v in cv_results["params"]:
            for p in param_labels:
                param_label = p.replace(param_prefix, "")
                new_param_values[p].append(_NNRunnerBase._sanitize_value(v[param_label]))

        cv_results.update(new_param_values)
        df = pd.DataFrame(cv_results)
        df.dropna(inplace=True)

        return df

    @staticmethod
    def build_grid_search_parameters(grid_search_parameters: dict, **kwargs) -> dict:
        """
        Builds a dictionary of grid search parameters by combining initial parameters and additional arguments.

        Parameters
        ----------
        grid_search_parameters : dict
            Initial grid search parameters.
        **kwargs :
            Additional hyperparameters for grid search.

        Returns
        -------
        dict
            Combined grid search parameters.
        """
        return {**grid_search_parameters, **kwargs}

    def _grid_search_score_intercept(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None, adjusted: bool = False
    ) -> float:
        """
        Intercepts the grid search scoring process to handle special cases, particularly in aborted runs.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        sample_weight : np.ndarray | None, optional, default=None
            Sample weights for scoring.
        adjusted : bool, optional, default=False
            Whether to adjust the score.

        Returns
        -------
        float
            The grid search score, or NaN if the run was aborted.
        """
        if not self.classifier.fit_started_ and self.has_aborted():
            return np.NaN

        return super()._grid_search_score_intercept(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, adjusted=adjusted)

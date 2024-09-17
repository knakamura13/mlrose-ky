"""Base class for running optimization experiments, including multiprocessing, logging, error handling, and result saving."""

# Authors: Andrew Rollings (modified by Kyle Nakamura)
# License: BSD 3-clause

import ctypes
import inspect
import itertools
import logging
import multiprocessing
import os
import pickle as pk
import signal
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from mlrose_ky.decorators import get_short_name
from mlrose_ky.runners.utils import build_data_filename


class _RunnerBase(ABC):
    """
    Abstract base class for running and managing optimization experiments.

    This class provides a framework for setting up, running, and managing the lifecycle
    of optimization experiments. It handles tasks such as logging, error handling,
    signal interruption, dynamic naming, and result saving to both pickle and CSV files.
    The class is designed to be extended by concrete subclasses that implement the
    `run` method, which defines the specific behavior of the experiment.

    Attributes
    ----------
    __abort : multiprocessing.Value
        Flag to signal abortion of the experiment.
    __spawn_count : multiprocessing.Value
        Tracks the number of spawned processes in parallel execution.
    __replay : multiprocessing.Value
        Indicates whether replay mode is active.
    __original_sigint_handler : Any
        Stores the original signal handler for Ctrl-C.
    __sigint_params : tuple[int, Any] | None
        Stores the signal and frame parameters when Ctrl-C is triggered.
    """

    __abort: multiprocessing.Value = multiprocessing.Value(ctypes.c_bool)
    __spawn_count: multiprocessing.Value = multiprocessing.Value(ctypes.c_uint)
    __replay: multiprocessing.Value = multiprocessing.Value(ctypes.c_bool)
    __original_sigint_handler: Any = None
    __sigint_params: tuple[int, Any] | None = None

    def __init__(
        self,
        problem: Any,
        experiment_name: str,
        seed: int,
        iteration_list: list[int],
        max_attempts: int = 500,
        generate_curves: bool = True,
        output_directory: str = None,
        copy_zero_curve_fitness_from_first: bool = False,
        replay: bool = False,
        override_ctrl_c_handler: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the runner with required parameters, including problem setup,
        iteration controls, and output settings.

        Parameters
        ----------
        problem : Any
            The optimization problem instance to solve.
        experiment_name : str
            The name of the experiment.
        seed : int
            Seed for random number generation.
        iteration_list : list[int]
            List of iterations to log results.
        max_attempts : int, optional
            Maximum number of attempts for optimization, default=500.
        generate_curves : bool, optional
            Whether to generate fitness curves, default=True.
        output_directory : str, optional
            Directory to save experiment result, default=None.
        copy_zero_curve_fitness_from_first : bool, optional, default=False
            Whether to copy the first curve fitness value to the zeroth iteration.
        replay : bool, optional
            Whether to enable replay mode, default=False.
        override_ctrl_c_handler : bool, optional, default=True
            Whether to override the Ctrl-C signal handler.
        **kwargs : Any
            Additional keyword arguments for experiment configuration.
        """
        self.problem: Any = problem
        self.seed: int = seed
        self.iteration_list: list[int] = iteration_list
        self.max_attempts: int = max_attempts
        self.generate_curves: bool = generate_curves
        self.parameter_description_dict: dict[str, str] = {}
        self.override_ctrl_c_handler: bool = override_ctrl_c_handler

        # Initialize output and state-tracking variables
        self.run_stats_df: pd.DataFrame | None = None
        self.curves_df: pd.DataFrame | None = None
        self._raw_run_stats: list[dict[str, Any]] = []
        self._fitness_curves: list[dict[str, Any]] = []
        self._curve_base: int = 0
        self._copy_zero_curve_fitness_from_first: bool = copy_zero_curve_fitness_from_first
        self._copy_zero_curve_fitness_from_first_original: bool = copy_zero_curve_fitness_from_first
        self._extra_args: dict[str, Any] = kwargs
        self._output_directory: str | None = output_directory
        self._dynamic_short_name: str | None = None
        self._experiment_name: str = experiment_name
        self._current_logged_algorithm_args: dict[str, Any] = {}
        self._run_start_time: float | None = None
        self._iteration_times: list[float] = []
        self._first_curve_synthesized: bool = False

        if replay:
            self.set_replay_mode()

        self._increment_spawn_count()

    @classmethod
    def runner_name(cls) -> str:
        """Get a short name for the runner class."""
        return get_short_name(cls)

    def dynamic_runner_name(self) -> str:
        """Get the dynamic name of the runner, if set, otherwise return the default runner name."""
        dynamic_runner_name = self._dynamic_short_name or self.runner_name()

        if not dynamic_runner_name:
            raise ValueError("dynamic_runner_name is None")

        return dynamic_runner_name

    def _set_dynamic_runner_name(self, name: str):
        """Set a dynamic runner name."""
        self._dynamic_short_name = name

    @staticmethod
    def _print_banner(text: str):
        """Print a formatted banner for logging."""
        logging.info("*" * len(text))
        logging.info(text)
        logging.info("*" * len(text))

    @staticmethod
    def _sanitize_value(value: Any) -> str:
        """Sanitize a value for logging, handling different types appropriately."""
        if isinstance(value, (tuple, list)):
            sanitized_value = str(value)
        elif isinstance(value, np.ndarray):
            sanitized_value = str(list(value))
        elif callable(value):
            sanitized_value = get_short_name(value)
        else:
            sanitized_value = str(value)  # Handle non-callable types like floats, ints, etc.

        return sanitized_value

    @abstractmethod
    def run(self):
        """Abstract method to be implemented by subclasses."""
        pass

    def _increment_spawn_count(self):
        """Increment the spawn count for the runner (used in parallel execution)."""
        with self.__spawn_count.get_lock():
            self.__spawn_count.value += 1

    def _decrement_spawn_count(self):
        """Decrement the spawn count for the runner."""
        with self.__spawn_count.get_lock():
            self.__spawn_count.value -= 1

    def _get_spawn_count(self) -> int:
        """Return the current spawn count and log it."""
        self._print_banner(f"*** Spawn Count Remaining: {self.__spawn_count.value} ***")
        return self.__spawn_count.value

    def abort(self):
        """Set the abort flag to signal that the runner should stop execution."""
        self._print_banner("*** ABORTING ***")

        with self.__abort.get_lock():
            self.__abort.value = True

    def has_aborted(self) -> bool:
        """Return whether the abort flag has been set."""
        return self.__abort.value

    def set_replay_mode(self, value: bool = True):
        """Enable or disable replay mode, which reuses previous results."""
        with self.__replay.get_lock():
            self.__replay.value = value

    def replay_mode(self) -> bool:
        """Check if replay mode is enabled."""
        return self.__replay.value

    def _setup(self):
        """Prepare the runner by clearing stats, setting up directories, and handling Ctrl-C interrupts."""
        self._raw_run_stats = []
        self._fitness_curves = []
        self._curve_base = 0

        self._iteration_times = []
        self._copy_zero_curve_fitness_from_first = self._copy_zero_curve_fitness_from_first_original
        self._current_logged_algorithm_args.clear()

        # Create the output directory if it doesn't exist
        if self._output_directory is not None:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)

        # Set up Ctrl-C handler if necessary
        if self.override_ctrl_c_handler:
            if self.__original_sigint_handler is None:
                self.__original_sigint_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._ctrl_c_handler)

    def _ctrl_c_handler(self, sig: int, frame: Any):
        """
        Handle Ctrl-C interruptions by saving progress and aborting the run.

        Parameters
        ----------
        sig : int
            Signal number (e.g., SIGINT).
        frame : Any
            Current stack frame.
        """
        logging.info("Interrupted - saving progress so far")
        self.__sigint_params = (sig, frame)
        self.abort()

    def _tear_down(self):
        """Clean up after the experiment, restoring signal handlers and managing resources."""
        if not self.override_ctrl_c_handler:
            return

        try:
            # Restore the original Ctrl-C handler
            self._decrement_spawn_count()
            if self.__original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self.__original_sigint_handler)
                if self.has_aborted() and self._get_spawn_count() == 0:
                    sig, frame = self.__sigint_params
                    self.__original_sigint_handler(sig, frame)
        except (ValueError, TypeError, AttributeError, Exception) as e:
            logging.error(f"Problem restoring SIGINT handler: {e}")

    def _log_current_argument(self, arg_name: str, arg_value: Any):
        """Log the current argument passed to the algorithm."""
        self._current_logged_algorithm_args[arg_name] = arg_value

    def run_experiment_(self, algorithm: Any, **kwargs: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Execute the experiment with the provided algorithm and log the results.

        Parameters
        ----------
        algorithm : Any
            The optimization algorithm to run.
        **kwargs : Any
            Additional arguments for the algorithm.

        Returns
        -------
        tuple[pd.DataFrame | None, pd.DataFrame | None]
            DataFrames containing the run statistics and fitness curves.
        """
        self._setup()

        # Generate all combinations of parameter values for the experiment
        values = [([(k, v) for v in vs]) for (k, (n, vs)) in kwargs.items() if vs is not None]

        self.parameter_description_dict = {k: n for (k, (n, vs)) in kwargs.items() if vs is not None}

        value_sets = list(itertools.product(*values))

        logging.info(f"Running {self.dynamic_runner_name()}")
        run_start = time.perf_counter()

        for value_set in value_sets:
            total_args = dict(value_set)

            if "max_iters" not in total_args:
                total_args["max_iters"] = int(max(self.iteration_list))

            self._run_one_experiment(algorithm, total_args)

        run_end = time.perf_counter()
        logging.info(f"Run time: {run_end - run_start:.2f} seconds")

        self._create_and_save_run_data_frames(final_save=True)
        self._tear_down()

        return self.run_stats_df, self.curves_df

    def _run_one_experiment(self, algorithm: Any, total_args: dict[str, Any], **params: Any):
        """
        Execute a single iteration of the experiment.

        Parameters
        ----------
        algorithm : Any
            The algorithm to run.
        total_args : dict[str, Any]
            The arguments passed to the algorithm.
        **params : Any
            Additional parameters for the experiment.
        """
        if self._extra_args:
            total_args.update(self._extra_args)

        total_args.update(params)
        user_info = [(k, v) for k, v in total_args.items()]

        self._invoke_algorithm(
            algorithm=algorithm,
            problem=self.problem,
            max_attempts=self.max_attempts,
            curve=self.generate_curves,
            user_info=user_info,
            **total_args,
        )

    def _create_and_save_run_data_frames(self, extra_data_frames: dict[str, pd.DataFrame] = None, final_save: bool = False):
        """
        Save the collected run statistics and fitness curves to disk.

        Parameters
        ----------
        extra_data_frames : dict[str, pd.DataFrame], optional
            Additional DataFrames to save.
        final_save : bool, optional
            Whether this is the final save of the experiment (default False).
        """
        self.run_stats_df = pd.DataFrame(self._raw_run_stats)
        self.curves_df = pd.DataFrame(self._fitness_curves)

        if self._output_directory:
            if not self.run_stats_df.empty:
                self._dump_df_to_disk(self.run_stats_df, df_name="run_stats_df", final_save=final_save)

            if self.generate_curves and not self.curves_df.empty:
                self._dump_df_to_disk(self.curves_df, df_name="curves_df", final_save=final_save)

            if isinstance(extra_data_frames, dict):
                for name, df in extra_data_frames.items():
                    self._dump_df_to_disk(df, df_name=name, final_save=final_save)

    def _dump_df_to_disk(self, df: pd.DataFrame, df_name: str, final_save: bool = False):
        """
        Save the DataFrame to disk as both a pickle and CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to save.
        df_name : str
            The name of the DataFrame.
        final_save : bool, optional
            Whether this is the final save (default False).
        """
        filename_root = self._dump_pickle_to_disk(object_to_pickle=df, name=df_name)
        df.to_csv(f"{filename_root}.csv")

        if final_save:
            logging.info(f"Saved: [{filename_root}.csv]")

    def _get_pickle_filename_root(self, name: str) -> str:
        """Generate the root filename for the pickle file based on experiment metadata."""
        return build_data_filename(
            output_directory=self._output_directory,
            runner_name=self.dynamic_runner_name(),
            experiment_name=self._experiment_name,
            df_name=name,
        )

    def _dump_pickle_to_disk(self, object_to_pickle: Any, name: str, final_save: bool = False) -> str | None:
        """
        Save an object as a pickle file on disk.

        Parameters
        ----------
        object_to_pickle : Any
            The object to pickle.
        name : str
            The name of the pickle file.
        final_save : bool, optional
            Whether this is the final save (default False).

        Returns
        -------
        str | None
            The root filename of the saved pickle file, or None if no directory is provided.
        """
        if self._output_directory is None:
            return None

        filename_root = self._get_pickle_filename_root(name)
        pk.dump(object_to_pickle, open(f"{filename_root}.p", "wb"))

        if final_save:
            logging.info(f"Saved: [{filename_root}.p]")

        return filename_root

    def _load_pickles(self) -> bool:
        """Load pickled fitness curves and run statistics from disk."""
        curves_df_filename = f"{self._get_pickle_filename_root('curves_df')}.p"
        run_stats_df_filename = f"{self._get_pickle_filename_root('run_stats_df')}.p"

        self.curves_df = None
        self.run_stats_df = None

        if os.path.exists(curves_df_filename):
            with open(curves_df_filename, "rb") as pickle_file:
                try:
                    self.curves_df = pk.load(pickle_file)
                except (OSError, IOError, pk.PickleError):
                    pass

        if os.path.exists(run_stats_df_filename):
            with open(run_stats_df_filename, "rb") as pickle_file:
                try:
                    self.run_stats_df = pk.load(pickle_file)
                except (OSError, IOError, pk.PickleError):
                    pass

        return self.curves_df is not None and self.run_stats_df is not None

    def _invoke_algorithm(
        self,
        algorithm: Any,
        problem: Any,
        max_attempts: int,
        curve: bool,
        user_info: list[tuple[str, Any]],
        additional_algorithm_args: dict[str, Any] = None,
        **total_args: Any,
    ) -> tuple | None:
        """
        Invoke the algorithm with the given parameters, either running it or loading previous results.

        Parameters
        ----------
        algorithm : Any
            The algorithm to run.
        problem : Any
            The problem instance to solve.
        max_attempts : int
            The maximum number of attempts allowed.
        curve : bool
            Whether to generate fitness curves.
        user_info : list[tuple[str, Any]]
            Additional information to log.
        additional_algorithm_args : dict[str, Any] | None, optional
            Extra algorithm arguments.
        **total_args : Any
            All arguments passed to the algorithm.

        Returns
        -------
        tuple | None
            None if the algorithm was run, or previously loaded results if replay mode is enabled.
        """
        self._current_logged_algorithm_args.update(total_args)
        if additional_algorithm_args is not None:
            self._current_logged_algorithm_args.update(additional_algorithm_args)

        # If replay mode is enabled and previous results are found, load them
        if self.replay_mode() and self._load_pickles():
            return None, None, None

        # Execute the algorithm
        self._print_banner("*** Run START ***")
        np.random.seed(self.seed)

        # Filter arguments to those accepted by the algorithm function signature
        valid_args = [k for k in inspect.signature(algorithm).parameters]
        kwargs = {k: v for k, v in total_args.items() if k in valid_args}

        # Reset the problem instance and run the algorithm
        self._start_run_timing()
        problem.reset()
        result = algorithm(
            problem=problem,
            max_attempts=max_attempts,
            curve=curve,
            random_state=self.seed,
            state_fitness_callback=self._save_state,
            callback_user_info=user_info,
            **kwargs,
        )

        self._print_banner("*** Run END ***")
        self._curve_base = len(self._fitness_curves)

        return result

    def _start_run_timing(self):
        """Start timing the experiment's execution."""
        self._run_start_time = time.perf_counter()

    @staticmethod
    def _create_curve_stat(
        iteration: int, curve_value: tuple[float, int] | dict[str, Any], curve_data: dict[str, Any], t: float = None
    ) -> dict[str, Any]:
        """
        Create a single fitness curve statistic for logging.

        Parameters
        ----------
        iteration : int
            The iteration number.
        curve_value : tuple[float, int] | dict[str, Any]
            The curve's fitness and evaluation values.
        curve_data : dict[str, Any]
            Additional data to log.
        t : float | None, optional
            Time elapsed for this iteration (default None).

        Returns
        -------
        dict[str, Any]
            The fitness curve statistic as a dictionary.
        """
        curve_fitness_value, curve_feval_value = curve_value
        curve_stat = {"Iteration": iteration, "Time": t, "Fitness": curve_fitness_value, "FEvals": curve_feval_value}

        curve_stat.update(curve_data)
        if isinstance(curve_value, dict):
            curve_stat.update(curve_value)

        return curve_stat

    def _save_state(
        self,
        iteration: int,
        state: Any,
        fitness: float,
        user_data: list[tuple[str, Any]],
        attempt: int = 0,
        done: bool = False,
        curve: list[tuple[float, int]] = None,
        fitness_evaluations: int = None,
    ) -> bool:
        """
        Save the state of the experiment, logging key data for each iteration.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        state : Any
            The current state of the problem.
        fitness : float
            The fitness value of the current state.
        user_data : list[tuple[str, Any]]
            Additional data to log.
        attempt : int, optional
            The current attempt number.
        done : bool, optional
            Whether the experiment is complete.
        curve : list[tuple[float, int]] | None, optional
            The fitness curve (default None).
        fitness_evaluations : int | None, optional
            The number of fitness evaluations (default None).

        Returns
        -------
        bool
            True if the experiment should continue, False otherwise.
        """
        # Log iteration timing
        end = time.perf_counter()
        t = end - self._run_start_time
        self._iteration_times.append(t)

        # Skip logging for non-final iterations not in the list
        if iteration > 0 and iteration not in self.iteration_list and not done:
            return True

        # Update logging with current algorithm and user data
        display_data = {**self._current_logged_algorithm_args}
        if user_data:
            display_data.update({n: v for n, v in user_data})
            data_desc = ", ".join([f"{n}:[{get_short_name(v)}]" for n, v in display_data.items()])
            logging.debug(data_desc)

        logging.debug(
            f"runner_name:[{self.dynamic_runner_name()}], experiment_name:[{self._experiment_name}], "
            + ("" if attempt is None else f"attempt:[{attempt}], ")
            + f"iteration:[{iteration}], done:[{done}], "
            f"time:[{t:.2f}], fitness:[{fitness:.4f}]"
        )

        # Log the state as a truncated string for easier viewing
        state_string = str(state).replace("\n", "//")[:200]
        logging.debug(f"\t{state_string}...")
        logging.debug("")

        # Sanitize and log additional user data
        def get_description(name: str) -> str:
            """Return the description for a parameter name."""
            return name if name not in self.parameter_description_dict else self.parameter_description_dict[name]

        def get_info(val: Any) -> dict[str, Any]:
            """Return additional info for a value if available."""
            return val.get_info__(t) if hasattr(val, "get_info__") else {}

        current_iteration_stats = {str(get_description(k)): self._sanitize_value(v) for k, v in self._current_logged_algorithm_args.items()}
        current_iteration_stats.update({str(get_description(k)): self._sanitize_value(v) for k, v in user_data})

        additional_info = {
            k: self._sanitize_value(v)
            for info_dict in (get_info(v) for v in current_iteration_stats.values())
            for k, v in info_dict.items()
        }

        # Determine which iterations to log
        if iteration > 0:
            remaining_iterations = [i for i in self.iteration_list if i >= iteration]
            iterations = [min(remaining_iterations)] if not done else remaining_iterations
        else:
            iterations = [0]

        # Log the run statistics for each iteration
        for i in iterations:
            run_stat = {"Iteration": i, "Fitness": fitness, "FEvals": fitness_evaluations, "Time": t, "State": self._sanitize_value(state)}
            run_stat.update(additional_info)
            run_stat.update(current_iteration_stats)
            self._raw_run_stats.append(run_stat)

        # Generate and save fitness curves if required
        if self.generate_curves and iteration == 0:
            if curve is None:
                curve = [(fitness, fitness_evaluations)]
                self._first_curve_synthesized = True

        if self.generate_curves and curve is not None:
            curve_stats_saved = len(self._fitness_curves)
            total_curve_stats = self._curve_base + len(curve)
            curve_stats_to_save = total_curve_stats - curve_stats_saved

            if self._first_curve_synthesized:
                curve_stats_to_save += 1

            ix_end = iteration + 1
            ix_start = ix_end - curve_stats_to_save
            if ix_start < 0:
                ix_start = 0

            curve_tuples = list(zip(range(ix_start, ix_end), curve[-curve_stats_to_save:]))

            curve_stats = [
                self._create_curve_stat(iteration=ix, curve_value=f, curve_data=current_iteration_stats, t=self._iteration_times[ix])
                for ix, f in curve_tuples
            ]

            self._fitness_curves.extend(curve_stats)

            # Copy the first fitness value to the zeroth iteration if specified
            if self._copy_zero_curve_fitness_from_first and len(self._fitness_curves) > 1:
                self._fitness_curves[0]["Fitness"] = self._fitness_curves[1]["Fitness"]
                self._copy_zero_curve_fitness_from_first = False
            self._create_and_save_run_data_frames()

        return not (self.has_aborted() or done)

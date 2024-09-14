"""
Class for running optimization experiments using mlrose_ky.NNClassifier, including grid search functionality.

Example usage:
    from mlrose_ky.runners import NNGSRunner

    grid_search_parameters = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
        'learning_rate': [0.001, 0.002, 0.003],                         # nn params
        'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    })

    nnr = NNGSRunner(x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     experiment_name='nn_test',
                     algorithm=mlrose_ky.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=500,
                     generate_curves=True,
                     seed=200972)

    results: skms.GridSearchCV = nnr.run()
"""

# Authors: Andrew Rollings (modified by Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Callable, Optional

import numpy as np
import sklearn.metrics as skmt

from mlrose_ky.decorators import short_name, get_short_name
from mlrose_ky.neural import NNClassifier
from mlrose_ky.runners._nn_runner_base import _NNRunnerBase


@short_name("nngs")
class NNGSRunner(_NNRunnerBase):
    """
    A runner for performing optimization experiments using the mlrose_ky.NNClassifier.

    This class extends _NNRunnerBase and provides grid search functionality for optimizing
    the hyperparameters of an NNClassifier model.

    Attributes
    ----------
    classifier : NNClassifier
        The classifier used for the experiment.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
        iteration_list: list[int],
        algorithm: str,
        grid_search_parameters: dict,
        grid_search_scorer_method: Optional[Callable] = skmt.balanced_accuracy_score,
        bias: bool = True,
        early_stopping: bool = True,
        clip_max: float = 1e10,
        max_attempts: int = 500,
        n_jobs: int = 1,
        cv: int = 5,
        generate_curves: bool = True,
        output_directory: str = None,
        **kwargs: Any,
    ):
        """
        Initialize the NNGSRunner class with training and testing data and various experiment parameters.

        Parameters
        ----------
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Target labels for training data.
        x_test : np.ndarray
            Test input data.
        y_test : np.ndarray
            Target labels for test data.
        experiment_name : str
            Name of the experiment.
        seed : int
            Random seed for reproducibility.
        iteration_list : list of int
            List of iterations for the experiment.
        algorithm: str
            The optimization algorithm to be used (e.g., simulated_annealing).
        grid_search_parameters : dict
            Parameters for grid search.
        grid_search_scorer_method : callable, optional
            Scoring method for grid search.
        bias : bool, optional
            Whether to use bias in the NNClassifier.
        early_stopping : bool, optional
            Whether to stop early if no improvement is detected.
        clip_max : float, optional
            Maximum value for gradient clipping.
        max_attempts : int, optional
            Maximum number of attempts without improvement before stopping.
        n_jobs : int, optional
            Number of jobs to run in parallel.
        cv : int, optional
            Number of cross-validation folds.
        generate_curves : bool, optional
            Whether to generate learning curves.
        output_directory : str, optional
            Directory to save output.
        """
        # Take a copy of the grid search parameters
        grid_search_parameters = {**grid_search_parameters}

        # Hack for compatibility purposes
        if "max_iter" in grid_search_parameters:
            grid_search_parameters["max_iter"] = grid_search_parameters.pop("max_iters")

        # Call the base class init
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_name=experiment_name,
            seed=seed,
            iteration_list=iteration_list,
            grid_search_parameters=grid_search_parameters,
            generate_curves=generate_curves,
            output_directory=output_directory,
            n_jobs=n_jobs,
            cv=cv,
            grid_search_scorer_method=grid_search_scorer_method,
            **kwargs,
        )

        # Build the classifier
        self.classifier = NNClassifier(
            runner=self,
            algorithm=algorithm,
            max_attempts=max_attempts,
            clip_max=clip_max,
            early_stopping=early_stopping,
            seed=seed,
            bias=bias,
        )

        # Update short name based on the algorithm
        self._set_dynamic_runner_name(f"{get_short_name(self)}_{get_short_name(algorithm)}")

    def run_one_experiment_(self, algorithm: Any, total_args: dict, **params: dict) -> tuple | None:
        """
        Run one instance of the experiment with the specified algorithm and parameters.

        Parameters
        ----------
        algorithm: Any
            The optimization algorithm to run.
        total_args : dict
            Dictionary of arguments to pass to the algorithm.
        **params : dict
            Additional parameters for the experiment.

        Returns
        -------
        tuple | None
            The results of the experiment.
        """
        if self._extra_args is not None and len(self._extra_args) > 0:
            params = {**params, **self._extra_args}

        total_args.update(params)
        total_args.pop("problem")
        user_info = [(k, v) for k, v in total_args.items()]

        return self._invoke_algorithm(
            algorithm=algorithm, curve=self.generate_curves, user_info=user_info, additional_algorithm_args=total_args, **params
        )

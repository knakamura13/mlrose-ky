"""Class for running optimization experiments using sklearn's MLPClassifier, including grid search functionality."""

# Authors: Andrew Rollings (modified by Kyle Nakamura)
# License: BSD 3-clause

import inspect
from typing import Any

import numpy as np
import sklearn.metrics as skmt
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier

import mlrose_ky.neural.activation as act
from mlrose_ky.decorators import short_name
from mlrose_ky.runners._nn_runner_base import _NNRunnerBase


@short_name("skmlp")
class SKMLPRunner(_NNRunnerBase):
    """
    A runner for performing optimization experiments using sklearn's MLPClassifier.

    This class extends _NNRunnerBase and provides grid search functionality for optimizing
    the hyperparameters of an MLPClassifier.

    Attributes
    ----------
    classifier : SKMLPRunner._MLPClassifier
        An instance of an internal MLPClassifier with extended functionality.
    """

    class _MLPClassifier(BaseEstimator):
        """
        Internal wrapper for MLPClassifier with additional callback functionality for tracking
        the training progress and storing intermediate results.

        Parameters
        ----------
        runner
            Reference to the parent runner class.
        **kwargs : dict
            Additional keyword arguments to be passed to MLPClassifier.
        """

        def __init__(self, runner: "SKMLPRunner", **kwargs):
            """
            Initialize the _MLPClassifier wrapper.

            Parameters
            ----------
            runner : SKMLPRunner
                The parent runner instance.
            **kwargs : dict
                Additional arguments to pass to MLPClassifier.
            """
            self.runner: SKMLPRunner = runner

            # Filter out invalid kwargs using inspect.signature
            valid_args = inspect.signature(MLPClassifier.__init__).parameters
            mlp_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

            self.mlp: MLPClassifier = MLPClassifier(**mlp_kwargs)
            self.state_callback = self.runner._save_state
            self.fit_started_: bool = False
            self.user_info_: list[tuple[str, Any]] | None = None
            self.kwargs_: dict[str, Any] = kwargs
            self.loss_: float = 1.0
            self.state_: list | None = None
            self.curve_: list[tuple[float, int | None]] = []

            # need to intercept the classifier so we can track statistics.
            if runner.generate_curves:
                if hasattr(self.mlp, "_update_no_improvement_count"):
                    # noinspection PyProtectedMember
                    self._mlp_update_no_improvement_count = self.mlp._update_no_improvement_count
                    self.mlp._update_no_improvement_count = self._update_no_improvement_count_intercept
                if hasattr(self.mlp, "_loss_grad_lbfgs"):
                    # noinspection PyProtectedMember
                    self._mlp_loss_grad_lbfgs = self.mlp._loss_grad_lbfgs
                    self.mlp._loss_grad_lbfgs = self._loss_grad_lbfgs_intercept

        def __getattr__(self, item: str, default: Any = None) -> Any:
            """Fetch an attribute from the MLPClassifier if not found in the class instance."""
            if "mlp" in self.__dict__ and hasattr(self.__dict__["mlp"], item):
                return self.__dict__["mlp"].__getattr__(item, default)

            return self.__dict__[item] if item in self.__dict__ else default

        def __setattr__(self, item: str, value: Any):
            """Set an attribute on the MLPClassifier if it exists, otherwise set it on the class instance."""
            if "mlp" in self.__dict__ and hasattr(self.__dict__["mlp"], item):
                self.__dict__["mlp"].__setattr__(item, value)

            self.__dict__[item] = value

        def get_params(self, deep: bool = True) -> dict:
            """
            Get parameters of the MLPClassifier and this wrapper.

            Parameters
            ----------
            deep : bool, optional
                If True, will return the parameters for this estimator and contained sub-objects that are estimators.

            Returns
            -------
            dict
                A dictionary of parameters.
            """
            out = super().get_params()
            out.update(self.mlp.get_params())

            # Exclude any that end with an underscore
            return {k: v for (k, v) in out.items() if not k[-1] == "_"}

        def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> MLPClassifier:
            """
            Fit the model to the training data.

            Parameters
            ----------
            x_train : np.ndarray
                Training input data.
            y_train : np.ndarray, optional
                Target labels for training data.

            Returns
            -------
            MLPClassifier
                The trained model.
            """
            self.fit_started_ = True
            self.runner._start_run_timing()

            # Make initial callback
            self._invoke_runner_callback()

            # noinspection PyTypeChecker
            return self.mlp.fit(x_train, y_train)

        def predict(self, x_test: np.ndarray) -> np.ndarray:
            """
            Predict the class labels for the provided data.

            Parameters
            ----------
            x_test : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input data.

            Returns
            -------
            y : ndarray, shape (n_samples,) or (n_samples, n_classes)
                The predicted classes.
            """
            return self.mlp.predict(x_test)

        def _update_no_improvement_count_intercept(self, early_stopping: bool, x_val: np.ndarray, y_val: np.ndarray) -> int:
            """
            Intercept the '_update_no_improvement_count' method to track state.

            Parameters
            ----------
            early_stopping : bool
                Whether early stopping is enabled.
            x_val : np.ndarray
                Validation input data.
            y_val : np.ndarray
                Validation target data.

            Returns
            -------
            int
                The result of the original '_update_no_improvement_count' method.
            """
            ret = self._mlp_update_no_improvement_count(early_stopping, x_val, y_val)

            self.state_ = self.mlp.coefs_ if hasattr(self.mlp, "coefs_") else []
            self.loss_ = self.mlp.loss_ if hasattr(self.mlp, "loss_") else 0

            if hasattr(self.mlp, "loss_curve_"):
                self.curve_ = [(_loss_val, None) for _loss_val in self.mlp.loss_curve_]
            else:
                self.curve_.append((self.loss_, None))

            self._invoke_runner_callback()

            return ret

        def _loss_grad_lbfgs_intercept(
            self,
            packed_coef_inter: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            activations: list,
            deltas: list,
            coef_grads: list,
            intercept_grads: list,
        ) -> tuple[float, np.ndarray | list]:
            """
            Intercept the '_loss_grad_lbfgs' method to track loss and state.

            Parameters
            ----------
            packed_coef_inter : ndarray
                A vector comprising the flattened coefficients and intercepts.

            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input data.

            y : ndarray of shape (n_samples,)
                The target values.

            activations : list, length = n_layers - 1
                The ith element of the list holds the values of the ith layer.

            deltas : list, length = n_layers - 1
                The ith element of the list holds the difference between the
                activations of the i + 1 layer and the backpropagated error.
                More specifically, deltas are gradients of loss with respect to z
                in each layer, where z = wx + b is the value of a particular layer
                before passing through the activation function

            coef_grads : list, length = n_layers - 1
                The ith element contains the amount of change used to update the
                coefficient parameters of the ith layer in an iteration.

            intercept_grads : list, length = n_layers - 1
                The ith element contains the amount of change used to update the
                intercept parameters of the ith layer in an iteration.

            Returns
            -------
            loss : float
            grad : array-like, shape (number of nodes of all layers,)
            """
            f, g = self._mlp_loss_grad_lbfgs(packed_coef_inter, x, y, activations, deltas, coef_grads, intercept_grads)

            self.loss_ = f
            self.state_ = g
            self.curve_.append((self.loss_, None))
            self._invoke_runner_callback()

            return f, g

        def _invoke_runner_callback(self):
            """Invoke the runner callback to save the current state of training."""
            # noinspection PyProtectedMember
            no_improvement_count = self.mlp._no_improvement_count if hasattr(self.mlp, "_no_improvement_count") else 0

            iterations = self.mlp.n_iter_ if hasattr(self.mlp, "n_iter_") else 0
            done = self.mlp.early_stopping and (no_improvement_count > self.mlp.n_iter_no_change) or iterations == self.mlp.max_iter

            # Check for early abort
            if self.runner.has_aborted():
                return

            if self.user_info_ is None:
                self.user_info_ = [(k, self.__dict__[k]) for k in self.kwargs_.keys() if hasattr(self, k)]
                for k, v in self.user_info_:
                    self.runner._log_current_argument(k, v)

            self.state_callback(
                iteration=iterations,
                state=self.state_,
                fitness=self.loss_,
                user_data=self.user_info_,
                attempt=no_improvement_count,
                done=done,
                curve=self.curve_,
            )

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
        iteration_list: list[int],
        grid_search_parameters: dict[str, Any],
        grid_search_scorer_method: callable = skmt.balanced_accuracy_score,
        early_stopping: bool = True,
        max_attempts: int = 500,
        n_jobs: int = 1,
        cv: int = 5,
        override_ctrl_c_handler: bool = True,
        generate_curves: bool = True,
        output_directory: str = None,
        replay: bool = False,
        **kwargs: dict,
    ):
        """
        Initialize the SKMLPRunner class with training and testing data and various experiment parameters.

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
        grid_search_parameters : dict
            Parameters for grid search.
        grid_search_scorer_method : callable, optional
            Scoring method for grid search.
        early_stopping : bool, optional
            Whether to stop early if no improvement is detected.
        max_attempts : int, optional
            Maximum number of attempts without improvement before stopping.
        n_jobs : int, optional
            Number of jobs to run in parallel.
        cv : int, optional
            Number of cross-validation folds.
        override_ctrl_c_handler : bool, optional
            Whether to override the Ctrl-C handler.
        generate_curves : bool, optional
            Whether to generate learning curves.
        output_directory : str, optional
            Directory to save output.
        replay : bool, optional
            Whether to replay the experiment.
        """
        grid_search_parameters = {**grid_search_parameters}

        # Hack for compatibility purposes
        if "max_iters" in grid_search_parameters:
            grid_search_parameters["max_iter"] = grid_search_parameters.pop("max_iters")

        if "max_attempts" in grid_search_parameters:
            grid_search_parameters["n_iter_no_change"] = grid_search_parameters.pop("max_attempts")

        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_name=experiment_name,
            seed=seed,
            iteration_list=iteration_list,
            grid_search_parameters=grid_search_parameters,
            grid_search_scorer_method=grid_search_scorer_method,
            generate_curves=generate_curves,
            output_directory=output_directory,
            override_ctrl_c_handler=override_ctrl_c_handler,
            replay=replay,
            n_jobs=n_jobs,
            cv=cv,
        )

        # Create a dictionary of default values
        default_kwargs = {
            "shuffle": True,
            "random_state": seed,
            "verbose": False,
            "warm_start": False,
            "early_stopping": early_stopping,
            "n_iter_no_change": max_attempts,
        }

        # Merge default_kwargs into kwargs, with kwargs taking precedence in case of conflicts
        kwargs = {**default_kwargs, **kwargs}

        # Instantiate the _MLPClassifier with runner and kwargs
        self.classifier = self._MLPClassifier(runner=self, **kwargs)  # Pass everything as kwargs

        self.classifier.runner = self

    @staticmethod
    def build_grid_search_parameters(grid_search_parameters: dict[str, Any], **kwargs: dict) -> dict[str, Any]:
        """
        Build and return grid search parameters, ensuring compatibility with sklearn's MLPClassifier.

        Parameters
        ----------
        grid_search_parameters : dict
            Initial grid search parameters.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        dict
            Final grid search parameters with compatible types.
        """
        all_grid_search_parameters = _NNRunnerBase.build_grid_search_parameters(grid_search_parameters, **kwargs)

        if "activation" in all_grid_search_parameters:
            activation_set = list(all_grid_search_parameters["activation"])

            for i in range(len(activation_set)):
                a = activation_set[i]
                if a == act.relu:
                    activation_set[i] = "relu"
                elif a == act.sigmoid:
                    activation_set[i] = "logistic"
                elif a == act.tanh:
                    activation_set[i] = "tanh"
                elif a == act.identity:
                    activation_set[i] = "identity"
                elif a == act.softmax:
                    activation_set[i] = "softmax"

            all_grid_search_parameters["activation"] = activation_set

        return all_grid_search_parameters

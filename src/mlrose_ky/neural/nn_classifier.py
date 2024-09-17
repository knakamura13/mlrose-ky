"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Optional, Callable

import numpy as np

from mlrose_ky.neural._nn_base import _NNBase
from mlrose_ky.neural.activation import tanh


class NNClassifier(_NNBase):
    """
    Neural Network Classifier class based on _NNBase.

    Parameters
    ----------
    runner : Any
        Object that controls the execution of training experiments.
    algorithm : str, optional
        The optimization algorithm to use for training (e.g., "random_hill_climb", "simulated_annealing", etc.).
    activation : Callable, optional, default=mlrose_ky.tanh
        Activation function to use in the neural network layers.
    hidden_layer_sizes : list[int], optional
        List defining the number of hidden layers and nodes in each layer.
    max_iters : int, default=100
        Maximum number of iterations for the optimization algorithm.
    max_attempts : int, default=10
        Maximum number of attempts to find a better state at each step.
    learning_rate_init : float, default=0.1
        Initial learning rate for optimization.
    bias : bool, default=True
        Whether to include bias nodes in the network.
    early_stopping : bool, default=False
        Whether to stop training early if no improvement is detected.
    clip_max : float, default=1e10
        Maximum value for clipping weights during optimization.
    seed : int or None, default=None
        Random seed for reproducibility.
    kwargs : dict, optional
        Additional arguments passed to the training functions.
    """

    def __init__(
        self,
        runner: Any,
        algorithm: str = None,
        activation: Optional[Callable] = tanh,
        hidden_layer_sizes: list[int] = None,
        max_iters: int = 100,
        max_attempts: int = 10,
        learning_rate_init: float = 0.1,
        bias: bool = True,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        seed: int = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.runner = runner
        self.grid_search_parameters = runner.grid_search_parameters

        # Neural network properties
        self.hidden_layer_sizes: list[int] = hidden_layer_sizes if hidden_layer_sizes is not None else []

        self.activation: Optional[Callable] = activation
        self.learning_rate_init: float = learning_rate_init
        self.max_iters: int = max_iters
        self.max_attempts: int = max_attempts

        self.bias: bool = bias
        self.early_stopping: bool = early_stopping
        self.clip_max: float = clip_max
        self.algorithm: str | None = algorithm

        # Result properties
        self.fitness_fn: Any = None
        self.problem: Any = None
        self.fitted_weights: np.ndarray | None = None
        self.output_activation: Optional[Callable] = None
        self.predicted_probabilities: np.ndarray | None = None
        self.node_list: list[int] | None = None
        self.node_count: int | None = None
        self.loss: float | None = None
        self.fit_started_: bool = False
        self.seed: int | None = seed

        # Extra parameters
        self.kwargs: dict[str, Any] = kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                # Ignore kwargs that are already defined as attributes
                continue
            self.__setattr__(k, v)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get the parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns the parameters for this estimator and its sub-objects.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        out = super().get_params(deep)

        # Exclude any attributes that end with an underscore
        out = {k: v for (k, v) in out.items() if not k.endswith("_")}
        ap = {k: self.__dict__.get(k, None) for k in self.grid_search_parameters if k not in out}
        out.update(ap)

        return out

    def _get_nodes(self, x_train: np.ndarray, y_train: np.ndarray) -> list[int]:
        """
        Get the number of nodes in each layer of the network.

        Parameters
        ----------
        x_train : np.ndarray
            The training feature set.
        y_train : np.ndarray
            The training label set.

        Returns
        -------
        list of int
            Number of nodes in each layer.
        """
        return _NNBase._build_node_list(X=x_train, y=y_train, hidden_nodes=self.hidden_layer_sizes, bias=self.bias)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None, init_weights: np.ndarray = None) -> "NNClassifier":
        """
        Fit the neural network classifier to the training data.

        Parameters
        ----------
        x_train : np.ndarray
            The training feature set.
        y_train : np.ndarray, optional
            The training label set.
        init_weights : np.ndarray, optional
            Initial weights for the network. If None, random weights are used.

        Returns
        -------
        NNClassifier
            The fitted classifier.
        """
        x_train, y_train = self._format_x_y_data(x_train, y_train)
        self.node_list = self._get_nodes(x_train, y_train)
        self.node_count = _NNBase._calculate_state_size(self.node_list)

        fitness, problem = _NNBase._build_problem_and_fitness_function(
            X=x_train,
            y=y_train,
            node_list=self.node_list,
            activation=self.activation,
            learning_rate=self.learning_rate_init,
            clip_max=self.clip_max,
            bias=self.bias,
        )
        self.fitness_fn = fitness
        self.problem = problem

        # If algorithm is None, skip the training process
        if self.algorithm is None:
            self.fitted_weights = None
            self.loss = None
            self.output_activation = None
            return self

        # Check for early abort or replay mode.
        if self.runner.has_aborted() or self.runner.replay_mode():
            self.fitted_weights = np.array([np.NaN] * self.node_count)
            self.loss = np.NaN
            self.output_activation = self.fitness_fn.get_output_activation()
            return self

        # Handle grid search or regular training
        params = {k: self.__getattribute__(k) for k in self.kwargs}
        if init_weights is None:
            np.random.seed(self.seed)
            init_weights = np.random.uniform(-1, 1, self.node_count)

        params["init_state"] = init_weights
        total_args = {
            "algorithm": self.algorithm,
            "activation": self.activation,
            "bias": self.bias,
            "early_stopping": self.early_stopping,
            "clip_max": self.clip_max,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "learning_rate_init": self.learning_rate_init,
        }
        max_attempts = self.max_attempts if self.early_stopping else self.max_iters
        self.fit_started_ = True

        fitted_weights, loss, _ = self.runner.run_one_experiment_(
            algorithm=self.algorithm, problem=problem, max_iters=self.max_iters, max_attempts=max_attempts, total_args=total_args, **params
        )

        # Save fitted weights
        self.fitted_weights = problem.get_state()
        self.loss = loss
        self.output_activation = self.fitness_fn.get_output_activation()

        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for the test feature set.

        Parameters
        ----------
        x_test : np.ndarray
            The test feature set.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if np.shape(x_test)[1] != (self.node_list[0] - self.bias):
            raise ValueError(f"The number of columns in X must equal {self.node_list[0] - self.bias}, got {np.shape(x_test)[1]}.")

        y_pred, pp = self._predict(
            X=x_test,
            fitted_weights=self.fitted_weights,
            node_list=self.node_list,
            input_activation=self.activation,
            output_activation=self.output_activation,
            bias=self.bias,
        )

        self.predicted_probabilities = pp

        return y_pred

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the test feature set.

        Parameters
        ----------
        x_test : np.ndarray
            The test feature set.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        self.predict(x_test)
        return self.predicted_probabilities

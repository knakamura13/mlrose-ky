"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from abc import abstractmethod
from typing import Callable, Optional

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from mlrose_ky.algorithms.decay import GeomDecay
from mlrose_ky.algorithms.ga import genetic_alg
from mlrose_ky.algorithms.rhc import random_hill_climb
from mlrose_ky.algorithms.sa import simulated_annealing
from mlrose_ky.neural._nn_base import _NNBase
from mlrose_ky.neural.activation import identity, relu, sigmoid, tanh
from mlrose_ky.neural.utils.weights import gradient_descent_original


class _NNCore(_NNBase):
    """
    Core class for neural networks.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        hidden_nodes: list[int] = None,
        activation: str = "relu",
        algorithm: str = "random_hill_climb",
        max_iters: int | float = 100,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: GeomDecay = GeomDecay(),
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: int = None,
        curve: bool = False,
    ):
        super().__init__()
        self.hidden_nodes: list[int] = hidden_nodes if hidden_nodes is not None else []
        self.activation_dict: dict[str, Callable] = {"identity": identity, "relu": relu, "sigmoid": sigmoid, "tanh": tanh}
        self.activation: str = activation
        self.algorithm: str = algorithm
        self.max_iters: int | float = max_iters
        self.bias: bool = bias
        self.is_classifier: bool = is_classifier
        self.learning_rate: float = learning_rate
        self.early_stopping: bool = early_stopping
        self.clip_max: float = clip_max
        self.restarts: int = restarts
        self.schedule: GeomDecay = schedule
        self.pop_size: int = pop_size
        self.mutation_prob: float = mutation_prob
        self.max_attempts: int = max_attempts
        self.random_state: int | None = random_state
        self.curve: bool = curve

        self.node_list: list[int] = []
        self.fitted_weights: np.ndarray = np.array([])
        self.loss: float = np.inf
        self.output_activation: Optional[Callable] = None
        self.predicted_probs: np.ndarray = np.array([])
        self.fitness_curve: list[float] = []

    def _validate(self):
        """Validate the model parameters."""
        if (not isinstance(self.max_iters, int) and self.max_iters != np.inf) or self.max_iters < 0:
            raise ValueError(f"max_iters must be a positive integer, got {self.max_iters}.")
        if not isinstance(self.bias, bool):
            raise ValueError(f"bias must be True or False, got {self.bias}.")
        if not isinstance(self.is_classifier, bool):
            raise ValueError(f"is_classifier must be True or False, got {self.is_classifier}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be greater than 0, got {self.learning_rate}.")
        if not isinstance(self.early_stopping, bool):
            raise ValueError(f"early_stopping must be True or False, got {self.early_stopping}.")
        if self.clip_max <= 0:
            raise ValueError(f"clip_max must be greater than 0, got {self.clip_max}.")
        if not isinstance(self.max_attempts, int) or self.max_attempts < 0:
            raise ValueError(f"max_attempts must be a positive integer, got {self.max_attempts}.")
        if self.pop_size < 0 or not isinstance(self.pop_size, int):
            raise ValueError(f"pop_size must be a positive integer, got {self.pop_size}.")
        if not (0 <= self.mutation_prob <= 1):
            raise ValueError(f"mutation_prob must be between 0 and 1, got {self.mutation_prob}.")
        if self.activation not in self.activation_dict:
            raise ValueError(f"Activation function must be one of: 'identity', 'relu', 'sigmoid', or 'tanh', got {self.activation}.")
        if self.algorithm not in ["random_hill_climb", "simulated_annealing", "genetic_alg", "gradient_descent"]:
            raise ValueError(
                f"Algorithm must be one of: "
                f"'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent', got {self.algorithm}."
            )

    def _validate_input(self, y: np.ndarray):
        """
        Add classes_ attribute based on classes present in y.

        Parameters
        ----------
        y : np.ndarray
            Data labels.
        """
        if not hasattr(self, "classes_"):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_

    def fit(self, X: np.ndarray, y: np.ndarray = None, init_weights: np.ndarray = None) -> "_NNCore":
        """
        Fit neural network to data.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset.
        y : np.ndarray, optional
            Data labels. Length must be the same as X.
        init_weights : np.ndarray, optional
            Initial weights for the algorithm. If None, random weights are used.

        Returns
        -------
        _NNCore
            Fitted model.
        """
        self._validate()
        self._validate_input(y)

        X, y = self._format_x_y_data(X, y)

        node_list = self._build_node_list(X, y, self.hidden_nodes, self.bias)
        num_nodes = self._calculate_state_size(node_list)

        if init_weights is not None and len(init_weights) != num_nodes:
            raise ValueError(f"init_weights must be None or have length {num_nodes}, got {len(init_weights)}.")

        if isinstance(self.random_state, int) and self.random_state > 0:
            np.random.seed(self.random_state)

        fitness, problem = self._build_problem_and_fitness_function(
            X, y, node_list, self.activation_dict[self.activation], self.learning_rate, self.clip_max, self.bias, self.is_classifier
        )

        if self.algorithm == "random_hill_climb":
            fitness_curve, fitted_weights, loss = self.__run_with_rhc(init_weights, num_nodes, problem)
        elif self.algorithm == "simulated_annealing":
            fitness_curve, fitted_weights, loss = self._run_with_sa(init_weights, num_nodes, problem)
        elif self.algorithm == "genetic_alg":
            fitness_curve, fitted_weights, loss = self._run_with_ga(problem)
        else:
            fitness_curve, fitted_weights, loss = self._run_with_gd(init_weights, num_nodes, problem)

        self.node_list = node_list
        self.fitted_weights = fitted_weights
        self.loss = loss
        self.output_activation = fitness.get_output_activation()

        if self.curve:
            self.fitness_curve = fitness_curve

        return self

    def _run_with_gd(self, init_weights: np.ndarray | None, num_nodes: int, problem) -> tuple[np.ndarray | list, np.ndarray, float]:
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)

        fitted_weights, loss, fitness_curve = gradient_descent_original(
            problem,
            max_attempts=self.max_attempts if self.early_stopping else self.max_iters,
            max_iters=self.max_iters,
            curve=self.curve,
            init_state=init_weights,
        )

        return fitness_curve if fitness_curve is not None else [], fitted_weights, loss

    def _run_with_ga(self, problem) -> tuple[np.ndarray | list, np.ndarray, float]:
        fitted_weights, loss, fitness_curve = genetic_alg(
            problem,
            pop_size=self.pop_size,
            mutation_prob=self.mutation_prob,
            max_attempts=self.max_attempts if self.early_stopping else self.max_iters,
            max_iters=self.max_iters,
            curve=self.curve,
        )

        return fitness_curve if fitness_curve is not None else [], fitted_weights, loss

    def _run_with_sa(self, init_weights: np.ndarray | None, num_nodes: int, problem) -> tuple[np.ndarray | list, np.ndarray, float]:
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)

        fitted_weights, loss, fitness_curve = simulated_annealing(
            problem,
            schedule=self.schedule,
            max_attempts=self.max_attempts if self.early_stopping else self.max_iters,
            max_iters=self.max_iters,
            init_state=init_weights,
            curve=self.curve,
        )

        return fitness_curve if fitness_curve is not None else [], fitted_weights, loss

    def __run_with_rhc(self, init_weights: np.ndarray | None, num_nodes: int, problem) -> tuple[np.ndarray | list, np.ndarray, float]:
        fitness_curve = []
        fitted_weights = []
        loss = np.inf

        for _ in range(self.restarts + 1):
            restart_weights = np.random.uniform(-1, 1, num_nodes) if init_weights is None else init_weights

            current_weights, current_loss, fitness_curve = random_hill_climb(
                problem,
                max_attempts=self.max_attempts if self.early_stopping else self.max_iters,
                max_iters=self.max_iters,
                init_state=restart_weights,
                curve=self.curve,
            )

            if not self.curve or fitness_curve is None:
                fitness_curve = []

            if current_loss < loss:
                fitted_weights = current_weights
                loss = current_loss

        return fitness_curve, fitted_weights, loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use model to predict data labels for given feature array.

        Parameters
        ----------
        X : np.ndarray
            Feature dataset.

        Returns
        -------
        np.ndarray
            Predicted data labels.
        """
        if np.shape(X)[1] != (self.node_list[0] - self.bias):
            raise ValueError(f"The number of columns in X must equal {self.node_list[0] - self.bias}, got {np.shape(X)[1]}.")

        y_pred, pp = self._predict(
            X=X,
            fitted_weights=self.fitted_weights,
            node_list=self.node_list,
            input_activation=self.activation_dict[self.activation],
            output_activation=self.output_activation,
            bias=self.bias,
            is_classifier=self.is_classifier,
        )

        self.predicted_probs = pp

        return y_pred

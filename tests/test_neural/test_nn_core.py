"""Unit tests for neural/_nn_core.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky.algorithms.decay import GeomDecay

# noinspection PyProtectedMember
from mlrose_ky.neural._nn_core import _NNCore


class TestNNCore:
    """Test cases for the _NNCore class."""

    def setup_method(self):
        """Set up method to initialize common variables for the tests."""
        self.hidden_nodes = [2, 3]
        self.activation = "relu"
        self.algorithm = "random_hill_climb"
        self.max_iters = 100
        self.bias = True
        self.is_classifier = True
        self.learning_rate = 0.1
        self.early_stopping = False
        self.clip_max = 1e10
        self.restarts = 0
        self.schedule = GeomDecay()
        self.pop_size = 200
        self.mutation_prob = 0.1
        self.max_attempts = 10
        self.random_state = 42
        self.curve = False
        self.X_train = np.array([[1, 0], [0, 1], [1, 1]])
        self.y_train = np.array([0, 1, 0])

    def test_initialization(self):
        """Test that _NNCore is initialized correctly."""
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes,
            activation=self.activation,
            algorithm=self.algorithm,
            max_iters=self.max_iters,
            bias=self.bias,
            is_classifier=self.is_classifier,
            learning_rate=self.learning_rate,
            early_stopping=self.early_stopping,
            clip_max=self.clip_max,
            restarts=self.restarts,
            schedule=self.schedule,
            pop_size=self.pop_size,
            mutation_prob=self.mutation_prob,
            max_attempts=self.max_attempts,
            random_state=self.random_state,
            curve=self.curve,
        )
        # Assert that all the parameters are correctly initialized
        assert nn.hidden_nodes == self.hidden_nodes
        assert nn.activation == self.activation
        assert nn.algorithm == self.algorithm
        assert nn.max_iters == self.max_iters
        assert nn.bias == self.bias
        assert nn.is_classifier == self.is_classifier

    def test_validate_correct(self):
        """Test that _validate works with correct parameters."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm)
        # Should not raise any errors
        nn._validate()

    # Add tests for all exceptions in _validate
    def test_validate_incorrect_max_iters(self):
        """Test that _validate raises ValueError with incorrect max_iters."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, max_iters=-1)  # Invalid value
        with pytest.raises(ValueError, match="max_iters must be a positive integer"):
            nn._validate()

    def test_validate_incorrect_bias(self):
        """Test that _validate raises ValueError with incorrect bias."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, bias="not_bool")  # Invalid value
        with pytest.raises(ValueError, match="bias must be True or False"):
            nn._validate()

    def test_validate_incorrect_is_classifier(self):
        """Test that _validate raises ValueError with incorrect is_classifier."""
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, is_classifier="not_bool"  # Invalid value
        )
        with pytest.raises(ValueError, match="is_classifier must be True or False"):
            nn._validate()

    def test_validate_incorrect_learning_rate(self):
        """Test that _validate raises ValueError with incorrect learning_rate."""
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, learning_rate=-0.1  # Invalid value
        )
        with pytest.raises(ValueError, match="learning_rate must be greater than 0"):
            nn._validate()

    def test_validate_incorrect_early_stopping(self):
        """Test that _validate raises ValueError with incorrect early_stopping."""
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, early_stopping="not_bool"  # Invalid value
        )
        with pytest.raises(ValueError, match="early_stopping must be True or False"):
            nn._validate()

    def test_validate_incorrect_clip_max(self):
        """Test that _validate raises ValueError with incorrect clip_max."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, clip_max=-1.0)  # Invalid value
        with pytest.raises(ValueError, match="clip_max must be greater than 0"):
            nn._validate()

    def test_validate_incorrect_max_attempts(self):
        """Test that _validate raises ValueError with incorrect max_attempts."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, max_attempts=-1)  # Invalid value
        with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
            nn._validate()

    def test_validate_incorrect_pop_size(self):
        """Test that _validate raises ValueError with incorrect pop_size."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, pop_size=-1)  # Invalid value
        with pytest.raises(ValueError, match="pop_size must be a positive integer"):
            nn._validate()

    def test_validate_incorrect_mutation_prob(self):
        """Test that _validate raises ValueError with incorrect mutation_prob."""
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, mutation_prob=1.1  # Invalid value
        )
        with pytest.raises(ValueError, match="mutation_prob must be between 0 and 1"):
            nn._validate()

    def test_validate_incorrect_activation(self):
        """Test that _validate raises ValueError with incorrect activation function."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation="invalid_activation", algorithm=self.algorithm)  # Invalid value
        with pytest.raises(ValueError, match="Activation function must be one of"):
            nn._validate()

    def test_validate_incorrect_algorithm(self):
        """Test that _validate raises ValueError with incorrect algorithm."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm="invalid_algorithm")  # Invalid value
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            nn._validate()

    def test_fit(self):
        """Test the fit method."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm, random_state=self.random_state)
        # Call fit method with training data
        nn.fit(self.X_train, self.y_train)
        # Check that model was fitted
        assert nn.fitted_weights is not None
        assert nn.loss is not None
        assert nn.output_activation is not None

    def test_fit_invalid_weights(self):
        """Test that fit raises a ValueError for incorrect initial weights."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm)
        # Pass in invalid initial weights
        init_weights = np.random.uniform(size=5)  # Wrong size
        with pytest.raises(ValueError, match="init_weights must be None or have length"):
            nn.fit(self.X_train, self.y_train, init_weights)

    def test_predict(self):
        """Test the predict method."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm)
        nn.fit(self.X_train, self.y_train)
        # Predict using the trained model
        y_pred = nn.predict(self.X_train)
        assert y_pred is not None

    def test_predict_invalid_input(self):
        """Test that predict raises ValueError for invalid input dimensions."""
        nn = _NNCore(hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm=self.algorithm)
        nn.fit(self.X_train, self.y_train)
        X_invalid = np.array([[1, 2, 3]])  # Invalid dimensions
        with pytest.raises(ValueError, match="The number of columns in X must equal"):
            nn.predict(X_invalid)

    def test_run_with_gd_init_weights_none(self):
        """Test _run_with_gd with init_weights=None."""
        # Create a mock instance of _NNCore
        nn = _NNCore(
            hidden_nodes=self.hidden_nodes, activation=self.activation, algorithm="gradient_descent", random_state=self.random_state
        )

        # Simulate data and problem setup (mimicking part of what fit() does)
        X_train = self.X_train
        y_train = self.y_train

        # Manually reshape y_train to ensure it is 2D (mimicking _format_x_y_data in fit())
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        # Manually create the problem object using _build_problem_and_fitness_function
        node_list = nn._build_node_list(X_train, y_train, nn.hidden_nodes, nn.bias)
        num_nodes = nn._calculate_state_size(node_list)
        activation_fn = nn.activation_dict[nn.activation]

        # Build the problem and fitness function as done in the fit method
        fitness, problem = nn._build_problem_and_fitness_function(
            X_train, y_train, node_list, activation_fn, nn.learning_rate, nn.clip_max, nn.bias, nn.is_classifier
        )

        # Call _run_with_gd with init_weights=None to cover the if statement
        fitness_curve_gd, fitted_weights_gd, loss_gd = nn._run_with_gd(None, num_nodes, problem)

        # Assertions to verify the output
        assert fitted_weights_gd is not None
        assert isinstance(fitted_weights_gd, np.ndarray)
        assert len(fitted_weights_gd) == num_nodes
        assert isinstance(fitness_curve_gd, list)
        assert isinstance(loss_gd, float)

        # Call _run_with_sa with init_weights=None to cover the if statement
        fitness_curve_sa, fitted_weights_sa, loss_sa = nn._run_with_sa(None, num_nodes, problem)

        # Assertions to verify the output
        assert fitted_weights_sa is not None
        assert isinstance(fitted_weights_sa, np.ndarray)
        assert len(fitted_weights_sa) == num_nodes
        assert isinstance(fitness_curve_sa, list)
        assert isinstance(loss_sa, float)

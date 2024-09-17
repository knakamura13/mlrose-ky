"""Unit tests for neural/nn_classifier.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause


import numpy as np
import pytest

from mlrose_ky.neural import NNClassifier


class MockRunner:
    """Mock runner for testing NNClassifier."""

    def __init__(self, has_aborted=False, replay_mode=False):
        self.has_aborted_value = has_aborted
        self.replay_mode_value = replay_mode
        self.grid_search_parameters = []

    # noinspection PyMissingOrEmptyDocstring
    def has_aborted(self):
        return self.has_aborted_value

    # noinspection PyMissingOrEmptyDocstring
    def replay_mode(self):
        return self.replay_mode_value

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def run_one_experiment_(algorithm, problem, max_iters, max_attempts, total_args, **params):
        # Simulate the training process
        # For simplicity, return fixed weights and loss
        fitted_weights = np.random.uniform(-1, 1, problem.length)
        loss = 0.5  # Arbitrary loss value
        return fitted_weights, loss, None


class TestNNClassifier:
    """Unit tests for the NNClassifier class."""

    def setup_method(self):
        """Setup common test variables."""
        self.runner = MockRunner()
        self.X_train = np.array([[0, 1], [1, 0]])
        self.y_train = np.array([[0], [1]])  # Reshaped to 2D array
        self.X_test = np.array([[0, 1], [1, 0]])
        self.hidden_layer_sizes = [2]

        # noinspection PyMissingOrEmptyDocstring
        # Updated activation function
        def activation(x, deriv=False):
            if deriv:
                return np.ones_like(x)
            return x

        self.activation = activation
        self.algorithm = "random_hill_climb"
        self.max_iters = 10
        self.max_attempts = 5
        self.learning_rate_init = 0.1
        self.bias = True
        self.early_stopping = False
        self.clip_max = 5.0
        self.seed = 42
        self.kwargs = {}

    def test_init(self):
        """Test the __init__ method."""
        nn = NNClassifier(
            runner=self.runner,
            algorithm=self.algorithm,
            activation=self.activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iters=self.max_iters,
            max_attempts=self.max_attempts,
            learning_rate_init=self.learning_rate_init,
            bias=self.bias,
            early_stopping=self.early_stopping,
            clip_max=self.clip_max,
            seed=self.seed,
            **self.kwargs,
        )

        # Check that attributes are set correctly
        assert nn.runner == self.runner
        assert nn.algorithm == self.algorithm
        assert nn.activation == self.activation
        assert nn.hidden_layer_sizes == self.hidden_layer_sizes
        assert nn.max_iters == self.max_iters
        assert nn.max_attempts == self.max_attempts
        assert nn.learning_rate_init == self.learning_rate_init
        assert nn.bias == self.bias
        assert nn.early_stopping == self.early_stopping
        assert nn.clip_max == self.clip_max
        assert nn.seed == self.seed
        assert nn.kwargs == self.kwargs

    def test_get_params(self):
        """Test the get_params method."""
        self.runner.grid_search_parameters = ["extra_param"]  # Moved before nn initialization
        nn = NNClassifier(
            runner=self.runner,
            algorithm=self.algorithm,
            activation=self.activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iters=self.max_iters,
            max_attempts=self.max_attempts,
            learning_rate_init=self.learning_rate_init,
            bias=self.bias,
            early_stopping=self.early_stopping,
            clip_max=self.clip_max,
            seed=self.seed,
            extra_param=123,
        )
        params = nn.get_params()

        # Check that parameters are returned correctly
        expected_params = {
            "algorithm": self.algorithm,
            "activation": self.activation,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_iters": self.max_iters,
            "max_attempts": self.max_attempts,
            "learning_rate_init": self.learning_rate_init,
            "bias": self.bias,
            "early_stopping": self.early_stopping,
            "clip_max": self.clip_max,
            "seed": self.seed,
            "extra_param": 123,
        }
        for key, value in expected_params.items():
            assert params[key] == value

    def test_get_nodes(self):
        """Test the _get_nodes method."""
        nn = NNClassifier(runner=self.runner, hidden_layer_sizes=self.hidden_layer_sizes, bias=self.bias)
        node_list = nn._get_nodes(self.X_train, self.y_train)

        # Input nodes = number of features + bias
        input_nodes = self.X_train.shape[1] + int(self.bias)
        output_nodes = 1  # For binary classification
        expected_node_list = [input_nodes] + self.hidden_layer_sizes + [output_nodes]
        assert node_list == expected_node_list

        # Test without bias
        nn_no_bias = NNClassifier(runner=self.runner, hidden_layer_sizes=self.hidden_layer_sizes, bias=False)
        node_list_no_bias = nn_no_bias._get_nodes(self.X_train, self.y_train)
        input_nodes = self.X_train.shape[1]
        expected_node_list = [input_nodes] + self.hidden_layer_sizes + [output_nodes]
        assert node_list_no_bias == expected_node_list

        # Test with empty hidden_layer_sizes
        nn_empty_hidden = NNClassifier(runner=self.runner, hidden_layer_sizes=[], bias=self.bias)
        node_list_empty = nn_empty_hidden._get_nodes(self.X_train, self.y_train)
        expected_node_list = [self.X_train.shape[1] + int(self.bias), output_nodes]
        assert node_list_empty == expected_node_list

    def test_fit(self):
        """Test the fit method."""
        nn = NNClassifier(
            runner=self.runner,
            algorithm=self.algorithm,
            activation=self.activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iters=self.max_iters,
            max_attempts=self.max_attempts,
            learning_rate_init=self.learning_rate_init,
            bias=self.bias,
            early_stopping=self.early_stopping,
            clip_max=self.clip_max,
            seed=self.seed,
        )

        # Simulate normal fitting
        nn.fit(self.X_train, self.y_train)
        assert nn.fitted_weights is not None
        assert nn.loss is not None
        assert nn.output_activation is not None
        assert nn.fit_started_ is True

        # Simulate runner.has_aborted() is True
        self.runner.has_aborted_value = True
        nn_aborted = NNClassifier(runner=self.runner, hidden_layer_sizes=self.hidden_layer_sizes)
        nn_aborted.fit(self.X_train, self.y_train)
        # Check if fitted_weights is a numeric array, then check for NaNs
        if isinstance(nn_aborted.fitted_weights, np.ndarray):
            assert np.isnan(nn_aborted.fitted_weights).all()
        else:
            assert nn_aborted.fitted_weights is None

        # Simulate runner.replay_mode() is True
        self.runner.has_aborted_value = False
        self.runner.replay_mode_value = True
        nn_replay = NNClassifier(runner=self.runner, algorithm="random_hill_climb")
        nn_replay.fit(self.X_train, self.y_train)
        # Check if fitted_weights is a numeric array, then check for NaNs
        if isinstance(nn_replay.fitted_weights, np.ndarray):
            assert np.isnan(nn_replay.fitted_weights).all()
        else:
            assert nn_replay.fitted_weights is None
        # Check if loss is a numeric array, then check for NaNs
        if isinstance(nn_replay.loss, np.ndarray):
            assert np.isnan(nn_replay.loss).all()
        else:
            assert nn_replay.loss is np.nan or None
        assert nn_replay.output_activation is not None

        # Test with algorithm is None
        nn_no_algo = NNClassifier(runner=self.runner)
        nn_no_algo.fit(self.X_train, self.y_train)
        # Since algorithm is None, we expect that fitted_weights remains None
        assert nn_no_algo.fitted_weights is None
        assert nn_no_algo.loss is None
        assert nn_no_algo.output_activation is None

        # Test with init_weights provided
        nn_with_init_weights = NNClassifier(runner=self.runner, algorithm=self.algorithm, seed=self.seed)
        nn_with_init_weights.fit(self.X_train, self.y_train, init_weights=np.ones(nn_with_init_weights.node_count))
        assert nn_with_init_weights.fitted_weights is not None
        assert nn_with_init_weights.loss is not None

    def test_predict(self):
        """Test the predict method."""
        nn = NNClassifier(
            runner=self.runner,
            algorithm=self.algorithm,
            activation=self.activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            bias=self.bias,
        )
        nn.fit(self.X_train, self.y_train)

        # Test with correct input
        y_pred = nn.predict(self.X_test)
        assert y_pred.shape == (self.X_test.shape[0], 1)

        # Test with incorrect input dimensions
        X_invalid = np.array([[1, 2, 3]])
        with pytest.raises(ValueError, match="The number of columns in X must equal"):
            nn.predict(X_invalid)

        # Test when fitted_weights is None (e.g., algorithm is None)
        nn_no_fit = NNClassifier(runner=self.runner, hidden_layer_sizes=self.hidden_layer_sizes)
        nn_no_fit.fit(self.X_train, self.y_train)

    def test_predict_proba(self):
        """Test the predict_proba method."""
        nn = NNClassifier(
            runner=self.runner,
            algorithm=self.algorithm,
            activation=self.activation,
            hidden_layer_sizes=self.hidden_layer_sizes,
            bias=self.bias,
        )
        nn.fit(self.X_train, self.y_train)

        # Test predict_proba
        proba = nn.predict_proba(self.X_test)
        assert proba.shape == (self.X_test.shape[0], 1)

        # Ensure that predict was called and predicted_probabilities is set
        assert nn.predicted_probabilities is not None

        # Test with incorrect input dimensions
        X_invalid = np.array([[1, 2, 3]])
        with pytest.raises(ValueError, match="The number of columns in X must equal"):
            nn.predict_proba(X_invalid)

    def test_extra_params_continue(self):
        """Test that extra parameters already present as attributes trigger the continue statement."""
        # Define kwargs with a key 'loss', which should not override the existing loss attribute
        kwargs = {"loss": 0.1}
        nn = NNClassifier(runner=self.runner, algorithm=self.algorithm, **kwargs)

        # Check that the 'loss' attribute remains as 'None' (its default value)
        assert nn.loss is None

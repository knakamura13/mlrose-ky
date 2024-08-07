"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3 clause

from .activation import (identity, relu, leaky_relu, sigmoid, softmax, tanh)
from .utils import (flatten_weights, unflatten_weights)
from .neural_network import NeuralNetwork
from .fitness.network_weights import NetworkWeights
from .logistic_regression import LogisticRegression
from .linear_regression import LinearRegression
from .nn_classifier import NNClassifier

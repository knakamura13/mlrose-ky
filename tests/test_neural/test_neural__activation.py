"""Unit tests for neural/activation/"""

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np

from mlrose_ky.neural.activation import identity, sigmoid, softmax, tanh, relu, leaky_relu


def test_identity():
    """Test identity activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    assert np.array_equal(identity(x), x)


def test_identity_deriv():
    """Test identity activation function derivative"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    assert np.array_equal(identity(x, deriv=True), np.ones([4, 3]))


def test_sigmoid():
    """Test sigmoid activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0.5, 0.73106, 0.95257], [0.26894, 0.5, 0.00669], [0.73106, 0.5, 0.95257], [0.99995, 0.00012, 0.00091]])
    assert np.allclose(sigmoid(x), y, atol=0.00001)


def test_sigmoid_deriv():
    """Test sigmoid activation function derivative"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0.25, 0.19661, 0.04518], [0.19661, 0.25, 0.00665], [0.19661, 0.25, 0.04518], [0.00005, 0.00012, 0.00091]])
    assert np.allclose(sigmoid(x, deriv=True), y, atol=0.00001)


def test_relu():
    """Test relu activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0, 1, 3], [0, 0, 0], [1, 0, 3], [10, 0, 0]])
    assert np.array_equal(relu(x), y)


def test_relu_deriv():
    """Test relu activation function derivative"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0]])
    assert np.array_equal(relu(x, deriv=True), y)


def test_leaky_relu():
    """Test relu activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0, 1, 3], [-0.3, 0, -1.5], [1, 0, 3], [10, -2.7, -2.1]])
    assert np.allclose(leaky_relu(x), y)


def test_leaky_relu_deriv():
    """Test relu activation function derivative"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0, 1, 1], [0.3, 0, 0.3], [1, 0, 1], [1, 0.3, 0.3]])
    assert np.allclose(leaky_relu(x, deriv=True), y)


def test_tanh():
    """Test tanh activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0, 0.76159, 0.99505], [-0.76159, 0, -0.99991], [0.76159, 0, 0.99505], [1.00000, -1.00000, -1.00000]])
    assert np.allclose(tanh(x), y, atol=0.00001)


def test_tanh_deriv():
    """Test tanh activation function derivative"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[1, 0.41998, 0.00988], [0.41998, 1, 0.00018], [0.41998, 1, 0.00988], [0, 0, -0]])
    assert np.allclose(tanh(x, deriv=True), y, atol=0.0001)


def test_softmax():
    """Test softmax activation function"""
    x = np.array([[0, 1, 3], [-1, 0, -5], [1, 0, 3], [10, -9, -7]])
    y = np.array([[0.04201, 0.11420, 0.84379], [0.26762, 0.72747, 0.00490], [0.11420, 0.04201, 0.84379], [1, 0, 0]])
    assert np.allclose(softmax(x), y, atol=0.00001)

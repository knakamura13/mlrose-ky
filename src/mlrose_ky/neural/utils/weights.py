"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import List, Tuple

import numpy as np


def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """
    Flatten list of weights arrays into a 1D array.

    Parameters
    ----------
    weights : list of np.ndarray
        List of 2D arrays for flattening.

    Returns
    -------
    np.ndarray
        1D array of flattened weights.
    """
    flat_weights = []

    for weight in weights:
        flat_weights += list(weight.flatten())

    return np.array(flat_weights)


def unflatten_weights(flat_weights: np.ndarray, node_list: List[int]) -> List[np.ndarray]:
    """
    Convert 1D weights array into list of 2D arrays.

    Parameters
    ----------
    flat_weights : np.ndarray
        1D weights array.

    node_list : list of int
        List giving the number of nodes in each layer of the network,
        including the input and output layers.

    Returns
    -------
    list of np.ndarray
        List of 2D arrays created from flat_weights.
    """
    nodes = sum(node_list[i] * node_list[i + 1] for i in range(len(node_list) - 1))

    if len(flat_weights) != nodes:
        raise ValueError(f"flat_weights must have length {nodes}, but got {len(flat_weights)}.")

    weights = []
    start = 0

    for i in range(len(node_list) - 1):
        end = start + node_list[i] * node_list[i + 1]
        weights.append(np.reshape(flat_weights[start:end], [node_list[i], node_list[i + 1]]))
        start = end

    return weights


def gradient_descent_original(
    problem,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
) -> Tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use gradient descent to find the optimal neural network weights.

    Parameters
    ----------
    problem : optimization object
        Object containing the optimization problem to be solved.
    max_attempts : int, default=10
        Maximum number of attempts to find a better state at each step.
    max_iters : int or float, default=np.inf
        Maximum number of iterations of the algorithm.
    init_state : np.ndarray, default=None
        Numpy array containing the starting state for the algorithm.
        If None, then a random state is used.
    curve : bool, default=False
        If True, returns a history of fitness values.
    random_state : int, default=None
        If provided, sets the random seed for reproducibility.

    Returns
    -------
    best_state : np.ndarray
        Numpy array containing the state that optimizes the fitness function.
    best_fitness : float
        Value of the fitness function at the best state.
    fitness_curve : np.ndarray or None
        Array containing the fitness at every iteration (if curve=True).
    """
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(f"max_attempts must be a positive integer, got {max_attempts}.")

    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError(f"max_iters must be a positive integer, got {max_iters}.")

    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError(f"init_state must have the same length as the problem, got {len(init_state)}.")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    fitness_curve = []
    attempts = 0
    iters = 0

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    while attempts < max_attempts and iters < max_iters:
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())
        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        if next_fitness > problem.get_fitness():
            attempts = 0
        else:
            attempts += 1

        if next_fitness > problem.get_maximize() * best_fitness:
            best_fitness = problem.get_maximize() * next_fitness
            best_state = next_state

        if curve:
            fitness_curve.append(problem.get_adjusted_fitness())

        problem.set_state(next_state)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness, None

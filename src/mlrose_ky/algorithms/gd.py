"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name
from mlrose_ky.neural.utils import flatten_weights


@short_name("gd")
def gradient_descent(
    problem: Any,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: Any = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Use gradient_descent to find the optimal neural network weights.

    Parameters
    ----------
    problem: optimization object
        Object containing optimization problem to be solved.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: np.ndarray, default: None
        Numpy array containing starting state for algorithm.
        If None, then a random state is used.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    state_fitness_callback: function taking five parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info: any, default: None
        User data passed as last parameter of callback.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing state that optimizes fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: np.ndarray
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    """
    # TODO: fix and uncomment these problematic raise statements
    # if not isinstance(max_attempts, int) or max_attempts < 0:
    #     raise ValueError(f"max_attempts must be a positive integer. Got {max_attempts}")
    # if not (isinstance(max_iters, int) or max_iters == np.inf) or (isinstance(max_iters, int) and max_iters < 0):
    #     raise ValueError(f"max_iters must be a positive integer or np.inf. Got {max_iters}")
    # if init_state is not None and len(init_state) != problem.get_length():
    #     raise ValueError(f"init_state must have the same length as problem. Got {len(init_state)}")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem
    fitness_curve = []
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)
    if state_fitness_callback is not None:
        state_fitness_callback(
            iteration=0,
            max_attempts_reached=False,
            state=problem.get_state(),
            fitness=problem.get_adjusted_fitness(),
            user_data=callback_user_info,
        )

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    attempts = 0
    iters = 0
    continue_iterating = True
    while attempts < max_attempts and iters < max_iters:
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())
        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

        # invoke callback
        if state_fitness_callback is not None:
            max_attempts_reached = attempts == max_attempts or iters == max_iters or problem.can_stop()
            continue_iterating = state_fitness_callback(
                iteration=iters,
                attempt=attempts + 1,
                done=max_attempts_reached,
                state=problem.get_state(),
                fitness=problem.get_adjusted_fitness(),
                curve=np.asarray(fitness_curve) if curve else None,
                user_data=callback_user_info,
            )

        if not continue_iterating:
            break

        if next_fitness > problem.get_maximize() * best_fitness:
            best_fitness = problem.get_maximize() * next_fitness
            best_state = next_state

        problem.set_state(next_state)

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None

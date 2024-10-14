"""Implementation of the Randomized Hill Climbing optimization algorithm."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name


@short_name("rhc")
def random_hill_climb(
    problem: Any,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    restarts: int = 0,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: dict = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use randomized hill climbing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing the optimization problem to be solved.
        For example, `DiscreteOpt()`, `ContinuousOpt()`, or `TSPOpt()`.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
        Must be a positive integer greater than 0.

    max_iters: int or float, default: np.inf
        Maximum number of iterations of the algorithm.
        Must be a positive integer greater than 0 or `np.inf`.

    restarts: int, default: 0
        Number of random restarts.
        Must be a non-negative integer.

    init_state: np.ndarray, default: None
        1-D Numpy array containing the starting state for the algorithm.
        If `None`, then a random state is used.

    curve: bool, default: False
        Whether to keep fitness values for a curve.
        If `False`, then no curve is stored.
        If `True`, then a history of fitness values is provided as a third return value.

    random_state: int, default: None
        Seed for the random number generator.

    state_fitness_callback: callable, default: None
        If specified, this callback function is invoked once per iteration with the following parameters:

        - iteration: int
          The current iteration number (starting from 0). `iteration=0` indicates the initial state before the optimization loop starts.
        - attempt: int
          The current number of consecutive unsuccessful attempts to find a better neighbor.
        - done: bool
          True if the algorithm is about to terminate (max attempts reached, max iterations reached, or `problem.can_stop()` returns True);
          False otherwise.
        - state: np.ndarray
          The current state vector.
        - fitness: float
          The current adjusted fitness value.
        - fitness_evaluations: int
          The cumulative number of fitness evaluations.
        - curve: np.ndarray or None
          The fitness curve up to the current iteration, or `None` if `curve=False`.
        - user_data: dict
          The user data passed in `callback_user_info`.

        The callback should return a boolean: `True` to continue iterating, or `False` to stop.

    callback_user_info: dict, default: None
        Dictionary of user-managed data passed as the `user_data` parameter of the callback function.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing the state that optimizes the fitness function.
    best_fitness: float
        Value of the fitness function at best state.
    fitness_curve: np.ndarray
        Numpy array of shape (n_iterations, 2), where each row represents:

        - Column 0: Adjusted fitness at the current iteration.
        - Column 1: Cumulative number of fitness evaluations.

        Only returned if the input argument `curve` is `True`.

    Notes
    -----
    - The `state_fitness_callback` function is also called before the optimization loop starts (iteration 0)
       with the initial state and fitness values.
    - The algorithm performs a total of `restarts + 1` runs, each time starting from a new random state (unless `init_state` is provided),
      and keeps track of the best state found across all restarts.

    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(f"max_attempts must be a positive integer. Got {max_attempts}")
    if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters < 0:
        raise ValueError(f"max_iters must be a positive integer or np.inf. Got {max_iters}")
    if not isinstance(restarts, int) or restarts < 0:
        raise ValueError(f"restarts must be a positive integer. Got {restarts}")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError(f"init_state must have the same length as the problem. Expected {problem.get_length()}, got {len(init_state)}")
    if callback_user_info is not None and not isinstance(callback_user_info, dict):
        raise TypeError(f"callback_user_info must be a dict. Got {type(callback_user_info).__name__}")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None
    best_fitness_curve = []
    all_curves = []

    problem.reset()
    for current_restart in range(restarts + 1):
        # Initialize optimization problem
        fevals = problem.fitness_evaluations
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)
        problem.fitness_evaluations = fevals

        fitness_curve = []

        # Prepare callback user data
        if state_fitness_callback is not None:
            if callback_user_info is None:
                callback_user_info = {}
            callback_user_info.update({"current_restart": current_restart})

            # Initial callback invocation
            continue_iterating = state_fitness_callback(
                iteration=0,
                attempt=0,
                done=False,
                state=problem.get_state(),
                fitness=problem.get_adjusted_fitness(),
                fitness_evaluations=problem.fitness_evaluations,
                curve=np.asarray(fitness_curve) if curve else None,
                user_data=callback_user_info,
            )
            if not continue_iterating:
                return problem.get_state(), best_fitness, np.asarray(best_fitness_curve) if curve else None

        attempts = 0
        iters = 0
        # Main optimization loop
        while attempts < max_attempts and iters < max_iters:
            iters += 1
            problem.current_iteration += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If next state is better, move to that state and reset attempts counter
            current_fitness = problem.get_fitness()
            if next_fitness > current_fitness:
                problem.set_state(next_state, next_fitness)
                attempts = 0
            else:
                attempts += 1

            if curve:
                adjusted_fitness = problem.get_adjusted_fitness()
                curve_value = (adjusted_fitness, problem.fitness_evaluations)
                fitness_curve.append(curve_value)
                all_curves.append(curve_value)

            # Invoke callback
            if state_fitness_callback is not None:
                max_attempts_reached = attempts == max_attempts or iters == max_iters or problem.can_stop()
                continue_iterating = state_fitness_callback(
                    iteration=iters,
                    attempt=attempts,
                    done=max_attempts_reached,
                    state=problem.get_state(),
                    fitness=problem.get_adjusted_fitness(),
                    fitness_evaluations=problem.fitness_evaluations,
                    curve=np.asarray(all_curves) if curve else None,
                    user_data=callback_user_info,
                )
                # Break out if requested
                if not continue_iterating:
                    break

            # Terminate if problem signals to stop
            if problem.can_stop():
                break

        # Update best state and best fitness if current is better
        current_fitness = problem.get_fitness()
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_state = problem.get_state()
            if curve:
                best_fitness_curve = [*fitness_curve]

        # Break out if problem signals to stop
        if problem.can_stop():
            break

    best_fitness *= problem.get_maximize()

    return best_state, best_fitness, np.asarray(best_fitness_curve) if curve else None

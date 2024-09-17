"""Functions to implement the randomized optimization and search algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name


@short_name("hc")
def hill_climb(
    problem: Any,
    max_iters: int | float = np.inf,
    restarts: int = 0,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: Any = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Use standard hill climbing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, `DiscreteOpt()`, `ContinuousOpt()` or `TSPOpt()`.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: np.ndarray, default: None
        1-D Numpy array containing starting state for algorithm.
        If None, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If False, then no curve is stored.
        If True, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    state_fitness_callback: callable, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info: any, default: None
        User data passed as last parameter of callback.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: np.ndarray
        Numpy array containing the fitness at every iteration.
        Only returned if input argument curve is True.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*, 3rd edition.
    Prentice Hall, New Jersey, USA.
    """
    # TODO: fix and uncomment these problematic raise statements
    # if not (isinstance(max_iters, int) or max_iters == np.inf or max_iters.is_integer()) or max_iters < 0:
    #     raise ValueError(f"max_iters must be a positive integer. Got {max_iters}")
    # if not (isinstance(restarts, int) or restarts.is_integer()) or restarts < 0:
    #     raise ValueError(f"restarts must be a positive integer. Got {restarts}")
    # if init_state is not None and len(init_state) != problem.get_length():
    #     raise ValueError(
    #         f"init_state must have the same length as the problem. " f"Expected length {problem.get_length()}, got {len(init_state)}"
    #     )

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None

    fitness_curve = []
    best_fitness_curve = []

    continue_iterating = True
    for current_restart in range(restarts + 1):
        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        callback_extra_data = None
        if state_fitness_callback is not None:
            callback_extra_data = callback_user_info + [("current_restart", current_restart)]
            # initial call with base data
            state_fitness_callback(
                iteration=0, state=problem.get_state(), fitness=problem.get_adjusted_fitness(), user_data=callback_extra_data
            )
        iters = 0
        while iters < max_iters:
            iters += 1

            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            if curve:
                fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

            # invoke callback
            if state_fitness_callback is not None:
                max_attempts_reached = (iters == max_iters) or problem.can_stop()
                continue_iterating = state_fitness_callback(
                    iteration=iters,
                    attempt=None,
                    done=max_attempts_reached,
                    state=problem.get_state(),
                    fitness=problem.get_adjusted_fitness(),
                    curve=np.asarray(fitness_curve) if curve else None,
                    user_data=callback_extra_data,
                )
                # break out if requested
                if not continue_iterating:
                    break

            # If best neighbor is an improvement, move to that state
            current_fitness = problem.get_fitness()
            if next_fitness > current_fitness:
                problem.set_state(next_state)
            else:
                break

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()
            if curve:
                best_fitness_curve = [*fitness_curve]
                fitness_curve = []

        # break out if requested
        if not continue_iterating:
            break

    best_fitness = problem.get_maximize() * best_fitness

    return best_state, best_fitness, np.asarray(best_fitness_curve) if curve else None

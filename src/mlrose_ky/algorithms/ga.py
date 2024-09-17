"""Functions to implement the randomized optimization and search algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name


def _get_hamming_distance_default(population: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hamming distance between a given individual and the rest of the population.

    Parameters
    ----------
    population : np.ndarray
        Population of individuals.
    p1 : np.ndarray
        Individual to compare with the population.

    Returns
    -------
    np.ndarray
        Array of Hamming distances.
    """
    return np.array([np.count_nonzero(p1 != p2) / len(p1) for p2 in population])


def _get_hamming_distance_float(population: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hamming distance (as a float) between a given individual and the rest of the population.

    Parameters
    ----------
    population : np.ndarray
        Population of individuals.
    p1 : np.ndarray
        Individual to compare with the population.

    Returns
    -------
    np.ndarray
        Array of Hamming distances.
    """
    return np.array([np.abs(p1 - p2) / len(p1) for p2 in population])


def _genetic_alg_select_parents(
    pop_size: int,
    problem: Any,
    get_hamming_distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    hamming_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select parents for the next generation in the genetic algorithm.

    Parameters
    ----------
    pop_size : int
        Size of the population.
    problem : optimization object
        The optimization problem instance.
    get_hamming_distance_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to calculate Hamming distance.
    hamming_factor : float, default: 0.0
        Factor to account for Hamming distance in parent selection.

    Returns
    -------
    tuple
        Selected parents (p1, p2) for reproduction.
    """
    mating_probabilities = problem.get_mate_probs()

    if get_hamming_distance_func is not None and hamming_factor > 0.01:
        population = problem.get_population()
        selected = np.random.choice(pop_size, p=mating_probabilities)
        p1 = population[selected]

        hamming_distances = get_hamming_distance_func(population, p1)
        hfa = hamming_factor / (1.0 - hamming_factor)
        hamming_distances = hamming_distances * hfa * mating_probabilities
        hamming_distances /= hamming_distances.sum()

        selected = np.random.choice(pop_size, p=hamming_distances)
        p2 = population[selected]

        return p1, p2

    selected = np.random.choice(pop_size, size=2, p=mating_probabilities)
    p1 = problem.get_population()[selected[0]]
    p2 = problem.get_population()[selected[1]]

    return p1, p2


@short_name("ga")
def genetic_alg(
    problem: Any,
    pop_size: int = 200,
    pop_breed_percent: float = 0.75,
    elite_dreg_ratio: float = 0.99,
    minimum_elites: int = 0,
    minimum_dregs: int = 0,
    mutation_prob: float = 0.1,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: Any = None,
    hamming_factor: float = 0.0,
    hamming_decay_factor: float = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use a standard genetic algorithm to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem : optimization object
        Object containing fitness function optimization problem to be solved.
    pop_size : int, default: 200
        Size of population to be used in genetic algorithm.
    pop_breed_percent : float, default 0.75
        Percentage of population to breed in each iteration.
        The remainder of the pop will be filled from the elite and dregs of the prior generation in a ratio specified by elite_dreg_ratio.
    elite_dreg_ratio : float, default:0.95
        The ratio of elites:dregs added directly to the next generation.
        For the default value, 95% of the added population will be elites, 5% will be dregs.
    minimum_elites : int, default: 0
        Minimum number of elites to be added to next generation
    minimum_dregs : int, default: 0
        Minimum number of dregs to be added to next generation
    mutation_prob : float, default: 0.1
        Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1.
    max_attempts : int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters : int | float, default: np.inf
        Maximum number of iterations of the algorithm.
    curve : bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a third return value.
    random_state : int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed(); otherwise, the random seed is not set.
    state_fitness_callback : Callable[..., Any], default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info : Any, default: None
        User data passed as last parameter of callback.
    hamming_factor : float, default: 0.0
        Factor to account for Hamming distance in parent selection.
    hamming_decay_factor : float, default: None
        Decay factor for the hamming_factor over iterations.

    Returns
    -------
    best_state : np.ndarray
        Numpy array containing state that optimizes the fitness function.
    best_fitness : float
        Value of fitness function at best state.
    fitness_curve : np.ndarray
        Numpy array of arrays containing the fitness of the entire population at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*, 3rd edition.
    Prentice Hall, New Jersey, USA.
    """
    # TODO: fix and uncomment these problematic raise statements
    # if not isinstance(pop_size, int) or pop_size < 0:
    #     raise ValueError(f"pop_size must be a positive integer. Got {pop_size}")
    # if not 0 <= pop_breed_percent <= 1:
    #     raise ValueError(f"pop_breed_percent must be between 0 and 1. Got {pop_breed_percent}")
    # if not 0 <= elite_dreg_ratio <= 1:
    #     raise ValueError(f"elite_dreg_ratio must be between 0 and 1. Got {elite_dreg_ratio}")
    # if not 0 <= mutation_prob <= 1:
    #     raise ValueError(f"mutation_prob must be between 0 and 1. Got {mutation_prob}")
    # if not isinstance(max_attempts, int) or max_attempts < 0:
    #     raise ValueError(f"max_attempts must be a positive integer. Got {max_attempts}")
    # if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters < 0:
    #     raise ValueError(f"max_iters must be a positive integer or np.inf. Got {max_iters}")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem
    fitness_curve = []
    problem.reset()
    problem.random_pop(pop_size)
    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(
            iteration=0,
            state=problem.get_state(),
            fitness=problem.get_adjusted_fitness(),
            fitness_evaluations=problem.fitness_evaluations,
            user_data=callback_user_info,
        )

    get_hamming_distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    if hamming_factor > 0:
        g1 = problem.get_population()[0][0]
        if isinstance(g1, float) or g1.dtype == "float64":
            get_hamming_distance_func = _get_hamming_distance_float
        else:
            get_hamming_distance_func = _get_hamming_distance_default

    # initialize survivor count, elite count and dreg count
    breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
    survivors_size = pop_size - breeding_pop_size
    dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
    elites_size = max(survivors_size - dregs_size, minimum_elites)
    if dregs_size + elites_size > survivors_size:
        over_population = dregs_size + elites_size - survivors_size
        breeding_pop_size -= over_population

    attempts = 0
    iters = 0
    continue_iterating = True
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1
        problem.current_iteration += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []
        for _ in range(breeding_pop_size):
            # Select parents
            parent_1, parent_2 = _genetic_alg_select_parents(
                pop_size=pop_size, problem=problem, hamming_factor=hamming_factor, get_hamming_distance_func=get_hamming_distance_func
            )

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        # fill remaining population with elites/dregs
        if survivors_size > 0:
            last_gen = list(zip(problem.get_population(), problem.get_pop_fitness()))
            sorted_parents = sorted(last_gen, key=lambda f: -f[1])
            best_parents = sorted_parents[:elites_size]
            next_gen.extend([p[0] for p in best_parents])
            if dregs_size > 0:
                worst_parents = sorted_parents[-dregs_size:]
                next_gen.extend([p[0] for p in worst_parents])

        next_gen = np.array(next_gen[:pop_size])
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement, move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
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
                fitness_evaluations=problem.fitness_evaluations,
                curve=np.asarray(fitness_curve) if curve else None,
                user_data=callback_user_info,
            )

        # decay hamming factor
        if hamming_decay_factor is not None and hamming_factor > 0.0:
            hamming_factor *= hamming_decay_factor
            hamming_factor = max(min(hamming_factor, 1.0), 0.0)

        # break out if requested
        if not continue_iterating:
            break

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None

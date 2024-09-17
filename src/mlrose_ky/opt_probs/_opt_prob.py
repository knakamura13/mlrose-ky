"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np


class _OptProb:
    """Base class for optimization problems.

    Parameters
    ----------
    length : int
        Number of elements in the state vector.
    fitness_fn : Any
        Object to implement the fitness function for optimization.
    maximize : bool, default=True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    Attributes
    ----------
    length : int
        Length of the state vector.
    fitness_fn : Any
        Fitness function for the optimization problem.
    maximize : float
        Maximization factor, 1.0 for maximization, -1.0 for minimization.
    state : np.ndarray
        Current state vector.
    neighbors : np.ndarray
        Array containing neighboring states.
    fitness : float
        Fitness value of the current state.
    population : np.ndarray
        Array containing the current population.
    pop_fitness : np.ndarray
        Array containing the fitness values for the current population.
    mate_probs : np.ndarray
        Array containing the mate probabilities for the current population.
    fevals : dict
        Dictionary for tracking function evaluations.
    fitness_evaluations : int
        Counter for the number of fitness evaluations.
    current_iteration : int
        Current iteration number in the optimization process.
    """

    def __init__(self, length: int, fitness_fn: Any, maximize: bool = True):
        if length < 0:
            raise ValueError("length must be a positive integer.")
        elif not isinstance(length, int):
            if length.is_integer():
                self.length: int = int(length)
            else:
                raise ValueError("length must be a positive integer.")
        else:
            self.length: int = length

        self.state: np.ndarray = np.array([0] * self.length)
        self.neighbors: np.ndarray = np.array([])
        self.fitness_fn: Any = fitness_fn
        self.fitness: float = 0.0
        self.population: np.ndarray = np.array([])
        self.pop_fitness: np.ndarray = np.array([])
        self.mate_probs: np.ndarray = np.array([])
        self.fevals: dict = {}
        self.fitness_evaluations: int = 0
        self.current_iteration: int = 0
        self.maximize: float = 1.0 if maximize else -1.0

    def best_child(self) -> np.ndarray:
        """Return the best state in the current population.

        Returns
        -------
        np.ndarray
            State vector defining the best child.
        """
        return self.population[np.argmax(self.pop_fitness)]

    def best_neighbor(self) -> np.ndarray:
        """Return the best neighbor of the current state.

        Returns
        -------
        np.ndarray
            State vector defining the best neighbor.
        """
        fitness_list = [self.eval_fitness(neigh) for neigh in self.neighbors]
        return self.neighbors[np.argmax(fitness_list)]

    def eval_fitness(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State vector for evaluation.

        Returns
        -------
        float
            Value of the fitness function.
        """
        if len(state) != self.length:
            raise ValueError(f"State length {len(state)} must match problem length {self.length}")

        fitness = self.maximize * self.fitness_fn.evaluate(state)
        self.fitness_evaluations += 1
        return fitness

    def eval_mate_probs(self) -> None:
        """Calculate the probability of each member of the population reproducing."""
        pop_fitness = np.copy(self.pop_fitness)

        # Set -1*inf values to 0 to avoid dividing by sum of infinity.
        # This forces mate_probs for these pop members to 0.
        pop_fitness[pop_fitness == -1.0 * np.inf] = 0

        # Account for maximize = False
        if self.maximize == -1.0:
            pop_fitness -= np.min(pop_fitness)

        if np.sum(pop_fitness) == 0:
            self.mate_probs = np.ones(len(pop_fitness)) / len(pop_fitness)
        else:
            self.mate_probs = pop_fitness / np.sum(pop_fitness)

    def get_fitness(self) -> float:
        """Return the fitness of the current state vector.

        Returns
        -------
        float
            Fitness value of the current state vector.
        """
        return self.fitness

    def get_adjusted_fitness(self) -> float:
        """Return maximization factor * fitness of the current state vector.

        Returns
        -------
        float
            Fitness value of the current state vector adjusted by the maximization factor.
        """
        return self.maximize * self.fitness

    def get_length(self) -> int:
        """Return the state vector length.

        Returns
        -------
        int
            Length of the state vector.
        """
        return self.length

    def get_mate_probs(self) -> np.ndarray:
        """Return the population mate probabilities.

        Returns
        -------
        np.ndarray
            Numpy array containing mate probabilities of the current population.
        """
        return self.mate_probs

    def get_maximize(self) -> float:
        """Return the maximization multiplier.

        Returns
        -------
        float
            Maximization multiplier.
        """
        return self.maximize

    def get_pop_fitness(self) -> np.ndarray:
        """Return the current population fitness array.

        Returns
        -------
        np.ndarray
            Numpy array containing the fitness values for the current population.
        """
        return self.pop_fitness

    def get_population(self) -> np.ndarray:
        """Return the current population.

        Returns
        -------
        np.ndarray
            Numpy array containing the current population.
        """
        return self.population

    def get_state(self) -> np.ndarray:
        """Return the current state vector.

        Returns
        -------
        np.ndarray
            Current state vector.
        """
        return self.state

    def set_population(self, new_population: np.ndarray) -> None:
        """Set a new population and evaluate its fitness.

        Parameters
        ----------
        new_population : np.ndarray
            Numpy array containing the new population.
        """
        self.population = new_population
        self.evaluate_population_fitness()

    def evaluate_population_fitness(self) -> None:
        """Evaluate the fitness of the current population."""
        self.pop_fitness = np.array([self.eval_fitness(indiv) for indiv in self.population])

    def set_state(self, new_state: np.ndarray) -> None:
        """Set a new state vector and evaluate its fitness.

        Parameters
        ----------
        new_state : np.ndarray
            New state vector.
        """
        if len(new_state) != self.length:
            raise ValueError(f"new_state length {len(new_state)} must match problem length {self.length}")

        self.state = new_state
        self.fitness = self.eval_fitness(self.state)

    def can_stop(self) -> bool:
        """Determine if the optimization process can stop.

        Returns
        -------
        bool
            Always returns False for this base class.
        """
        return False

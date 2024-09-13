"""TSP Crossover implementation for Genetic Algorithms (GA).

This module defines a TSP-specific crossover operation used in genetic algorithms,
which handles the mating of parent solutions to produce offspring that respect the TSP
constraints.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Sequence

import numpy as np

from mlrose_ky.algorithms.crossovers._crossover_base import _CrossoverBase


class TSPCrossover(_CrossoverBase):
    """
    Crossover operation tailored for the Travelling Salesperson Problem (TSP) in genetic algorithms.

    Implements specific crossover techniques that ensure valid TSP routes in the offspring.
    The crossover handles distinct city sequences without repetitions and uses specialized
    logic to combine parental genes.

    Inherits from:
    _CrossoverBase : Abstract base class for crossover operations.
    """

    def __init__(self, optimization_problem: Any):
        """
        Initialize the TSPCrossover with the given optimization problem.

        Parameters
        ----------
        optimization_problem : Any
            An instance of the optimization problem related to the genetic algorithm.
        """
        super().__init__(optimization_problem)

    def mate(self, parent1: Sequence[int], parent2: Sequence[int]) -> np.ndarray:
        """
        Perform the crossover (mating) between two parent sequences to produce offspring.

        Chooses between two internal methods to generate offspring based on TSP-specific
        constraints and optimizations.

        Parameters
        ----------
        parent1 : Sequence[int]
            The first parent representing a TSP route.
        parent2 : Sequence[int]
            The second parent representing a TSP route.

        Returns
        -------
        np.ndarray
            The offspring representing a new TSP route.
        """
        return self._mate_fill(parent1, parent2)

    def _mate_fill(self, parent1: Sequence[int], parent2: Sequence[int]) -> np.ndarray:
        """
        Perform a fill-based crossover using a segment of the first parent and filling
        the rest with non-repeated cities from the second parent.

        Parameters
        ----------
        parent1 : Sequence[int]
            The first parent representing a TSP route.
        parent2 : Sequence[int]
            The second parent representing a TSP route.

        Returns
        -------
        np.ndarray
            The offspring TSP route.
        """
        if self._length > 1:
            n = 1 + np.random.randint(self._length - 1)
            child = np.array([0] * self._length)
            child[:n] = parent1[:n]
            unvisited = [city for city in parent2 if city not in parent1[:n]]
            child[n:] = unvisited
        else:
            child = np.copy(parent1 if np.random.randint(2) == 0 else parent2)

        return child

    def _mate_traverse(self, parent1: Sequence[int], parent2: Sequence[int]) -> np.ndarray:
        """
        Perform a traversal-based crossover using city adjacency considerations from
        both parents to construct a viable TSP route.

        The method determines the next city to visit based on the adjacency in both
        parents' routes, considering previously visited cities and selecting based
        on fitness values where applicable.

        Parameters
        ----------
        parent1 : Sequence[int]
            The first parent representing a TSP route.
        parent2 : Sequence[int]
            The second parent representing a TSP route.

        Returns
        -------
        np.ndarray
            The offspring TSP route.
        """
        if self._length > 1:
            next_city_parent1 = np.append(parent1[1:], parent1[0])
            next_city_parent2 = np.append(parent2[1:], parent2[0])

            visited_cities = [False] * self._length
            offspring_route = np.array([0] * self._length)

            starting_city = np.random.randint(len(parent1))
            offspring_route[0] = starting_city
            visited_cities[starting_city] = True

            for index in range(1, len(offspring_route)):
                current_city = offspring_route[index - 1]
                next_city1 = next_city_parent1[current_city]
                next_city2 = next_city_parent2[current_city]

                visited_city1 = visited_cities[next_city1]
                visited_city2 = visited_cities[next_city2]

                if visited_city1 and not visited_city2:
                    next_city = next_city2
                elif not visited_city1 and visited_city2:
                    next_city = next_city1
                elif not visited_city1 and not visited_city2:
                    fitness1 = self._opt_prob.fitness_fn.calculate_fitness([current_city, next_city1])
                    fitness2 = self._opt_prob.fitness_fn.calculate_fitness([current_city, next_city2])
                    next_city = next_city2 if fitness1 > fitness2 else next_city1  # Choose the smaller distance
                else:
                    while True:
                        next_city = np.random.randint(len(parent1))
                        if not visited_cities[next_city]:
                            break

                offspring_route[index] = next_city
                visited_cities[next_city] = True
        else:
            offspring_route = np.copy(parent1 if np.random.randint(2) == 0 else parent2)

        return offspring_route

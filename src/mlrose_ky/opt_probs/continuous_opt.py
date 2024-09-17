"""Classes for defining optimization problem objects."""

from typing import Any

import numpy as np

from mlrose_ky.opt_probs._opt_prob import _OptProb


# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause


class ContinuousOpt(_OptProb):
    """Class for defining continuous-state optimization problems.

    Parameters
    ----------
    length : int
        Number of elements in the state vector.

    fitness_fn : Any
        Fitness function for optimization.

    maximize : bool, default=True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    min_val : float, default=0
        Minimum value that each element of the state vector can take.

    max_val : float, default=1
        Maximum value that each element of the state vector can take.

    step : float, default=0.
        Step size used in determining neighbors of the current state.

    Attributes
    ----------
    min_val : float
        Minimum value for the state vector elements.
    max_val : float
        Maximum value for the state vector elements.
    step : float
        Step size used in neighbor determination.
    prob_type : str
        Problem type; always 'continuous' for this class.
    """

    def __init__(self, length: int, fitness_fn: Any, maximize: bool = True, min_val: float = 0.0, max_val: float = 1.0, step: float = 0.1):
        super().__init__(length, fitness_fn, maximize=maximize)

        if self.fitness_fn.get_prob_type() not in {"continuous", "either"}:
            raise ValueError(
                "fitness_fn must have problem type 'continuous' or 'either'. "
                "Define problem as DiscreteOpt or use an appropriate fitness function."
            )

        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val.")
        if step <= 0:
            raise ValueError("step size must be positive.")
        if (max_val - min_val) < step:
            raise ValueError(f"step size must be less than (max_val - min_val).")

        self.prob_type: str = "continuous"
        self.min_val: float = min_val
        self.max_val: float = max_val
        self.step: float = step

    def calculate_updates(self) -> list:
        """Calculate gradient descent updates.

        Returns
        -------
        list
            List of back-propagation weight updates.
        """
        return self.fitness_fn.calculate_updates()

    def find_neighbors(self):
        """Find all neighbors of the current state."""
        # Pre-allocate a NumPy array for neighbors (maximum of 2 * length neighbors)
        neighbors_matrix = np.zeros((2 * self.length, self.length))

        neighbor_count = 0  # Track how many valid neighbors we find

        for i in range(self.length):
            base_neighbor = np.copy(self.state)

            for step_dir in [-self.step, self.step]:
                neighbor = base_neighbor.copy()
                neighbor[i] += step_dir

                # Clip the value to ensure it remains within bounds
                neighbor[i] = np.clip(neighbor[i], self.min_val, self.max_val)

                # Only add if different from the current state
                if not np.array_equal(neighbor, self.state):
                    neighbors_matrix[neighbor_count] = neighbor
                    neighbor_count += 1

        # Store only the valid neighbors (slice the array up to the count of valid ones)
        self.neighbors = neighbors_matrix[:neighbor_count]

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Problem type.
        """
        return self.prob_type

    def random(self) -> np.ndarray:
        """Generate and return a random state vector.

        Returns
        -------
        np.ndarray
            Randomly generated state vector.
        """
        return np.random.uniform(self.min_val, self.max_val, self.length)

    def random_neighbor(self) -> np.ndarray:
        """Return a random neighbor of the current state vector.

        Returns
        -------
        np.ndarray
            State vector of the random neighbor.
        """
        while True:
            neighbor = np.copy(self.state)
            i = np.random.randint(0, self.length)
            neighbor[i] += self.step * np.random.choice([-1, 1])

            if neighbor[i] > self.max_val:
                neighbor[i] = self.max_val
            elif neighbor[i] < self.min_val:
                neighbor[i] = self.min_val

            if not np.array_equal(neighbor, self.state):
                break

        return neighbor

    def random_pop(self, pop_size: int):
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size : int
            Size of the population to be created.

        Raises
        ------
        ValueError
            If pop_size is not a positive integer.
        """
        if pop_size <= 0 or not isinstance(pop_size, int):
            raise ValueError("pop_size must be a positive integer.")

        population = []
        pop_fitness = []

        for _ in range(pop_size):
            state = self.random()
            fitness = self.eval_fitness(state)
            population.append(state)
            pop_fitness.append(fitness)

        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)

    def reproduce(self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float = 0.1) -> np.ndarray:
        """Create a child state vector from two parent state vectors.

        Parameters
        ----------
        parent_1 : np.ndarray
            State vector for parent 1.

        parent_2 : np.ndarray
            State vector for parent 2.

        mutation_prob : float, default=0.1
            Probability of a mutation at each state vector element during reproduction.

        Returns
        -------
        np.ndarray
            Child state vector produced from parents 1 and 2.

        Raises
        ------
        ValueError
            If the lengths of the parents do not match the problem length,
            or if mutation_prob is not between 0 and 1.
        """
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise ValueError("Lengths of parents must match problem length.")

        if not (0 <= mutation_prob <= 1):
            raise ValueError("mutation_prob must be between 0 and 1.")

        if self.length > 1:
            _n = np.random.randint(self.length - 1)
            child = np.zeros(self.length)
            child[0 : _n + 1] = parent_1[0 : _n + 1]
            child[_n + 1 :] = parent_2[_n + 1 :]
        else:
            child = np.copy(parent_1 if np.random.randint(2) == 0 else parent_2)

        # Mutate child
        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]

        for i in mutate:
            child[i] = np.random.uniform(self.min_val, self.max_val)

        return child

    def reset(self):
        """Set the current state vector to a random value and reset its fitness."""
        self.state = self.random()
        self.fitness_evaluations = 0
        self.fitness = self.eval_fitness(self.state)

    def update_state(self, updates: np.ndarray) -> np.ndarray:
        """Update the current state given a vector of updates.

        Parameters
        ----------
        updates : np.ndarray
            Array of updates.

        Returns
        -------
        np.ndarray
            Current state adjusted for updates.

        Raises
        ------
        ValueError
            If the length of updates does not match the problem length.
        """
        if len(updates) != self.length:
            raise ValueError("Length of updates must match problem length.")

        updated_state = self.state + updates
        updated_state[updated_state > self.max_val] = self.max_val
        updated_state[updated_state < self.min_val] = self.min_val

        return updated_state

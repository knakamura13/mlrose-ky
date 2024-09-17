"""Classes for defining optimization problem objects."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
from sklearn.metrics import mutual_info_score

from mlrose_ky.algorithms.crossovers import UniformCrossOver, TSPCrossOver
from mlrose_ky.algorithms.mutators import SwapMutator
from mlrose_ky.opt_probs._opt_prob import _OptProb


class DiscreteOpt(_OptProb):
    """Class for defining discrete-state optimization problems.

    Parameters
    ----------
    length : int
        Number of elements in the state vector.

    fitness_fn : Any
        Object to implement the fitness function for optimization.

    maximize : bool, default=True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    max_val : int, default=2
        Number of unique values that each element in the state vector can take.
        Assumes values are integers in the range 0 to (max_val - 1), inclusive.

    crossover : UniformCrossOver | TSPCrossOver, default=None
        Crossover operation used for reproduction. If None, defaults to `UniformCrossOver`.

    mutator : SwapMutator, default=None
        Mutation operation used for reproduction. If None, defaults to `SwapMutator`.

    Attributes
    ----------
    keep_sample : np.ndarray
        Array of samples from the top percentile of the population.
    node_probs : np.ndarray
        Probability density estimates for each node.
    parent_nodes : np.ndarray
        Parent nodes based on the minimum spanning tree.
    sample_order : list[int]
        Order of generating sample vector elements.
    prob_type : str
        Problem type; always 'discrete' for this class.
    noise : float
        Noise factor for probability density estimation.
    _crossover : UniformCrossOver
        Crossover operation for reproduction.
    _mutator : SwapMutator
        Mutation operation for reproduction.
    _mut_mask : np.ndarray, default=None
        Mask for mutual information computation in fast mode.
    _mut_inf : np.ndarray, default=None
        Mutual information matrix.
    """

    def __init__(
        self,
        length: int,
        fitness_fn: Any,
        maximize: bool = True,
        max_val: int = 2,
        crossover: UniformCrossOver | TSPCrossOver = None,
        mutator: "SwapMutator" = None,
    ):
        self._get_mutual_info_impl = self._get_mutual_info_slow

        super().__init__(length, fitness_fn, maximize)

        if self.fitness_fn.get_prob_type() == "continuous":
            raise ValueError(
                "fitness_fn must have problem type 'discrete', 'either', or 'tsp'."
                " Define problem as ContinuousOpt or use an appropriate fitness function."
            )

        if not max_val or max_val < 0:
            raise ValueError(f"max_val must be a positive integer. Got {max_val}")
        elif not isinstance(max_val, int):
            if max_val.is_integer():
                self.max_val: int = int(max_val)
            else:
                raise ValueError(f"max_val must be a positive integer. Got {max_val}")
        else:
            self.max_val: int = max_val

        self.prob_type: str = "discrete"
        self.keep_sample: np.ndarray = np.array([])
        self.node_probs: np.ndarray = np.zeros([self.length, self.max_val, self.max_val])
        self.parent_nodes: np.ndarray = np.array([])
        self.sample_order: list[int] = []
        self.noise: float = 0

        self._crossover: UniformCrossOver | TSPCrossOver = UniformCrossOver(self) if crossover is None else crossover
        self._mutator: SwapMutator = SwapMutator(self) if mutator is None else mutator

        self._mut_mask: np.ndarray | None = None
        self._mut_inf: np.ndarray | None = None

    def eval_node_probs(self) -> None:
        """Update probability density estimates."""
        mutual_info = self._get_mutual_info_impl()

        # Find minimum spanning tree of mutual info matrix
        csr_mx = csr_matrix(mutual_info)
        # noinspection PyTypeChecker
        mst = minimum_spanning_tree(csr_mx)

        # Convert MST to depth first tree with node 0 as root
        dft = depth_first_tree(csr_matrix(mst.toarray()), 0, directed=False)
        dft = np.round(dft.toarray(), 10)

        # Determine parent of each node
        parent = np.argmin(dft[:, 1:], axis=0)

        probs = np.zeros([self.length, self.max_val, self.max_val])
        probs[0] = np.histogram(self.keep_sample[:, 0], np.arange(self.max_val + 1), density=True)[0]

        for i in range(1, self.length):
            for j in range(self.max_val):
                subset = self.keep_sample[np.where(self.keep_sample[:, parent[i - 1]] == j)[0]]

                if not len(subset):
                    probs[i, j] = 1 / self.max_val
                else:
                    temp_probs = np.histogram(subset[:, i], np.arange(self.max_val + 1), density=True)[0]
                    if self.noise > 0:
                        temp_probs += self.noise
                        temp_probs = np.divide(temp_probs, np.sum(temp_probs))
                        if sum(temp_probs) != 1.0:
                            temp_probs = np.divide(temp_probs, np.sum(temp_probs))
                    probs[i, j] = temp_probs

        self.node_probs = probs
        self.parent_nodes = parent

    def set_mimic_fast_mode(self, fast_mode: bool) -> None:
        """Enable or disable MIMIC fast mode."""
        if fast_mode:
            mut_mask = np.zeros([self.length, self.length], dtype=bool)

            for i in range(0, self.length):
                for j in range(i, self.length):
                    mut_mask[i, j] = True

            mut_mask = mut_mask.reshape((self.length * self.length))
            self._mut_mask = mut_mask

            np.seterr(divide="ignore", invalid="ignore")
            self._get_mutual_info_impl = self._get_mutual_info_fast
            self._mut_inf = np.zeros([self.length * self.length])
        else:
            self._mut_mask = None
            self._get_mutual_info_impl = self._get_mutual_info_slow
            self._mut_inf = None

    def _get_mutual_info_slow(self) -> np.ndarray:
        mutual_info = np.zeros([self.length, self.length])

        for i in range(self.length - 1):
            for j in range(i + 1, self.length):
                mutual_info[i, j] = -1 * mutual_info_score(self.keep_sample[:, i], self.keep_sample[:, j])

        return mutual_info

    def _get_mutual_info_fast(self) -> np.ndarray:
        if self._mut_inf is None:
            self._get_mutual_info_impl = self._get_mutual_info_slow
            return self._get_mutual_info_impl()

        len_sample_kept = self.keep_sample.shape[0]
        len_prob = self.keep_sample.shape[1]

        b = np.repeat(self.keep_sample, self.length).reshape(len_sample_kept, len_prob * len_prob)
        d = np.hstack(([self.keep_sample] * len_prob))

        self._mut_inf.fill(0)
        U = {}
        V = {}
        U_sum = {}
        V_sum = {}
        for i in range(0, self.max_val):
            U[i] = d == i
            V[i] = b == i
            U_sum[i] = np.sum(d == i, axis=0)
            V_sum[i] = np.sum(b == i, axis=0)

        for i in range(0, self.max_val):
            for j in range(0, self.max_val):
                coeff = np.sum(U[i] * V[j], axis=0)
                UV_length = U_sum[i] * V_sum[j]

                temp = np.log(coeff) - np.log(UV_length) + np.log(len_sample_kept)
                temp[np.isnan(temp)] = 0
                temp[np.isneginf(temp)] = 0

                div = temp * np.divide(coeff, len_sample_kept)
                div[self._mut_mask] = 0
                self._mut_inf += div

        self._mut_inf = -self._mut_inf.reshape(self.length, self.length)

        mutual_info = self._mut_inf.T
        self._mut_inf = self._mut_inf.reshape(self.length * self.length)

        return mutual_info

    def find_neighbors(self) -> None:
        """Find all neighbors of the current state."""
        self.neighbors = []

        if self.max_val == 2:
            for i in range(self.length):
                neighbor = np.copy(self.state)
                neighbor[i] = np.abs(neighbor[i] - 1)
                self.neighbors.append(neighbor)
        else:
            for i in range(self.length):
                vals = list(np.arange(self.max_val))
                vals.remove(self.state[i])

                for j in vals:
                    neighbor = np.copy(self.state)
                    neighbor[i] = j
                    self.neighbors.append(neighbor)

    def find_sample_order(self) -> None:
        """Determine order in which to generate sample vector elements."""
        sample_order = []
        last = [0]
        parent = self.parent_nodes

        while len(sample_order) < self.length:
            inds = []

            if len(last) == 0:
                inds = [np.random.choice(list(set(np.arange(self.length)) - set(sample_order)))]
            else:
                for i in last:
                    inds += list(np.where(parent == i)[0] + 1)

            sample_order += last
            last = inds

        self.sample_order = sample_order

    def find_top_pct(self, keep_pct: float) -> None:
        """Select samples with fitness in the top keep_pct percentile.

        Parameters
        ----------
        keep_pct : float
            Proportion of samples to keep.
        """
        if not (0 <= keep_pct <= 1):
            raise ValueError("keep_pct must be between 0 and 1.")

        theta = np.percentile(self.pop_fitness, 100 * (1 - keep_pct))
        keep_inds = np.where(self.pop_fitness >= theta)[0]
        self.keep_sample = self.population[keep_inds]

    def get_keep_sample(self) -> np.ndarray:
        """Return the keep sample.

        Returns
        -------
        np.ndarray
            Numpy array containing samples with fitness in the top keep_pct percentile.
        """
        return self.keep_sample

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Returns problem type.
        """
        return self.prob_type

    def random(self) -> np.ndarray:
        """Return a random state vector.

        Returns
        -------
        np.ndarray
            Randomly generated state vector.
        """
        return np.random.randint(0, self.max_val, self.length)

    def random_neighbor(self) -> np.ndarray:
        """Return random neighbor of current state vector.

        Returns
        -------
        np.ndarray
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)

        if self.max_val == 2:
            neighbor[i] = np.abs(neighbor[i] - 1)
        else:
            vals = list(np.arange(self.max_val))
            vals.remove(neighbor[i])
            neighbor[i] = vals[np.random.randint(0, self.max_val - 1)]

        return neighbor

    def random_pop(self, pop_size: int) -> None:
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size : int
            Size of population to be created.
        """
        if pop_size <= 0:
            raise ValueError("pop_size must be a positive integer.")

        population = []
        pop_fitness = []

        for _ in range(pop_size):
            state = self.random()
            population.append(state)
            fitness = self.eval_fitness(state)
            pop_fitness.append(fitness)

        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)

    def reproduce(self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float = 0.1) -> np.ndarray:
        """Create child state vector from two parent state vectors.

        Parameters
        ----------
        parent_1 : np.ndarray
            State vector for parent 1.
        parent_2 : np.ndarray
            State vector for parent 2.
        mutation_prob : float
            Probability of a mutation at each state element during reproduction.

        Returns
        -------
        np.ndarray
            Child state vector produced from parents 1 and 2.
        """
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise ValueError("Lengths of parents must match problem length.")

        if not (0 <= mutation_prob <= 1):
            raise ValueError("mutation_prob must be between 0 and 1.")

        child = self._crossover.mate(parent_1, parent_2)
        return self._mutator.mutate(child, mutation_prob)

    def reset(self) -> None:
        """Set the current state vector to a random value and get its fitness."""
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)
        self.fevals = {}
        self.fitness_evaluations = 0
        self.current_iteration = 0

    def sample_pop(self, sample_size: int) -> np.ndarray:
        """Generate new sample from probability density.

        Parameters
        ----------
        sample_size : int
            Size of sample to be generated.

        Returns
        -------
        np.ndarray
            Numpy array containing new sample.
        """
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer.")

        new_sample = np.zeros([sample_size, self.length])
        new_sample[:, 0] = np.random.choice(self.max_val, sample_size, p=self.node_probs[0, 0])

        self.find_sample_order()
        sample_order = self.sample_order[1:]

        for i in sample_order:
            par_ind = self.parent_nodes[i - 1]

            for j in range(self.max_val):
                inds = np.where(new_sample[:, par_ind] == j)[0]
                new_sample[inds, i] = np.random.choice(self.max_val, len(inds), p=self.node_probs[i, j])

        return new_sample

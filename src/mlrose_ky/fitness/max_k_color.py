"""Class defining the Max-K Color fitness function for use with optimization algorithms."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

import numpy as np


class MaxKColor:
    """Fitness function for Max-K color optimization problem.

    Evaluates the fitness of an n-dimensional state vector
    .. math::

        x = [x_{0}, x_{1}, \\ldots, x_{n-1}]

    where :math:`x_{i}` represents the color of node i, as the number of pairs of adjacent nodes
    of the same color.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        List of all pairs of connected nodes. Order does not matter, so (a, b)
        and (b, a) are considered to be the same.

    maximize : bool, optional, default=False
        Whether to maximize or minimize the fitness function.

    Examples
    --------
    >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    >>> fitness = MaxKColor(edges)
    >>> state_vector = np.array([0, 1, 0, 1, 1])
    >>> fitness.evaluate(state_vector)
    3.0

    Note
    ----
    The MaxKColor fitness function is suitable for use in discrete-state
    optimization problems *only*.

    If this is a cost minimization problem: lower scores are better than
    higher scores. That is, for a given graph, and a given number of colors,
    the challenge is to assign a color to each node in the graph such that
    the number of pairs of adjacent nodes of the same color is minimized.

    If this is a cost maximization problem: higher scores are better than
    lower scores. That is, for a given graph, and a given number of colors,
    the challenge is to assign a color to each node in the graph such that
    the number of pairs of adjacent nodes of different colors are maximized.
    """

    def __init__(self, edges: list[tuple[int, int]], maximize: bool = False):
        """
        Initialize the MaxKColor fitness function.

        Parameters
        ----------
        edges : List[Tuple[int, int]]
            List of all pairs of connected nodes.

        maximize : bool, optional, default=False
            Whether to maximize or minimize the fitness function.
        """
        self.prob_type: str = "discrete"
        self.maximize: bool = maximize
        self.graph_edges: list[tuple[int, int]] | None = None

        # Remove any duplicates from list
        # noinspection PyTypeChecker
        self.edges: list[tuple[int, int]] = list({tuple(sorted(edge)) for edge in edges})

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state : np.ndarray
            State array for evaluation.

        Returns
        -------
        float
            Value of fitness function.

        Raises
        ------
        TypeError
            If `state` is not an instance of `np.ndarray`.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state_vector to be np.ndarray, got {type(state).__name__} instead.")

        edges = self.graph_edges if self.graph_edges is not None else self.edges

        if self.maximize:
            # Maximize the number of adjacent nodes not of the same color.
            return float(sum(state[n1] != state[n2] for (n1, n2) in edges))

        # Minimize the number of adjacent nodes of the same color.
        return float(sum(state[n1] == state[n2] for (n1, n2) in edges))

    def get_prob_type(self) -> str:
        """Return the problem type.

        Returns
        -------
        str
            Specifies problem type as 'discrete'.
        """
        return self.prob_type

    def set_graph(self, graph):
        """Set the graph edges from an external graph representation.

        Parameters
        ----------
        graph : Any
            A graph object with an `edges()` method that returns a list of edges.
        """
        self.graph_edges = [e for e in graph.edges()]

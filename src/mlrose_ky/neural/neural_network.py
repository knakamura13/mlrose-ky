"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from sklearn.base import ClassifierMixin

from mlrose_ky.algorithms.decay import GeomDecay
from ._nn_core import _NNCore


class NeuralNetwork(_NNCore, ClassifierMixin):
    """
    Class for defining neural network classifier weights optimization problem.

    Parameters
    ----------
    hidden_nodes : list of int, optional
        List giving the number of nodes in each hidden layer.

    activation : str, default='relu'
        Activation function for each of the hidden layers. Must be one of:
        'identity', 'relu', 'sigmoid', or 'tanh'.

    algorithm : str, default='random_hill_climb'
        Algorithm used to find optimal network weights. Must be one of:
        'random_hill_climb', 'simulated_annealing', 'genetic_alg', or
        'gradient_descent'.

    max_iters : int, default=100
        Maximum number of iterations used to fit the weights.

    bias : bool, default=True
        Whether to include a bias term.

    is_classifier : bool, default=True
        Whether the network is for classification or regression. Set
        :code:`True` for classification and :code:`False` for regression.

    learning_rate : float, default=0.1
        Learning rate for gradient descent or step size for randomized
        optimization algorithms.

    early_stopping : bool, default=False
        Whether to terminate the algorithm early if the loss is not improving.

    clip_max : float, default=1e+10
        Used to limit weights to the range [-1 * clip_max, clip_max].

    restarts : int, default=0
        Number of random restarts. Only required if :code:`algorithm='random_hill_climb'`.

    schedule : GeomDecay, default=mlrose_ky.GeomDecay()
        Schedule used to determine the value of the temperature parameter.
        Only required if :code=`algorithm='simulated_annealing'`.

    pop_size : int, default=200
        Size of population. Only required if :code=`algorithm='genetic_alg'`.

    mutation_prob : float, default=0.1
        Probability of a mutation at each element of the state vector during
        reproduction, expressed as a value between 0 and 1. Only required if
        :code=`algorithm='genetic_alg'`.

    max_attempts : int, default=10
        Maximum number of attempts to find a better state. Only required if
        :code=`early_stopping=True`.

    random_state : int or None, default=None
        If random_state is a positive integer, it is used as the seed by
        np.random.seed(); otherwise, the random seed is not set.

    curve : bool, default=False
        If True, fitness_curve containing the fitness at each training
        iteration is returned.

    Attributes
    ----------
    fitted_weights : np.ndarray
        Numpy array giving the fitted weights after :code=`fit`.

    loss : float
        Value of the loss function for fitted weights after :code=`fit`.

    predicted_probs : np.ndarray
        Numpy array giving the predicted probabilities for each class when
        :code=`predict` is performed for multi-class classification data; or
        the predicted probability for class 1 when :code=`predict` is performed
        for binary classification data.

    fitness_curve : np.ndarray
        Numpy array giving the fitness at each training iteration.
    """

    def __init__(
        self,
        hidden_nodes: list[int] = None,
        activation: str = "relu",
        algorithm: str = "random_hill_climb",
        max_iters: int = 100,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: GeomDecay = GeomDecay(),
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: int = None,
        curve: bool = False,
    ):
        # Initialize the NeuralNetwork model with the given parameters
        super().__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve,
        )
"""mlrose-ky initialization file."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

# noinspection PyUnresolvedReferences
from .algorithms.ga import genetic_alg

# noinspection PyUnresolvedReferences
from .algorithms.sa import simulated_annealing

# noinspection PyUnresolvedReferences
from .algorithms.hc import hill_climb

# noinspection PyUnresolvedReferences
from .algorithms.rhc import random_hill_climb

# noinspection PyUnresolvedReferences
from .algorithms.gd import gradient_descent

# noinspection PyUnresolvedReferences
from .algorithms.mimic import mimic

# noinspection PyUnresolvedReferences
from .algorithms.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule

# noinspection PyUnresolvedReferences
from .algorithms.crossovers import OnePointCrossOver, UniformCrossOver, TSPCrossOver

# noinspection PyUnresolvedReferences
from .algorithms.mutators import ChangeOneMutator, DiscreteMutator, SwapMutator, ShiftOneMutator

# noinspection PyUnresolvedReferences
from .fitness import OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks, Knapsack, TravellingSales, Queens, MaxKColor, CustomFitness

# noinspection PyUnresolvedReferences,PyProtectedMember
from .neural import NeuralNetwork, LinearRegression, LogisticRegression, NNClassifier, _nn_core

# noinspection PyUnresolvedReferences
from .neural.activation import identity, relu, leaky_relu, sigmoid, softmax, tanh

# noinspection PyUnresolvedReferences
from .neural.fitness import NetworkWeights

# noinspection PyUnresolvedReferences
from .neural.utils.weights import flatten_weights, unflatten_weights

# noinspection PyUnresolvedReferences
from .gridsearch import GridSearchMixin

# noinspection PyUnresolvedReferences
from .opt_probs import DiscreteOpt, ContinuousOpt, KnapsackOpt, TSPOpt, QueensOpt, FlipFlopOpt, MaxKColorOpt

# noinspection PyUnresolvedReferences
from .runners import GARunner, MIMICRunner, RHCRunner, SARunner, NNGSRunner, SKMLPRunner, build_data_filename

# noinspection PyUnresolvedReferences
from .generators import MaxKColorGenerator, QueensGenerator, FlipFlopGenerator, TSPGenerator, KnapsackGenerator, ContinuousPeaksGenerator

# noinspection PyUnresolvedReferences
from .samples import SyntheticData, plot_synthetic_dataset

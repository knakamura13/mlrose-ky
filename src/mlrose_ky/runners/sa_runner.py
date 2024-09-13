"""
Class for running optimization experiments using Simulated Annealing (SA), including grid search functionality.

Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)

    # note that you can also initialize a temperature_list this way
    # temperature_list = [mlrose_ky.GeomDecay(init_temp=t, decay=d) for (t, d) in [(1, 0.99), (1e2, 0.999)]]
    # if you use this form, the decay_list parameter is ignored.

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                  decay_list=[mlrose_ky.GeomDecay])

    df_run_stats, df_run_curves = sa.run()
"""

# Authors: Andrew Rollings (modified by Kyle Nakamura)
# License: BSD 3-clause

from typing import Any, Callable

import numpy as np
import pandas as pd

import mlrose_ky
from mlrose_ky.decorators import short_name
from mlrose_ky.runners._runner_base import _RunnerBase


@short_name("sa")
class SARunner(_RunnerBase):
    """
    A runner for performing optimization experiments using Simulated Annealing (SA).

    This class extends _RunnerBase and provides functionality for running experiments
    with the SA algorithm, including grid search over hyperparameters such as
    temperature and decay schedules.

    Attributes
    ----------
    temperature_list : list[float]
        List of temperatures to test in the grid search.
    decay_list : list[Callable], optional
        List of decay schedules to use if raw temperature values are not provided.
    use_raw_temp : bool
        Whether to use raw temperature values or decay schedules.
    """

    def __init__(
        self,
        problem: Any,
        experiment_name: str,
        seed: int,
        iteration_list: list[int],
        temperature_list: list[float],
        decay_list: list[Callable] | None = None,
        max_attempts: int = 500,
        generate_curves: bool = True,
        **kwargs: dict,
    ):
        """
        Initialize the SARunner class with problem data and various experiment parameters.

        Parameters
        ----------
        problem : Any
            The optimization problem to be solved.
        experiment_name : str
            Name of the experiment.
        seed : int
            Random seed for reproducibility.
        iteration_list : list of int
            List of iterations for the experiment.
        temperature_list : list of float
            List of temperature values to test in the grid search.
        decay_list : list of callable, optional
            List of decay schedules to use, if raw temperature values are not provided.
        max_attempts : int, optional
            Maximum number of attempts without improvement before stopping.
        generate_curves : bool, optional
            Whether to generate learning curves.
        """
        super().__init__(
            problem=problem,
            experiment_name=experiment_name,
            seed=seed,
            iteration_list=iteration_list,
            max_attempts=max_attempts,
            generate_curves=generate_curves,
            **kwargs,
        )
        self.use_raw_temp = True
        self.temperature_list: list[float] = temperature_list

        # Use decay schedules if provided
        if all([np.isscalar(x) for x in temperature_list]):
            if decay_list is None:
                decay_list = [mlrose_ky.GeomDecay]

            self.decay_list: list[Callable] = decay_list
            self.use_raw_temp = False

    def run(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Run the Simulated Annealing (SA) experiment.

        This method performs grid search over the provided temperature values and
        decay schedules, and returns the statistics and curves generated by the experiment.

        Returns
        -------
        tuple
            A tuple containing two DataFrames: run statistics and run curves.
        """
        temperatures = (
            self.temperature_list if self.use_raw_temp else [d(init_temp=t) for t in self.temperature_list for d in self.decay_list]
        )
        return super().run_experiment(algorithm=mlrose_ky.simulated_annealing, schedule=("Temperature", temperatures))

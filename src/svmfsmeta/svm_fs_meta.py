from copy import deepcopy
from math import isfinite
from numbers import Integral, Real
from time import time
from matplotlib.pyplot import show, figure, xlabel, ylabel, title, grid, scatter
import os
import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .population import Population
import logging

logger = logging.getLogger(__name__)


def _is_finite_real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


class SVMFsMeta:
    """
    NSGA-II based algorithm for simultaneous Support Vector Machine training and Feature Selection (SVM + FS).

    Parameters
    ----------
    method : str
        The multi-objective optimization method ('DIST-EPS', 'MC').
    data : pandas.DataFrame
        The dataset to be used.
    population_size : int
        Integer number of individuals in the population, at least 3.
    inputs : list
        List of input (feature) column names.
    output : str
        Name of the output (target) column.
    num_selected_features : int
        Exact integer number of selected features, from 1 to len(inputs).
    allow_clones : int, optional
        Whether to allow cloned individuals in the population (default is 0).
    p_mutation : float, optional
        Global mutation probability (default is 0.7).
    p_mutate_individual : float, optional
        Probability of mutating an individual (default is 0.4).
    p_mutate_feature : float, optional
        Probability of mutating a feature selection (default is 0.4).
    p_mutate_coord : float, optional
        Probability of mutating the coordinate (SVM hyperparameters) (default is 0.2).
    coord_mutation_type : int, optional
        Coordinate mutation type: 0 (scaling) or 1 (replacement).

    Raises
    ------
    TypeError, ValueError
        If configuration is invalid. Predictors must be finite real numeric
        columns, the response must contain at least two samples of each class
        (-1 and 1), and probabilities must be finite real numbers in [0, 1].
        Index labels must be unique.
    """

    def __init__(self, method, data, population_size, inputs, output, num_selected_features,
                 allow_clones=0, p_mutation=0.7, p_mutate_individual=0.4,
                 p_mutate_feature=0.4, p_mutate_coord=0.2, coord_mutation_type=0, logger=None):
        self.method = method
        self.data = data
        self.inputs = inputs
        self.output = output
        self.population_size = population_size
        self.num_selected_features = num_selected_features
        self.allow_clones = allow_clones
        self.p_mutation = p_mutation
        self.p_mutate_individual = p_mutate_individual
        self.p_mutate_feature = p_mutate_feature
        self.p_mutate_coord = p_mutate_coord
        self.coord_mutation_type = coord_mutation_type
        self.custom_logger = logger

        self._validate_configuration()
        self.population = None
        self._population_context = None

    def _validate_configuration(self):
        """Reject invalid configuration before any population is generated."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if self.data.empty:
            raise ValueError("data must contain rows and columns.")
        if not self.data.columns.is_unique:
            raise ValueError("data must have unique column names.")
        if not self.data.index.is_unique:
            raise ValueError("data must have unique index labels.")
        if (not isinstance(self.inputs, (list, tuple)) or not self.inputs or
                not all(isinstance(name, str) and name for name in self.inputs) or
                len(set(self.inputs)) != len(self.inputs) or
                not all(name in self.data.columns for name in self.inputs)):
            raise ValueError("inputs must be a non-empty list or tuple of unique existing column names.")
        if (not isinstance(self.output, str) or not self.output or
                self.output not in self.data.columns):
            raise ValueError("output must name one existing column in data.")
        if self.output in self.inputs:
            raise ValueError("inputs must not include output.")
        if not isinstance(self.method, str) or self.method not in ('DIST-EPS', 'MC'):
            raise ValueError("method must be 'DIST-EPS' or 'MC'.")

        for name, minimum in (('population_size', 3), ('num_selected_features', 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
        if self.num_selected_features > len(self.inputs):
            raise ValueError("num_selected_features must not exceed the number of inputs.")
        for name in ('p_mutation', 'p_mutate_individual', 'p_mutate_feature', 'p_mutate_coord'):
            value = getattr(self, name)
            if not _is_finite_real(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        for name in ('allow_clones', 'coord_mutation_type'):
            value = getattr(self, name)
            if not isinstance(value, (Integral, np.bool_)) or value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1 (False or True).")
        if (self.custom_logger is not None and not callable(self.custom_logger) and
                not callable(getattr(self.custom_logger, 'info', None))):
            raise TypeError("logger must be None, a callable, or an object with a callable info method.")

        def real_numeric(series):
            return (is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype) and
                    not is_complex_dtype(series.dtype))

        if not all(real_numeric(self.data[name]) for name in self.inputs):
            raise ValueError("inputs must select real numeric columns with only finite values.")
        predictors = self.data[list(self.inputs)].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(predictors).all():
            raise ValueError("inputs must select real numeric columns with only finite values.")
        response = self.data[self.output]
        if (not real_numeric(response) or response.isna().any() or
                not response.isin([-1, 1]).all() or response.nunique() != 2):
            raise ValueError("output must contain both numeric classes -1 and 1, with no other values.")
        if (response.eq(-1).sum() < 2) or (response.eq(1).sum() < 2):
            raise ValueError("output must contain at least two observations in each class.")

    def run(self, train='iter', num_iter=None):
        """
        Execute the NSGA-II optimization process.

        Parameters
        ----------
        train : str, optional
            Training mode: 'sec' (run for a number of seconds) or 'iter' (run for a number of iterations).
        num_iter : int or float, optional
            Positive integer iterations for 'iter', or positive finite seconds
            for 'sec'. None selects 10 iterations or 300 seconds, respectively.

        Returns
        -------
        SVMFsMeta
            This optimizer. Repeated calls continue its population when data
            and structural configuration are unchanged.

        Raises
        ------
        TypeError, ValueError
            If the configuration or training budget is invalid. Validation
            happens before replacing the population or generating candidates.
        """
        self._validate_configuration()
        if not isinstance(train, str) or train not in ('iter', 'sec'):
            raise ValueError("train must be 'sec' or 'iter'.")
        if num_iter is None:
            num_iter = 10 if train == 'iter' else 300
        if train == 'iter':
            if (isinstance(num_iter, (bool, np.bool_)) or
                    not isinstance(num_iter, Integral) or num_iter < 1):
                raise ValueError("num_iter must be a positive integer when train='iter'.")
        elif not _is_finite_real(num_iter) or num_iter <= 0:
            raise ValueError("num_iter must be a positive finite number when train='sec'.")

        context = (self.method, self.population_size, self.num_selected_features,
                   tuple(self.inputs), self.output,
                   pd.util.hash_pandas_object(self.data[list(self.inputs) + [self.output]],
                                             index=True).values.tobytes())
        if (not isinstance(self.population, Population) or
                len(self.population.solutions) != self.population_size or
                (self._population_context is not None and context != self._population_context)):
            self.population = Population()
        self.population.generate_initial_population(
            self.population_size, self.data, self.inputs, self.output,
            self.num_selected_features, self.allow_clones, self.method
        )
        self.population.fnds(self.method)
        self._population_context = context
        self._log(f"Initial population: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")

        i = 1
        if train == 'sec':
            max_time = time() + num_iter
            while time() < max_time:
                self.nsga2()
                self._log(f"Iteration {i}: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")
                i += 1
        elif train == 'iter':
            while i <= num_iter:
                self.nsga2()
                self._log(f"Iteration {i}: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")
                i += 1
        return self

    def nsga2(self):
        """
        Perform one iteration of NSGA-II: generate offspring, evaluate, and select the next generation.
        """

        self.population.create_new_population(
            self.population_size, self.p_mutation, self.data, self.inputs, self.output,
            self.p_mutate_individual, self.p_mutate_feature, self.p_mutate_coord,
            self.coord_mutation_type, self.allow_clones, self.method
        )
        self.population.fnds(self.method)
        reduced = self.population.reduce_population(self.population_size, self.method)
        self.population = deepcopy(reduced)

    def draw_solution(self):
        """
        Visualize the Pareto front (front 0) of the current population.

        Raises
        ------
        ValueError
            If the optimization method is not recognized.
        """

        front0 = self.population.solutions_df[self.population.solutions_df.FRONT == 0]

        config = {
            'DIST-EPS': {
                'x': 'DIST', 'y': 'EPS', 'z': None,
                'xlabel': 'DIST', 'ylabel': 'EPS', 'title': 'Pareto Front'
            },
            'MC': {
                'x': 'MC_POS', 'y': 'MC_NEG', 'z': None,
                'xlabel': 'MC_POS', 'ylabel': 'MC_NEG', 'title': 'Pareto Front'
            }
        }

        cfg = config.get(self.method)
        if not cfg:
            raise ValueError(f"Unrecognized method: {self.method}")

        x, y = front0[cfg['x']], front0[cfg['y']]

        if cfg['z'] is None:
            scatter(x, y, color='blue', marker='o')
            xlabel(cfg['xlabel'])
            ylabel(cfg['ylabel'])
            title(cfg['title'])
            grid(True)
            show()
        else:
            z = front0[cfg['z']]
            fig = figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, color='blue', marker='o')
            ax.set_xlabel(cfg['xlabel'])
            ax.set_ylabel(cfg['ylabel'])
            ax.set_zlabel(cfg['zlabel'])
            ax.set_title(cfg['title'])
            show()

    def save_results(self, dataset_name, output_dir="results", iteration_val=None, iteration_type=None):
        """
        Save the Pareto front (front 0) to a CSV file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (used for directory naming).
        output_dir : str, optional
            Base directory to save the results (default is "results").
        iteration_val : int, optional
            Number of iterations or seconds.
        iteration_type : str, optional
            Training mode ('iter' or 'sec').
        """
        results = self.population.solutions_df
        pareto_front = results[results.FRONT == 0]

        directory = os.path.join(output_dir, dataset_name, self.method)
        os.makedirs(directory, exist_ok=True)

        base_filename = f"{self.population_size}"
        if iteration_type and iteration_val:
            base_filename = f"{base_filename}_{iteration_type}_{iteration_val}"

        output_file = os.path.join(directory, f"{base_filename}.csv")

        counter = 1
        filename, extension = os.path.splitext(output_file)
        while os.path.exists(output_file):
            output_file = f"{filename}_{counter}{extension}"
            counter += 1

        pareto_front.to_csv(output_file, index=False)
        self._log(f"Pareto front saved in: {output_file}")

    def _log(self, message):
        """Helper to log messages using standard logger and/or custom callback."""
        logger.info(message)
        if self.custom_logger:
            if callable(self.custom_logger):
                self.custom_logger(message)
            elif hasattr(self.custom_logger, 'info'):
                self.custom_logger.info(message)

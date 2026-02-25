from copy import deepcopy
from time import time
from matplotlib.pyplot import show, figure, xlabel, ylabel, title, grid, scatter
import os
import pandas as pd

from .population import Population
import logging

logger = logging.getLogger(__name__)

class NSGA2_SVMFS:
    """
    NSGA-II based algorithm for simultaneous Support Vector Machine training and Feature Selection (SVM + FS).

    Parameters
    ----------
    method : str
        The multi-objective optimization method ('DIST-EPS', 'MC', etc.).
    data : pandas.DataFrame
        The dataset to be used.
    costs : pandas.Series or ndarray
        The cost associated with each feature.
    population_size : int
        Number of individuals in the population.
    inputs : list
        List of input (feature) column names.
    output : str
        Name of the output (target) column.
    num_selected_features : int
        Maximum number of features to be selected.
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
        Type of mutation to apply on the coordinates (default is 0).
    """

    def __init__(self, method, data, costs, population_size, inputs, output, num_selected_features,
                 allow_clones=0, p_mutation=0.7, p_mutate_individual=0.4,
                 p_mutate_feature=0.4, p_mutate_coord=0.2, coord_mutation_type=0):
        self.method = method
        self.data = data
        self.costs = costs
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

        self.population = None
    def run(self, train='sec', num_iter=10):
        """
        Execute the NSGA-II optimization process.

        Parameters
        ----------
        train : str, optional
            Training mode: 'sec' (run for a number of seconds) or 'iter' (run for a number of iterations).
        num_iter : int, optional
            Number of iterations or seconds (depending on 'train') (default is 10).
        """

        self.population = Population()
        self.population.generate_initial_population(
            self.population_size, self.data, self.inputs, self.output,
            self.costs, self.num_selected_features, self.allow_clones, self.method
        )
        self.population.fnds(self.method)
        logger.info(f"Initial population: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")

        i = 1
        if train == 'sec':
            max_time = time() + num_iter
            while time() < max_time:
                self.nsga2()
                logger.info(f"Iteration {i}: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")
                i += 1
        elif train == 'iter':
            while i <= num_iter:
                self.nsga2()
                logger.info(f"Iteration {i}: {len(self.population.solutions_df[self.population.solutions_df.FRONT == 0])} solutions in front 0")
                i += 1
        else:
            raise ValueError("Invalid training method. Choose 'sec' or 'iter'.")

    def nsga2(self):
        """
        Perform one iteration of NSGA-II: generate offspring, evaluate, and select the next generation.
        """

        self.population.create_new_population(
            self.population_size, self.p_mutation, self.data, self.inputs, self.output, self.costs,
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
            },
            'DIST-EPS-COST': {
                'x': 'DIST', 'y': 'EPS', 'z': 'COST',
                'xlabel': 'DIST', 'ylabel': 'EPS', 'zlabel': 'COST', 'title': '3D Pareto Front'
            },
            'MC-COST': {
                'x': 'MC_POS', 'y': 'MC_NEG', 'z': 'COST',
                'xlabel': 'MC_POS', 'ylabel': 'MC_NEG', 'zlabel': 'COST', 'title': '3D Pareto Front'
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
        print(f"Pareto front saved in: {output_file}")

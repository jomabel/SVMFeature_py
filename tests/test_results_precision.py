from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestResultsPrecision(unittest.TestCase):
    expected = {
        'DIST': 1.23456789,
        'EPS': 0.004,
        'MC_POS': 1,
        'MC_NEG': 2,
        'ACCURACY': 0.987654321,
        'PRECISION': 0.976543219,
        'RECALL': 0.965432198,
        'F1': 0.954321987,
        'KAPPA': 0.943219876,
        'AUC': 0.9951,
    }

    def make_population(self, method):
        best = Solution(0)
        best.objective = [self.expected[key] for key in ('DIST', 'EPS', 'MC_POS', 'MC_NEG')]
        for attribute in ('accuracy', 'precision', 'recall', 'f1', 'kappa', 'auc'):
            setattr(best, attribute, self.expected[attribute.upper()])
        dominated = Solution(1)
        dominated.objective = [0.9, 0.4, 2, 3]
        population = Population()
        population.solutions = [best, dominated]
        population.fnds(method)
        # Reverse front order to exercise sorting as well as export.
        population.solutions.reverse()
        population.generate_solutions_df()
        return population

    def test_population_table_preserves_objective_and_metric_precision(self):
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method):
                population = self.make_population(method)
                self.assertEqual(population.solutions_df['SOL'].tolist(), [0, 1])
                row = population.solutions_df.iloc[0]
                for column, expected in self.expected.items():
                    self.assertEqual(row[column], expected, column)

    def test_csv_preserves_precision_and_exports_only_the_first_front(self):
        data = pd.DataFrame({'x': [-1., -.5, .5, 1.], 'y': [-1, -1, 1, 1]})
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method), TemporaryDirectory() as directory:
                optimizer = SVMFsMeta(method, data, 3, ['x'], 'y', 1)
                optimizer.population = self.make_population(method)
                optimizer.save_results('precision', output_dir=directory,
                                       iteration_val=1, iteration_type='iter')
                output = Path(directory) / 'precision' / method / '3_iter_1.csv'
                exported = pd.read_csv(output, float_precision='round_trip')
                self.assertEqual(exported['SOL'].tolist(), [0])
                self.assertEqual(exported['FRONT'].tolist(), [0])
                for column, expected in self.expected.items():
                    self.assertEqual(exported.iloc[0][column], expected, column)


if __name__ == '__main__':
    unittest.main()

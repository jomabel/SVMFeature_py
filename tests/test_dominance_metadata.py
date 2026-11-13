from ast import literal_eval
from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestDominanceMetadata(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)

    @staticmethod
    def chain(method):
        population = Population()
        for i in range(4):
            solution = Solution(i)
            solution.objective = [i + 1, 4 - i, 4 - i, 4 - i]
            population.solutions.append(solution)
        population.fnds(method)
        return population

    def assert_graph(self, population, method):
        ids = {s.solution_id for s in population.solutions}
        self.assertEqual(ids, set(range(len(population.solutions))))
        for solution in population.solutions:
            expected_dominates = {other.solution_id for other in population.solutions
                                  if other is not solution and solution.dominates_solution(other, method)}
            expected_dominated_by = {other.solution_id for other in population.solutions
                                     if other is not solution and other.dominates_solution(solution, method)}
            self.assertEqual(set(solution.dominates_list), expected_dominates)
            self.assertEqual(set(solution.dominated_by_list), expected_dominated_by)
            self.assertEqual(solution.num_dominated_by, len(expected_dominated_by))
            row = population.solutions_df.set_index('SOL').loc[solution.solution_id]
            self.assertEqual(row['NUM_DOMINATED_BY'], len(expected_dominated_by))
            self.assertEqual(set(row['DOMINATES']), expected_dominates)
            self.assertEqual(set(row['DOMINATED_BY']), expected_dominated_by)
            self.assertIn(solution.solution_id, population.fronts[solution.front].solutions)

    def test_sorting_preserves_actual_counts_across_repeated_calls(self):
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method):
                population = self.chain(method)
                for _ in range(2):
                    self.assertEqual([s.front for s in population.solutions], [3, 2, 1, 0])
                    self.assertEqual([s.num_dominated_by for s in population.solutions], [3, 2, 1, 0])
                    self.assert_graph(population, method)
                    population.fnds(method)

    def test_reduction_rebuilds_survivor_relations_without_renumbering_the_source(self):
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method):
                population = self.chain(method)
                before = deepcopy(population.solutions_df)
                reduced = population.reduce_population(2, method)
                self.assertEqual([s.solution_id for s in population.solutions], [0, 1, 2, 3])
                self.assertEqual([s.objective[0] for s in reduced.solutions], [4, 3])
                self.assertEqual(reduced.solutions[0].dominates_list, [1])
                self.assertEqual(reduced.solutions[1].dominated_by_list, [0])
                self.assertEqual(reduced.solutions[1].num_dominated_by, 1)
                self.assert_graph(reduced, method)
                self.assert_graph(population, method)
                pd.testing.assert_frame_equal(population.solutions_df, before)
                for survivor in reduced.solutions:
                    self.assertTrue(all(survivor is not parent for parent in population.solutions))

    def test_partial_front_reduction_retains_only_current_identifiers(self):
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method):
                population = Population()
                for i, (distance, epsilon, fn, fp) in enumerate(
                        ((1, 0, 0, 2), (2, 1, 1, 1), (3, 2, 2, 0), (2, 1, 1, 1))):
                    solution = Solution(i)
                    solution.objective = [distance, epsilon, fn, fp]
                    population.solutions.append(solution)
                population.fnds(method)
                reduced = population.reduce_population(2, method)
                self.assert_graph(reduced, method)
                self.assert_graph(population, method)
                self.assertEqual([s.front for s in reduced.solutions], [0, 0])

    def test_pareto_csv_uses_current_population_identifiers(self):
        data = pd.DataFrame({'x': [-1., -.5, .5, 1.], 'y': [-1, -1, 1, 1]})
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method), TemporaryDirectory() as directory:
                optimizer = SVMFsMeta(method, data, 3, ['x'], 'y', 1)
                optimizer.population = self.chain(method).reduce_population(3, method)
                optimizer.save_results('metadata', output_dir=directory)
                exported = pd.read_csv(Path(directory) / 'metadata' / method / '3.csv')
                self.assertEqual(exported['SOL'].tolist(), [0])
                self.assertEqual(literal_eval(exported.loc[0, 'DOMINATES']), [1, 2])
                self.assertEqual(literal_eval(exported.loc[0, 'DOMINATED_BY']), [])
                self.assertEqual(exported.loc[0, 'NUM_DOMINATED_BY'], 0)

    def test_optimizer_metadata_remains_consistent_after_multiple_generations(self):
        data = pd.DataFrame(np.random.default_rng(1003).normal(size=(20, 4)),
                            columns=['x1', 'x2', 'x3', 'x4'])
        data['y'] = [-1] * 10 + [1] * 10
        for method in ('DIST-EPS', 'MC'):
            with self.subTest(method=method):
                random.seed(0)
                optimizer = SVMFsMeta(method, data, 6, list(data.columns[:-1]), 'y', 2,
                                       allow_clones=1)
                optimizer.run(train='iter', num_iter=3)
                self.assert_graph(optimizer.population, method)


if __name__ == '__main__':
    unittest.main()

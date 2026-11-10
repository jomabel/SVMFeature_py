import math
import random
import unittest
from unittest.mock import patch

import pandas as pd

from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestCrossoverValidity(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(17)
        self.data = pd.DataFrame({
            'y': [-1, -1, 1, 1],
            'x1': [0.0, 0.5, 1.0, 0.0],
            'x2': [2.0, 0.0, 0.0, 1.0],
        })
        self.inputs = ['x1', 'x2']

    def make_parents(self, data=None):
        data = self.data if data is None else data
        father, mother = Solution(0), Solution(1)
        father.features, father.plane_coords, father.vectors = ['x1'], [1.0], [0, 2]
        mother.features, mother.plane_coords, mother.vectors = ['x2'], [1.0], [1, 3]
        for parent in (father, mother):
            self.assertEqual(parent.evaluate_solution(data, 'y', self.inputs), 1)
        return father, mother

    @staticmethod
    def key(child):
        return tuple(child.features), tuple(child.vectors)

    def test_two_valid_children_survive_even_when_invalid_children_have_zero_mc(self):
        # Children 1 and 4 have coincident margin planes. Children 2 and 3
        # are valid, with MC objectives (0, 1) and (1, 0), respectively.
        for method in ('MC', 'DIST-EPS'):
            with self.subTest(method=method):
                father, mother = self.make_parents()
                selected = Population().crossover_solutions(
                    father, mother, 2, method, self.data, self.inputs, 'y'
                )
                self.assertEqual(
                    {self.key(child) for child in selected},
                    {(('x2',), (0, 3)), (('x1',), (1, 2))}
                )
                self.assertEqual([child.solution_id for child in selected], [2, 3])
                self.assertEqual(
                    {tuple(child.objective[2:]) for child in selected}, {(0, 1), (1, 0)}
                )
                for child in selected:
                    self.assertEqual(child.evaluate_solution(self.data, 'y', self.inputs), 1)

    def test_failed_evaluation_clears_previous_objectives_and_metrics(self):
        for reason in ('zero_normal', 'coincident_planes'):
            with self.subTest(reason=reason):
                child, _ = self.make_parents()
                self.assertIsNotNone(child.conf_matrix)
                if reason == 'zero_normal':
                    child.plane_coords = [0.0]
                else:
                    child.vectors = [0, 3]
                self.assertEqual(child.evaluate_solution(self.data, 'y', self.inputs), 0)
                self.assertTrue(all(math.isnan(value) for value in child.objective))
                for attribute in ('conf_matrix', 'accuracy', 'precision', 'recall', 'f1', 'kappa', 'auc'):
                    self.assertIsNone(getattr(child, attribute), attribute)

    def test_valid_child_is_preferred_in_either_comparison_order(self):
        valid, invalid = self.make_parents()
        invalid.plane_coords = [0.0]
        self.assertEqual(invalid.evaluate_solution(self.data, 'y', self.inputs), 0)
        for method in ('MC', 'DIST-EPS'):
            for first, second in ((valid, invalid), (invalid, valid)):
                with self.subTest(method=method, first=first.solution_id):
                    with patch.object(Solution, 'compare_dominance', side_effect=AssertionError(
                        'Invalid objectives must not enter dominance comparisons'
                    )):
                        winner, loser = Population.select_best_offspring(first, second, method)
                    self.assertIs(winner, valid)
                    self.assertIs(loser, invalid)

    def test_zero_classification_errors_are_still_a_valid_result(self):
        data = pd.DataFrame({'y': [-1, -1, 1, 1], 'x1': [-2., -1., 1., 2.]})
        valid = Solution(0)
        valid.features, valid.plane_coords, valid.vectors = ['x1'], [1.0], [0, 3]
        self.assertEqual(valid.evaluate_solution(data, 'y', ['x1']), 1)
        self.assertEqual(valid.objective[2:], [0, 0])
        invalid = Solution(1)
        invalid.features, invalid.plane_coords, invalid.vectors = ['x1'], [0.0], [0, 3]
        self.assertEqual(invalid.evaluate_solution(data, 'y', ['x1']), 0)
        winner, _ = Population.select_best_offspring(invalid, valid, 'MC')
        self.assertIs(winner, valid)

    def test_one_valid_child_is_kept_and_second_slot_can_be_repaired(self):
        data = self.data.copy()
        data['x2'] = [2.0, 0.0, 0.0, 2.0]
        for method in ('MC', 'DIST-EPS'):
            with self.subTest(method=method):
                father, mother = self.make_parents(data)
                selected = Population().crossover_solutions(
                    father, mother, 2, method, data, self.inputs, 'y'
                )
                self.assertEqual(len(selected), 2)
                self.assertEqual(self.key(selected[0]), (('x1',), (1, 2)))
                self.assertTrue(all(math.isfinite(value) for value in selected[0].objective))
                self.assertTrue(all(math.isnan(value) for value in selected[1].objective))

    def test_all_invalid_children_tie_without_comparing_objectives(self):
        data = pd.DataFrame({
            'y': [-1, -1, 1, 1], 'x1': [0., 1., 1., 0.], 'x2': [0., 1., 1., 0.]
        })
        for method in ('MC', 'DIST-EPS'):
            with self.subTest(method=method):
                father, mother = self.make_parents(data)
                with patch.object(Solution, 'compare_dominance', side_effect=AssertionError(
                    'Two invalid candidates must tie without comparing objectives'
                )):
                    selected = Population().crossover_solutions(
                        father, mother, 2, method, data, self.inputs, 'y'
                    )
                self.assertEqual(len(selected), 2)
                self.assertIsNot(selected[0], selected[1])
                self.assertEqual([child.solution_id for child in selected], [2, 3])
                for child in selected:
                    self.assertTrue(all(math.isnan(value) for value in child.objective))

    def test_unknown_method_is_rejected_even_when_both_children_are_invalid(self):
        first, second = self.make_parents()
        for child in (first, second):
            child.plane_coords = [0.0]
            self.assertEqual(child.evaluate_solution(self.data, 'y', self.inputs), 0)
        with self.assertRaises(ValueError):
            Population.select_best_offspring(first, second, 'unknown')


if __name__ == '__main__':
    unittest.main()
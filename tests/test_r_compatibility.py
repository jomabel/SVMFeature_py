from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population
from svmfsmeta.preprocessing import minmax_scale
from svmfsmeta.solution import Solution


class TestRCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / 'fixtures' / 'r_reference.json'
        cls.reference = json.loads(path.read_text(encoding='utf-8'))

    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)
        self.data = pd.DataFrame({'x1': [-2., -1., 1., 2.],
                                  'x2': [2., 0., 1., 3.], 'y': [-1, -1, 1, 1]})

    def optimizer(self, **changes):
        kwargs = dict(method='MC', data=self.data, population_size=3,
                      inputs=['x1', 'x2'], output='y', num_selected_features=1)
        kwargs.update(changes)
        return SVMFsMeta(**kwargs)

    def test_54_paired_candidate_evaluations_match_r_reference(self):
        for case in self.reference['cases']:
            with self.subTest(case=case['id']):
                data = pd.DataFrame(self.reference['datasets'][case['dataset']])
                candidate = Solution(0)
                candidate.features = case['features'].copy()
                candidate.vectors = case['vectors'].copy()
                candidate.plane_coords = case['coords'].copy()
                expected = case['expected']
                with np.errstate(over='ignore', invalid='ignore'):
                    valid = candidate.evaluate_solution(data, 'y', list(data.columns.drop('y')))
                self.assertEqual(bool(valid), expected['valid'])
                if valid:
                    np.testing.assert_allclose(candidate.objective, expected['objectives'], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(candidate.plane_term_b, expected['intercepts'], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(candidate.conf_matrix, expected['conf_matrix'])
                    for name, value in expected['metrics'].items():
                        np.testing.assert_allclose(getattr(candidate, name), value, rtol=1e-12, atol=1e-12)
                else:
                    self.assertTrue(all(np.isnan(value) for value in candidate.objective))
                    self.assertIsNone(candidate.conf_matrix)
                    self.assertIsNone(candidate.accuracy)

    def test_infinite_decision_scores_keep_r_auc_ranking(self):
        data = pd.DataFrame({'x': [-1e308, -9.5e307, -8e307, 1e308], 'y': [-1, -1, 1, 1]})
        candidate = Solution(0)
        candidate.features, candidate.vectors, candidate.plane_coords = ['x'], [0, 2], [1.]
        with np.errstate(over='ignore', invalid='ignore'):
            self.assertEqual(candidate.evaluate_solution(data, 'y', ['x']), 1)
        np.testing.assert_allclose(candidate.objective, [2e307, 5e306, 0, 0], rtol=1e-12)
        self.assertEqual(candidate.accuracy, 1)
        self.assertEqual(candidate.auc, 1)

    def test_same_mutation_decisions_produce_r_normal_and_classification(self):
        data = pd.DataFrame({'x1': [0., 0., 1., 0.], 'x2': [2., .5, 0., 1.], 'y': [-1, -1, 1, 1]})
        candidate = Solution(0)
        candidate.features, candidate.vectors = ['x1', 'x2'], [0, 2]
        candidate.plane_coords = [-.95, .2]
        with patch('svmfsmeta.solution.uniform', side_effect=[.9, .1, .2, .25, .9]) as uniform:
            candidate.mutate_solution(0, 0, .5, 0, data, ['x1', 'x2'], 'y')
        uniform.assert_any_call(0, .25)
        expected = self.reference['controlled_mutation']
        np.testing.assert_allclose(candidate.plane_coords, expected['coords'], rtol=1e-14)
        self.assertEqual(candidate.evaluate_solution(data, 'y', ['x1', 'x2']), 1)
        np.testing.assert_allclose(candidate.objective, expected['objectives'], rtol=1e-12)
        np.testing.assert_array_equal(candidate.conf_matrix, expected['conf_matrix'])

    def test_proportional_mutation_checks_both_signs_and_directions(self):
        for value, percentage, direction, expected in ((.95, .2, .25, .114),
                (-.95, .2, .25, -.114), (1.2, .1, .75, .108),
                (-1.2, .1, .75, -.108), (.8, 0., .25, .8)):
            with self.subTest(value=value, percentage=percentage, direction=direction):
                candidate = Solution(0)
                candidate.features, candidate.plane_coords, candidate.vectors = ['x1'], [value], [0, 2]
                with patch('svmfsmeta.solution.uniform', side_effect=[.9, .1, percentage, direction]):
                    candidate.mutate_solution(0, 0, .5, 0, self.data, ['x1'], 'y')
                self.assertAlmostEqual(candidate.plane_coords[0], expected)

    def test_selecting_all_predictors_preserves_r_input_order(self):
        candidate = Solution(0)
        with patch.object(Solution, 'get_class_vector', side_effect=[0, 2]), patch(
                'svmfsmeta.solution.sample', side_effect=AssertionError('No feature sampling is needed')):
            candidate.generate_random_solution(self.data, ['x2', 'x1'], 'y', 2)
        self.assertEqual(candidate.features, ['x2', 'x1'])

    def test_forbidden_clones_are_rejected_without_extra_mutation(self):
        population = Population()
        for i in range(3):
            candidate = Solution(i)
            candidate.features, candidate.plane_coords, candidate.vectors = ['x1'], [.5], [0, 2]
            candidate.evaluate_solution(self.data, 'y', ['x1'])
            population.solutions.append(candidate)
        def crossover(*args, **kwargs):
            children = []
            for i in range(2):
                candidate = Solution(i)
                candidate.features, candidate.plane_coords, candidate.vectors = ['x1'], [.5], [0, 2]
                children.append(candidate)
            return children
        with patch.object(population, 'select_two_distinct_parents', return_value=tuple(population.solutions[:2])), patch.object(
                population, 'crossover_solutions', side_effect=crossover), patch.object(
                Solution, 'generate_random_solution', side_effect=AssertionError('Unexpected regeneration')), patch.object(
                Solution, 'mutate_solution', side_effect=AssertionError('Unexpected repair mutation')):
            with self.assertRaisesRegex(RuntimeError, '100.*consecutive'):
                population.create_new_population(3, 0, self.data, ['x1'], 'y', 0, 0, 0, 0, 0, 'MC')
        self.assertEqual(len(population.solutions), 3)

    def test_public_minima_match_r(self):
        with self.assertRaisesRegex(ValueError, 'population_size'):
            self.optimizer(population_size=2)
        for y in ([-1, 1, 1, 1], [-1, -1, -1, 1]):
            data = self.data.copy()
            data['y'] = y
            with self.subTest(y=y), self.assertRaisesRegex(ValueError, 'at least two'):
                self.optimizer(data=data)
        self.optimizer(population_size=3)

    def test_default_run_uses_ten_iterations_and_returns_optimizer(self):
        optimizer = self.optimizer()
        with patch.object(optimizer, 'nsga2') as step:
            returned = optimizer.run()
        self.assertIs(returned, optimizer)
        self.assertEqual(step.call_count, 10)

    def test_default_time_budget_is_three_hundred_seconds(self):
        optimizer = self.optimizer()
        with patch.object(optimizer, 'nsga2') as step, patch(
                'svmfsmeta.svm_fs_meta.time', side_effect=[100., 399.9, 400.1]):
            optimizer.run(train='sec')
        self.assertEqual(step.call_count, 1)

    def test_repeated_run_continues_population_without_new_random_candidates(self):
        optimizer = self.optimizer()
        with patch.object(optimizer, 'nsga2'):
            optimizer.run(train='iter', num_iter=1)
            previous = optimizer.population
            state = random.getstate()
            with patch.object(Solution, 'generate_random_solution', side_effect=AssertionError('Must reuse the population')):
                self.assertIs(optimizer.run(train='iter', num_iter=1), optimizer)
        self.assertIs(optimizer.population, previous)
        self.assertEqual(random.getstate(), state)

    def test_data_or_structural_changes_start_a_new_population(self):
        for change in ('data', 'features', 'population_size', 'method'):
            with self.subTest(change=change):
                optimizer = self.optimizer()
                with patch.object(optimizer, 'nsga2'):
                    optimizer.run(train='iter', num_iter=1)
                    previous = optimizer.population
                    if change == 'data':
                        optimizer.data = optimizer.data.copy()
                        optimizer.data.loc[0, 'x1'] += .25
                    elif change == 'features':
                        optimizer.num_selected_features = 2
                    elif change == 'population_size':
                        optimizer.population_size = 4
                    else:
                        optimizer.method = 'DIST-EPS'
                    optimizer.run(train='iter', num_iter=1)
                self.assertIsNot(optimizer.population, previous)
                self.assertEqual(len(optimizer.population.solutions), optimizer.population_size)

    def test_explicit_population_reset_starts_a_new_run(self):
        optimizer = self.optimizer()
        with patch.object(optimizer, 'nsga2'):
            optimizer.run(train='iter', num_iter=1)
            previous = optimizer.population
            optimizer.population = None
            optimizer.run(train='iter', num_iter=1)
        self.assertIsNot(optimizer.population, previous)

    def test_undefined_precision_matches_r_auxiliary_metric(self):
        candidate = Solution(0)
        candidate.features, candidate.plane_coords = ['x1'], [1.]
        candidate.plane_term_b = [-10., -12., -11.]
        data = pd.DataFrame({'x1': [-4., -3., -2., -1.], 'y': [-1, -1, 1, 1]})
        candidate.calculate_epsilon_objective(data, 'y')
        self.assertTrue(np.isnan(candidate.precision))
        self.assertEqual(candidate.recall, 0)
        self.assertEqual(candidate.f1, 0)
        self.assertEqual(candidate.auc, 1)

    def test_normalizer_matches_r_constants_and_preserves_input(self):
        data = pd.DataFrame({'y': [-1, -1, 1, 1], 'x1': [10, 20, 30, 40], 'x2': [7, 7, 7, 7],
                             'unused': ['a', 'b', 'c', 'd']}, index=['d', 'c', 'b', 'a'])
        before = data.copy(deep=True)
        normalized = minmax_scale(data, ['x2', 'x1'])
        np.testing.assert_allclose(normalized['x1'], [0, 1/3, 2/3, 1])
        np.testing.assert_array_equal(normalized['x2'], [.5, .5, .5, .5])
        pd.testing.assert_series_equal(normalized['y'], data['y'])
        pd.testing.assert_series_equal(normalized['unused'], data['unused'])
        pd.testing.assert_frame_equal(data, before)
        tiny = data.copy()
        tiny['x1'] = np.arange(4) * 1e-16
        np.testing.assert_allclose(minmax_scale(tiny, ['x1'])['x1'], [0, 1/3, 2/3, 1])

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'example_usage.py').is_file(),
                         'Repository examples are not part of the core distribution')
    def test_example_uses_canonical_colon_loader(self):
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo))
        self.addCleanup(sys.path.remove, str(repo))
        data = pd.DataFrame(np.random.default_rng(7).normal(size=(20, 6)), columns=[f'x{i}' for i in range(1, 7)])
        data['x6'] = 7
        data.insert(0, 'y', [-1] * 10 + [1] * 10)
        with patch('main.load_colon', return_value=data) as load, patch.object(
                SVMFsMeta, 'save_results') as save, patch.object(SVMFsMeta, 'draw_solution'), redirect_stdout(StringIO()):
            result = runpy.run_path(str(repo / 'example_usage.py'), run_name='__main__')
        load.assert_called_once_with()
        save.assert_called_once_with('colon', iteration_val=5, iteration_type='iter')
        np.testing.assert_array_equal(result['optimizer'].data['x6'], [.5] * len(data))


if __name__ == '__main__':
    unittest.main()

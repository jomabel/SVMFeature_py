import logging
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population


class TestOptimizerValidation(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)
        self.data = pd.DataFrame({
            'x1': np.linspace(-2, 2, 20),
            'x2': np.random.default_rng(7).normal(size=20),
            'y': [-1] * 10 + [1] * 10,
        })

    def optimizer(self, **changes):
        kwargs = dict(method='MC', data=self.data, population_size=6,
                      inputs=['x1', 'x2'], output='y', num_selected_features=1)
        kwargs.update(changes)
        return SVMFsMeta(**kwargs)

    def test_probabilities_reject_out_of_range_nonfinite_and_nonnumeric_values(self):
        for name in ('p_mutation', 'p_mutate_individual', 'p_mutate_feature', 'p_mutate_coord'):
            for value in (-.01, 1.01, np.nan, np.inf, -np.inf, True, '0.5', None, [], 1j):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.optimizer(**{name: value})

    def test_probability_boundaries_and_numpy_scalars_are_accepted(self):
        for value in (0, 1, 0.0, np.float64(1.0)):
            with self.subTest(value=value):
                optimizer = self.optimizer(**{name: value for name in (
                    'p_mutation', 'p_mutate_individual', 'p_mutate_feature', 'p_mutate_coord')})
                self.assertEqual(optimizer.p_mutation, value)
        optimizer = self.optimizer(population_size=np.int64(6), num_selected_features=np.int64(1),
                                   inputs=('x1', 'x2'), allow_clones=np.bool_(True),
                                   coord_mutation_type=np.int64(1))
        self.assertEqual(optimizer.population_size, 6)

    def test_counts_require_integers_and_features_fit_available_inputs(self):
        for name, values in (
            ('population_size', (0, 1, -2, 2.5, 6.0, True, np.inf, '6', None)),
            ('num_selected_features', (0, -1, 3, 1.5, True, np.nan, '1', None)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.optimizer(**{name: value})

    def test_method_and_flags_reject_unknown_choices(self):
        for value in ('mc', '', 'other', None, [], 0):
            with self.subTest(method=value), self.assertRaisesRegex(ValueError, 'method'):
                self.optimizer(method=value)
        for name in ('allow_clones', 'coord_mutation_type'):
            for value in (-1, 2, .5, '0', None, []):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.optimizer(**{name: value})
        for value in (False, True, 0, 1):
            self.optimizer(allow_clones=value, coord_mutation_type=value)

    def test_predictor_and_response_columns_are_validated(self):
        for inputs in ([], ['missing'], ['x1', 'x1'], ['y'], 'x1', None, [1], ['']):
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ValueError, 'inputs'):
                self.optimizer(inputs=inputs)
        for output in ('missing', '', None, [], 0):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'output'):
                self.optimizer(output=output)

    def test_dataframe_and_unique_labels_are_required(self):
        for data in (None, [], np.zeros((4, 3))):
            with self.subTest(data=type(data)), self.assertRaisesRegex(TypeError, 'data'):
                self.optimizer(data=data)
        duplicate_columns = self.data.copy()
        duplicate_columns.columns = ['x1', 'x1', 'y']
        duplicate_index = self.data.copy()
        duplicate_index.index = [0] * len(duplicate_index)
        for data in (pd.DataFrame(), self.data.iloc[:0], duplicate_columns, duplicate_index):
            with self.subTest(shape=data.shape), self.assertRaisesRegex(ValueError, 'data'):
                self.optimizer(data=data)

    def test_predictors_must_be_real_numeric_and_finite(self):
        for value in (np.nan, np.inf, -np.inf, 'invalid', complex(1, 2), None):
            data = self.data.copy()
            data['x1'] = data['x1'].astype(object) if isinstance(value, (str, complex)) else data['x1']
            data.loc[0, 'x1'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'inputs'):
                self.optimizer(data=data)
        for dtype in (bool, complex):
            data = self.data.copy()
            data['x1'] = data['x1'].astype(dtype)
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, 'inputs'):
                self.optimizer(data=data)

    def test_response_requires_both_numeric_classes_and_no_missing_values(self):
        for values in ([0] * 10 + [1] * 10, [-1] * 20, [1] * 20,
                       ['-1'] * 10 + ['1'] * 10, [False] * 10 + [True] * 10,
                       [-1] * 10 + [1] * 9 + [np.nan], [-1] * 10 + [1] * 9 + [np.inf],
                       [-1] * 10 + [1] * 9 + [2]):
            data = self.data.copy()
            data['y'] = values
            with self.subTest(values=values[-2:]), self.assertRaisesRegex(ValueError, 'output'):
                self.optimizer(data=data)

    def test_nullable_numeric_columns_and_unused_nonnumeric_columns_are_accepted(self):
        data = self.data.astype({'x1': 'Float64', 'y': 'Int64'})
        data['unused'] = [None] * len(data)
        data.index = pd.MultiIndex.from_arrays([['sample'] * len(data), range(len(data))])
        before = data.copy(deep=True)
        self.optimizer(data=data)
        pd.testing.assert_frame_equal(data, before)

    def test_logger_accepts_callbacks_and_logger_objects(self):
        for value in (None, lambda message: None, logging.getLogger('svmfsmeta-test')):
            self.optimizer(logger=value)
        for value in (0, False, [], SimpleNamespace(info=1)):
            with self.subTest(logger=value), self.assertRaisesRegex(TypeError, 'logger'):
                self.optimizer(logger=value)

    def test_invalid_training_budget_fails_before_population_or_rng_changes(self):
        optimizer = self.optimizer()
        previous = object()
        optimizer.population = previous
        state = random.getstate()
        cases = [('iter', value) for value in (-1, 0, 1.5, 1.0, True, np.nan, np.inf, None, '1')]
        cases += [('sec', value) for value in (-1, 0, True, np.nan, np.inf, None, '1', 2 ** 2000)]
        cases += [(value, 1) for value in ('invalid', None, [], 1)]
        with patch.object(Population, 'generate_initial_population') as generate:
            for train, count in cases:
                with self.subTest(train=train, count=count), self.assertRaisesRegex(ValueError, 'train|num_iter'):
                    optimizer.run(train=train, num_iter=count)
                self.assertIs(optimizer.population, previous)
                self.assertEqual(random.getstate(), state)
            generate.assert_not_called()

    def test_configuration_changes_are_revalidated_before_training(self):
        optimizer = self.optimizer()
        optimizer.p_mutation = 2
        with patch.object(Population, 'generate_initial_population') as generate:
            with self.assertRaisesRegex(ValueError, 'p_mutation'):
                optimizer.run(train='iter', num_iter=1)
            generate.assert_not_called()
        optimizer.p_mutation = .7
        optimizer.data.loc[0, 'y'] = 0
        with self.assertRaisesRegex(ValueError, 'output'):
            optimizer.run(train='iter', num_iter=1)

    def test_fractional_seconds_are_accepted(self):
        optimizer = self.optimizer()
        with patch('svmfsmeta.svm_fs_meta.time', side_effect=[100., 101.]), patch.object(optimizer, 'nsga2') as step:
            optimizer.run(train='sec', num_iter=np.float64(.125))
        step.assert_not_called()
        self.assertEqual(len(optimizer.population.solutions), 6)


if __name__ == '__main__':
    unittest.main()
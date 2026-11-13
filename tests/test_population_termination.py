from itertools import cycle
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestPopulationTermination(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)
        self.data = pd.DataFrame({"x": [0., 1., 3., 4.], "y": [-1, -1, 1, 1]})

    @staticmethod
    def candidate(solution_id):
        candidate = Solution(solution_id)
        candidate.features = ["x"]
        candidate.plane_coords = [1.]
        candidate.vectors = [0, 2]
        candidate.objective = [3., 1., 0, 0]
        return candidate

    def population(self, size=3):
        population = Population()
        population.solutions = [self.candidate(i) for i in range(size)]
        return population

    def expand(self, population, allow_clones=False, method="DIST-EPS"):
        size = len(population.solutions)
        population.create_new_population(
            size, 0., self.data, ["x"], "y", 0., 0., 0., 0,
            allow_clones, method
        )

    def fixed_parents(self, population):
        return patch.object(
            population, "tournament_select_parent",
            side_effect=cycle(population.solutions[:2])
        )

    def guarded_crossover(self, population, rejection=None, max_batches=60):
        proposals = 0
        batches = 0

        def crossover(father, mother, solution_id, method, data, inputs, output):
            nonlocal proposals, batches
            batches += 1
            if batches > max_batches:
                raise AssertionError("Offspring generation did not terminate.")
            children = []
            for offset in range(2):
                proposals += 1
                child = self.candidate(solution_id + offset)
                child.reject_for_test = bool(rejection and rejection(proposals))
                children.append(child)
            return children

        return patch.object(population, "crossover_solutions", side_effect=crossover)

    def test_constant_predictors_stop_initialization_with_an_informative_error(self):
        population = Population()
        data = pd.DataFrame({"x": [0., 0., 0., 0.], "y": [-1, -1, 1, 1]})
        evaluate = Solution.evaluate_solution
        attempts = 0

        def guarded_evaluation(candidate, data, output, inputs):
            nonlocal attempts
            attempts += 1
            if attempts > 101:
                raise AssertionError("Initialization did not stop after invalid candidates.")
            return evaluate(candidate, data, output, inputs)

        with patch.object(Solution, "evaluate_solution", autospec=True,
                          side_effect=guarded_evaluation):
            with self.assertRaisesRegex(RuntimeError, "100.*consecutive"):
                population.generate_initial_population(3, data, ["x"], "y", 1, False, "DIST-EPS")
        self.assertEqual(attempts, 100)
        self.assertEqual(population.solutions, [])

    def test_forbidden_initial_clones_have_a_rejection_limit(self):
        population = Population()
        evaluate = Solution.evaluate_solution
        attempts = 0

        def fixed_candidate(candidate, data, inputs, output, num_features):
            candidate.features = ["x"]
            candidate.plane_coords = [1.]
            candidate.vectors = [0, 2]

        def guarded_evaluation(candidate, data, output, inputs):
            nonlocal attempts
            attempts += 1
            if attempts > 102:
                raise AssertionError("Initialization did not stop after duplicate candidates.")
            return evaluate(candidate, data, output, inputs)

        with patch.object(Solution, "generate_random_solution", autospec=True,
                          side_effect=fixed_candidate), patch.object(
                              Solution, "evaluate_solution", autospec=True,
                              side_effect=guarded_evaluation):
            with self.assertRaisesRegex(RuntimeError, "100.*consecutive"):
                population.generate_initial_population(3, self.data, ["x"], "y", 1, False, "DIST-EPS")
        self.assertEqual(len(population.solutions), 1)
        self.assertEqual(attempts, 101)

    def test_acceptance_resets_the_initialization_rejection_counter(self):
        population = Population()
        evaluate = Solution.evaluate_solution
        attempts = 0

        def intermittent_evaluation(candidate, data, output, inputs):
            nonlocal attempts
            attempts += 1
            if attempts > 201:
                raise AssertionError("Initialization did not retain accepted candidates.")
            return evaluate(candidate, data, output, inputs) if attempts % 100 == 0 else 0

        with patch.object(Solution, "evaluate_solution", autospec=True,
                          side_effect=intermittent_evaluation):
            population.generate_initial_population(2, self.data, ["x"], "y", 1, True, "DIST-EPS")
        self.assertEqual(attempts, 200)
        self.assertEqual([s.solution_id for s in population.solutions], [0, 1])

    def test_allowed_clones_are_added_and_odd_populations_stop_at_twice_the_size(self):
        for method in ("DIST-EPS", "MC"):
            with self.subTest(method=method):
                population = self.population()
                with self.fixed_parents(population), self.guarded_crossover(population), patch.object(
                    population, "check_clones", return_value=True
                ), patch.object(Solution, "evaluate_solution", return_value=1):
                    self.expand(population, allow_clones=True, method=method)
                self.assertEqual(len(population.solutions), 6)
                self.assertEqual([s.solution_id for s in population.solutions], list(range(6)))

    def test_persistent_forbidden_clones_stop_offspring_generation(self):
        population = self.population()
        with self.fixed_parents(population), self.guarded_crossover(population), patch.object(
            population, "check_clones", return_value=True
        ), patch.object(Solution, "evaluate_solution", return_value=1), patch.object(
            Solution, "mutate_solution", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "100.*consecutive"):
                self.expand(population)
        self.assertEqual(len(population.solutions), 3)

    def test_persistent_invalid_children_stop_without_regeneration(self):
        population = self.population()
        regenerations = 0

        def regenerate(candidate, data, inputs, output, num_features):
            nonlocal regenerations
            regenerations += 1
            if regenerations > 1001:
                raise AssertionError("Invalid offspring regeneration did not terminate.")
            self.assertIs(data, self.data)
            self.assertEqual(inputs, ["x"])
            self.assertEqual(output, "y")
            self.assertEqual(num_features, 1)

        with self.fixed_parents(population), self.guarded_crossover(population), patch.object(
            Solution, "evaluate_solution", return_value=0
        ), patch.object(Solution, "generate_random_solution", autospec=True,
                        side_effect=regenerate):
            with self.assertRaisesRegex(RuntimeError, "100.*consecutive"):
                self.expand(population)
        self.assertEqual(len(population.solutions), 3)
        self.assertEqual(regenerations, 0)

    def test_acceptance_resets_the_offspring_rejection_counter(self):
        population = self.population()
        with self.fixed_parents(population), self.guarded_crossover(
            population, rejection=lambda proposal: proposal % 100 != 0, max_batches=151
        ), patch.object(population, "check_clones", side_effect=lambda child, method: child.reject_for_test), patch.object(
            Solution, "evaluate_solution", return_value=1
        ), patch.object(Solution, "mutate_solution", return_value=None):
            self.expand(population)
        self.assertEqual(len(population.solutions), 6)
        self.assertEqual([s.solution_id for s in population.solutions], list(range(6)))

    def test_rejected_children_do_not_leave_gaps_in_solution_ids(self):
        population = self.population()
        with self.fixed_parents(population), self.guarded_crossover(
            population, rejection=lambda proposal: proposal % 2 == 1
        ), patch.object(population, "check_clones", side_effect=lambda child, method: child.reject_for_test), patch.object(
            Solution, "evaluate_solution", return_value=1
        ), patch.object(Solution, "mutate_solution", return_value=None):
            self.expand(population)
        self.assertEqual([s.solution_id for s in population.solutions], list(range(6)))
        population.fnds("DIST-EPS")
        self.assertEqual([s.front for s in population.solutions], [0] * 6)

    def test_unselectable_second_parent_stops_with_an_informative_error(self):
        population = self.population()
        calls = 0

        def repeated_winner(population_size):
            nonlocal calls
            calls += 1
            if calls > 102:
                raise AssertionError("Parent selection did not terminate.")
            return population.solutions[0]

        with patch.object(population, "tournament_select_parent", side_effect=repeated_winner):
            with self.assertRaisesRegex(RuntimeError, "distinct parents"):
                population.select_two_distinct_parents(3)
        self.assertLessEqual(calls, 101)

    def test_tournament_rejects_a_population_with_only_one_candidate(self):
        population = self.population(1)
        calls = 0

        def repeated_index(low, high):
            nonlocal calls
            calls += 1
            if calls > 102:
                raise AssertionError("A one-candidate tournament did not terminate.")
            return 0

        with patch("svmfsmeta.population.randint", side_effect=repeated_index, create=True):
            with self.assertRaises(ValueError):
                population.tournament_select_parent(1)

    def test_public_optimizer_can_complete_a_generation_with_clones_allowed(self):
        data = pd.DataFrame(np.random.default_rng(1003).normal(size=(20, 4)),
                            columns=["x1", "x2", "x3", "x4"])
        data["y"] = [-1] * 10 + [1] * 10
        crossover = Population.crossover_solutions
        for method in ("DIST-EPS", "MC"):
            with self.subTest(method=method):
                random.seed(0)
                calls = 0
                tournaments = 0

                def fixed_parent(population, population_size):
                    nonlocal tournaments
                    parent = population.solutions[tournaments % 2]
                    tournaments += 1
                    return parent

                def bounded_crossover(population, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls > 31:
                        raise AssertionError("Clones were not added during optimization.")
                    return crossover(population, *args, **kwargs)

                optimizer = SVMFsMeta(method, data, 6, list(data.columns[:-1]), "y", 2,
                                       allow_clones=1)
                # Fix only parent selection to isolate acceptance of allowed clones.
                with patch.object(Population, "tournament_select_parent", autospec=True,
                                  side_effect=fixed_parent), patch.object(
                                      Population, "crossover_solutions", autospec=True,
                                      side_effect=bounded_crossover):
                    optimizer.run(train="iter", num_iter=1)
                self.assertEqual(len(optimizer.population.solutions), 6)
                self.assertIsNotNone(optimizer.population.solutions_df)


if __name__ == "__main__":
    unittest.main()

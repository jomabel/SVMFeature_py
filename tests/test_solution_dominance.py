import unittest

from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestSolutionDominance(unittest.TestCase):
    def make_solution(self, solution_id, objectives):
        solution = Solution(solution_id)
        solution.objective = objectives
        return solution

    def test_equal_objectives_do_not_dominate_each_other(self):
        cases = {
            "DIST-EPS": [3, 2, 0, 0],
            "MC": [0, 0, 2, 3],
        }

        for method, objectives in cases.items():
            with self.subTest(method=method):
                first = self.make_solution(0, objectives.copy())
                second = self.make_solution(1, objectives.copy())

                self.assertFalse(first.dominates_solution(second, method))
                self.assertFalse(second.dominates_solution(first, method))
                self.assertEqual(first.compare_dominance(second, method), 0)

    def test_tradeoff_solutions_are_incomparable(self):
        cases = {
            "DIST-EPS": ([3, 4, 0, 0], [2, 2, 0, 0]),
            "MC": ([0, 0, 2, 4], [0, 0, 3, 2]),
        }

        for method, (first_objectives, second_objectives) in cases.items():
            with self.subTest(method=method):
                first = self.make_solution(0, first_objectives)
                second = self.make_solution(1, second_objectives)

                self.assertFalse(first.dominates_solution(second, method))
                self.assertFalse(second.dominates_solution(first, method))
                self.assertEqual(first.compare_dominance(second, method), 0)
                self.assertEqual(second.compare_dominance(first, method), 0)

    def test_strictly_dominating_solution_is_reported_in_both_objectives(self):
        cases = {
            "DIST-EPS": ([3, 2, 0, 0], [2, 3, 0, 0]),
            "MC": ([0, 0, 1, 2], [0, 0, 2, 3]),
        }

        for method, (better_objectives, worse_objectives) in cases.items():
            with self.subTest(method=method):
                better = self.make_solution(0, better_objectives)
                worse = self.make_solution(1, worse_objectives)

                self.assertTrue(better.dominates_solution(worse, method))
                self.assertFalse(worse.dominates_solution(better, method))
                self.assertEqual(better.compare_dominance(worse, method), 1)
                self.assertEqual(worse.compare_dominance(better, method), 2)

    def test_equal_and_tradeoff_solutions_share_first_front(self):
        cases = {
            "DIST-EPS": (
                [
                    [3, 2, 0, 0],
                    [3, 2, 0, 0],
                    [4, 4, 0, 0],
                    [1, 6, 0, 0],
                ],
                [0, 0, 0, 1],
            ),
            "MC": (
                [
                    [0, 0, 2, 3],
                    [0, 0, 2, 3],
                    [0, 0, 1, 5],
                    [0, 0, 4, 1],
                    [0, 0, 5, 6],
                ],
                [0, 0, 0, 0, 1],
            ),
        }

        for method, (objectives, expected_fronts) in cases.items():
            with self.subTest(method=method):
                population = Population()
                population.solutions = [
                    self.make_solution(solution_id, values)
                    for solution_id, values in enumerate(objectives)
                ]

                population.fnds(method)

                self.assertEqual(
                    [solution.front for solution in population.solutions],
                    expected_fronts,
                )


if __name__ == "__main__":
    unittest.main()

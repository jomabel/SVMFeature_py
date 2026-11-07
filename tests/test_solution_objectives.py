import unittest

import pandas as pd

from svmfsmeta.solution import Solution


class TestClassificationObjectives(unittest.TestCase):
    def make_solution(self):
        solution = Solution(solution_id=1)
        solution.features = ["x"]
        solution.plane_coords = [1]
        solution.plane_term_b = [2, -2, 0]
        return solution

    def test_points_inside_margin_but_correctly_classified_are_not_mc_errors(self):
        data = pd.DataFrame({
            "x": [-2, -1, 1, 2],
            "y": [-1, -1, 1, 1],
        })
        solution = self.make_solution()

        solution.calculate_epsilon_objective(data, "y")

        self.assertEqual(solution.objective, [0, 2, 0, 0])

    def test_mc_objectives_count_false_negatives_and_false_positives(self):
        data = pd.DataFrame({
            "x": [-2, 2, 0.5, -0.5],
            "y": [-1, 1, -1, 1],
        })
        solution = self.make_solution()

        solution.calculate_epsilon_objective(data, "y")

        self.assertEqual(solution.objective, [0, 5, 1, 1])


if __name__ == "__main__":
    unittest.main()

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

    def test_equivalent_plane_orientations_give_identical_predictions(self):
        data = pd.DataFrame({
            "x": [2, 8, 4, 10],
            "y": [-1, -1, 1, 1],
        })
        results = []

        for weight, intercepts in ((1, [-2, -10, -6]), (-1, [2, 10, 6])):
            solution = self.make_solution()
            solution.plane_coords = [weight]
            solution.plane_term_b = intercepts

            solution.calculate_epsilon_objective(data, "y")
            results.append(solution)

        self.assertEqual(results[0].objective, [0, 12, 1, 1])
        self.assertEqual(results[1].objective, results[0].objective)
        self.assertEqual(results[1].conf_matrix.tolist(), results[0].conf_matrix.tolist())
        self.assertEqual(results[1].accuracy, results[0].accuracy)
        self.assertEqual(results[1].auc, results[0].auc)

    def test_zero_sum_nonzero_normal_is_valid(self):
        data = pd.DataFrame({
            "x1": [-1, 1],
            "x2": [0, 0],
            "y": [-1, 1],
        })
        solution = Solution(solution_id=1)
        solution.features = ["x1", "x2"]
        solution.vectors = [0, 1]
        solution.plane_coords = [1, -1]

        result = solution.evaluate_solution(data, "y", ["x1", "x2"])

        self.assertEqual(result, 1)
        self.assertAlmostEqual(solution.objective[0], 2 ** 0.5)
        self.assertEqual(solution.objective[1:], [0, 0, 0])

    def test_zero_normal_is_invalid(self):
        data = pd.DataFrame({
            "x1": [-1, 1],
            "x2": [0, 0],
            "y": [-1, 1],
        })
        solution = Solution(solution_id=1)
        solution.features = ["x1", "x2"]
        solution.vectors = [0, 1]
        solution.plane_coords = [0, 0]

        self.assertEqual(solution.evaluate_solution(data, "y", ["x1", "x2"]), 0)

    def test_coincident_margin_planes_are_invalid(self):
        data = pd.DataFrame({
            "x1": [0, 0],
            "x2": [-1, 1],
            "y": [-1, 1],
        })
        solution = Solution(solution_id=1)
        solution.features = ["x1", "x2"]
        solution.vectors = [0, 1]
        solution.plane_coords = [1, 0]

        self.assertEqual(solution.evaluate_solution(data, "y", ["x1", "x2"]), 0)


if __name__ == "__main__":
    unittest.main()

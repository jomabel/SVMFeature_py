from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import pandas as pd

from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestCrossoverIsolation(unittest.TestCase):
    def setUp(self):
        random_state = random.getstate()
        self.addCleanup(random.setstate, random_state)
        random.seed(17)
        self.data = pd.DataFrame({
            "x1": [-2, -1, 0, 1, 2, 3],
            "x2": [0, 1, -1, 1, 0, 2],
            "x3": [1, 0, 2, -1, 1, 3],
            "y": [-1, -1, -1, 1, 1, 1],
        })
        self.inputs = ["x1", "x2", "x3"]

    def make_parents(self):
        father = Solution(0)
        father.features = ["x1", "x2"]
        father.plane_coords = [0.4, 0.2]
        father.vectors = [0, 4]

        mother = Solution(1)
        mother.features = ["x2", "x3"]
        mother.plane_coords = [-0.3, 0.5]
        mother.vectors = [1, 5]

        for parent in (father, mother):
            self.assertEqual(
                parent.evaluate_solution(self.data, "y", self.inputs), 1
            )
        return father, mother

    @staticmethod
    def snapshot(solution):
        state = deepcopy(solution.to_dict())
        state["CONFUSION_MATRIX"] = solution.conf_matrix.tolist()
        return state

    def check_mutation_isolation(self, feature_probability, coordinate_probability,
                                 changed_field):
        for method in ("DIST-EPS", "MC"):
            for child_index in range(4):
                with self.subTest(method=method, child=child_index):
                    father, mother = self.make_parents()
                    children = []
                    evaluate = Solution.evaluate_solution

                    def record_evaluation(child, data, output, inputs):
                        children.append(child)
                        return evaluate(child, data, output, inputs)

                    # Capture all four candidates while preserving real evaluation
                    # and selection, regardless of which two children are returned.
                    with patch.object(
                        Solution, "evaluate_solution", autospec=True,
                        side_effect=record_evaluation
                    ):
                        Population().crossover_solutions(
                            father, mother, 2, method,
                            self.data, self.inputs, "y"
                        )
                    self.assertEqual(len(children), 4)
                    parents_before = [self.snapshot(p) for p in (father, mother)]
                    children_before = [self.snapshot(c) for c in children]
                    target = children[child_index]

                    def mutation_draw(low, high):
                        return 0.75 if (low, high) == (-1, 1) else 0.5

                    with patch("svmfsmeta.solution.uniform", side_effect=mutation_draw):
                        target.mutate_solution(
                            0, feature_probability, coordinate_probability, 1,
                            self.data, self.inputs, "y"
                        )

                    self.assertNotEqual(
                        self.snapshot(target)[changed_field],
                        children_before[child_index][changed_field]
                    )
                    for parent, before in zip((father, mother), parents_before):
                        self.assertEqual(self.snapshot(parent), before)
                    for index, child in enumerate(children):
                        if index != child_index:
                            self.assertEqual(
                                self.snapshot(child), children_before[index]
                            )

    def test_feature_mutation_preserves_parents_and_siblings(self):
        self.check_mutation_isolation(1, 0, "FEATURES")

    def test_coordinate_mutation_preserves_parents_and_siblings(self):
        self.check_mutation_isolation(0, 1, "PLANO_COOR")


if __name__ == "__main__":
    unittest.main()

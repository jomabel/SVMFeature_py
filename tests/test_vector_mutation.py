import random
import unittest
from unittest.mock import patch

import pandas as pd

from svmfsmeta.solution import Solution


class TestVectorMutation(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(3)

    def guarded_sampling(self):
        sample = Solution.get_class_vector
        attempts = 0

        def guarded(class_label, data, output):
            nonlocal attempts
            attempts += 1
            if attempts > 12:
                raise AssertionError("Vector mutation kept resampling without an alternative.")
            return sample(class_label, data, output)

        return patch.object(Solution, "get_class_vector", side_effect=guarded)

    def test_singleton_class_keeps_its_anchor_and_other_class_still_mutates(self):
        data = pd.DataFrame({"x": [-1., 1., 2., 3.], "y": [-1, 1, 1, 1]},
                            index=["negative", "positive-a", "positive-b", "positive-c"])
        candidate = Solution(0)
        candidate.features = ["x"]
        candidate.plane_coords = [1.]
        candidate.vectors = ["negative", "positive-a"]
        with self.guarded_sampling():
            candidate.mutate_vectors(data, "y")
        self.assertEqual(candidate.vectors[0], "negative")
        self.assertNotEqual(candidate.vectors[1], "positive-a")
        self.assertEqual([data.loc[label, "y"] for label in candidate.vectors], [-1, 1])
        self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)

    def test_two_singleton_classes_allow_full_individual_mutation(self):
        data = pd.DataFrame({"x": [-1., 1.], "y": [-1, 1]}, index=[10, 20])
        original = data.copy(deep=True)
        candidate = Solution(0)
        candidate.features = ["x"]
        candidate.plane_coords = [1.]
        candidate.vectors = [10, 20]
        self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)
        objectives = list(candidate.objective)
        with self.guarded_sampling(), patch("svmfsmeta.solution.uniform", return_value=0.5):
            candidate.mutate_solution(1., 0., 0., 0, data, ["x"], "y")
        self.assertEqual(candidate.vectors, [10, 20])
        self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)
        self.assertEqual(candidate.objective, objectives)
        pd.testing.assert_frame_equal(data, original)

    def test_classes_with_alternatives_change_to_another_label_of_the_same_class(self):
        for index in ([2, 3, 0, 1], ["n-a", "n-b", "p-a", "p-b"]):
            with self.subTest(index=index):
                data = pd.DataFrame({"x": [-2., -1., 1., 2.], "y": [-1, -1, 1, 1]},
                                    index=index)
                candidate = Solution(0)
                candidate.vectors = [index[0], index[2]]
                with self.guarded_sampling():
                    candidate.mutate_vectors(data, "y")
                self.assertEqual(candidate.vectors, [index[1], index[3]])


if __name__ == "__main__":
    unittest.main()

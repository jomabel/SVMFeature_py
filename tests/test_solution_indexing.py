from copy import deepcopy
import random
import unittest

import numpy as np
import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.solution import Solution


class TestSolutionIndexing(unittest.TestCase):
    def setUp(self):
        random_state = random.getstate()
        self.addCleanup(random.setstate, random_state)
        self.data = pd.DataFrame({
            "x": [0.0, 1.0, 3.0, 4.0],
            "y": [-1, -1, 1, 1],
        })

    @staticmethod
    def labeled_frames(data):
        size = len(data)
        indices = {
            "default": pd.RangeIndex(size),
            "shifted": pd.Index([10 * (i + 1) for i in range(size)]),
            "permuted": pd.Index(list(range(2, size)) + [0, 1]),
            "text": pd.Index([f"sample-{i}" for i in range(size)]),
            "dates": pd.date_range("2020-01-01", periods=size),
            "multiindex": pd.MultiIndex.from_tuples(
                [("batch", i + 10) for i in range(size)]
            ),
        }
        for name, index in indices.items():
            frame = data.copy()
            frame.index = index
            yield name, frame
        doubled = data.loc[data.index.repeat(2)].reset_index(drop=True)
        yield "filtered", doubled.loc[doubled.index % 2 == 0].copy()

    @staticmethod
    def signature(solution, data):
        state = deepcopy(solution.to_dict())
        state["VECTORS"] = [data.index.get_loc(label) for label in solution.vectors]
        state["CONFUSION_MATRIX"] = solution.conf_matrix.tolist()
        return state

    def evaluate_candidate(self, data):
        candidate = Solution(0)
        candidate.features = ["x"]
        candidate.vectors = [data.index[0], data.index[2]]
        candidate.plane_coords = [1.0]
        self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)
        return candidate

    def test_evaluation_uses_anchor_labels_for_every_index_type(self):
        baseline = self.evaluate_candidate(self.data)
        self.assertEqual(baseline.objective, [3.0, 1.0, 0, 0])
        self.assertEqual(baseline.accuracy, 1.0)
        expected = self.signature(baseline, self.data)
        for name, data in self.labeled_frames(self.data):
            with self.subTest(index=name):
                original = data.copy(deep=True)
                candidate = self.evaluate_candidate(data)
                pd.testing.assert_frame_equal(
                    candidate.solution_data, data.loc[candidate.vectors, ["x"]]
                )
                self.assertEqual(self.signature(candidate, data), expected)
                pd.testing.assert_frame_equal(data, original)

    def test_generation_and_mutation_keep_class_anchor_labels(self):
        for name, data in self.labeled_frames(self.data):
            with self.subTest(index=name):
                random.seed(7)
                candidate = Solution(0)
                candidate.generate_random_solution(data, ["x"], "y", 1)
                self.assertEqual(
                    [data.loc[label, "y"] for label in candidate.vectors], [-1, 1]
                )
                self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)
                previous = list(candidate.vectors)
                candidate.mutate_vectors(data, "y")
                self.assertTrue(all(a != b for a, b in zip(candidate.vectors, previous)))
                self.assertEqual(
                    [data.loc[label, "y"] for label in candidate.vectors], [-1, 1]
                )
                self.assertEqual(candidate.evaluate_solution(data, "y", ["x"]), 1)
                normalized = data.reset_index(drop=True)
                equivalent = Solution(0)
                equivalent.features = list(candidate.features)
                equivalent.vectors = [
                    data.index.get_loc(label) for label in candidate.vectors
                ]
                equivalent.plane_coords = list(candidate.plane_coords)
                self.assertEqual(equivalent.evaluate_solution(normalized, "y", ["x"]), 1)
                self.assertEqual(
                    self.signature(candidate, data),
                    self.signature(equivalent, normalized)
                )

    def test_public_optimization_is_independent_of_index_labels(self):
        data = pd.DataFrame(
            np.random.default_rng(1003).normal(size=(20, 4)),
            columns=["x1", "x2", "x3", "x4"]
        )
        data["y"] = [-1] * 10 + [1] * 10
        inputs = ["x1", "x2", "x3", "x4"]

        def optimize(frame, method):
            random.seed(0)
            optimizer = SVMFsMeta(method, frame, 6, inputs, "y", 2)
            optimizer.run(train="iter", num_iter=2)
            return [
                self.signature(candidate, frame)
                for candidate in optimizer.population.solutions
            ]

        for method in ("DIST-EPS", "MC"):
            expected = optimize(data, method)
            for name, frame in self.labeled_frames(data):
                with self.subTest(method=method, index=name):
                    original = frame.copy(deep=True)
                    self.assertEqual(optimize(frame, method), expected)
                    pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()

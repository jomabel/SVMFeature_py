import random
import unittest

import pandas as pd

from svmfsmeta import SVMFsMeta
from svmfsmeta.population import Population
from svmfsmeta.solution import Solution


class TestTwoParentSelection(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(2)

    def test_dominated_second_individual_is_used_as_the_other_parent(self):
        for method in ('DIST-EPS', 'MC'):
            for best_id in (0, 1):
                with self.subTest(method=method, best_id=best_id):
                    population = Population()
                    for i in range(2):
                        candidate = Solution(i)
                        candidate.objective = [2, 0, 0, 0] if i == best_id else [1, 1, 1, 1]
                        population.solutions.append(candidate)
                    population.fnds(method)
                    father, mother = population.select_two_distinct_parents(2)
                    self.assertIs(father, population.solutions[best_id])
                    self.assertIs(mother, population.solutions[1 - best_id])

    def test_public_optimizer_trains_two_individuals_for_multiple_generations(self):
        data = pd.DataFrame({'y': [-1, -1, 1, 1],
                             'x1': [0., .5, 1., 0.], 'x2': [2., 0., 0., 1.]})
        for method in ('MC', 'DIST-EPS'):
            for allow_clones in (0, 1):
                with self.subTest(method=method, allow_clones=allow_clones):
                    random.seed(2)
                    optimizer = SVMFsMeta(method, data, 2, ['x1', 'x2'], 'y', 2,
                                           allow_clones=allow_clones)
                    optimizer.run(train='iter', num_iter=3)
                    self.assertEqual(len(optimizer.population.solutions), 2)
                    self.assertEqual([s.solution_id for s in optimizer.population.solutions], [0, 1])
                    for candidate in optimizer.population.solutions:
                        self.assertEqual(candidate.evaluate_solution(data, 'y', ['x1', 'x2']), 1)


if __name__ == '__main__':
    unittest.main()
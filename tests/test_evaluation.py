import unittest

from job_agent.evaluation import (
    average_precision_at_k,
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_interactions,
)
from tests.fixtures import get_small_bundle


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = get_small_bundle()

    def test_metrics(self):
        ranked = [0, 1, 2]
        positive = {0, 2}
        self.assertAlmostEqual(precision_at_k(ranked, positive, 3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(ranked, positive, 3), 1.0)
        self.assertAlmostEqual(average_precision_at_k(ranked, positive, 3), 5 / 6)
        self.assertAlmostEqual(ndcg_at_k(ranked, positive, 3), 0.91972, places=4)

    def test_split_disjoint(self):
        train, val, test = split_interactions(self.bundle["interactions"], seed=42)
        self.assertEqual(len(train) + len(val) + len(test), len(self.bundle["interactions"]))
        for uid in self.bundle["users"]["user_id"].head(10):
            tr = set(train.loc[train["user_id"] == uid, "job_id"])
            te = set(test.loc[test["user_id"] == uid, "job_id"])
            self.assertFalse(tr & te)

    def test_bootstrap(self):
        lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4], n_boot=100)
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()


import tempfile
import unittest
from pathlib import Path

import numpy as np

from job_agent.recommender import LearnedRanker
from tests.fixtures import get_small_bundle


class RecommenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = get_small_bundle()
        cls.engine = cls.bundle["engine"]
        cls.user = cls.bundle["users"].iloc[0].to_dict()
        cls.uid = cls.user["user_id"]

    def test_score_arrays(self):
        arrays = self.engine.score_arrays(self.user, self.uid)
        for key in ["content", "cf", "svd", "cluster", "popularity"]:
            self.assertEqual(arrays[key].shape[0], len(self.bundle["jobs"]))
            self.assertTrue(np.isfinite(arrays[key]).all())

    def test_rank_constraints_and_exclusion(self):
        recs = self.engine.rank(self.user, user_id=self.uid, method="proposed", top_k=5, apply_constraints=True, use_mmr=False)
        self.assertGreater(len(recs), 0)
        self.assertEqual(len({r["job_index"] for r in recs}), len(recs))
        mask = self.bundle["builder"].hard_mask(self.user, relax=0)
        train_items = self.engine.train_items_by_user.get(self.uid, set())
        for rec in recs:
            self.assertTrue(mask[rec["job_index"]])
            self.assertNotIn(rec["job_index"], train_items)

    def test_explanations(self):
        contributions = self.engine.explanation_contributions(self.user, 0)
        self.assertGreater(len(contributions), 0)
        self.assertIn("feature", contributions[0])
        self.assertIn("contribution", contributions[0])

    def test_ranker_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranker.joblib"
            self.engine.ranker.save(path)
            loaded = LearnedRanker.load(path)
            X = np.zeros((2, 25))
            self.assertTrue(np.allclose(self.engine.ranker.score_matrix(X), loaded.score_matrix(X)))


if __name__ == "__main__":
    unittest.main()

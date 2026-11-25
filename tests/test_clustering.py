import tempfile
import unittest
from pathlib import Path

import numpy as np

from job_agent.clustering import UserClusterer
from tests.fixtures import get_small_bundle


class ClusteringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = get_small_bundle()
        cls.clusterer = cls.bundle["clusterer"]

    def test_profiles_and_affinity(self):
        self.assertIn(self.clusterer.k, {3, 4, 5, 6})
        self.assertEqual(len(self.clusterer.profiles), self.clusterer.k)
        sums = self.clusterer.family_affinity.sum(axis=1)
        self.assertTrue(np.allclose(sums, 1.0))
        fractions = sum(p["fraction"] for p in self.clusterer.profiles)
        self.assertAlmostEqual(fractions, 1.0, places=5)

    def test_predict_and_similarity(self):
        user = self.bundle["users"].iloc[0].to_dict()
        cluster = self.clusterer.predict(user, self.bundle["builder"])
        self.assertGreaterEqual(cluster, 0)
        self.assertLess(cluster, self.clusterer.k)
        sim = self.clusterer.job_skill_similarity(cluster, self.bundle["builder"])
        self.assertEqual(sim.shape[0], len(self.bundle["jobs"]))
        self.assertTrue(np.all(sim >= -1e-9))
        self.assertTrue(np.all(sim <= 1.0 + 1e-9))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clusterer.joblib"
            self.clusterer.save_with_builder(path, self.bundle["builder"])
            loaded = UserClusterer.load(path, self.bundle["builder"], self.bundle["tax"])
            user = self.bundle["users"].iloc[0].to_dict()
            self.assertEqual(
                loaded.predict(user, self.bundle["builder"]),
                self.clusterer.predict(user, self.bundle["builder"]),
            )


if __name__ == "__main__":
    unittest.main()

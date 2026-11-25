import unittest

import numpy as np

from job_agent.features import PAIR_FEATURE_NAMES
from tests.fixtures import get_small_bundle


class FeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = get_small_bundle()

    def test_hard_constraints(self):
        bundle = self.bundle
        job = bundle["jobs"].iloc[0]
        user = {
            "city": job["city"],
            "target_cities": [job["city"]],
            "expected_salary": float(job["salary_min"]),
            "experience_years": int(job["experience_min"]),
            "education": job["education"],
            "remote_pref": False,
            "skills": list(job["skills"]),
            "preferred_families": [job["family"]],
        }
        mask = bundle["builder"].hard_mask(user, relax=0)
        self.assertTrue(mask[0])
        user["target_cities"] = ["拉萨"]
        self.assertFalse(bundle["builder"].hard_mask(user, relax=0)[0])
        user["target_cities"] = [job["city"]]
        user["expected_salary"] = float(job["salary_max"]) * 3
        self.assertFalse(bundle["builder"].hard_mask(user, relax=0)[0])

    def test_pair_matrix(self):
        bundle = self.bundle
        user = bundle["users"].iloc[0].to_dict()
        n = len(bundle["jobs"])
        zeros = np.zeros(n)
        matrix = bundle["builder"].build_pair_matrix(
            user, zeros, zeros, zeros, zeros, zeros, zeros
        )
        self.assertEqual(matrix.shape, (n, len(PAIR_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_cluster_vector(self):
        vector = self.bundle["builder"].build_user_cluster_vector(self.bundle["users"].iloc[0].to_dict())
        self.assertTrue(np.isfinite(vector).all())
        self.assertGreater(len(vector), 10)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from job_agent.demo import SAMPLE_COMPLETE, SAMPLE_NO_MATCH
from tests.fixtures import build_small_agent, get_small_bundle


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.agent = build_small_agent(cls.tmp.name)
        cls.bundle = get_small_bundle()
        cls.jobs = cls.bundle["jobs"].set_index("job_id")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_need_more_info(self):
        response = self.agent.run("我想找数据分析工作，期望15k")
        self.assertEqual(response["status"], "need_more_info")
        self.assertGreater(len(response["questions"]), 0)
        self.assertEqual(response["recommendations"], [])

    def test_invalid_salary(self):
        response = self.agent.run({**SAMPLE_COMPLETE, "expected_salary": -5})
        self.assertEqual(response["status"], "need_more_info")
        self.assertTrue(any("期望月薪" in q for q in response["questions"]))

    def test_successful_recommendations(self):
        user = self.bundle["users"].iloc[0].to_dict()
        payload = {
            "city": user["city"], "target_cities": user["target_cities"],
            "expected_salary": user["expected_salary"], "experience_years": user["experience_years"],
            "education": user["education"], "remote_pref": user["remote_pref"],
            "skills": user["skills"], "preferred_families": user["preferred_families"],
            "bio": user["bio"],
        }
        response = self.agent.run(payload, top_k=5, history_user_id=user["user_id"])
        self.assertEqual(response["status"], "ok")
        self.assertGreater(len(response["recommendations"]), 0)
        profile_skills = set(response["profile"]["skills"])
        for rec in response["recommendations"]:
            job = self.jobs.loc[rec["job_id"]]
            job_skills = set(job["skills"])
            self.assertTrue(set(rec["evidence"]["matched_skills"]).issubset(profile_skills))
            self.assertTrue(set(rec["evidence"]["matched_skills"]).issubset(job_skills))
            self.assertTrue(set(rec["evidence"]["missing_skills"]).issubset(job_skills))
            self.assertFalse(set(rec["evidence"]["missing_skills"]) & profile_skills)
            self.assertIn("constraints", rec["evidence"])
            self.assertGreater(len(rec["explanation"]), 20)

    def test_no_match(self):
        response = self.agent.run(SAMPLE_NO_MATCH)
        self.assertIn(response["status"], {"no_match", "need_more_info"})

    def test_feedback_and_export(self):
        user = self.bundle["users"].iloc[1].to_dict()
        payload = {
            "city": user["city"], "target_cities": user["target_cities"],
            "expected_salary": user["expected_salary"], "experience_years": user["experience_years"],
            "education": user["education"], "remote_pref": user["remote_pref"],
            "skills": user["skills"], "preferred_families": user["preferred_families"],
        }
        response = self.agent.run(payload, top_k=3)
        self.assertEqual(response["status"], "ok")
        job_id = response["recommendations"][0]["job_id"]
        fb = self.agent.record_feedback(response["session_id"], job_id, "like")
        self.assertEqual(fb["status"], "ok")
        self.assertTrue(Path(self.agent.feedback_store.path).exists())
        export_path = Path(self.tmp.name) / "export.csv"
        self.agent.export_csv(response, export_path)
        self.assertTrue(export_path.exists())
        lines = export_path.read_text(encoding="utf-8-sig").strip().splitlines()
        self.assertEqual(len(lines), len(response["recommendations"]) + 1)


if __name__ == "__main__":
    unittest.main()

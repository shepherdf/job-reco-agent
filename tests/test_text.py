import unittest

from job_agent.taxonomy import get_taxonomy
from job_agent.text import (
    extract_city,
    extract_education,
    extract_experience,
    extract_remote,
    extract_salary,
    infer_families,
)


class TextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = get_taxonomy()

    def test_extract_salary(self):
        self.assertEqual(extract_salary("期望月薪 15k"), 15.0)
        self.assertEqual(extract_salary("月薪15000"), 15.0)
        self.assertEqual(extract_salary("期望1.5万"), 15.0)
        self.assertIsNone(extract_salary("我有3年经验"))

    def test_extract_experience(self):
        self.assertEqual(extract_experience("3年经验"), 3)
        self.assertEqual(extract_experience("应届生"), 0)
        self.assertIsNone(extract_experience("熟悉Python"))

    def test_extract_basic_fields(self):
        self.assertEqual(extract_city("我在杭州，想去上海", self.tax.cities), "杭州")
        self.assertEqual(extract_education("最高学历本科", self.tax.education_levels), "本科")
        self.assertTrue(extract_remote("希望远程办公"))

    def test_skill_aliases_and_boundaries(self):
        text = "熟悉Java、JavaScript、C++、Go语言，了解大模型和RAG，做过推荐系统"
        skills = set(self.tax.extract_skills(text))
        for expected in ["Java", "JavaScript", "C++", "Go", "大模型", "RAG", "推荐系统"]:
            self.assertIn(expected, skills)
        self.assertNotIn("Angular", skills)

    def test_skill_family_inference(self):
        families = infer_families("我想找数据分析工作，熟悉SQL和Python", self.tax)
        self.assertIn("data", families)


if __name__ == "__main__":
    unittest.main()

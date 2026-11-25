from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .taxonomy import Taxonomy, get_taxonomy

PAIR_FEATURE_NAMES = [
    "content_sim",
    "cf_score",
    "svd_score",
    "cluster_skill_sim",
    "cluster_family_affinity",
    "skill_jaccard",
    "job_skill_coverage",
    "user_skill_coverage",
    "city_match",
    "is_same_city",
    "salary_fit",
    "salary_gap_ratio",
    "salary_ok",
    "exp_fit",
    "edu_fit",
    "remote_fit",
    "popularity",
    "job_salary_mid",
    "job_exp_min",
    "job_edu_rank",
    "job_age_days",
    "content_x_skill",
    "cf_x_content",
    "is_preferred_family",
    "is_related_family",
]


def skill_matrix(values: List[List[str]], tax: Taxonomy) -> np.ndarray:
    index = {s: i for i, s in enumerate(tax.skill_names)}
    mat = np.zeros((len(values), len(tax.skill_names)), dtype=float)
    for i, skills in enumerate(values):
        for skill in skills:
            j = index.get(skill)
            if j is not None:
                mat[i, j] = 1.0
    return mat


def family_matrix(values: List[List[str]], tax: Taxonomy) -> np.ndarray:
    index = {f: i for i, f in enumerate(tax.family_ids)}
    mat = np.zeros((len(values), len(tax.family_ids)), dtype=float)
    for i, families in enumerate(values):
        for fid in families:
            j = index.get(fid)
            if j is not None:
                mat[i, j] = 1.0
    return mat


class FeatureBuilder:
    def __init__(self, jobs: pd.DataFrame, tax: Taxonomy | None = None):
        self.tax = tax or get_taxonomy()
        self.jobs = jobs.reset_index(drop=True)
        self.n_jobs = len(self.jobs)
        self.job_skill = skill_matrix(self.jobs["skills"].tolist(), self.tax)
        self.job_family = family_matrix([[f] for f in self.jobs["family"]], self.tax)
        self.job_skill_sum = self.job_skill.sum(axis=1)
        self.job_salary_mid = ((self.jobs["salary_min"] + self.jobs["salary_max"]) / 2).to_numpy(dtype=float)
        self.job_salary_max = self.jobs["salary_max"].to_numpy(dtype=float)
        self.job_exp = self.jobs["experience_min"].to_numpy(dtype=float)
        self.job_edu = self.jobs["education"].map(self.tax.education_rank).to_numpy(dtype=float)
        self.job_remote = self.jobs["remote"].to_numpy(dtype=bool)
        self.job_city = self.jobs["city"].to_numpy()
        self.job_family_id = self.jobs["family"].to_numpy()
        self.job_age = self.jobs["posted_days_ago"].to_numpy(dtype=float)
        self.job_company = self.jobs["company"].to_numpy()
        self.n_skills = len(self.tax.skill_names)
        self.user_scaler = None

    def user_skill_vector(self, user: dict) -> np.ndarray:
        return skill_matrix([list(user.get("skills", []))], self.tax)[0]

    def user_family_vector(self, user: dict) -> np.ndarray:
        return family_matrix([list(user.get("preferred_families", []))], self.tax)[0]

    def hard_mask(self, user: dict, relax: int = 0) -> np.ndarray:
        salary_ratio = [0.9, 0.8, 0.7, 0.6, 0.5][min(relax, 4)]
        exp_slack = [1, 2, 3, 4, 5][min(relax, 4)]
        cities = list(user.get("target_cities", []))
        if relax >= 3 or not cities:
            cities = list(self.tax.cities)
        expected = float(user.get("expected_salary") or 1.0)
        experience = float(user.get("experience_years") or 0.0)
        user_edu = self.tax.education_rank(user.get("education") or "本科")
        mask = (
            np.isin(self.job_city, cities)
            & (self.job_salary_max >= expected * salary_ratio)
            & (self.job_exp <= experience + exp_slack)
            & (self.job_edu <= user_edu)
        )
        return mask

    def violation_details(self, user: dict, idx: int) -> List[str]:
        job = self.jobs.iloc[int(idx)]
        violations: List[str] = []
        if job["city"] not in set(user.get("target_cities", [])):
            violations.append(f"城市不在期望范围（{job['city']}）")
        if float(job["salary_max"]) < float(user.get("expected_salary") or 0) * 0.9:
            violations.append("薪资上限低于期望")
        if float(job["experience_min"]) > float(user.get("experience_years") or 0) + 1:
            violations.append("经验要求偏高")
        if self.tax.education_rank(job["education"]) > self.tax.education_rank(user.get("education") or "本科"):
            violations.append("学历要求偏高")
        return violations

    def build_pair_matrix(
        self,
        user: dict,
        content_sim: np.ndarray,
        cf_score: np.ndarray,
        svd_score: np.ndarray,
        cluster_skill_sim: np.ndarray,
        cluster_family_affinity: np.ndarray,
        popularity: np.ndarray,
    ) -> np.ndarray:
        user_skill = self.user_skill_vector(user)
        user_family = self.user_family_vector(user)
        user_skill_sum = float(user_skill.sum())
        inter = self.job_skill @ user_skill
        union = self.job_skill_sum + user_skill_sum - inter
        skill_jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        job_skill_coverage = np.divide(inter, self.job_skill_sum, out=np.zeros_like(inter), where=self.job_skill_sum > 0)
        user_skill_coverage = np.divide(inter, user_skill_sum, out=np.zeros_like(inter), where=user_skill_sum > 0)
        expected = max(float(user.get("expected_salary") or 1.0), 1.0)
        experience = float(user.get("experience_years") or 0.0)
        user_edu = self.tax.education_rank(user.get("education") or "本科")
        target_cities = set(user.get("target_cities", []))
        city_match = np.where(self.job_city == user.get("city"), 1.0, np.where(np.isin(self.job_city, list(target_cities)), 0.65, 0.0))
        salary_fit = np.clip(1.0 - np.abs(self.job_salary_mid - expected) / expected, -1.0, 1.0)
        salary_gap = np.clip((self.job_salary_mid - expected) / expected, -1.0, 2.0)
        salary_ok = (self.job_salary_max >= expected * 0.9).astype(float)
        exp_fit = np.where(self.job_exp <= experience, 1.0, np.maximum(0.0, 1.0 - (self.job_exp - experience) / 5.0))
        edu_fit = (self.job_edu <= user_edu).astype(float)
        remote_fit = self.job_remote.astype(float) if user.get("remote_pref") else np.full(self.n_jobs, 0.5)
        preferred = set(user.get("preferred_families", []))
        related: set[str] = set()
        for fid in preferred:
            related |= self.tax.related_families(fid)
        is_preferred = np.isin(self.job_family_id, list(preferred)).astype(float)
        is_related = np.isin(self.job_family_id, list(related)).astype(float)
        content_x_skill = content_sim * skill_jaccard
        cf_x_content = cf_score * content_sim
        cols = [
            content_sim, cf_score, svd_score, cluster_skill_sim, cluster_family_affinity,
            skill_jaccard, job_skill_coverage, user_skill_coverage, city_match,
            (self.job_city == user.get("city")).astype(float), salary_fit, salary_gap,
            salary_ok, exp_fit, edu_fit, remote_fit, popularity, self.job_salary_mid,
            self.job_exp, self.job_edu, self.job_age, content_x_skill, cf_x_content,
            is_preferred, is_related,
        ]
        return np.column_stack(cols)

    def _user_numeric_frame(self, users: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            users["experience_years"].to_numpy(dtype=float) / 8.0,
            users["expected_salary"].to_numpy(dtype=float) / 40.0,
            users["education"].map(self.tax.education_rank).to_numpy(dtype=float) / 4.0,
            users["remote_pref"].to_numpy(dtype=float),
        ])

    def build_user_cluster_matrix(self, users: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
        skills = skill_matrix(users["skills"].tolist(), self.tax)
        families = family_matrix(users["preferred_families"].tolist(), self.tax) * 2.5
        numeric = self._user_numeric_frame(users)
        self.user_scaler = StandardScaler()
        numeric_scaled = self.user_scaler.fit_transform(numeric)
        matrix = np.hstack([skills, families, numeric_scaled])
        names = [f"skill:{s}" for s in self.tax.skill_names] + [f"family:{f}" for f in self.tax.family_ids] + [
            "experience", "salary", "education", "remote"
        ]
        return matrix, names

    def build_user_cluster_vector(self, user: dict) -> np.ndarray:
        if self.user_scaler is None:
            raise RuntimeError("user_scaler is not fitted")
        skills = self.user_skill_vector(user)
        families = self.user_family_vector(user) * 2.5
        numeric = np.array([[
            float(user.get("experience_years") or 0.0) / 8.0,
            float(user.get("expected_salary") or 1.0) / 40.0,
            self.tax.education_rank(user.get("education") or "本科") / 4.0,
            1.0 if user.get("remote_pref") else 0.0,
        ]])
        numeric_scaled = self.user_scaler.transform(numeric)[0]
        return np.concatenate([skills, families, numeric_scaled])

    def build_job_cluster_matrix(self) -> np.ndarray:
        return np.hstack([self.job_skill, self.job_family * 2.5])


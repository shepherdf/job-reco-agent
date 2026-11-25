import csv
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .clustering import UserClusterer
from .data_gen import load_processed
from .evaluation import split_interactions
from .features import FeatureBuilder
from .recommender import (
    ItemCFRecommender,
    PopularityRecommender,
    RecommendationEngine,
    SVDRecommender,
)
from .taxonomy import PROJECT_ROOT, Taxonomy, get_taxonomy
from .text import (
    ChineseTextModel,
    extract_cities,
    extract_city,
    extract_education,
    extract_experience,
    extract_remote,
    extract_salary,
    infer_families,
)


FEATURE_LABELS = {
    "content_sim": "岗位文本与画像相似度",
    "cluster_skill_sim": "分群技能相似度",
    "cluster_family_affinity": "分群方向偏好",
    "skill_jaccard": "技能重合度",
    "job_skill_coverage": "岗位技能覆盖率",
    "city_match": "城市匹配",
    "salary_fit": "薪资匹配",
    "exp_fit": "经验匹配",
    "edu_fit": "学历匹配",
    "remote_fit": "远程偏好",
    "popularity": "岗位热度",
    "is_preferred_family": "目标方向匹配",
    "is_related_family": "相关方向匹配",
    "cf_score": "协同过滤分数",
    "svd_score": "矩阵分解分数",
}


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in re.split(r"[,\uFF0C\u3001|;；/]+", str(value)) if x.strip()]


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "是", "远程"}


class Trace:
    def __init__(self):
        self.started = time.perf_counter()
        self.steps = []

    def add(self, name: str, status: str, detail: dict | None = None) -> None:
        self.steps.append({
            "step": name,
            "status": status,
            "detail": detail or {},
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 2),
        })

    def to_list(self) -> List[dict]:
        return self.steps


class FeedbackStore:
    def __init__(self, path: Path | str = PROJECT_ROOT / "artifacts" / "sessions" / "feedback.csv"):
        self.path = Path(path)

    def record(self, session_id: str, job_id: str, action: str, note: str = "") -> dict:
        if action not in {"like", "skip", "apply_click", "not_interested"}:
            raise ValueError("unsupported feedback action")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "session_id", "job_id", "action", "note"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), session_id, job_id, action, note])
        return {"status": "ok", "session_id": session_id, "job_id": job_id, "action": action}

    def summary(self) -> dict:
        if not self.path.exists():
            return {"total": 0, "by_action": {}}
        df = pd.read_csv(self.path, encoding="utf-8-sig")
        return {"total": int(len(df)), "by_action": df["action"].value_counts().to_dict()}


class LLMExplainer:
    def __init__(self, enabled: bool = False, tax: Taxonomy | None = None):
        self.enabled = enabled
        self.tax = tax or get_taxonomy()
        self.api_key = os.environ.get("JOB_AGENT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = os.environ.get("JOB_AGENT_LLM_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("JOB_AGENT_LLM_MODEL", "gpt-4o-mini")

    def generate(self, facts: dict, fallback: str) -> str:
        if not self.enabled or not self.api_key:
            return fallback
        try:
            import requests
            prompt = (
                "你是求职推荐解释器。只能使用 FACTS 中出现的技能、城市、薪资和分群信息，"
                "不得编造技能、公司或薪资，不得承诺录用。用 2-3 句中文解释推荐理由，并提示信息以招聘方为准。\n"
                f"FACTS={json.dumps(facts, ensure_ascii=False)}"
            )
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "只输出解释文本，不要输出 Markdown。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 220,
                },
                timeout=20,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            allowed = set(facts.get("matched_skills", [])) | set(facts.get("missing_skills", [])) | set(facts.get("job_skills", []))
            mentioned = set(self.tax.extract_skills(text))
            if mentioned - allowed or len(text) > 500:
                return fallback
            return text
        except Exception:
            return fallback


@dataclass
class JobAgent:
    jobs: pd.DataFrame
    users: pd.DataFrame
    builder: FeatureBuilder
    text_model: ChineseTextModel
    clusterer: UserClusterer
    popularity: PopularityRecommender
    cf: ItemCFRecommender
    svd: SVDRecommender
    engine: RecommendationEngine
    tax: Taxonomy
    feedback_store: FeedbackStore = field(default_factory=FeedbackStore)
    llm: LLMExplainer = field(default_factory=LLMExplainer)

    @classmethod
    def from_artifacts(cls, project_root: Path | str = PROJECT_ROOT, use_llm: bool = False) -> "JobAgent":
        root = Path(project_root)
        models = root / "artifacts" / "models"
        required = ["text_model.joblib", "clusterer.joblib", "popularity.joblib", "cf.joblib", "svd.joblib", "ranker.joblib"]
        missing = [name for name in required if not (models / name).exists()]
        if missing:
            raise FileNotFoundError(
                "缺少模型文件：" + ", ".join(missing) + "。请先运行 python scripts/train_eval.py --seed 42"
            )
        jobs, users, interactions = load_processed()
        builder = FeatureBuilder(jobs)
        text_model = joblib.load(models / "text_model.joblib")
        clusterer = UserClusterer.load(models / "clusterer.joblib", builder)
        popularity = joblib.load(models / "popularity.joblib")
        cf = joblib.load(models / "cf.joblib")
        svd = joblib.load(models / "svd.joblib")
        train_path = root / "artifacts" / "train_interactions.csv"
        train = pd.read_csv(train_path, encoding="utf-8-sig") if train_path.exists() else split_interactions(interactions, seed=42)[0]
        engine = RecommendationEngine(
            jobs, users, builder, text_model, clusterer, popularity, cf, svd, train_interactions=train
        )
        engine.load_ranker(models / "ranker.joblib")
        tax = get_taxonomy()
        return cls(
            jobs=jobs, users=users, builder=builder, text_model=text_model, clusterer=clusterer,
            popularity=popularity, cf=cf, svd=svd, engine=engine, tax=tax,
            llm=LLMExplainer(enabled=use_llm, tax=tax),
        )

    def parse_profile(self, payload: dict | str, session_id: str) -> tuple[dict, List[str], List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []
        if isinstance(payload, str):
            text = payload
            profile = {
                "user_id": session_id,
                "name": "匿名求职者",
                "city": extract_city(text, self.tax.cities),
                "target_cities": extract_cities(text, self.tax.cities),
                "expected_salary": extract_salary(text),
                "experience_years": extract_experience(text),
                "education": extract_education(text, self.tax.education_levels),
                "remote_pref": extract_remote(text),
                "skills": self.tax.extract_skills(text),
                "preferred_families": infer_families(text, self.tax),
                "bio": text,
            }
        else:
            p = dict(payload or {})
            profile = {
                "user_id": str(p.get("user_id") or session_id),
                "name": str(p.get("name") or "匿名求职者"),
                "city": p.get("city"),
                "target_cities": _as_list(p.get("target_cities")),
                "expected_salary": p.get("expected_salary"),
                "experience_years": p.get("experience_years"),
                "education": p.get("education"),
                "remote_pref": _to_bool(p.get("remote_pref", False)),
                "skills": _as_list(p.get("skills")),
                "preferred_families": _as_list(p.get("preferred_families")),
                "bio": str(p.get("bio") or ""),
            }
        if profile.get("expected_salary") is not None:
            try:
                profile["expected_salary"] = float(profile["expected_salary"])
            except (TypeError, ValueError):
                errors.append("期望薪资无法识别，请填写数字（单位 k）。")
                profile["expected_salary"] = None
        if profile.get("experience_years") is not None:
            try:
                profile["experience_years"] = float(profile["experience_years"])
                if profile["experience_years"].is_integer():
                    profile["experience_years"] = int(profile["experience_years"])
            except (TypeError, ValueError):
                errors.append("工作经验无法识别，请填写年数（应届生填 0）。")
                profile["experience_years"] = None
        if isinstance(profile.get("city"), str):
            profile["city"] = profile["city"].strip()
        if not profile.get("target_cities") and profile.get("city"):
            profile["target_cities"] = [profile["city"]]
        if profile.get("city") is None and profile.get("target_cities"):
            profile["city"] = profile["target_cities"][0]
        if profile.get("city") is None and profile.get("bio"):
            profile["city"] = extract_city(profile["bio"], self.tax.cities)
            if profile.get("city") and not profile.get("target_cities"):
                profile["target_cities"] = [profile["city"]]
        if profile.get("expected_salary") is None and profile.get("bio"):
            profile["expected_salary"] = extract_salary(profile["bio"])
        if profile.get("experience_years") is None and profile.get("bio"):
            profile["experience_years"] = extract_experience(profile["bio"])
        if profile.get("education") is None and profile.get("bio"):
            profile["education"] = extract_education(profile["bio"], self.tax.education_levels)
        raw_skills = list(profile.get("skills", []))
        profile["skills"] = self.tax.normalize_skills(raw_skills)
        if raw_skills and not profile["skills"]:
            warnings.append("未能识别填写的技能，已忽略未知项。")
        if not profile["skills"] and profile.get("bio"):
            profile["skills"] = self.tax.extract_skills(profile["bio"])
        families = []
        for value in profile.get("preferred_families", []):
            fid = self.tax.family_id(value) or self.tax.family_id(self.tax.family_name(value))
            if fid and fid not in families:
                families.append(fid)
        profile["preferred_families"] = families
        if not profile["preferred_families"] and profile.get("bio"):
            profile["preferred_families"] = infer_families(profile["bio"], self.tax)
        if not profile["preferred_families"] and profile["skills"]:
            counts = {}
            for skill in profile["skills"]:
                for fid in self.tax.skill_family_map.get(skill, set()):
                    counts[fid] = counts.get(fid, 0) + 1
            profile["preferred_families"] = [fid for fid, _ in sorted(counts.items(), key=lambda x: -x[1])[:2]]
        unknown_cities = [c for c in profile.get("target_cities", []) if c not in self.tax.cities]
        if unknown_cities:
            errors.append("暂不支持这些城市：" + "、".join(unknown_cities) + "。")
            profile["target_cities"] = [c for c in profile["target_cities"] if c in self.tax.cities]
        if profile.get("city") and profile["city"] not in self.tax.cities:
            errors.append(f"暂不支持城市「{profile['city']}」，请从支持城市中选择。")
        return profile, errors, warnings, []

    def validate_profile(self, profile: dict, errors: List[str], warnings: List[str]) -> tuple[List[str], List[str]]:
        if profile.get("expected_salary") is not None and not (3 <= float(profile["expected_salary"]) <= 100):
            errors.append("期望月薪需在 3–100k 之间，请重新填写。")
            profile["expected_salary"] = None
        if profile.get("experience_years") is not None and not (0 <= float(profile["experience_years"]) <= 40):
            errors.append("工作经验需在 0–40 年之间，请重新填写。")
            profile["experience_years"] = None
        if profile.get("education") is not None and profile["education"] not in self.tax.education_levels:
            errors.append("学历需为大专、本科、硕士或博士。")
            profile["education"] = None
        if profile.get("expected_salary") is not None and profile["expected_salary"] > 50:
            warnings.append("期望薪资较高，可能会过滤掉部分匹配岗位。")
        if profile.get("remote_pref"):
            warnings.append("远程岗位占比较低，远程偏好会作为加分项而非硬性过滤。")
        if len(profile.get("target_cities", [])) > 5:
            warnings.append("目标城市较多，推荐结果可能更分散。")
        missing = []
        if not profile.get("city"):
            missing.append("city")
        if profile.get("expected_salary") is None:
            missing.append("expected_salary")
        if profile.get("experience_years") is None:
            missing.append("experience_years")
        if not profile.get("education"):
            missing.append("education")
        if not profile.get("skills") and not profile.get("preferred_families"):
            missing.append("skills_or_family")
        return errors, missing

    def _questions(self, errors: List[str], missing: List[str]) -> List[str]:
        questions = list(errors)
        mapping = {
            "city": "你希望在哪个城市工作？支持：北京、上海、深圳、杭州、广州、成都、武汉、南京、西安、苏州、重庆、长沙。",
            "expected_salary": "你的期望月薪是多少 k？（例如 12 表示 12k/月）",
            "experience_years": "你有几年相关工作经验？（应届生填 0）",
            "education": "你的最高学历是？（大专/本科/硕士/博士）",
            "skills_or_family": "请补充 3–5 个核心技能，或说明你想找的岗位方向（如数据分析、后端开发）。",
        }
        for key in missing:
            if key in mapping:
                questions.append(mapping[key])
        return list(dict.fromkeys(questions))

    def _cluster_info(self, profile: dict) -> dict:
        cluster_id = self.clusterer.predict(profile, self.builder)
        info = self.clusterer.profile(cluster_id)
        return {
            "id": int(cluster_id),
            "name": info["name"],
            "size": info["size"],
            "top_skills": info["top_skills"],
            "top_preferred_families": info["top_preferred_families"],
        }

    def _salary_text(self, job: pd.Series) -> str:
        return f"{float(job['salary_min']):.1f}-{float(job['salary_max']):.1f}k"

    def _confidence(self, score: float, margin: float) -> str:
        if score >= 0.40 and margin >= 0.05:
            return "high"
        if score >= 0.30 and margin >= 0.025:
            return "medium"
        return "low"

    def _job_confidence(self, score: float) -> str:
        if score >= 0.42:
            return "高"
        if score >= 0.33:
            return "中"
        return "低"

    def _constraint_evidence(self, profile: dict, job: pd.Series) -> dict:
        expected = float(profile.get("expected_salary") or 0)
        user_exp = float(profile.get("experience_years") or 0)
        user_edu = self.tax.education_rank(profile.get("education") or "本科")
        job_edu = self.tax.education_rank(job["education"])
        return {
            "city_ok": job["city"] in set(profile.get("target_cities", [])),
            "salary_ok": float(job["salary_max"]) >= expected * 0.9,
            "experience_ok": float(job["experience_min"]) <= user_exp + 1,
            "education_ok": job_edu <= user_edu,
            "remote": bool(job["remote"]),
        }

    def _explanation(self, profile: dict, job: pd.Series, cluster_info: dict, contributions: List[dict]) -> tuple[str, dict]:
        user_skills = set(profile.get("skills", []))
        job_skills = list(job["skills"])
        matched = [s for s in job_skills if s in user_skills]
        missing = [s for s in job_skills if s not in user_skills]
        constraints = self._constraint_evidence(profile, job)
        expected = float(profile.get("expected_salary") or 0)
        salary_gap = float(job["salary_max"]) - expected
        if salary_gap >= 0:
            salary_line = f"薪资上限 {float(job['salary_max']):.1f}k 不低于你的期望 {expected:.1f}k"
        elif salary_gap >= -expected * 0.15:
            salary_line = f"薪资上限 {float(job['salary_max']):.1f}k 与你的期望 {expected:.1f}k 接近"
        else:
            salary_line = f"薪资上限 {float(job['salary_max']):.1f}k 低于你的期望 {expected:.1f}k"
        if matched:
            match_line = f"你的技能「{'、'.join(matched[:3])}」与该岗位要求直接重合"
        else:
            match_line = "该岗位与你的目标方向一致，但技能重合较少"
        cluster_line = (
            f"你属于「{cluster_info['name']}」分群，该群体常见方向为"
            f"「{'、'.join(cluster_info['top_preferred_families'][:2])}」"
        )
        gap_parts = []
        if missing:
            gap_parts.append(f"岗位还要求 {('、'.join(missing[:3]))}，你的画像中尚未填写")
        if float(job["experience_min"]) > float(profile.get("experience_years") or 0) + 1:
            gap_parts.append(
                f"岗位要求 {int(job['experience_min'])} 年经验，你目前 {profile.get('experience_years')} 年，属于冲刺型机会"
            )
        gap_line = "；".join(gap_parts)
        factors = [FEATURE_LABELS.get(c["feature"], c["feature"]) for c in contributions[:2] if c["contribution"] > 0]
        factor_line = ("模型主要依据：" + "、".join(factors)) if factors else ""
        text = (
            f"{match_line}；工作地点 {job['city']} 在期望范围内；{salary_line}。"
            f"{cluster_line}。"
            + (f"需注意：{gap_line}。" if gap_line else "")
            + (f"{factor_line}。" if factor_line else "")
            + "该结果由离线模型排序，不会自动投递；薪资与岗位信息以招聘方为准。"
        )
        facts = {
            "matched_skills": matched,
            "missing_skills": missing,
            "job_skills": job_skills,
            "city": job["city"],
            "salary": self._salary_text(job),
            "expected_salary": expected,
            "cluster": cluster_info["name"],
            "constraints": constraints,
        }
        text = self.llm.generate(facts, text)
        evidence = {
            "matched_skills": matched,
            "missing_skills": missing,
            "constraints": constraints,
            "cluster": cluster_info["name"],
            "top_factors": [
                {"feature": c["feature"], "label": FEATURE_LABELS.get(c["feature"], c["feature"]), "contribution": round(c["contribution"], 4)}
                for c in contributions[:3]
            ],
            "salary_gap": round(salary_gap, 2),
        }
        return text, evidence

    def _fallback_scores(self, profile: dict) -> np.ndarray:
        user_skill = self.builder.user_skill_vector(profile)
        inter = self.builder.job_skill @ user_skill
        union = self.builder.job_skill_sum + float(user_skill.sum()) - inter
        jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        preferred = set(profile.get("preferred_families", []))
        related = set()
        for fid in preferred:
            related |= self.tax.related_families(fid)
        family = np.isin(self.builder.job_family_id, list(preferred)).astype(float) * 0.25
        family += np.isin(self.builder.job_family_id, list(related)).astype(float) * 0.08
        city = np.isin(self.builder.job_city, list(profile.get("target_cities", []))).astype(float) * 0.12
        expected = max(float(profile.get("expected_salary") or 1), 1.0)
        salary = np.clip(1.0 - np.abs(self.builder.job_salary_mid - expected) / expected, 0, 1) * 0.1
        return 0.55 * jaccard + family + city + salary

    def _recommendation(self, profile: dict, job_index: int, score: float, cluster_info: dict) -> dict:
        job = self.jobs.iloc[int(job_index)]
        try:
            contributions = self.engine.explanation_contributions(profile, int(job_index))
        except Exception:
            contributions = []
        explanation, evidence = self._explanation(profile, job, cluster_info, contributions)
        return {
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            "city": job["city"],
            "salary": self._salary_text(job),
            "salary_min": float(job["salary_min"]),
            "salary_max": float(job["salary_max"]),
            "experience_min": int(job["experience_min"]),
            "education": job["education"],
            "remote": bool(job["remote"]),
            "family": job["family_name"],
            "skills": list(job["skills"]),
            "score": round(float(score), 4),
            "match_level": self._job_confidence(float(score)),
            "explanation": explanation,
            "evidence": evidence,
        }

    def run(
        self,
        payload: dict | str,
        session_id: str | None = None,
        top_k: int = 10,
        use_mmr: bool = False,
        use_llm: bool | None = None,
        history_user_id: str | None = None,
        relax_if_needed: bool = True,
    ) -> dict:
        started = time.perf_counter()
        session_id = session_id or f"S{datetime.now():%Y%m%d%H%M%S}{uuid.uuid4().hex[:4]}"
        trace = Trace()
        profile, errors, warnings, _ = self.parse_profile(payload, session_id)
        trace.add("parse_profile", "ok" if not errors else "warning", {"skills": len(profile.get("skills", []))})
        errors, missing = self.validate_profile(profile, errors, warnings)
        trace.add("validate_profile", "need_more_info" if (errors or missing) else "ok", {"errors": len(errors), "missing": missing})
        if errors or missing:
            return {
                "status": "need_more_info",
                "session_id": session_id,
                "message": "还需要补充或修正一些信息，才能给出可靠推荐。",
                "profile": profile,
                "questions": self._questions(errors, missing),
                "warnings": warnings,
                "recommendations": [],
                "trace": trace.to_list(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        if not profile.get("preferred_families") and profile.get("skills"):
            counts = {}
            for skill in profile["skills"]:
                for fid in self.tax.skill_family_map.get(skill, set()):
                    counts[fid] = counts.get(fid, 0) + 1
            profile["preferred_families"] = [fid for fid, _ in sorted(counts.items(), key=lambda x: -x[1])[:2]]
        if not profile.get("target_cities"):
            profile["target_cities"] = [profile["city"]]
        trace.add("infer_preferences", "ok", {"target_cities": profile["target_cities"], "families": profile["preferred_families"]})
        cluster_info = self._cluster_info(profile)
        trace.add("user_clustering", "ok", {"cluster": cluster_info})
        relax = 0
        mask = self.builder.hard_mask(profile, relax=0)
        counts = [int(mask.sum())]
        if counts[0] < top_k and relax_if_needed:
            for level in range(1, 5):
                mask = self.builder.hard_mask(profile, relax=level)
                counts.append(int(mask.sum()))
                relax = level
                if counts[-1] >= top_k:
                    break
            if relax > 0:
                warnings.append(
                    f"严格条件下仅有 {counts[0]} 个候选，已按第 {relax} 级放宽薪资/经验/城市范围；请确认放宽后的结果。"
                )
        trace.add("hard_constraint_filter", "ok" if counts[-1] > 0 else "no_match", {"candidate_counts": counts, "relax_level": relax})
        if counts[-1] == 0:
            return {
                "status": "no_match",
                "session_id": session_id,
                "message": "当前条件下没有找到可推荐的岗位。",
                "profile": profile,
                "cluster": cluster_info,
                "questions": [
                    "是否增加目标城市（如北京、上海、深圳、杭州）？",
                    "是否将期望薪资下调 10%–20%？",
                    "是否补充技能或接受相关岗位方向？",
                ],
                "warnings": warnings,
                "recommendations": [],
                "trace": trace.to_list(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        history_id = history_user_id if history_user_id in self.engine.train_items_by_user else None
        used_fallback = False
        try:
            recs = self.engine.rank(
                profile, user_id=history_id, method="proposed", top_k=top_k,
                apply_constraints=True, use_mmr=use_mmr, relax=relax,
            )
        except Exception:
            used_fallback = True
            warnings.append("排序模型调用失败，已回退到内容+技能规则排序。")
            scores = self._fallback_scores(profile)
            scores = np.where(self.builder.hard_mask(profile, relax=relax), scores, -np.inf)
            order = [int(i) for i in np.argsort(-scores) if np.isfinite(scores[int(i)])][:top_k]
            recs = [{"job_index": i, "job_id": self.jobs.iloc[i]["job_id"], "score": float(scores[i])} for i in order]
        trace.add("learned_ranking", "fallback" if used_fallback else "ok", {"returned": len(recs)})
        if not recs:
            return {
                "status": "no_match",
                "session_id": session_id,
                "message": "排序后没有得到可用岗位，请调整条件。",
                "profile": profile,
                "cluster": cluster_info,
                "questions": ["是否放宽城市或薪资条件？", "是否补充技能或目标方向？"],
                "warnings": warnings,
                "recommendations": [],
                "trace": trace.to_list(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        scores = [float(r["score"]) for r in recs]
        top_score = scores[0]
        margin = top_score - (scores[4] if len(scores) > 4 else scores[-1])
        confidence = self._confidence(top_score, margin)
        if confidence == "low":
            warnings.append("候选岗位的排序区分度较低，建议补充技能/城市偏好或进行人工复核。")
        recommendations = [
            self._recommendation(profile, r["job_index"], r["score"], cluster_info)
            for r in recs
        ]
        for i, rec in enumerate(recommendations, start=1):
            rec["rank"] = i
        trace.add("explanation_generation", "ok", {"explanations": len(recommendations)})
        trace.add("confidence_check", confidence, {"top_score": round(top_score, 4), "margin": round(margin, 4)})
        return {
            "status": "ok",
            "session_id": session_id,
            "message": f"已为「{cluster_info['name']}」画像生成 {len(recommendations)} 个岗位建议。",
            "profile": profile,
            "cluster": cluster_info,
            "recommendations": recommendations,
            "warnings": warnings,
            "questions": [],
            "confidence": confidence,
            "next_actions": [
                {"action": "adjust_constraints", "label": "放宽城市或薪资"},
                {"action": "add_skills", "label": "补充技能"},
                {"action": "export_requires_confirmation", "label": "确认后导出投递清单"},
            ],
            "trace": trace.to_list(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def record_feedback(self, session_id: str, job_id: str, action: str, note: str = "") -> dict:
        return self.feedback_store.record(session_id, job_id, action, note)

    def export_csv(self, response: dict, path: Path | str) -> Path:
        if response.get("status") != "ok" or not response.get("recommendations"):
            raise ValueError("only successful responses can be exported")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "job_id", "title", "company", "city", "salary", "family", "score", "match_level", "explanation"])
            for rec in response["recommendations"]:
                writer.writerow([
                    rec["rank"], rec["job_id"], rec["title"], rec["company"], rec["city"],
                    rec["salary"], rec["family"], rec["score"], rec["match_level"], rec["explanation"],
                ])
        return out

    def save_session(self, response: dict, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        return out


import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .taxonomy import PROJECT_ROOT, Taxonomy, get_taxonomy


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

COMPANIES = [
    "星云科技", "远航数据", "智联未来", "云帆网络", "极光智能", "山海软件", "青禾科技", "北辰信息",
    "微光互联", "深蓝数据", "拾光科技", "同舟网络", "晨曦智能", "万象云", "知行科技", "原点软件",
    "光年信息", "匠心网络", "灵犀智能", "峰瑞科技", "木棉数据", "澜舟科技", "白泽智能", "长风软件",
    "一川网络", "麦田科技", "新芽数据", "云栖信息", "海纳智能", "星桥科技", "跃迁网络", "敏行软件",
    "恒河数据", "明途科技", "启明信息", "advantage 科技", "博观智能", "简一网络", "澄明数据", "合流科技",
]

INDUSTRIES = ["互联网", "人工智能", "金融科技", "电子商务", "企业服务", "在线教育", "医疗健康", "智能制造", "文化传媒", "物流供应链"]
COMPANY_SIZES = ["20-99人", "100-499人", "500-999人", "1000-9999人", "10000人以上"]
SOFT_SKILLS = ["沟通协作", "逻辑思维", "学习能力", "责任心", "抗压能力", "跨团队协作", "文档表达", "问题拆解"]

RESPONSIBILITIES: Dict[str, List[str]] = {
    "backend": ["负责核心业务服务的设计、开发与维护", "参与接口设计与数据库建模", "优化系统性能与稳定性", "参与代码评审和技术方案讨论"],
    "frontend": ["负责Web端和移动端页面开发", "与设计师协作还原交互与视觉稿", "优化前端性能和兼容性", "参与组件库和工程化建设"],
    "algorithm": ["负责机器学习模型的训练、评估与上线", "参与特征工程和样本建设", "跟踪线上效果并迭代模型", "探索大模型在业务中的应用"],
    "data": ["负责业务数据的清洗、分析与可视化", "搭建核心指标体系并输出分析报告", "设计A/B实验并评估效果", "与产品和运营协作定位问题"],
    "qa": ["负责产品功能测试和回归测试", "编写测试用例并维护自动化脚本", "跟踪缺陷闭环并推动质量改进", "参与接口和性能测试"],
    "devops": ["负责云资源的部署、监控与故障处理", "建设持续集成和持续交付流程", "优化容器化平台和发布效率", "参与安全加固和容量规划"],
    "product": ["负责需求调研、分析和产品方案设计", "撰写PRD并推动研发落地", "跟踪产品数据并持续迭代", "协调设计、研发和运营团队"],
    "operations": ["负责用户/内容/活动运营策略制定", "策划并执行增长活动", "分析运营数据并复盘效果", "维护社群和用户关系"],
    "design": ["负责产品界面和交互设计", "输出设计规范和高保真原型", "参与用户体验研究和可用性测试", "与前端协作保证设计还原"],
    "hr": ["负责招聘需求分析和人才寻访", "组织面试并维护候选人体验", "参与绩效、员工关系和培训项目", "维护人力资源数据和流程"],
    "finance": ["负责日常会计核算和账务处理", "编制财务报表并进行财务分析", "参与预算、成本和税务管理", "配合审计和内部控制工作"],
    "sales": ["负责客户开发和商务谈判", "维护客户关系并完成销售目标", "开展市场调研和竞品分析", "协同市场团队制定推广方案"],
}

CITY_WEIGHTS = {
    "北京": 0.18, "上海": 0.18, "深圳": 0.16, "杭州": 0.12, "广州": 0.1, "成都": 0.08,
    "武汉": 0.05, "南京": 0.05, "西安": 0.03, "苏州": 0.03, "重庆": 0.01, "长沙": 0.01,
}


def _join(values: List[str]) -> str:
    return "|".join(values)


def _split(value: str) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [x for x in str(value).split("|") if x]


def _weighted_choice(rng: np.random.Generator, items: List[str], weights: List[float] | None = None) -> str:
    probs = np.array(weights, dtype=float) if weights is not None else None
    if probs is not None:
        probs = probs / probs.sum()
    return str(rng.choice(items, p=probs))


def _sample_skills(rng: np.random.Generator, tax: Taxonomy, family_id: str, n_skills: int) -> List[str]:
    core = tax.core_skills(family_id)
    optional = tax.optional_skills(family_id)
    n_core = min(len(core), max(2, int(round(n_skills * 0.6))))
    chosen = list(rng.choice(core, size=n_core, replace=False))
    pool = [s for s in core + optional if s not in chosen]
    if pool and len(chosen) < n_skills:
        k = min(len(pool), n_skills - len(chosen))
        chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    related = sorted(tax.related_families(family_id))
    if related and rng.random() < 0.35:
        rel = str(rng.choice(related))
        rel_pool = [s for s in tax.all_family_skills(rel) if s not in chosen]
        if rel_pool:
            chosen.append(str(rng.choice(rel_pool)))
    return list(dict.fromkeys(str(s) for s in chosen))


def _make_description(rng: np.random.Generator, row: dict, family_name: str) -> str:
    duties = RESPONSIBILITIES.get(row["family"], ["负责相关业务工作"])
    chosen = rng.choice(duties, size=min(3, len(duties)), replace=False).tolist()
    skills = "、".join(row["skills"][:6])
    soft = "、".join(rng.choice(SOFT_SKILLS, size=2, replace=False).tolist())
    return (
        f"{row['company']}正在招聘{row['title']}，工作地点位于{row['city']}。"
        f"岗位职责：{'；'.join(chosen)}。"
        f"任职要求：熟悉{skills}；{row['experience_min']}年以上相关经验；"
        f"{row['education']}及以上学历；具备良好的{soft}。"
        f"我们提供有竞争力的薪酬和{family_name}方向的成长空间。"
    )


def generate_jobs(n_jobs: int = 3000, seed: int = 42, tax: Taxonomy | None = None) -> pd.DataFrame:
    tax = tax or get_taxonomy()
    rng = np.random.default_rng(seed)
    families = tax.family_ids
    counts = {fid: n_jobs // len(families) for fid in families}
    for fid in families[: n_jobs % len(families)]:
        counts[fid] += 1
    cities = list(CITY_WEIGHTS.keys())
    city_probs = np.array([CITY_WEIGHTS[c] for c in cities], dtype=float)
    city_probs = city_probs / city_probs.sum()
    rows: List[dict] = []
    job_no = 1
    for fid in families:
        fam = tax.families[fid]
        for _ in range(counts[fid]):
            title = str(rng.choice(fam["titles"]))
            senior = "高级" in title or "经理" in title
            experience_min = int(rng.choice([5, 6, 8])) if senior else int(rng.choice([0, 1, 3, 3, 5]))
            if senior and rng.random() < 0.5:
                experience_min = 5
            level = min(experience_min / 8.0, 1.0)
            base_low, base_high = fam["salary_base"]
            mid = base_low + (base_high - base_low) * (0.15 + 0.75 * level)
            mid *= float(rng.normal(1.0, 0.08))
            salary_min = max(4.0, round(mid * 0.82 * 2) / 2)
            salary_max = max(salary_min + 1.0, round(mid * 1.18 * 2) / 2)
            edu_probs = [0.08, 0.72, 0.2] if fid != "algorithm" else [0.02, 0.58, 0.4]
            education = str(rng.choice(["大专", "本科", "硕士"], p=edu_probs))
            n_skills = int(rng.integers(5, 10))
            skills = _sample_skills(rng, tax, fid, n_skills)
            city = str(rng.choice(cities, p=city_probs))
            row = {
                "job_id": f"J{job_no:05d}",
                "title": title,
                "company": str(rng.choice(COMPANIES)),
                "city": city,
                "salary_min": float(salary_min),
                "salary_max": float(salary_max),
                "experience_min": experience_min,
                "education": education,
                "remote": bool(rng.random() < 0.12),
                "family": fid,
                "family_name": fam["name"],
                "skills": skills,
                "company_size": str(rng.choice(COMPANY_SIZES, p=[0.2, 0.3, 0.2, 0.2, 0.1])),
                "industry": str(rng.choice(INDUSTRIES)),
                "posted_days_ago": int(rng.integers(0, 61)),
            }
            row["description"] = _make_description(rng, row, fam["name"])
            rows.append(row)
            job_no += 1
    return pd.DataFrame(rows)


def generate_users(n_users: int = 400, seed: int = 123, tax: Taxonomy | None = None) -> pd.DataFrame:
    tax = tax or get_taxonomy()
    rng = np.random.default_rng(seed)
    families = tax.family_ids
    cities = list(CITY_WEIGHTS.keys())
    city_probs = np.array([CITY_WEIGHTS[c] for c in cities], dtype=float)
    city_probs = city_probs / city_probs.sum()
    exp_choices = [0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 8]
    rows: List[dict] = []
    for i in range(1, n_users + 1):
        fid = str(rng.choice(families))
        fam = tax.families[fid]
        experience = int(rng.choice(exp_choices))
        if experience <= 1:
            edu_probs = [0.15, 0.8, 0.05]
        elif experience <= 4:
            edu_probs = [0.08, 0.75, 0.17]
        else:
            edu_probs = [0.03, 0.62, 0.35]
        education = str(rng.choice(["大专", "本科", "硕士"], p=edu_probs))
        n_skills = int(rng.integers(3, 9))
        skills = _sample_skills(rng, tax, fid, n_skills)
        current_city = str(rng.choice(cities, p=city_probs))
        n_targets = int(rng.integers(2, 5))
        target_cities = [current_city]
        remaining = [c for c in cities if c != current_city]
        extra = rng.choice(remaining, size=min(n_targets - 1, len(remaining)), replace=False, p=None).tolist()
        target_cities.extend(str(c) for c in extra)
        target_cities = list(dict.fromkeys(target_cities))
        base_low, base_high = fam["salary_base"]
        expected = base_low + (base_high - base_low) * min(experience / 8.0, 1.0)
        expected *= float(rng.normal(1.0, 0.1))
        expected = max(4.0, round(expected * 2) / 2)
        preferred = [fid]
        related = sorted(tax.related_families(fid))
        if related and rng.random() < 0.55:
            preferred.append(str(rng.choice(related)))
        remote_pref = bool(rng.random() < 0.15)
        skill_text = "、".join(skills)
        bio = (
            f"我有{experience}年{tax.family_name(fid)}相关经验，熟悉{skill_text}。"
            f"目前居住在{current_city}，希望在{target_cities[0]}等地寻找{tax.family_name(fid)}方向的工作，"
            f"期望月薪{expected:.1f}k左右。"
        )
        rows.append({
            "user_id": f"U{i:04d}",
            "name": f"求职者{i:04d}",
            "city": current_city,
            "target_cities": target_cities,
            "expected_salary": float(expected),
            "experience_years": experience,
            "education": education,
            "remote_pref": remote_pref,
            "skills": skills,
            "preferred_families": preferred,
            "bio": bio,
        })
    return pd.DataFrame(rows)


def _skill_overlap(user_skills: List[str], job_skills: List[str]) -> float:
    a, b = set(user_skills), set(job_skills)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _constraint_mask(user: dict, jobs: pd.DataFrame, tax: Taxonomy, relax: int = 0) -> np.ndarray:
    salary_ratio = 0.9 - 0.1 * min(relax, 2)
    exp_slack = 1 + relax
    cities = set(user["target_cities"])
    if relax >= 3:
        cities = set(tax.cities)
    mask = (
        jobs["city"].isin(cities).to_numpy()
        & (jobs["salary_max"].to_numpy() >= user["expected_salary"] * salary_ratio)
        & (jobs["experience_min"].to_numpy() <= user["experience_years"] + exp_slack)
        & (jobs["education"].map(tax.education_rank).to_numpy() <= tax.education_rank(user["education"]))
    )
    return mask


def generate_interactions(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    seed: int = 2024,
    tax: Taxonomy | None = None,
    min_candidates: int = 120,
) -> pd.DataFrame:
    tax = tax or get_taxonomy()
    rng = np.random.default_rng(seed)
    job_family = jobs["family"].to_numpy()
    job_skills = jobs["skills"].tolist()
    job_city = jobs["city"].to_numpy()
    job_salary_mid = ((jobs["salary_min"] + jobs["salary_max"]) / 2).to_numpy()
    job_salary_max = jobs["salary_max"].to_numpy()
    job_exp = jobs["experience_min"].to_numpy()
    job_edu = jobs["education"].map(tax.education_rank).to_numpy()
    job_remote = jobs["remote"].to_numpy()
    attractiveness = rng.normal(0, 0.55, size=len(jobs))
    rows: List[dict] = []
    for user in users.to_dict("records"):
        mask = _constraint_mask(user, jobs, tax, relax=0)
        relax = 0
        while mask.sum() < min_candidates and relax < 3:
            relax += 1
            mask = _constraint_mask(user, jobs, tax, relax=relax)
        candidate_idx = np.flatnonzero(mask)
        if len(candidate_idx) == 0:
            continue
        user_skills = set(user["skills"])
        preferred = set(user["preferred_families"])
        related: set[str] = set()
        for fid in preferred:
            related |= tax.related_families(fid)
        utility = np.zeros(len(jobs), dtype=float)
        for idx in candidate_idx:
            fam_affinity = 1.8 if job_family[idx] in preferred else (0.5 if job_family[idx] in related else -0.45)
            overlap = _skill_overlap(user["skills"], job_skills[idx])
            city_match = 1.0 if job_city[idx] == user["city"] else 0.55
            salary_fit = max(-1.0, min(1.0, 1.0 - abs(job_salary_mid[idx] - user["expected_salary"]) / max(user["expected_salary"], 1.0)))
            salary_bonus = 0.25 if job_salary_max[idx] >= user["expected_salary"] else -0.3
            exp_gap = job_exp[idx] - user["experience_years"]
            exp_fit = 0.45 if exp_gap <= 0 else -0.45 * exp_gap
            edu_fit = 0.2 if job_edu[idx] <= tax.education_rank(user["education"]) else -0.6
            remote_fit = 0.25 if (user["remote_pref"] and job_remote[idx]) else 0.0
            utility[idx] = (
                fam_affinity + 2.6 * overlap + 0.9 * city_match + 0.6 * salary_fit + salary_bonus
                + exp_fit + edu_fit + remote_fit + 0.4 * attractiveness[idx]
            )
        utility[candidate_idx] += rng.normal(0, 0.45, size=len(candidate_idx))
        scaled = (utility[candidate_idx] - np.max(utility[candidate_idx])) / 0.9
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        n_interactions = int(rng.integers(25, 41))
        n_interactions = min(n_interactions, len(candidate_idx))
        picked = rng.choice(candidate_idx, size=n_interactions, replace=False, p=probs)
        for idx in picked:
            event = str(rng.choice(["view", "click", "apply"], p=[0.42, 0.33, 0.25]))
            weight = {"view": 1, "click": 2, "apply": 3}[event]
            rows.append({
                "user_id": user["user_id"],
                "job_id": jobs.iloc[int(idx)]["job_id"],
                "event_type": event,
                "weight": weight,
                "days_ago": int(rng.integers(0, 91)),
            })
    return pd.DataFrame(rows)


def save_dataset(jobs: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jobs_out = jobs.copy()
    users_out = users.copy()
    jobs_out["skills"] = jobs_out["skills"].map(_join)
    users_out["skills"] = users_out["skills"].map(_join)
    users_out["target_cities"] = users_out["target_cities"].map(_join)
    users_out["preferred_families"] = users_out["preferred_families"].map(_join)
    jobs_out.to_csv(PROCESSED_DIR / "jobs.csv", index=False, encoding="utf-8-sig")
    users_out.to_csv(PROCESSED_DIR / "users.csv", index=False, encoding="utf-8-sig")
    interactions.to_csv(PROCESSED_DIR / "interactions.csv", index=False, encoding="utf-8-sig")


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = pd.read_csv(PROCESSED_DIR / "jobs.csv", encoding="utf-8-sig")
    users = pd.read_csv(PROCESSED_DIR / "users.csv", encoding="utf-8-sig")
    interactions = pd.read_csv(PROCESSED_DIR / "interactions.csv", encoding="utf-8-sig")
    jobs["skills"] = jobs["skills"].map(_split)
    users["skills"] = users["skills"].map(_split)
    users["target_cities"] = users["target_cities"].map(_split)
    users["preferred_families"] = users["preferred_families"].map(_split)
    jobs["remote"] = jobs["remote"].astype(bool)
    users["remote_pref"] = users["remote_pref"].astype(bool)
    return jobs, users, interactions


def build_dataset(n_jobs: int = 3000, n_users: int = 400, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tax = get_taxonomy()
    jobs = generate_jobs(n_jobs=n_jobs, seed=seed, tax=tax)
    users = generate_users(n_users=n_users, seed=seed + 1000, tax=tax)
    interactions = generate_interactions(users, jobs, seed=seed + 2000, tax=tax)
    save_dataset(jobs, users, interactions)
    stats = {
        "n_jobs": int(len(jobs)),
        "n_users": int(len(users)),
        "n_interactions": int(len(interactions)),
        "density": float(len(interactions) / (len(jobs) * len(users))),
        "jobs_per_family": jobs.groupby("family_name").size().to_dict(),
        "users_per_family": users.explode("preferred_families").groupby("preferred_families").size().to_dict(),
        "cities": jobs.groupby("city").size().to_dict(),
        "education_jobs": jobs.groupby("education").size().to_dict(),
        "experience_jobs": jobs.groupby("experience_min").size().to_dict(),
        "salary_min": float(jobs["salary_min"].min()),
        "salary_max": float(jobs["salary_max"].max()),
        "remote_rate": float(jobs["remote"].mean()),
        "events": interactions.groupby("event_type").size().to_dict(),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump({"n_jobs": n_jobs, "n_users": n_users, "seed": seed, "stats": stats}, f, ensure_ascii=False, indent=2)
    return jobs, users, interactions




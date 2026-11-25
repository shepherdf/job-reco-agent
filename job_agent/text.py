import re
from typing import Iterable, List

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .taxonomy import Taxonomy, get_taxonomy

jieba.setLogLevel(20)

STOPWORDS = set("""
的 了 和 与 及 或 在 是 有 为 对 从 到 中 上 下 我们 你们 他们 我 你 他 她 它
一个 一种 一些 相关 负责 参与 进行 以及 具备 良好 以上 要求 职责 岗位 招聘 工作 地点 位于
熟悉 优先 能够 能力 经验 方向 提供 竞争 力 成长 空间 公司 团队 业务 产品 服务 系统 平台
使用 通过 基于 包括 并 等 年 月 日 本科 硕士 大专 博士 以上学历 相关经验
""".split())


def _clean_token(token: str) -> str:
    token = token.strip().lower()
    if not token or token in STOPWORDS:
        return ""
    if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
        return ""
    if len(token) == 1 and not re.fullmatch(r"[a-z0-9]", token):
        return ""
    if not re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", token):
        return ""
    return token


class ChineseTextModel:
    def __init__(self, tax: Taxonomy | None = None, mode: str = "word"):
        self.tax = tax or get_taxonomy()
        self.mode = mode
        self.vectorizer = None
        self.job_matrix = None
        self.job_texts: List[str] = []
        if mode == "word":
            for word in self.tax.jieba_words():
                jieba.add_word(word, freq=20000)
            for fid in self.tax.family_ids:
                jieba.add_word(self.tax.family_name(fid), freq=30000)
                for title in self.tax.families[fid]["titles"]:
                    jieba.add_word(title, freq=20000)
        self._fit_vectorizer()

    def _fit_vectorizer(self) -> None:
        if self.mode == "word":
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize,
                token_pattern=None,
                lowercase=False,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True,
                max_features=8000,
            )
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=12000,
            )

    def tokenize(self, text: str) -> List[str]:
        tokens = [_clean_token(t) for t in jieba.lcut(str(text))]
        return [t for t in tokens if t]

    @staticmethod
    def build_job_text(job: dict) -> str:
        skills = job.get("skills", [])
        if isinstance(skills, str):
            skills = [s for s in skills.split("|") if s]
        return " ".join([
            str(job.get("title", "")),
            str(job.get("family_name", "")),
            str(job.get("city", "")),
            str(job.get("industry", "")),
            " ".join(skills),
            str(job.get("description", "")),
        ])

    @staticmethod
    def build_user_text(user: dict) -> str:
        skills = user.get("skills", [])
        if isinstance(skills, str):
            skills = [s for s in skills.split("|") if s]
        families = user.get("preferred_families", [])
        if isinstance(families, str):
            families = [s for s in families.split("|") if s]
        return " ".join([
            " ".join(skills),
            " ".join(families),
            str(user.get("bio", "")),
            str(user.get("city", "")),
        ])

    def fit_jobs(self, jobs: pd.DataFrame) -> "ChineseTextModel":
        self.job_texts = [self.build_job_text(row) for row in jobs.to_dict("records")]
        self.job_matrix = self.vectorizer.fit_transform(self.job_texts)
        return self

    def transform_users(self, users: pd.DataFrame) -> np.ndarray:
        texts = [self.build_user_text(row) for row in users.to_dict("records")]
        return self.vectorizer.transform(texts)

    def transform_text(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text])

    def job_scores(self, user_vector) -> np.ndarray:
        scores = self.job_matrix @ user_vector.T
        return np.asarray(scores.toarray()).ravel()

    def text_scores(self, text: str) -> np.ndarray:
        return self.job_scores(self.transform_text(text))

    def job_job_similarity(self, candidate_idx: Iterable[int], selected_idx: Iterable[int]) -> np.ndarray:
        c = self.job_matrix[list(candidate_idx)]
        s = self.job_matrix[list(selected_idx)]
        return np.asarray((c @ s.T).toarray())


def extract_salary(text: str) -> float | None:
    if not text:
        return None
    with_unit = re.findall(r"(\d+(?:\.\d+)?)\s*(万|w|k|千)", text, flags=re.IGNORECASE)
    values: List[float] = []
    for raw, unit in with_unit:
        value = float(raw)
        unit = unit.lower()
        if unit in {"万", "w"}:
            value *= 10
        if 2 <= value <= 150:
            values.append(value)
    if values:
        return values[0]
    contextual = re.findall(r"(?:月薪|薪资|期望|工资|待遇)[^\d]{0,8}(\d+(?:\.\d+)?)", text)
    for raw in contextual:
        value = float(raw)
        if value >= 1000:
            value = value / 1000
        if 2 <= value <= 150:
            return value
    return None


def extract_experience(text: str) -> int | None:
    if not text:
        return None
    if "应届" in text or "无经验" in text or "0年" in text:
        return 0
    m = re.search(r"(\d+)\s*年", text)
    return int(m.group(1)) if m else None


def extract_cities(text: str, cities: Iterable[str]) -> List[str]:
    found = []
    for order, city in enumerate(cities):
        pos = text.find(city)
        if pos >= 0:
            found.append((pos, order, city))
    found.sort()
    return [city for _, _, city in found]


def extract_city(text: str, cities: Iterable[str]) -> str | None:
    values = extract_cities(text, cities)
    return values[0] if values else None


def extract_education(text: str, levels: Iterable[str]) -> str | None:
    for level in ["博士", "硕士", "本科", "大专"]:
        if level in text and level in levels:
            return level
    return None


def extract_remote(text: str) -> bool:
    return any(k in text.lower() for k in ["远程", "remote", "居家"])


def infer_families(text: str, tax: Taxonomy | None = None) -> List[str]:
    tax = tax or get_taxonomy()
    scores = {fid: 0.0 for fid in tax.family_ids}
    for fid in tax.family_ids:
        fam = tax.families[fid]
        if fam["name"] in text:
            scores[fid] += 3.0
        for title in fam["titles"]:
            if title in text:
                scores[fid] += 1.5
    for skill in tax.extract_skills(text):
        for fid in tax.skill_family_map.get(skill, set()):
            scores[fid] += 0.6
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [fid for fid, score in ranked if score > 0][:3]

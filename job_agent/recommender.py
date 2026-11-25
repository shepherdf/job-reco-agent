from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from .clustering import UserClusterer
from .features import PAIR_FEATURE_NAMES, FeatureBuilder
from .taxonomy import Taxonomy, get_taxonomy
from .text import ChineseTextModel

DEFAULT_RANKER_FEATURES = [
    "content_sim",
    "cluster_skill_sim",
    "cluster_family_affinity",
    "skill_jaccard",
    "job_skill_coverage",
    "city_match",
    "salary_fit",
    "exp_fit",
    "edu_fit",
    "remote_fit",
    "popularity",
    "is_preferred_family",
    "is_related_family",
]


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _maxnorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    hi = float(np.max(values))
    if hi <= 1e-12:
        return np.zeros_like(values)
    return np.clip(values / hi, 0.0, 1.0)


class PopularityRecommender:
    def __init__(self):
        self.scores = None

    def fit(self, train: pd.DataFrame, jobs: pd.DataFrame) -> "PopularityRecommender":
        counts = np.zeros(len(jobs), dtype=float)
        index = {jid: i for i, jid in enumerate(jobs["job_id"])}
        for jid, weight in zip(train["job_id"], train["weight"]):
            i = index.get(jid)
            if i is not None:
                counts[i] += float(weight)
        self.scores = _maxnorm(np.log1p(counts))
        return self

    def score(self, user_id: str | None = None) -> np.ndarray:
        return self.scores.copy()


class ContentRecommender:
    def __init__(self, text_model: ChineseTextModel, users: pd.DataFrame):
        self.text_model = text_model
        self.users = users.reset_index(drop=True)
        self.user_matrix = None

    def fit(self, jobs: pd.DataFrame) -> "ContentRecommender":
        self.text_model.fit_jobs(jobs)
        self.user_matrix = self.text_model.transform_users(self.users)
        return self

    def score_user(self, user: dict, user_id: str | None = None) -> np.ndarray:
        vector = self.text_model.transform_text(self.text_model.build_user_text(user))
        return self.text_model.job_scores(vector)


class ItemCFRecommender:
    def __init__(self):
        self.user_index: Dict[str, int] = {}
        self.job_index: Dict[str, int] = {}
        self.matrix = None
        self.similarity = None
        self.n_jobs = 0

    def fit(self, train: pd.DataFrame, jobs: pd.DataFrame) -> "ItemCFRecommender":
        self.user_index = {uid: i for i, uid in enumerate(sorted(train["user_id"].unique()))}
        self.job_index = {jid: i for i, jid in enumerate(jobs["job_id"])}
        self.n_jobs = len(jobs)
        rows, cols, vals = [], [], []
        for row in train.itertuples(index=False):
            u = self.user_index.get(row.user_id)
            j = self.job_index.get(row.job_id)
            if u is not None and j is not None:
                rows.append(u)
                cols.append(j)
                vals.append(float(row.weight))
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.user_index), self.n_jobs))
        item_user = normalize(self.matrix.T, norm="l2", axis=1)
        self.similarity = (item_user @ item_user.T).tocsr()
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()
        return self

    def score_user(self, user_id: str | None) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(self.n_jobs, dtype=float)
        row = self.matrix[self.user_index[user_id]]
        scores = (row @ self.similarity).toarray().ravel()
        return _maxnorm(scores)


class SVDRecommender:
    def __init__(self, n_components: int = 32, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.user_index: Dict[str, int] = {}
        self.user_factors = None
        self.item_factors = None
        self.n_jobs = 0

    def fit(self, train: pd.DataFrame, jobs: pd.DataFrame) -> "SVDRecommender":
        self.user_index = {uid: i for i, uid in enumerate(sorted(train["user_id"].unique()))}
        job_index = {jid: i for i, jid in enumerate(jobs["job_id"])}
        self.n_jobs = len(jobs)
        rows, cols, vals = [], [], []
        for row in train.itertuples(index=False):
            u = self.user_index.get(row.user_id)
            j = job_index.get(row.job_id)
            if u is not None and j is not None:
                rows.append(u)
                cols.append(j)
                vals.append(float(row.weight))
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.user_index), self.n_jobs))
        k = int(min(self.n_components, max(2, min(matrix.shape) - 1)))
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        return self

    def score_user(self, user_id: str | None) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(self.n_jobs, dtype=float)
        scores = self.user_factors[self.user_index[user_id]] @ self.item_factors.T
        return _minmax(scores)


class ClusterPriorRecommender:
    def __init__(self, clusterer: UserClusterer, builder: FeatureBuilder):
        self.clusterer = clusterer
        self.builder = builder

    def score_user(self, user: dict) -> np.ndarray:
        cluster = self.clusterer.predict(user, self.builder)
        skill_sim = self.clusterer.job_skill_similarity(cluster, self.builder)
        family_aff = self.clusterer.job_family_affinity(cluster, self.builder)
        return _minmax(0.6 * skill_sim + 0.4 * family_aff)

    def cluster_for(self, user: dict) -> int:
        return self.clusterer.predict(user, self.builder)


class LearnedRanker:
    def __init__(self, feature_names: List[str], c_value: float = 1.0):
        self.feature_names = feature_names
        self.feature_indices = [PAIR_FEATURE_NAMES.index(n) for n in feature_names]
        self.c_value = c_value
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "LearnedRanker":
        Xs = X[:, self.feature_indices]
        self.scaler = StandardScaler()
        Xz = self.scaler.fit_transform(Xs)
        self.model = LogisticRegression(
            C=self.c_value,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        self.model.fit(Xz, y)
        return self

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        Xs = X[:, self.feature_indices]
        Xz = self.scaler.transform(Xs)
        return self.model.predict_proba(Xz)[:, 1]

    def contributions(self, x: np.ndarray) -> List[dict]:
        Xs = np.asarray(x, dtype=float)[self.feature_indices].reshape(1, -1)
        z = self.scaler.transform(Xs)[0]
        coef = self.model.coef_[0]
        values = coef * z
        order = np.argsort(-values)
        result = []
        for i in order:
            result.append({
                "feature": self.feature_names[i],
                "contribution": float(values[i]),
                "value": float(Xs[0, i]),
            })
        return result

    def save(self, path: Path | str) -> None:
        joblib.dump({
            "feature_names": self.feature_names,
            "c_value": self.c_value,
            "scaler": self.scaler,
            "model": self.model,
        }, path)

    @classmethod
    def load(cls, path: Path | str) -> "LearnedRanker":
        state = joblib.load(path)
        obj = cls(state["feature_names"], state["c_value"])
        obj.scaler = state["scaler"]
        obj.model = state["model"]
        return obj


class RecommendationEngine:
    def __init__(
        self,
        jobs: pd.DataFrame,
        users: pd.DataFrame,
        builder: FeatureBuilder,
        text_model: ChineseTextModel,
        clusterer: UserClusterer,
        popularity: PopularityRecommender,
        cf: ItemCFRecommender,
        svd: SVDRecommender,
        train_interactions: pd.DataFrame | None = None,
        content_blend: float = 0.7,
    ):
        self.content_blend = float(min(1.0, max(0.0, content_blend)))
        self.jobs = jobs.reset_index(drop=True)
        self.users = users.reset_index(drop=True)
        self.builder = builder
        self.text_model = text_model
        self.clusterer = clusterer
        self.popularity = popularity
        self.cf = cf
        self.svd = svd
        self.cluster_prior = ClusterPriorRecommender(clusterer, builder)
        self.ranker: LearnedRanker | None = None
        self.train_interactions = train_interactions
        self.job_id_to_idx = {jid: i for i, jid in enumerate(self.jobs["job_id"])}
        self.user_by_id = {row["user_id"]: row for row in self.users.to_dict("records")}
        self.train_items_by_user: Dict[str, set] = {}
        if train_interactions is not None:
            for uid, group in train_interactions.groupby("user_id"):
                self.train_items_by_user[uid] = {self.job_id_to_idx[j] for j in group["job_id"] if j in self.job_id_to_idx}

    def score_arrays(self, user: dict, user_id: str | None = None) -> Dict[str, np.ndarray]:
        content = self.text_model.text_scores(self.text_model.build_user_text(user))
        cf = self.cf.score_user(user_id)
        svd = self.svd.score_user(user_id)
        cluster = self.cluster_prior.score_user(user)
        pop = self.popularity.score(user_id)
        return {"content": content, "cf": cf, "svd": svd, "cluster": cluster, "popularity": pop}

    def method_scores(self, method: str, arrays: Dict[str, np.ndarray], user: dict) -> np.ndarray:
        if method == "popularity":
            return arrays["popularity"]
        if method == "content":
            return arrays["content"]
        if method == "itemcf":
            return arrays["cf"]
        if method == "svd":
            return arrays["svd"]
        if method == "cluster":
            return arrays["cluster"]
        if method == "hybrid":
            return (
                0.45 * arrays["content"] + 0.25 * arrays["cf"] + 0.15 * arrays["svd"]
                + 0.10 * arrays["cluster"] + 0.05 * arrays["popularity"]
            )
        if method == "proposed":
            if self.ranker is None:
                return (
                    0.45 * arrays["content"] + 0.25 * arrays["cf"] + 0.15 * arrays["svd"]
                    + 0.10 * arrays["cluster"] + 0.05 * arrays["popularity"]
                )
            X = self.builder.build_pair_matrix(
                user, arrays["content"], arrays["cf"], arrays["svd"], arrays["cluster"],
                self.clusterer.job_family_affinity(self.cluster_prior.cluster_for(user), self.builder),
                arrays["popularity"],
            )
            ranker_scores = self.ranker.score_matrix(X)
            return self.content_blend * arrays["content"] + (1.0 - self.content_blend) * ranker_scores
        raise ValueError(f"unknown method: {method}")

    def score_all(self, user: dict, user_id: str | None, method: str, apply_constraints: bool = True, relax: int = 0) -> np.ndarray:
        arrays = self.score_arrays(user, user_id)
        scores = self.method_scores(method, arrays, user)
        if apply_constraints:
            mask = self.builder.hard_mask(user, relax=relax)
            scores = np.where(mask, scores, -np.inf)
        if user_id in self.train_items_by_user:
            exclude = list(self.train_items_by_user[user_id])
            scores[exclude] = -np.inf
        return scores

    def rank(
        self,
        user: dict,
        user_id: str | None = None,
        method: str = "proposed",
        top_k: int = 10,
        apply_constraints: bool = True,
        use_mmr: bool = True,
        relax: int = 0,
        mmr_lambda: float = 0.8,
    ) -> List[dict]:
        arrays = self.score_arrays(user, user_id)
        scores = self.method_scores(method, arrays, user)
        if apply_constraints:
            mask = self.builder.hard_mask(user, relax=relax)
            scores = np.where(mask, scores, -np.inf)
        if user_id in self.train_items_by_user:
            scores[list(self.train_items_by_user[user_id])] = -np.inf
        if not np.isfinite(scores).any():
            return []
        pool_size = max(top_k * 10, 60)
        pool = np.argsort(-scores)[:pool_size]
        pool = pool[np.isfinite(scores[pool])]
        if use_mmr and len(pool) > top_k:
            selected = self._mmr(pool, scores, self.text_model, top_k, mmr_lambda)
        else:
            selected = pool[:top_k].tolist()
        results = []
        for idx in selected:
            results.append({
                "job_index": int(idx),
                "job_id": self.jobs.iloc[int(idx)]["job_id"],
                "score": float(scores[int(idx)]),
            })
        return results

    def _mmr(self, pool: np.ndarray, scores: np.ndarray, text_model: ChineseTextModel, top_k: int, lambda_: float) -> List[int]:
        selected: List[int] = []
        company_counts: Dict[str, int] = {}
        normalized = _minmax(scores[pool])
        score_map = {int(idx): float(normalized[i]) for i, idx in enumerate(pool)}
        while len(selected) < top_k and len(selected) < len(pool):
            best_idx = None
            best_value = -1e9
            for idx in pool:
                idx = int(idx)
                if idx in selected:
                    continue
                company = str(self.jobs.iloc[idx]["company"])
                if company_counts.get(company, 0) >= 2:
                    continue
                if selected:
                    sim = text_model.job_job_similarity([idx], selected).ravel()
                    max_sim = float(np.max(sim)) if len(sim) else 0.0
                else:
                    max_sim = 0.0
                value = lambda_ * score_map[idx] - (1.0 - lambda_) * max_sim
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            company = str(self.jobs.iloc[best_idx]["company"])
            company_counts[company] = company_counts.get(company, 0) + 1
        return selected

    def make_training_pairs(
        self,
        train: pd.DataFrame,
        all_interactions: pd.DataFrame,
        users: pd.DataFrame,
        neg_per_pos: int = 3,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        rows: List[np.ndarray] = []
        labels: List[int] = []
        job_family = self.jobs["family"].to_numpy()
        for user in users.to_dict("records"):
            uid = user["user_id"]
            pos_ids = train.loc[train["user_id"] == uid, "job_id"].tolist()
            if not pos_ids:
                continue
            pos_idx = [self.job_id_to_idx[j] for j in pos_ids if j in self.job_id_to_idx]
            if not pos_idx:
                continue
            arrays = self.score_arrays(user, uid)
            interacted = set(all_interactions.loc[all_interactions["user_id"] == uid, "job_id"])
            mask = self.builder.hard_mask(user, relax=0)
            pool = np.array([i for i in range(len(self.jobs)) if mask[i] and self.jobs.iloc[i]["job_id"] not in interacted])
            if len(pool) < max(10, len(pos_idx)):
                mask = self.builder.hard_mask(user, relax=2)
                pool = np.array([i for i in range(len(self.jobs)) if mask[i] and self.jobs.iloc[i]["job_id"] not in interacted])
            if len(pool) == 0:
                continue
            preferred = set(user.get("preferred_families", []))
            hard_pool = np.array([i for i in pool if job_family[i] in preferred])
            n_neg = min(len(pool), max(len(pos_idx) * neg_per_pos, 20))
            n_hard = min(len(hard_pool), n_neg // 2)
            chosen: List[int] = []
            if n_hard > 0:
                chosen.extend(rng.choice(hard_pool, size=n_hard, replace=False).tolist())
            remaining = n_neg - len(chosen)
            if remaining > 0:
                rest = np.array([i for i in pool if i not in set(chosen)])
                if len(rest) > 0:
                    chosen.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())
            idx = np.array(pos_idx + chosen, dtype=int)
            fam_aff = self.clusterer.job_family_affinity(self.cluster_prior.cluster_for(user), self.builder)
            X = self.builder.build_pair_matrix(
                user, arrays["content"], arrays["cf"], arrays["svd"], arrays["cluster"], fam_aff, arrays["popularity"]
            )[idx]
            rows.append(X)
            labels.extend([1] * len(pos_idx) + [0] * len(chosen))
        if not rows:
            raise RuntimeError("no training pairs generated")
        return np.vstack(rows), np.asarray(labels, dtype=int)

    def fit_ranker(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        all_interactions: pd.DataFrame,
        users: pd.DataFrame,
        feature_names: List[str] | None = None,
        neg_per_pos: int = 3,
        seed: int = 42,
        tune_c: bool = True,
        c_value: float | None = None,
    ) -> Dict[str, float]:
        feature_names = feature_names or list(DEFAULT_RANKER_FEATURES)
        X_all, y_all = self.make_training_pairs(train, all_interactions, users, neg_per_pos=neg_per_pos, seed=seed)
        if tune_c:
            c_candidates = [0.1, 0.3, 1.0, 3.0]
        else:
            c_candidates = [float(c_value) if c_value is not None else 1.0]
        best_c = c_candidates[0]
        best_ndcg = -1.0
        for c in c_candidates:
            ranker = LearnedRanker(feature_names, c_value=c).fit(X_all, y_all, seed=seed)
            ndcg = self._validation_ndcg(ranker, val, users, k=10)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_c = c
        self.ranker = LearnedRanker(feature_names, c_value=best_c).fit(X_all, y_all, seed=seed)
        return {"c_value": float(best_c), "val_ndcg@10": float(best_ndcg), "train_pairs": int(len(y_all))}

    def _validation_ndcg(self, ranker: LearnedRanker, val: pd.DataFrame, users: pd.DataFrame, k: int = 10) -> float:
        old = self.ranker
        self.ranker = ranker
        scores: List[float] = []
        for user in users.to_dict("records"):
            uid = user["user_id"]
            pos = set(val.loc[val["user_id"] == uid, "job_id"])
            if not pos:
                continue
            ranked = self.score_all(user, uid, "proposed", apply_constraints=True, relax=0)
            order = np.argsort(-ranked)[:k]
            hits = [1 if self.jobs.iloc[int(i)]["job_id"] in pos else 0 for i in order]
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(pos), k)))
            scores.append(dcg / ideal if ideal > 0 else 0.0)
        self.ranker = old
        return float(np.mean(scores)) if scores else 0.0

    def explanation_contributions(self, user: dict, job_index: int) -> List[dict]:
        arrays = self.score_arrays(user, None)
        fam_aff = self.clusterer.job_family_affinity(self.cluster_prior.cluster_for(user), self.builder)
        X = self.builder.build_pair_matrix(
            user, arrays["content"], arrays["cf"], arrays["svd"], arrays["cluster"], fam_aff, arrays["popularity"]
        )
        if self.ranker is not None:
            return self.ranker.contributions(X[job_index])
        return []

    def save_ranker(self, path: Path | str) -> None:
        if self.ranker is None:
            raise RuntimeError("ranker is not fitted")
        self.ranker.save(path)

    def load_ranker(self, path: Path | str) -> None:
        self.ranker = LearnedRanker.load(path)


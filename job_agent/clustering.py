from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .features import FeatureBuilder
from .taxonomy import Taxonomy, get_taxonomy


FAMILY_TO_BASE = {
    "backend": "后端与工程效能型",
    "devops": "后端与工程效能型",
    "qa": "后端与工程效能型",
    "algorithm": "数据与算法型",
    "data": "数据与算法型",
    "frontend": "前端与体验型",
    "design": "前端与体验型",
    "product": "产品与运营型",
    "operations": "产品与运营型",
    "sales": "商业与职能型",
    "hr": "商业与职能型",
    "finance": "商业与职能型",
}


class UserClusterer:
    def __init__(self, tax: Taxonomy | None = None):
        self.tax = tax or get_taxonomy()
        self.kmeans: KMeans | None = None
        self.k: int | None = None
        self.skill_centroids: np.ndarray | None = None
        self.family_affinity: np.ndarray | None = None
        self.profiles: List[dict] = []
        self.metrics: Dict[str, float] = {}
        self.feature_names: List[str] = []

    def fit(
        self,
        users: pd.DataFrame,
        builder: FeatureBuilder,
        interactions: pd.DataFrame | None = None,
        jobs: pd.DataFrame | None = None,
        k_range=range(2, 11),
        selection_range=range(4, 9),
        seed: int = 42,
    ) -> "UserClusterer":
        X, names = builder.build_user_cluster_matrix(users)
        self.feature_names = names
        records = []
        for k in k_range:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = model.fit_predict(X)
            sil = float(silhouette_score(X, labels))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            sizes = np.bincount(labels, minlength=k) / len(labels)
            records.append({
                "k": int(k),
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "inertia": float(model.inertia_),
                "min_cluster_fraction": float(sizes.min()),
            })
        eligible = [r for r in records if r["k"] in selection_range and r["min_cluster_fraction"] >= 0.04]
        if not eligible:
            eligible = [r for r in records if r["min_cluster_fraction"] >= 0.04] or records
        best = max(eligible, key=lambda r: r["silhouette"])
        self.k = int(best["k"])
        self.kmeans = KMeans(n_clusters=self.k, n_init=30, random_state=seed).fit(X)
        labels = self.kmeans.labels_
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        gmm = GaussianMixture(n_components=self.k, covariance_type="diag", n_init=5, random_state=seed)
        gmm_labels = gmm.fit_predict(X)
        gmm_sil = float(silhouette_score(X, gmm_labels))
        stability = []
        for delta in range(1, 6):
            alt = KMeans(n_clusters=self.k, n_init=20, random_state=seed + delta).fit_predict(X)
            stability.append(float(adjusted_rand_score(labels, alt)))
        self.metrics = {
            "k": self.k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "inertia": float(self.kmeans.inertia_),
            "gmm_bic": float(gmm.bic(X)),
            "gmm_aic": float(gmm.aic(X)),
            "gmm_silhouette": gmm_sil,
            "ari_kmeans_gmm": float(adjusted_rand_score(labels, gmm_labels)),
            "stability_ari_mean": float(np.mean(stability)),
            "stability_ari_std": float(np.std(stability)),
            "k_selection": records,
        }
        skill_cols = [i for i, n in enumerate(names) if n.startswith("skill:")]
        skill_part = X[:, skill_cols]
        self.skill_centroids = np.vstack([skill_part[labels == c].mean(axis=0) for c in range(self.k)])
        norms = np.linalg.norm(self.skill_centroids, axis=1, keepdims=True)
        self.skill_centroids = self.skill_centroids / np.maximum(norms, 1e-9)
        self.family_affinity = self._family_affinity(users, labels, interactions, jobs)
        self.profiles = self._build_profiles(users, labels, interactions, jobs)
        return self

    def _family_affinity(
        self,
        users: pd.DataFrame,
        labels: np.ndarray,
        interactions: pd.DataFrame | None,
        jobs: pd.DataFrame | None,
    ) -> np.ndarray:
        affinity = np.zeros((self.k, len(self.tax.family_ids)), dtype=float)
        fam_index = {f: i for i, f in enumerate(self.tax.family_ids)}
        for i, fams in enumerate(users["preferred_families"]):
            for fam in fams:
                if fam in fam_index:
                    affinity[labels[i], fam_index[fam]] += 1.0
        affinity += 0.5
        return affinity / affinity.sum(axis=1, keepdims=True)

    def _build_profiles(
        self,
        users: pd.DataFrame,
        labels: np.ndarray,
        interactions: pd.DataFrame | None,
        jobs: pd.DataFrame | None,
    ) -> List[dict]:
        profiles: List[dict] = []
        fam_index = {f: i for i, f in enumerate(self.tax.family_ids)}
        merged = None
        if interactions is not None and jobs is not None:
            merged = interactions.merge(jobs[["job_id", "family"]], on="job_id", how="left")
            merged["cluster"] = merged["user_id"].map(dict(zip(users["user_id"], labels)))
        for c in range(self.k):
            subset = users[labels == c]
            skill_counter = Counter(s for skills in subset["skills"] for s in skills)
            pref_counter = Counter(f for fams in subset["preferred_families"] for f in fams)
            top_skills = [s for s, _ in skill_counter.most_common(5)]
            top_pref = [f for f, _ in pref_counter.most_common(3)]
            top_job_fams: List[str] = []
            if merged is not None:
                job_fam_counter = Counter(merged.loc[merged["cluster"] == c, "family"].dropna())
                top_job_fams = [f for f, _ in job_fam_counter.most_common(3)]
            name = self._cluster_name(c, top_pref, top_job_fams, top_skills, fam_index)
            profiles.append({
                "cluster": int(c),
                "name": name,
                "size": int(len(subset)),
                "fraction": float(len(subset) / len(users)),
                "top_skills": top_skills,
                "top_preferred_families": [self.tax.family_name(f) for f in top_pref],
                "top_job_families": [self.tax.family_name(f) for f in top_job_fams],
                "mean_experience": float(subset["experience_years"].mean()),
                "mean_expected_salary": float(subset["expected_salary"].mean()),
                "education_distribution": subset["education"].value_counts(normalize=True).round(3).to_dict(),
                "remote_rate": float(subset["remote_pref"].mean()),
            })
        return profiles

    def _cluster_name(self, cluster: int, top_pref: List[str], top_job_fams: List[str], top_skills: List[str], fam_index: dict) -> str:
        signals = top_pref[:2] or top_job_fams[:2]
        base = "综合探索型"
        for fid in signals:
            if fid in FAMILY_TO_BASE:
                base = FAMILY_TO_BASE[fid]
                break
        suffix = "/".join(top_skills[:2]) if top_skills else f"C{cluster}"
        return f"{base}（{suffix}）"

    def predict(self, user: dict, builder: FeatureBuilder) -> int:
        if self.kmeans is None:
            raise RuntimeError("clusterer is not fitted")
        vector = builder.build_user_cluster_vector(user).reshape(1, -1)
        return int(self.kmeans.predict(vector)[0])

    def profile(self, cluster: int) -> dict:
        return self.profiles[int(cluster)]

    def job_skill_similarity(self, cluster: int, builder: FeatureBuilder) -> np.ndarray:
        centroid = self.skill_centroids[int(cluster)]
        inter = builder.job_skill @ centroid
        job_norm = np.linalg.norm(builder.job_skill, axis=1)
        denom = np.maximum(job_norm * max(np.linalg.norm(centroid), 1e-9), 1e-9)
        return inter / denom

    def job_family_affinity(self, cluster: int, builder: FeatureBuilder) -> np.ndarray:
        fam_index = {f: i for i, f in enumerate(self.tax.family_ids)}
        values = np.array([self.family_affinity[int(cluster), fam_index[f]] for f in builder.job_family_id])
        return values

    def save(self, path: Path | str) -> None:
        state = {
            "k": self.k,
            "kmeans": self.kmeans,
            "skill_centroids": self.skill_centroids,
            "family_affinity": self.family_affinity,
            "profiles": self.profiles,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "user_scaler": None,
        }
        joblib.dump(state, path)

    def save_with_builder(self, path: Path | str, builder: FeatureBuilder) -> None:
        state = {
            "k": self.k,
            "kmeans": self.kmeans,
            "skill_centroids": self.skill_centroids,
            "family_affinity": self.family_affinity,
            "profiles": self.profiles,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "user_scaler": builder.user_scaler,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Path | str, builder: FeatureBuilder, tax: Taxonomy | None = None) -> "UserClusterer":
        state = joblib.load(path)
        obj = cls(tax)
        obj.k = state["k"]
        obj.kmeans = state["kmeans"]
        obj.skill_centroids = state["skill_centroids"]
        obj.family_affinity = state["family_affinity"]
        obj.profiles = state["profiles"]
        obj.metrics = state["metrics"]
        obj.feature_names = state["feature_names"]
        if state.get("user_scaler") is not None:
            builder.user_scaler = state["user_scaler"]
        return obj

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .recommender import LearnedRanker, RecommendationEngine


def split_interactions(
    interactions: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in interactions.groupby("user_id", sort=False):
        n = len(group)
        order = rng.permutation(n)
        if n >= 5:
            n_test = max(2, int(round(n * test_frac)))
            n_val = max(1, int(round(n * val_frac)))
        else:
            n_test = max(1, n // 2)
            n_val = 0
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_test = max(1, n - n_train)
            n_val = 0
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        train_parts.append(group.iloc[train_idx])
        if len(val_idx):
            val_parts.append(group.iloc[val_idx])
        test_parts.append(group.iloc[test_idx])
    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else interactions.iloc[0:0]
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else interactions.iloc[0:0]
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else interactions.iloc[0:0]
    return train, val, test


def dcg_at_k(hits: List[int], k: int) -> float:
    return float(sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k])))


def ndcg_at_k(ranked: List[int], positive: set, k: int) -> float:
    hits = [1 if idx in positive else 0 for idx in ranked[:k]]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(positive), k)))
    return dcg_at_k(hits, k) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked: List[int], positive: set, k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for idx in ranked[:k] if idx in positive) / k


def recall_at_k(ranked: List[int], positive: set, k: int) -> float:
    if not positive:
        return 0.0
    return sum(1 for idx in ranked[:k] if idx in positive) / min(len(positive), k)


def average_precision_at_k(ranked: List[int], positive: set, k: int) -> float:
    if not positive:
        return 0.0
    hits = 0
    total = 0.0
    for i, idx in enumerate(ranked[:k], start=1):
        if idx in positive:
            hits += 1
            total += hits / i
    return total / min(len(positive), k)


def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(sample)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _rank_to_indices(engine: RecommendationEngine, user: dict, user_id: str, method: str, k: int, apply_constraints: bool, use_mmr: bool) -> tuple[List[int], np.ndarray]:
    if use_mmr:
        recs = engine.rank(
            user, user_id=user_id, method=method, top_k=k,
            apply_constraints=apply_constraints, use_mmr=True, relax=0,
        )
        ranked = [r["job_index"] for r in recs]
        scores = np.array([])
        return ranked, scores
    scores = engine.score_all(user, user_id, method, apply_constraints=apply_constraints, relax=0)
    order = np.argsort(-scores)
    ranked = [int(i) for i in order if np.isfinite(scores[int(i)])][:k]
    return ranked, scores


def evaluate_method(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    k: int = 10,
    apply_constraints: bool = True,
    use_mmr: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    job_id_to_idx = engine.job_id_to_idx
    test_by_user = {uid: group["job_id"].tolist() for uid, group in test.groupby("user_id")}
    per_user = []
    top_jobs: set[int] = set()
    start = time.perf_counter()
    for user in users.to_dict("records"):
        uid = user["user_id"]
        pos_ids = test_by_user.get(uid, [])
        if not pos_ids:
            continue
        mask = engine.builder.hard_mask(user, relax=0)
        pos_idx = {job_id_to_idx[j] for j in pos_ids if j in job_id_to_idx and mask[job_id_to_idx[j]]}
        if not pos_idx:
            continue
        ranked, _ = _rank_to_indices(engine, user, uid, method, k, apply_constraints, use_mmr)
        if not ranked:
            continue
        top_jobs.update(ranked[:k])
        violation = float(np.mean([0.0 if mask[idx] else 1.0 for idx in ranked[:k]])) if ranked else 0.0
        per_user.append({
            "user_id": uid,
            "precision@5": precision_at_k(ranked, pos_idx, 5),
            "recall@10": recall_at_k(ranked, pos_idx, 10),
            "ndcg@10": ndcg_at_k(ranked, pos_idx, 10),
            "map@10": average_precision_at_k(ranked, pos_idx, 10),
            "violation_rate": violation,
        })
    elapsed = time.perf_counter() - start
    if not per_user:
        return {"n_users": 0, "n_jobs": len(engine.jobs)}
    frame = pd.DataFrame(per_user)
    values = {
        "n_users": int(len(frame)),
        "precision@5": float(frame["precision@5"].mean()),
        "recall@10": float(frame["recall@10"].mean()),
        "ndcg@10": float(frame["ndcg@10"].mean()),
        "map@10": float(frame["map@10"].mean()),
        "coverage@10": float(len(top_jobs) / len(engine.jobs)),
        "violation_rate": float(frame["violation_rate"].mean()),
        "latency_ms": float(elapsed / len(frame) * 1000.0),
    }
    lo, hi = bootstrap_ci(frame["ndcg@10"].tolist(), seed=seed)
    values["ndcg@10_ci95"] = [lo, hi]
    values["_per_user"] = per_user
    return values


def evaluate_methods(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    test: pd.DataFrame,
    methods: Dict[str, dict] | None = None,
    seed: int = 42,
) -> Dict[str, dict]:
    if methods is None:
        methods = {
            "Popularity": {"method": "popularity", "apply_constraints": False, "use_mmr": False},
            "Content": {"method": "content", "apply_constraints": False, "use_mmr": False},
            "ItemCF": {"method": "itemcf", "apply_constraints": False, "use_mmr": False},
            "SVD": {"method": "svd", "apply_constraints": False, "use_mmr": False},
            "ClusterPrior": {"method": "cluster", "apply_constraints": False, "use_mmr": False},
            "HybridRule": {"method": "hybrid", "apply_constraints": False, "use_mmr": False},
            "Proposed": {"method": "proposed", "apply_constraints": True, "use_mmr": False},
        }
    result = {}
    for name, cfg in methods.items():
        metrics = evaluate_method(
            engine, users, test, cfg["method"], apply_constraints=cfg["apply_constraints"],
            use_mmr=cfg["use_mmr"], seed=seed,
        )
        result[name] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        result[name]["_per_user"] = metrics.get("_per_user", [])
    return result


def common_candidate_evaluation(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    test: pd.DataFrame,
    methods: Dict[str, str],
    k: int = 10,
) -> Dict[str, dict]:
    job_id_to_idx = engine.job_id_to_idx
    test_by_user = {uid: group["job_id"].tolist() for uid, group in test.groupby("user_id")}
    result = {}
    for name, method in methods.items():
        per_user = []
        for user in users.to_dict("records"):
            uid = user["user_id"]
            pos_ids = test_by_user.get(uid, [])
            if not pos_ids:
                continue
            mask = engine.builder.hard_mask(user, relax=0)
            pos_idx = {job_id_to_idx[j] for j in pos_ids if j in job_id_to_idx and mask[job_id_to_idx[j]]}
            if not pos_idx:
                continue
            scores = engine.score_all(user, uid, method, apply_constraints=True, relax=0)
            ranked = [int(i) for i in np.argsort(-scores) if np.isfinite(scores[int(i)])][:k]
            per_user.append({
                "ndcg@10": ndcg_at_k(ranked, pos_idx, 10),
                "recall@10": recall_at_k(ranked, pos_idx, 10),
            })
        if per_user:
            result[name] = {
                "n_users": len(per_user),
                "ndcg@10": float(np.mean([x["ndcg@10"] for x in per_user])),
                "recall@10": float(np.mean([x["recall@10"] for x in per_user])),
            }
    return result


def cold_start_evaluation(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_cold: int = 40,
    seed: int = 42,
) -> Dict[str, dict]:
    from .recommender import ItemCFRecommender, PopularityRecommender, SVDRecommender

    test_counts = test.groupby("user_id").size()
    candidates = [uid for uid, count in test_counts.items() if count >= 3]
    rng = np.random.default_rng(seed)
    cold_users = list(rng.choice(candidates, size=min(n_cold, len(candidates)), replace=False))
    cold_set = set(str(u) for u in cold_users)
    warm_train = engine.train_interactions[~engine.train_interactions["user_id"].isin(cold_set)]
    warm_val = val[~val["user_id"].isin(cold_set)]
    warm_users = users[~users["user_id"].isin(cold_set)]
    all_interactions = pd.concat([engine.train_interactions, val, test], ignore_index=True)
    popularity = PopularityRecommender().fit(warm_train, engine.jobs)
    cf = ItemCFRecommender().fit(warm_train, engine.jobs)
    svd = SVDRecommender(n_components=32, seed=seed).fit(warm_train, engine.jobs)
    cold_engine = RecommendationEngine(
        engine.jobs, engine.users, engine.builder, engine.text_model, engine.clusterer,
        popularity, cf, svd, train_interactions=warm_train,
    )
    cold_engine.fit_ranker(
        warm_train, warm_val, all_interactions, warm_users,
        neg_per_pos=3, seed=seed, tune_c=False, c_value=engine.ranker.c_value,
    )
    cold_users_df = users[users["user_id"].isin(cold_set)]
    methods = {
        "Popularity": {"method": "popularity", "apply_constraints": False, "use_mmr": False},
        "Content": {"method": "content", "apply_constraints": False, "use_mmr": False},
        "ClusterPrior": {"method": "cluster", "apply_constraints": False, "use_mmr": False},
        "HybridRule": {"method": "hybrid", "apply_constraints": False, "use_mmr": False},
        "Proposed": {"method": "proposed", "apply_constraints": True, "use_mmr": False},
    }
    return evaluate_methods(cold_engine, cold_users_df, test, methods, seed=seed)


def ablation_evaluation(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    all_interactions: pd.DataFrame,
) -> Dict[str, dict]:
    full_features = list(engine.ranker.feature_names)
    c_value = engine.ranker.c_value
    configs = {
        "Full": (full_features, True, False, engine.content_blend),
        "NoCluster": ([f for f in full_features if f not in {"cluster_skill_sim", "cluster_family_affinity"}], True, False, engine.content_blend),
        "NoContent": ([f for f in full_features if f not in {"content_sim"}], True, False, 0.0),
        "NoSkill": ([f for f in full_features if f not in {"skill_jaccard", "job_skill_coverage", "is_preferred_family", "is_related_family"}], True, False, engine.content_blend),
        "WithCF": (full_features + ["cf_score", "svd_score"], True, False, engine.content_blend),
        "WithMMR": (full_features, True, True, engine.content_blend),
        "NoHardFilter": (full_features, False, False, engine.content_blend),
    }
    result = {}
    for name, (features, apply_constraints, use_mmr, blend) in configs.items():
        temp = RecommendationEngine(
            engine.jobs, engine.users, engine.builder, engine.text_model, engine.clusterer,
            engine.popularity, engine.cf, engine.svd, train_interactions=engine.train_interactions,
            content_blend=blend,
        )
        X_all, y_all = temp.make_training_pairs(temp.train_interactions, all_interactions, users, seed=42)
        temp.ranker = LearnedRanker(features, c_value=c_value).fit(X_all, y_all, seed=42)
        metrics = evaluate_method(
            temp, users, test, "proposed", apply_constraints=apply_constraints, use_mmr=use_mmr
        )
        result[name] = {k: v for k, v in metrics.items() if not k.startswith("_")}
    return result


def tokenizer_ablation(
    engine: RecommendationEngine,
    jobs: pd.DataFrame,
    users: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    all_interactions: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, dict]:
    from .recommender import DEFAULT_RANKER_FEATURES, ContentRecommender
    from .text import ChineseTextModel

    result = {}
    for mode in ["word", "char"]:
        if mode == "word":
            temp = engine
        else:
            text_model = ChineseTextModel(mode="char")
            ContentRecommender(text_model, users).fit(jobs)
            temp = RecommendationEngine(
                jobs, users, engine.builder, text_model, engine.clusterer,
                engine.popularity, engine.cf, engine.svd, train_interactions=train,
            )
            temp.fit_ranker(
                train, val, all_interactions, users, feature_names=DEFAULT_RANKER_FEATURES,
                tune_c=False, c_value=engine.ranker.c_value,
            )
        fair = evaluate_method(temp, users, test, "proposed", apply_constraints=True, use_mmr=False)
        end = evaluate_method(temp, users, test, "proposed", apply_constraints=True, use_mmr=False)
        content_fair = common_candidate_evaluation(temp, users, test, {"Content": "content"})
        result[mode] = {
            "fair_ndcg@10": float(fair["ndcg@10"]),
            "end_ndcg@10": float(end["ndcg@10"]),
            "end_recall@10": float(end["recall@10"]),
            "content_fair_ndcg@10": float(content_fair.get("Content", {}).get("ndcg@10", 0.0)),
        }
    return result


def error_cases(
    engine: RecommendationEngine,
    users: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    n_cases: int = 6,
) -> List[dict]:
    job_id_to_idx = engine.job_id_to_idx
    test_by_user = {uid: group["job_id"].tolist() for uid, group in test.groupby("user_id")}
    rows = []
    for user in users.to_dict("records"):
        uid = user["user_id"]
        pos_ids = test_by_user.get(uid, [])
        if not pos_ids:
            continue
        mask = engine.builder.hard_mask(user, relax=0)
        pos_idx = {job_id_to_idx[j] for j in pos_ids if j in job_id_to_idx and mask[job_id_to_idx[j]]}
        if not pos_idx:
            continue
        recs = engine.rank(user, user_id=uid, method="proposed", top_k=k, apply_constraints=True, use_mmr=False)
        ranked = [r["job_index"] for r in recs]
        score = ndcg_at_k(ranked, pos_idx, k)
        rows.append({
            "user_id": uid,
            "ndcg@10": score,
            "user": {
                "city": user["city"],
                "target_cities": user["target_cities"],
                "expected_salary": user["expected_salary"],
                "experience_years": user["experience_years"],
                "education": user["education"],
                "skills": user["skills"],
                "preferred_families": user["preferred_families"],
            },
            "top_recommendations": [
                {
                    "job_id": engine.jobs.iloc[i]["job_id"],
                    "title": engine.jobs.iloc[i]["title"],
                    "company": engine.jobs.iloc[i]["company"],
                    "city": engine.jobs.iloc[i]["city"],
                    "salary": f"{engine.jobs.iloc[i]['salary_min']:.1f}-{engine.jobs.iloc[i]['salary_max']:.1f}k",
                    "family": engine.jobs.iloc[i]["family_name"],
                    "skills": engine.jobs.iloc[i]["skills"],
                    "score": float(recs[j]["score"]),
                }
                for j, i in enumerate(ranked[:5])
            ],
            "test_positives": [
                {
                    "job_id": engine.jobs.iloc[i]["job_id"],
                    "title": engine.jobs.iloc[i]["title"],
                    "company": engine.jobs.iloc[i]["company"],
                    "city": engine.jobs.iloc[i]["city"],
                    "salary": f"{engine.jobs.iloc[i]['salary_min']:.1f}-{engine.jobs.iloc[i]['salary_max']:.1f}k",
                    "family": engine.jobs.iloc[i]["family_name"],
                    "skills": engine.jobs.iloc[i]["skills"],
                }
                for i in list(pos_idx)[:5]
            ],
        })
    rows.sort(key=lambda x: x["ndcg@10"])
    return rows[:n_cases]

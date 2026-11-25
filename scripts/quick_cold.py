import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from job_agent.clustering import UserClusterer
from job_agent.data_gen import load_processed
from job_agent.evaluation import evaluate_method, split_interactions
from job_agent.features import FeatureBuilder
from job_agent.recommender import (
    DEFAULT_RANKER_FEATURES,
    ItemCFRecommender,
    PopularityRecommender,
    RecommendationEngine,
    SVDRecommender,
)


def main():
    jobs, users, interactions = load_processed()
    builder = FeatureBuilder(jobs)
    text_model = joblib.load(ROOT / "artifacts" / "models" / "text_model.joblib")
    clusterer = UserClusterer.load(ROOT / "artifacts" / "models" / "clusterer.joblib", builder)
    train, val, test = split_interactions(interactions, seed=42)
    test_counts = test.groupby("user_id").size()
    candidates = [uid for uid, count in test_counts.items() if count >= 3]
    rng = np.random.default_rng(42)
    cold_users = list(rng.choice(candidates, size=40, replace=False))
    cold_set = set(str(u) for u in cold_users)
    warm_train = train[~train["user_id"].isin(cold_set)]
    warm_val = val[~val["user_id"].isin(cold_set)]
    warm_users = users[~users["user_id"].isin(cold_set)]
    cold_users_df = users[users["user_id"].isin(cold_set)]
    all_interactions = pd.concat([train, val, test], ignore_index=True)
    pop = PopularityRecommender().fit(warm_train, jobs)
    cf = ItemCFRecommender().fit(warm_train, jobs)
    svd = SVDRecommender(32, seed=42).fit(warm_train, jobs)
    engine = RecommendationEngine(jobs, users, builder, text_model, clusterer, pop, cf, svd, train_interactions=warm_train)
    configs = {
        "Full": DEFAULT_RANKER_FEATURES,
        "NoCluster": [f for f in DEFAULT_RANKER_FEATURES if f not in {"cluster_skill_sim", "cluster_family_affinity"}],
        "NoSkill": [f for f in DEFAULT_RANKER_FEATURES if f not in {"skill_jaccard", "job_skill_coverage", "is_preferred_family", "is_related_family"}],
        "WithCF": DEFAULT_RANKER_FEATURES + ["cf_score", "svd_score"],
    }
    for name, features in configs.items():
        engine.fit_ranker(warm_train, warm_val, all_interactions, warm_users, feature_names=features, tune_c=False, c_value=0.1)
        m = evaluate_method(engine, cold_users_df, test, "proposed", apply_constraints=True, use_mmr=False)
        print(f"{name:12s} NDCG={m['ndcg@10']:.4f} Recall={m['recall@10']:.4f} Cov={m['coverage@10']:.3f}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from job_agent.clustering import UserClusterer
from job_agent.data_gen import load_processed
from job_agent.evaluation import common_candidate_evaluation, evaluate_method, split_interactions
from job_agent.features import FeatureBuilder
from job_agent.recommender import RecommendationEngine


def main():
    jobs, users, interactions = load_processed()
    builder = FeatureBuilder(jobs)
    text_model = joblib.load(ROOT / "artifacts" / "models" / "text_model.joblib")
    clusterer = UserClusterer.load(ROOT / "artifacts" / "models" / "clusterer.joblib", builder)
    popularity = joblib.load(ROOT / "artifacts" / "models" / "popularity.joblib")
    cf = joblib.load(ROOT / "artifacts" / "models" / "cf.joblib")
    svd = joblib.load(ROOT / "artifacts" / "models" / "svd.joblib")
    train, val, test = split_interactions(interactions, seed=42)
    engine = RecommendationEngine(jobs, users, builder, text_model, clusterer, popularity, cf, svd, train_interactions=train)

    all_features = [
        "content_sim", "cf_score", "svd_score", "cluster_skill_sim", "cluster_family_affinity",
        "skill_jaccard", "job_skill_coverage", "user_skill_coverage", "city_match", "is_same_city",
        "salary_fit", "salary_gap_ratio", "salary_ok", "exp_fit", "edu_fit", "remote_fit",
        "popularity", "job_salary_mid", "job_exp_min", "job_edu_rank", "job_age_days",
        "content_x_skill", "cf_x_content", "is_preferred_family", "is_related_family",
    ]
    subsets = {
        "All": all_features,
        "Clean": [
            "content_sim", "cf_score", "svd_score", "cluster_skill_sim", "cluster_family_affinity",
            "skill_jaccard", "job_skill_coverage", "city_match", "salary_fit", "exp_fit",
            "edu_fit", "remote_fit", "popularity", "is_preferred_family", "is_related_family",
        ],
        "NoRaw": [
            "content_sim", "cf_score", "svd_score", "cluster_skill_sim", "cluster_family_affinity",
            "skill_jaccard", "job_skill_coverage", "city_match", "salary_fit", "exp_fit",
            "edu_fit", "remote_fit", "popularity", "is_preferred_family", "is_related_family",
            "content_x_skill", "cf_x_content",
        ],
        "ContentHeavy": [
            "content_sim", "skill_jaccard", "job_skill_coverage", "user_skill_coverage",
            "city_match", "salary_fit", "exp_fit", "edu_fit", "remote_fit", "popularity",
            "is_preferred_family", "is_related_family", "cluster_skill_sim", "cluster_family_affinity",
        ],
        "NoCF": [
            "content_sim", "cluster_skill_sim", "cluster_family_affinity", "skill_jaccard",
            "job_skill_coverage", "city_match", "salary_fit", "exp_fit", "edu_fit",
            "remote_fit", "popularity", "is_preferred_family", "is_related_family",
        ],
        "NoContent": [
            "cf_score", "svd_score", "cluster_skill_sim", "cluster_family_affinity",
            "skill_jaccard", "job_skill_coverage", "city_match", "salary_fit", "exp_fit",
            "edu_fit", "remote_fit", "popularity", "is_preferred_family", "is_related_family",
        ],
    }
    for name, features in subsets.items():
        info = engine.fit_ranker(train, val, interactions, users, feature_names=features, tune_c=False, c_value=1.0)
        fair = evaluate_method(engine, users, test, "proposed", apply_constraints=True, use_mmr=False)
        end = evaluate_method(engine, users, test, "proposed", apply_constraints=True, use_mmr=True)
        print(f"{name:12s} fair={fair['ndcg@10']:.4f} end={end['ndcg@10']:.4f} recall={end['recall@10']:.4f} cov={end['coverage@10']:.3f} val={info['val_ndcg@10']:.4f}", flush=True)


if __name__ == "__main__":
    main()

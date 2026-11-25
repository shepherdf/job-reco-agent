import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_agent.clustering import UserClusterer
from job_agent.data_gen import load_processed
from job_agent.evaluation import (
    ablation_evaluation,
    cold_start_evaluation,
    common_candidate_evaluation,
    error_cases,
    evaluate_methods,
    split_interactions,
    tokenizer_ablation,
)
from job_agent.features import FeatureBuilder
from job_agent.recommender import (
    ContentRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    RecommendationEngine,
    SVDRecommender,
)
from job_agent.text import ChineseTextModel


def strip_private(result: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-cold-start", action="store_true")
    parser.add_argument("--no-tokenizer", action="store_true")
    args = parser.parse_args()
    started = time.time()
    jobs, users, interactions = load_processed()
    print(f"数据：{len(jobs)} 岗位，{len(users)} 用户，{len(interactions)} 条交互", flush=True)

    builder = FeatureBuilder(jobs)
    text_model = ChineseTextModel(mode="word")
    content = ContentRecommender(text_model, users).fit(jobs)
    train, val, test = split_interactions(interactions, seed=args.seed)
    print(f"划分：train={len(train)}, val={len(val)}, test={len(test)}", flush=True)

    popularity = PopularityRecommender().fit(train, jobs)
    cf = ItemCFRecommender().fit(train, jobs)
    svd = SVDRecommender(n_components=32, seed=args.seed).fit(train, jobs)
    clusterer = UserClusterer().fit(users, builder, interactions=train, jobs=jobs, seed=args.seed)
    print(f"聚类：k={clusterer.k}, silhouette={clusterer.metrics['silhouette']:.4f}, stability={clusterer.metrics['stability_ari_mean']:.4f}", flush=True)

    engine = RecommendationEngine(
        jobs, users, builder, text_model, clusterer, popularity, cf, svd, train_interactions=train
    )
    print("训练排序模型...", flush=True)
    ranker_info = engine.fit_ranker(train, val, interactions, users, seed=args.seed)
    print(f"排序模型：{ranker_info}", flush=True)

    print("评估主实验...", flush=True)
    main = evaluate_methods(engine, users, test, seed=args.seed)
    fair = common_candidate_evaluation(
        engine, users, test,
        {"Content": "content", "ItemCF": "itemcf", "SVD": "svd", "ClusterPrior": "cluster", "HybridRule": "hybrid", "Proposed": "proposed"},
    )
    print("主实验结果：", flush=True)
    for name, m in main.items():
        print(f"  {name:12s} NDCG@10={m['ndcg@10']:.4f}  Recall@10={m['recall@10']:.4f}  Coverage={m['coverage@10']:.3f}  Viol={m['violation_rate']:.3f}", flush=True)

    metrics = {
        "dataset": {
            "n_jobs": int(len(jobs)),
            "n_users": int(len(users)),
            "n_interactions": int(len(interactions)),
            "density": float(len(interactions) / (len(jobs) * len(users))),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "clustering": clusterer.metrics,
        "cluster_profiles": clusterer.profiles,
        "ranker": ranker_info,
        "main": strip_private(main),
        "fair_ranking": fair,
        "elapsed_seconds_before_extra": float(time.time() - started),
    }

    if not args.no_ablation:
        print("消融实验...", flush=True)
        metrics["ablation"] = ablation_evaluation(engine, users, val, test, interactions)
    if not args.no_cold_start:
        print("冷启动实验...", flush=True)
        metrics["cold_start"] = strip_private(cold_start_evaluation(engine, users, val, test, seed=args.seed))
    if not args.no_tokenizer:
        print("分词与表示消融...", flush=True)
        metrics["tokenizer_ablation"] = tokenizer_ablation(engine, jobs, users, train, val, test, interactions, seed=args.seed)
    print("错误案例分析...", flush=True)
    metrics["error_cases"] = error_cases(engine, users, test, n_cases=6)
    metrics["elapsed_seconds"] = float(time.time() - started)

    out_dir = ROOT / "artifacts"
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    per_user = {k: v.get("_per_user", []) for k, v in main.items()}
    with open(out_dir / "per_user_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_user, f, ensure_ascii=False, indent=2)
    test.to_csv(out_dir / "test_interactions.csv", index=False, encoding="utf-8-sig")
    train.to_csv(out_dir / "train_interactions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(text_model, model_dir / "text_model.joblib")
    joblib.dump(popularity, model_dir / "popularity.joblib")
    joblib.dump(cf, model_dir / "cf.joblib")
    joblib.dump(svd, model_dir / "svd.joblib")
    clusterer.save_with_builder(model_dir / "clusterer.joblib", builder)
    engine.save_ranker(model_dir / "ranker.joblib")
    print(f"完成，用时 {time.time() - started:.1f}s，结果写入 artifacts/metrics.json", flush=True)


if __name__ == "__main__":
    main()

from pathlib import Path
from tempfile import TemporaryDirectory

from job_agent.agent import FeedbackStore, JobAgent
from job_agent.clustering import UserClusterer
from job_agent.data_gen import generate_interactions, generate_jobs, generate_users
from job_agent.evaluation import split_interactions
from job_agent.features import FeatureBuilder
from job_agent.recommender import (
    ContentRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    RecommendationEngine,
    SVDRecommender,
)
from job_agent.taxonomy import get_taxonomy
from job_agent.text import ChineseTextModel


_CACHE = {}


def get_small_bundle():
    if "bundle" in _CACHE:
        return _CACHE["bundle"]
    tax = get_taxonomy()
    jobs = generate_jobs(n_jobs=180, seed=7, tax=tax)
    users = generate_users(n_users=40, seed=8, tax=tax)
    interactions = generate_interactions(users, jobs, seed=9, tax=tax, min_candidates=20)
    builder = FeatureBuilder(jobs, tax)
    text_model = ChineseTextModel(tax=tax, mode="word")
    content = ContentRecommender(text_model, users).fit(jobs)
    train, val, test = split_interactions(interactions, seed=42)
    popularity = PopularityRecommender().fit(train, jobs)
    cf = ItemCFRecommender().fit(train, jobs)
    svd = SVDRecommender(n_components=8, seed=42).fit(train, jobs)
    clusterer = UserClusterer(tax).fit(
        users, builder, interactions=train, jobs=jobs,
        k_range=range(2, 7), selection_range=range(3, 6), seed=42,
    )
    engine = RecommendationEngine(
        jobs, users, builder, text_model, clusterer, popularity, cf, svd, train_interactions=train
    )
    engine.fit_ranker(train, val, interactions, users, tune_c=False, c_value=1.0)
    bundle = {
        "tax": tax, "jobs": jobs, "users": users, "interactions": interactions,
        "builder": builder, "text_model": text_model, "content": content,
        "train": train, "val": val, "test": test, "popularity": popularity,
        "cf": cf, "svd": svd, "clusterer": clusterer, "engine": engine,
    }
    _CACHE["bundle"] = bundle
    return bundle


def build_small_agent(tmp_dir: str | None = None) -> JobAgent:
    bundle = get_small_bundle()
    feedback_path = Path(tmp_dir or TemporaryDirectory().name) / "feedback.csv"
    return JobAgent(
        jobs=bundle["jobs"], users=bundle["users"], builder=bundle["builder"],
        text_model=bundle["text_model"], clusterer=bundle["clusterer"],
        popularity=bundle["popularity"], cf=bundle["cf"], svd=bundle["svd"],
        engine=bundle["engine"], tax=bundle["tax"],
        feedback_store=FeedbackStore(feedback_path),
    )

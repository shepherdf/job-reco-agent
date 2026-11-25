import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from job_agent.data_gen import load_processed
from job_agent.evaluation import ndcg_at_k, split_interactions
from job_agent.text import ChineseTextModel


def eval_scores(jobs, users, test, score_fn, k=10):
    test_by_user = {u: g["job_id"].tolist() for u, g in test.groupby("user_id")}
    jid_to_idx = {j: i for i, j in enumerate(jobs["job_id"])}
    values = []
    for user in users.to_dict("records"):
        pos = {jid_to_idx[j] for j in test_by_user.get(user["user_id"], []) if j in jid_to_idx}
        if not pos:
            continue
        scores = score_fn(user)
        ranked = [int(i) for i in np.argsort(-scores)[:k]]
        values.append(ndcg_at_k(ranked, pos, k))
    return float(np.mean(values))


class Variant:
    def __init__(self, jobs, users, mode, repeat=1, skills_only=False):
        self.jobs = jobs
        self.users = users
        self.mode = mode
        self.repeat = repeat
        self.skills_only = skills_only
        self.model = ChineseTextModel(mode=mode)
        self.model.job_matrix = self.model.vectorizer.fit_transform([
            self.job_text(j) for j in jobs.to_dict("records")
        ])

    def job_text(self, job):
        skills = " ".join(job["skills"] * self.repeat)
        if self.skills_only:
            return " ".join([job["title"], job["family_name"], skills])
        return " ".join([job["title"], job["family_name"], job["city"], job["industry"], skills, job["description"]])

    def user_text(self, user):
        skills = " ".join(user["skills"] * self.repeat)
        fams = " ".join(user["preferred_families"])
        if self.skills_only:
            return " ".join([skills, fams])
        return " ".join([skills, fams, user["bio"], user["city"]])

    def scores(self, user):
        return self.model.text_scores(self.user_text(user))


def main():
    jobs, users, interactions = load_processed()
    train, val, test = split_interactions(interactions, seed=42)
    for mode in ["word", "char"]:
        for repeat in [1, 2, 3]:
            for skills_only in [False, True]:
                v = Variant(jobs, users, mode, repeat, skills_only)
                ndcg = eval_scores(jobs, users, test, v.scores)
                print(mode, repeat, skills_only, round(ndcg, 4), flush=True)


if __name__ == "__main__":
    main()

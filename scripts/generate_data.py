import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_agent.data_gen import build_dataset, load_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3000)
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    jobs, users, interactions = build_dataset(args.jobs, args.users, args.seed)
    jobs, users, interactions = load_processed()
    print(json.dumps({
        "jobs": len(jobs),
        "users": len(users),
        "interactions": len(interactions),
        "density": round(len(interactions) / (len(jobs) * len(users)), 4),
        "families": jobs["family_name"].value_counts().to_dict(),
        "events": interactions["event_type"].value_counts().to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.retrieval.experiment import run as run_experiment


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare(query: str, policies: list[str], mode: str, top_k: int | None) -> dict:
    settings = load_settings()
    runs = [run_experiment(query, policy, mode, top_k) for policy in policies]
    rank_table = build_rank_table(runs)
    report = {
        "query": query,
        "mode": mode,
        "policies": policies,
        "rank_table": rank_table,
        "runs": runs,
    }
    slug = "".join(character if character.isalnum() else "-" for character in query.lower())[:60].strip("-")
    write_json(settings.reports_dir / f"retrieval_compare_{mode}_{slug}.json", report)
    return report


def build_rank_table(runs: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for run in runs:
        policy = run["policy"]
        for result in run["results"]:
            key = result["chunk_id"]
            if key not in rows:
                rows[key] = {
                    "chunk_id": key,
                    "work_id": result["work_id"],
                    "title": result["title"],
                    "year": result["year"],
                    "cluster_id": result["cluster_id"],
                    "ranks": {},
                    "scores": {},
                }
            rows[key]["ranks"][policy] = result["rank"]
            rows[key]["scores"][policy] = result["final_score"]
    return sorted(
        rows.values(),
        key=lambda row: min(row["ranks"].values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval policies for one query.")
    parser.add_argument("query")
    parser.add_argument("--policies", default="lexical,balanced,importance,recency")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    print(json.dumps(compare(args.query, policies, args.mode, args.top_k), indent=2))


if __name__ == "__main__":
    main()

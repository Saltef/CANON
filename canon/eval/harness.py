from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from canon.claims.conflict import load_or_detect_conflicts
from canon.config import load_settings
from canon.retrieval.compare import compare


def load_gold(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def average_components(results: list[dict]) -> dict[str, float]:
    component_names = sorted({name for result in results for name in result["components"]})
    averages = {}
    for name in component_names:
        values = [float(result["components"].get(name, 0.0)) for result in results]
        averages[name] = round(mean(values), 6) if values else 0.0
    return averages


def rank_overlap(baseline: list[dict], candidate: list[dict]) -> float:
    baseline_ids = {result["chunk_id"] for result in baseline}
    candidate_ids = {result["chunk_id"] for result in candidate}
    if not baseline_ids and not candidate_ids:
        return 1.0
    if not baseline_ids or not candidate_ids:
        return 0.0
    return round(len(baseline_ids & candidate_ids) / len(baseline_ids | candidate_ids), 6)


def average_rank_delta(baseline: list[dict], candidate: list[dict]) -> float:
    baseline_ranks = {result["chunk_id"]: result["rank"] for result in baseline}
    candidate_ranks = {result["chunk_id"]: result["rank"] for result in candidate}
    shared = sorted(set(baseline_ranks) & set(candidate_ranks))
    if not shared:
        return 0.0
    return round(
        mean(abs(baseline_ranks[chunk_id] - candidate_ranks[chunk_id]) for chunk_id in shared),
        6,
    )


def summarize_run(run: dict, baseline_run: dict | None) -> dict:
    results = run["results"]
    return {
        "policy": run["policy"],
        "result_count": len(results),
        "distinct_works": len({result["work_id"] for result in results}),
        "distinct_sources": len({result["source_name"] for result in results}),
        "distinct_clusters": len(
            {result["cluster_id"] for result in results if result.get("cluster_id") is not None}
        ),
        "average_components": average_components(results),
        "top_sources": dict(Counter(result["source_name"] or "UNKNOWN" for result in results).most_common(10)),
        "rank_overlap_with_lexical": (
            rank_overlap(baseline_run["results"], results) if baseline_run else 1.0
        ),
        "average_rank_delta_with_lexical": (
            average_rank_delta(baseline_run["results"], results) if baseline_run else 0.0
        ),
    }


def evaluate(mode: str, top_k: int | None, policies: list[str]) -> dict:
    settings = load_settings()
    gold = load_gold(settings.root / "gold" / "seed_queries.json")
    conflicts = load_or_detect_conflicts(mode)
    query_reports = []
    for query in gold:
        comparison = compare(query["query"], policies, mode, top_k)
        baseline = next((run for run in comparison["runs"] if run["policy"] == "lexical"), None)
        query_reports.append(
            {
                "id": query["id"],
                "type": query["type"],
                "query": query["query"],
                "importance_expectation": query["importance_expectation"],
                "rank_table": comparison["rank_table"],
                "policy_summaries": [
                    summarize_run(run, baseline)
                    for run in comparison["runs"]
                ],
            }
        )
    report = {
        "mode": mode,
        "top_k": top_k,
        "policies": policies,
        "query_count": len(query_reports),
        "claim_count": conflicts["claim_count"],
        "conflict_count": conflicts["conflict_count"],
        "queries": query_reports,
    }
    write_json(settings.reports_dir / f"eval_seed_{mode}.json", report)
    return report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed-query retrieval evaluation.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,importance,recency")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    print(json.dumps(evaluate(args.mode, args.top_k, policies), indent=2))


if __name__ == "__main__":
    main()

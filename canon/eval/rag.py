from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.harness import load_gold
from canon.synthesis.answer import synthesize


def evaluate_rag(mode: str, top_k: int | None, policies: list[str]) -> dict:
    settings = load_settings()
    gold = load_gold(settings.root / "gold" / "seed_queries.json")
    query_reports = []
    for query in gold:
        policy_runs = [
            summarize_synthesis(synthesize(query["query"], policy, mode, top_k))
            for policy in policies
        ]
        query_reports.append(
            {
                "id": query["id"],
                "type": query["type"],
                "query": query["query"],
                "importance_expectation": query["importance_expectation"],
                "policy_runs": policy_runs,
            }
        )
    report = {
        "mode": mode,
        "top_k": top_k,
        "policies": policies,
        "query_count": len(query_reports),
        "queries": query_reports,
        "policy_summary": policy_summary(query_reports, policies),
    }
    write_json(settings.reports_dir / f"rag_eval_{mode}.json", report)
    return report


def summarize_synthesis(report: dict) -> dict:
    evidence = report["evidence"]
    return {
        "policy": report["policy"],
        "answer": report["answer"],
        "citation_count": len(report["citations"]),
        "claim_citation_count": sum(1 for item in evidence if item["claim"]),
        "conflict_note_count": len(report["conflict_notes"]),
        "distinct_works": report["importance_summary"]["distinct_works"],
        "distinct_sources": report["importance_summary"]["distinct_sources"],
        "distinct_clusters": report["importance_summary"]["distinct_clusters"],
        "average_components": report["importance_summary"]["average_components"],
        "limitations": report["limitations"],
        "citations": report["citations"],
    }


def policy_summary(query_reports: list[dict], policies: list[str]) -> dict[str, dict]:
    summary = {}
    for policy in policies:
        runs = [
            run
            for query in query_reports
            for run in query["policy_runs"]
            if run["policy"] == policy
        ]
        if not runs:
            summary[policy] = {}
            continue
        summary[policy] = {
            "average_citation_count": average(run["citation_count"] for run in runs),
            "average_claim_citation_count": average(run["claim_citation_count"] for run in runs),
            "average_conflict_note_count": average(run["conflict_note_count"] for run in runs),
            "average_distinct_clusters": average(run["distinct_clusters"] for run in runs),
            "runs_with_limitations": sum(1 for run in runs if run["limitations"]),
        }
    return summary


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end CANON RAG evaluation.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    print(json.dumps(evaluate_rag(args.mode, args.top_k, policies), indent=2))


if __name__ == "__main__":
    main()

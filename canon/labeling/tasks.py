from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.rag import evaluate_rag


LABEL_SCHEMA = {
    "citation_support": ["unsupported", "partially_supported", "supported"],
    "answer_faithfulness": ["hallucinated", "mixed", "faithful"],
    "importance_fit": ["misses_importance", "mixed", "fits_expectation"],
    "paradigm_coverage": ["narrow", "mixed", "plural"],
    "evidence_role_fit": ["wrong_role", "partly_right", "right_role"],
    "source_trust_fit": ["overtrusted", "unclear", "appropriate"],
    "safety_handling": ["unsafe_exposed", "unclear", "safe"],
    "corroboration_handling": ["misses_conflict_or_echo", "partial", "handles_conflict_or_echo"],
}


def build_label_tasks(mode: str, policies: list[str], top_k: int | None = None) -> dict:
    settings = load_settings()
    rag_report = evaluate_rag(mode, top_k, policies)
    tasks = []
    for query in rag_report["queries"]:
        for run in query["policy_runs"]:
            tasks.append(task_for_run(query, run))
    report = {
        "mode": mode,
        "schema": LABEL_SCHEMA,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    write_json(settings.reports_dir / f"label_tasks_{mode}.json", report)
    return report


def task_for_run(query: dict, run: dict) -> dict:
    return {
        "id": f"{query['id']}::{run['policy']}",
        "query_id": query["id"],
        "policy": run["policy"],
        "query": query["query"],
        "importance_expectation": query["importance_expectation"],
        "answer": run["answer"],
        "citations": run["citations"],
        "metrics": {
            "citation_count": run["citation_count"],
            "claim_citation_count": run["claim_citation_count"],
            "conflict_note_count": run["conflict_note_count"],
            "distinct_works": run.get("distinct_works", 0),
            "distinct_sources": run.get("distinct_sources", 0),
            "distinct_clusters": run["distinct_clusters"],
            "average_components": run.get("average_components", {}),
        },
        "average_components": run.get("average_components", {}),
        "limitations": run.get("limitations", []),
        "anchor_questions": [
            "Are the cited chunks relevant to the query rather than merely source-prestigious?",
            "Is the answer using methods, limitations, or contradictions when the query calls for them?",
            "Does the answer overtrust a source because it has citations, author prominence, a PDF, or open access?",
            "Did unsafe retrieved instructions remain excluded or sanitized as evidence data?",
        ],
        "labels": {name: None for name in LABEL_SCHEMA},
        "notes": "",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON human-label tasks.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    print(json.dumps(build_label_tasks(args.mode, policies, args.top_k), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.rag import evaluate_rag
from canon.labeling.calibration import technical_labels_for_task
from canon.labeling.tasks import task_for_run


TECHNICAL_DIMENSIONS = [
    "citation_support",
    "answer_faithfulness",
    "importance_fit",
    "paradigm_coverage",
    "evidence_role_fit",
    "source_trust_fit",
    "safety_handling",
    "corroboration_handling",
]


def evaluate_technical_calibration(
    mode: str,
    policies: list[str],
    top_k: int | None = None,
) -> dict[str, Any]:
    rag_report = evaluate_rag(mode, top_k, policies)
    rows = []
    for query in rag_report["queries"]:
        for run in query["policy_runs"]:
            task = task_for_run(query, run)
            task["average_components"] = run.get("average_components", {})
            task["limitations"] = run.get("limitations", [])
            rows.append(technical_row(task, run))
    report = {
        "mode": mode,
        "top_k": top_k,
        "policies": policies,
        "task_count": len(rows),
        "dimensions": TECHNICAL_DIMENSIONS,
        "policy_summary": policy_summary(rows),
        "rows": rows,
        "limitations": [
            "This is technical-only calibration; it does not use human labels.",
            "Proxy labels are useful for stress testing but must be compared with human/expert anchors.",
            "Source trust and evidence role proxies are especially provisional.",
        ],
    }
    settings = load_settings()
    write_json(settings.reports_dir / f"technical_calibration_{mode}.json", report)
    return report


def technical_row(task: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    labels = technical_labels_for_task(task)
    scores = numeric_scores(labels)
    return {
        "task_id": task["id"],
        "query_id": task["query_id"],
        "policy": task["policy"],
        "technical_labels": labels,
        "dimension_scores": scores,
        "technical_calibration_score": round(sum(scores.values()) / len(scores), 6) if scores else 0.0,
        "metrics": task["metrics"],
        "average_components": run.get("average_components", {}),
        "limitations": run.get("limitations", []),
    }


def numeric_scores(labels: dict[str, str]) -> dict[str, float]:
    scores = {}
    for label, value in labels.items():
        if value in {"supported", "faithful", "fits_expectation", "plural", "right_role", "appropriate", "safe", "handles_conflict_or_echo"}:
            scores[label] = 1.0
        elif value in {"partially_supported", "mixed", "partly_right", "unclear", "partial"}:
            scores[label] = 0.5
        else:
            scores[label] = 0.0
    return scores


def policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted({row["policy"] for row in rows})
    summary = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "task_count": len(policy_rows),
            "average_technical_calibration_score": average(
                row["technical_calibration_score"] for row in policy_rows
            ),
            "average_dimension_scores": {
                dimension: average(row["dimension_scores"].get(dimension, 0.0) for row in policy_rows)
                for dimension in TECHNICAL_DIMENSIONS
            },
            "failure_counts": {
                dimension: sum(1 for row in policy_rows if row["dimension_scores"].get(dimension, 0.0) == 0.0)
                for dimension in TECHNICAL_DIMENSIONS
            },
        }
    return summary


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run technical-only calibration over RAG outputs.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    print(json.dumps(evaluate_technical_calibration(args.mode, policies, args.top_k), indent=2))


if __name__ == "__main__":
    main()

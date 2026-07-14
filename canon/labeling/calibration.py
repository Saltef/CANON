from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.labeling.tasks import LABEL_SCHEMA


LABEL_ORDER = {
    "citation_support": ["unsupported", "partially_supported", "supported"],
    "answer_faithfulness": ["hallucinated", "mixed", "faithful"],
    "importance_fit": ["misses_importance", "mixed", "fits_expectation"],
    "paradigm_coverage": ["narrow", "mixed", "plural"],
    "evidence_role_fit": ["wrong_role", "partly_right", "right_role"],
    "source_trust_fit": ["overtrusted", "unclear", "appropriate"],
    "safety_handling": ["unsafe_exposed", "unclear", "safe"],
    "corroboration_handling": ["misses_conflict_or_echo", "partial", "handles_conflict_or_echo"],
}


def technical_labels_for_task(task: dict[str, Any]) -> dict[str, str]:
    metrics = task.get("metrics") or {}
    citation_count = int(metrics.get("citation_count") or 0)
    claim_count = int(metrics.get("claim_citation_count") or 0)
    conflict_count = int(metrics.get("conflict_note_count") or 0)
    clusters = int(metrics.get("distinct_clusters") or 0)
    sources = int(metrics.get("distinct_sources") or 0)
    average_components = task.get("average_components") or metrics.get("average_components") or {}
    relevance = max(
        float(average_components.get("relevance", 0.0) or 0.0),
        float(average_components.get("semantic_similarity", 0.0) or 0.0),
    )
    source_trust = float(average_components.get("source_quality", 0.0) or 0.0)
    limitations = " ".join(str(item).lower() for item in task.get("limitations") or [])

    return {
        "citation_support": citation_support_label(citation_count, claim_count),
        "answer_faithfulness": faithfulness_label(citation_count, claim_count, relevance),
        "importance_fit": importance_fit_label(relevance, claim_count, conflict_count, task.get("query", "")),
        "paradigm_coverage": paradigm_label(clusters, conflict_count),
        "evidence_role_fit": evidence_role_label(claim_count, conflict_count, task.get("query", "")),
        "source_trust_fit": source_trust_label(source_trust, sources),
        "safety_handling": safety_label(limitations),
        "corroboration_handling": corroboration_label(clusters, conflict_count),
    }


def citation_support_label(citation_count: int, claim_count: int) -> str:
    if citation_count >= 3 and claim_count >= 1:
        return "supported"
    if citation_count > 0:
        return "partially_supported"
    return "unsupported"


def faithfulness_label(citation_count: int, claim_count: int, relevance: float) -> str:
    if citation_count >= 2 and claim_count >= 1 and relevance >= 0.25:
        return "faithful"
    if citation_count > 0:
        return "mixed"
    return "hallucinated"


def importance_fit_label(relevance: float, claim_count: int, conflict_count: int, query: str) -> str:
    query_lower = query.lower()
    disagreement_query = any(term in query_lower for term in ["disagree", "debate", "conflict", "challenge", "mixed"])
    if disagreement_query and conflict_count <= 0:
        return "mixed" if relevance >= 0.25 else "misses_importance"
    if relevance >= 0.35 and claim_count >= 1:
        return "fits_expectation"
    if relevance > 0.0:
        return "mixed"
    return "misses_importance"


def paradigm_label(clusters: int, conflict_count: int) -> str:
    if clusters >= 2 and conflict_count >= 1:
        return "plural"
    if clusters >= 2:
        return "mixed"
    return "narrow"


def evidence_role_label(claim_count: int, conflict_count: int, query: str) -> str:
    query_lower = query.lower()
    wants_conflict = any(term in query_lower for term in ["disagree", "debate", "conflict", "challenge", "mixed"])
    if wants_conflict and conflict_count >= 1:
        return "right_role"
    if claim_count >= 1:
        return "right_role"
    if conflict_count >= 1:
        return "partly_right"
    return "wrong_role"


def source_trust_label(source_trust: float, sources: int) -> str:
    if source_trust >= 0.45 and sources >= 2:
        return "appropriate"
    if source_trust >= 0.25 or sources >= 1:
        return "unclear"
    return "overtrusted"


def safety_label(limitations: str) -> str:
    if "unsafe" in limitations or "quarantined" in limitations or "blocked" in limitations:
        return "unsafe_exposed"
    if "weak support" in limitations:
        return "unclear"
    return "safe"


def corroboration_label(clusters: int, conflict_count: int) -> str:
    if conflict_count >= 1:
        return "handles_conflict_or_echo"
    if clusters >= 2:
        return "partial"
    return "misses_conflict_or_echo"


def calibrate_label_tasks(tasks_report: dict[str, Any]) -> dict[str, Any]:
    tasks = list(tasks_report.get("tasks") or [])
    rows = [calibration_row(task) for task in tasks]
    completed = [row for row in rows if row["human_label_count"] > 0]
    return {
        "mode": tasks_report.get("mode"),
        "task_count": len(tasks),
        "completed_human_task_count": len(completed),
        "schema": LABEL_SCHEMA,
        "technical_label_distribution": label_distribution([row["technical_labels"] for row in rows]),
        "agreement": agreement_summary(rows),
        "rows": rows,
        "limitations": [
            "Technical labels are proxy judgments, not ground truth.",
            "Agreement is only meaningful once human labels are populated.",
            "The source-trust proxy uses available summary metrics and cannot judge method validity.",
        ],
    }


def calibration_row(task: dict[str, Any]) -> dict[str, Any]:
    human_labels = {
        key: value
        for key, value in (task.get("labels") or {}).items()
        if key in LABEL_SCHEMA and value is not None
    }
    technical = technical_labels_for_task(task)
    agreements = {
        key: human_labels[key] == technical.get(key)
        for key in sorted(human_labels)
        if key in technical
    }
    return {
        "task_id": task.get("id"),
        "query_id": task.get("query_id"),
        "policy": task.get("policy"),
        "human_label_count": len(human_labels),
        "human_labels": human_labels,
        "technical_labels": technical,
        "agreement": agreements,
        "exact_agreement_rate": average(1.0 if value else 0.0 for value in agreements.values()),
        "ordinal_distance": ordinal_distance_summary(human_labels, technical),
    }


def agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {}
    for label in LABEL_SCHEMA:
        comparisons = [
            row["agreement"][label]
            for row in rows
            if label in row["agreement"]
        ]
        by_label[label] = {
            "comparison_count": len(comparisons),
            "exact_agreement_rate": average(1.0 if value else 0.0 for value in comparisons),
        }
    all_comparisons = [
        value
        for row in rows
        for value in row["agreement"].values()
    ]
    return {
        "overall_comparison_count": len(all_comparisons),
        "overall_exact_agreement_rate": average(1.0 if value else 0.0 for value in all_comparisons),
        "by_label": by_label,
    }


def label_distribution(label_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    distribution = {label: {option: 0 for option in options} for label, options in LABEL_SCHEMA.items()}
    for labels in label_rows:
        for label, value in labels.items():
            if label in distribution and value in distribution[label]:
                distribution[label][value] += 1
    return distribution


def ordinal_distance_summary(human_labels: dict[str, str], technical: dict[str, str]) -> dict[str, Any]:
    distances = {}
    for label, human_value in human_labels.items():
        tech_value = technical.get(label)
        order = LABEL_ORDER.get(label) or []
        if human_value in order and tech_value in order:
            distances[label] = abs(order.index(human_value) - order.index(tech_value))
    return {
        "mean_distance": average(distances.values()),
        "by_label": distances,
    }


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def load_tasks(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate human labels against technical proxy labels.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--tasks", default=None)
    args = parser.parse_args()
    settings = load_settings()
    tasks_path = Path(args.tasks) if args.tasks else settings.reports_dir / f"label_tasks_{args.mode}.json"
    report = calibrate_label_tasks(load_tasks(tasks_path))
    write_json(settings.reports_dir / f"label_calibration_{report.get('mode') or args.mode}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.labeling.unstructured_tasks import UNSTRUCTURED_LABEL_SCHEMA


def calibrate_unstructured_label_tasks(tasks_report: dict[str, Any]) -> dict[str, Any]:
    tasks = list(tasks_report.get("tasks") or [])
    rows = [calibration_row(task) for task in tasks]
    completed_rows = [row for row in rows if row["completed_human_label_count"] > 0]
    completed_label_count = sum(row["completed_human_label_count"] for row in rows)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "unstructured_label_calibration_v1",
        "source_report_id": tasks_report.get("report_id"),
        "status": "calibrated" if completed_label_count > 0 else "needs_labels",
        "task_count": len(tasks),
        "completed_task_count": len(completed_rows),
        "completed_label_count": completed_label_count,
        "schema": UNSTRUCTURED_LABEL_SCHEMA,
        "technical_label_distribution": label_distribution(row["technical_labels"] for row in rows),
        "agreement": agreement_summary(rows),
        "label_family_summary": label_family_summary(rows),
        "rows": rows,
        "claim_boundaries": claim_boundaries(),
        "limitations": limitations(),
    }
    report["contract_validation"] = validate_report(report, "unstructured_label_calibration")
    return report


def calibration_row(task: dict[str, Any]) -> dict[str, Any]:
    human_labels = {
        key: value
        for key, value in (task.get("human_labels") or {}).items()
        if key in UNSTRUCTURED_LABEL_SCHEMA and value is not None
    }
    technical = technical_labels_for_task(task)
    agreements = {
        key: human_labels[key] == technical.get(key)
        for key in sorted(human_labels)
        if key in technical
    }
    return {
        "task_id": task.get("id"),
        "task_family": task.get("task_family"),
        "mode": task.get("mode"),
        "domain": task.get("domain"),
        "document_type": task.get("document_type"),
        "label_families": task.get("label_families") or [],
        "completed_human_label_count": len(human_labels),
        "human_labels": human_labels,
        "technical_labels": technical,
        "agreement": agreements,
        "exact_agreement_rate": average(1.0 if value else 0.0 for value in agreements.values()),
        "disagreement_labels": sorted(label for label, value in agreements.items() if not value),
    }


def technical_labels_for_task(task: dict[str, Any]) -> dict[str, str]:
    signals = task.get("deterministic_signals") or {}
    document_profile = signals.get("document_type_profile") or {}
    source_trust = signals.get("source_trust") or {}
    domain_fit = signals.get("domain_fit") or {}
    safety = signals.get("safety") or {}
    chunk_importance = signals.get("chunk_importance") or {}
    chunking_comparison = signals.get("chunking_variant_comparison") or {}
    committee_decision = str(signals.get("committee_decision") or "")
    allowed_uses = set(signals.get("allowed_uses") or [])
    proxy_risk = str(signals.get("proxy_usefulness_risk") or "")
    conflict_tags = set(signals.get("conflict_tags") or [])

    safety_decision = str(safety.get("decision") or safety_from_committee_tags(conflict_tags) or "allow")
    labels = {
        "document_type_correctness": document_type_correctness(document_profile),
        "primary_evidence_role": primary_evidence_role(task, chunk_importance, safety_decision, committee_decision),
        "source_trust_tier": source_trust_tier(source_trust, committee_decision),
        "source_independence": source_independence(task, conflict_tags),
        "safety_disposition": safety_decision if safety_decision in UNSTRUCTURED_LABEL_SCHEMA["safety_disposition"] else "uncertain",
        "generator_use": generator_use(task, safety_decision, chunk_importance, committee_decision, allowed_uses),
        "answer_usefulness": answer_usefulness(task, safety_decision, proxy_risk, chunk_importance),
        "exclusion_correctness": exclusion_correctness(task, safety_decision, proxy_risk, committee_decision),
        "chunking_harmful_merge": chunking_harmful_merge(chunking_comparison),
        "qualifier_preservation": qualifier_preservation(chunking_comparison),
        "fragmentation": fragmentation_label(chunking_comparison),
        "preferred_chunking_variant": preferred_chunking_variant(chunking_comparison),
    }
    return {
        key: value
        for key, value in labels.items()
        if key in (task.get("human_labels") or {}) or key in UNSTRUCTURED_LABEL_SCHEMA
    }


def document_type_correctness(document_profile: dict[str, Any]) -> str:
    confidence = float(document_profile.get("confidence", 0.0) or 0.0)
    if confidence >= 0.75:
        return "correct"
    if confidence >= 0.5:
        return "partly_correct"
    return "uncertain"


def primary_evidence_role(
    task: dict[str, Any],
    importance: dict[str, Any],
    safety_decision: str,
    committee_decision: str,
) -> str:
    if safety_decision in {"sanitize", "quarantine", "block"}:
        return "instruction_like_text"
    if task.get("document_type") in {"social_media_post", "forum_comment"}:
        return "public_expression"
    if committee_decision == "aggregate_only":
        return "public_expression"
    if float(importance.get("method_signal", 0.0) or 0.0) >= 0.25:
        return "method"
    if float(importance.get("limitation_signal", 0.0) or 0.0) >= 0.25:
        return "limitation"
    if float(importance.get("evidence_specificity", 0.0) or 0.0) >= 0.25:
        return "result"
    if float(importance.get("claim_density", 0.0) or 0.0) > 0.0:
        return "claim"
    if task.get("task_family") == "committee_gate_abstention_review":
        return "context"
    return "context"


def source_trust_tier(source_trust: dict[str, Any], committee_decision: str) -> str:
    tier = str(source_trust.get("trust_tier") or "")
    if tier in UNSTRUCTURED_LABEL_SCHEMA["source_trust_tier"]:
        return tier
    if committee_decision == "reject":
        return "low"
    return "uncertain"


def source_independence(task: dict[str, Any], conflict_tags: set[str]) -> str:
    provenance = str(task.get("provenance") or "")
    if "echo_risk" in conflict_tags or "dependent_echo" in conflict_tags:
        return "dependent_echo"
    if provenance in {"official", "archive", "academic", "repository"}:
        return "partly_independent"
    if provenance in {"social_media", "comment", "web"}:
        return "unclear"
    return "unclear"


def safety_from_committee_tags(conflict_tags: set[str]) -> str | None:
    if "safety_exfiltration" in conflict_tags or "safety_instruction_override" in conflict_tags:
        return "block"
    if "relevant_but_unsafe" in conflict_tags:
        return "quarantine"
    return None


def generator_use(
    task: dict[str, Any],
    safety_decision: str,
    importance: dict[str, Any],
    committee_decision: str,
    allowed_uses: set[str],
) -> str:
    if safety_decision in {"quarantine", "block"}:
        return "exclude"
    if "generator_context" in allowed_uses:
        return "allow_context"
    if committee_decision == "aggregate_only" or importance.get("generation_context_role") == "aggregate_only":
        return "aggregate_only"
    if committee_decision in {"inspect_only", "reject"}:
        return "inspect_only" if committee_decision == "inspect_only" else "exclude"
    if task.get("document_type") in {"social_media_post", "forum_comment"}:
        return "aggregate_only"
    return "allow_context"


def answer_usefulness(
    task: dict[str, Any],
    safety_decision: str,
    proxy_risk: str,
    importance: dict[str, Any],
) -> str:
    if safety_decision == "block":
        return "harmful"
    if proxy_risk == "potentially_useful_loss":
        return "essential"
    if proxy_risk == "review_needed_loss":
        return "useful_with_caution"
    if task.get("task_family") == "committee_gate_abstention_review":
        return "useful_with_caution"
    if float(importance.get("method_signal", 0.0) or 0.0) >= 0.25 or float(importance.get("limitation_signal", 0.0) or 0.0) >= 0.25:
        return "useful"
    if task.get("document_type") in {"social_media_post", "forum_comment", "web_page"}:
        return "useful_with_caution"
    return "uncertain"


def exclusion_correctness(
    task: dict[str, Any],
    safety_decision: str,
    proxy_risk: str,
    committee_decision: str,
) -> str:
    if safety_decision in {"quarantine", "block"} or committee_decision in {"reject", "aggregate_only"}:
        return "correctly_excluded"
    if proxy_risk in {"potentially_useful_loss", "review_needed_loss"}:
        return "uncertain"
    if task.get("task_family") == "committee_gate_loss_review":
        return "uncertain"
    return "uncertain"


def chunking_harmful_merge(comparison: dict[str, Any]) -> str:
    fixed = comparison.get("fixed_window") or {}
    fixed_merges = int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0)
    unsafe_present = bool((fixed.get("unsafe_isolation") or {}).get("unsafe_span_present"))
    if fixed_merges > 0 and unsafe_present:
        return "harmful_merge"
    if fixed_merges > 0:
        return "possible_harmful_merge"
    if comparison:
        return "no_harmful_merge"
    return "uncertain"


def qualifier_preservation(comparison: dict[str, Any]) -> str:
    variants = [
        comparison.get(name) or {}
        for name in ["fixed_window", "boundary_aware", "semantic_cohesion", "embedding_semantic", "atomic_sentence"]
    ]
    if not any(variants):
        return "uncertain"
    missing_counts = [len(row.get("missing_spans") or []) for row in variants]
    if min(missing_counts) == 0 and max(missing_counts) == 0:
        return "preserved"
    if min(missing_counts) == 0:
        return "partly_preserved"
    return "missing_qualifier"


def fragmentation_label(comparison: dict[str, Any]) -> str:
    fixed = comparison.get("fixed_window") or {}
    boundary = comparison.get("boundary_aware") or {}
    atomic = comparison.get("atomic_sentence") or {}
    fixed_count = int(fixed.get("chunk_count") or 0)
    boundary_count = int(boundary.get("chunk_count") or 0)
    atomic_count = int(atomic.get("chunk_count") or 0)
    if not comparison:
        return "uncertain"
    if fixed_count <= 1 and int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0) > 0:
        return "under_fragmented"
    if atomic_count >= max(4, boundary_count + 3):
        return "over_fragmented"
    return "appropriate"


def preferred_chunking_variant(comparison: dict[str, Any]) -> str:
    winners = comparison.get("best_variant_by_objective") or {}
    winner = str(winners.get("balanced_utility") or winners.get("safety_first") or "")
    if winner in UNSTRUCTURED_LABEL_SCHEMA["preferred_chunking_variant"]:
        return winner
    return "uncertain"


def agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [value for row in rows for value in row["agreement"].values()]
    by_label: dict[str, dict[str, Any]] = {}
    for label in UNSTRUCTURED_LABEL_SCHEMA:
        label_values = [row["agreement"][label] for row in rows if label in row["agreement"]]
        by_label[label] = {
            "comparison_count": len(label_values),
            "exact_agreement_rate": average(1.0 if value else 0.0 for value in label_values),
        }
    return {
        "overall_comparison_count": len(comparisons),
        "overall_exact_agreement_rate": average(1.0 if value else 0.0 for value in comparisons),
        "by_label": by_label,
    }


def label_family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter()
    completed = Counter()
    for row in rows:
        for family in row.get("label_families") or []:
            families[str(family)] += 1
        for label in row.get("human_labels") or {}:
            completed[label] += 1
    return {
        "task_label_family_counts": dict(sorted(families.items())),
        "completed_human_label_counts": dict(sorted(completed.items())),
    }


def label_distribution(label_rows) -> dict[str, dict[str, int]]:
    distribution = {label: {option: 0 for option in options} for label, options in UNSTRUCTURED_LABEL_SCHEMA.items()}
    for labels in label_rows:
        for label, value in labels.items():
            if label in distribution and value in distribution[label]:
                distribution[label][value] += 1
    return distribution


def claim_boundaries() -> list[str]:
    return [
        "Allowed: use this report to measure completed human-label coverage and disagreement with deterministic proxies.",
        "Allowed: use disagreement rows to prioritize rule, model, or domain-profile revisions.",
        "Not allowed: treat technical labels as human ground truth.",
        "Not allowed: claim calibration while status is `needs_labels`.",
    ]


def limitations() -> list[str]:
    return [
        "Agreement is only meaningful after humans or approved expert workflows fill label fields.",
        "Technical labels are deterministic proxies intended for comparison, not truth.",
        "Empty labels intentionally produce a `needs_labels` status.",
    ]


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    text = f"""# Unstructured Label Calibration

- Status: {report["status"]}
- Tasks: {report["task_count"]}
- Completed tasks: {report["completed_task_count"]}
- Completed labels: {report["completed_label_count"]}
- Overall comparisons: {report["agreement"]["overall_comparison_count"]}
- Overall exact agreement: {report["agreement"]["overall_exact_agreement_rate"]}

## Claim Boundaries

{chr(10).join(f"- {item}" for item in report["claim_boundaries"])}

## Limitations

{chr(10).join(f"- {item}" for item in report["limitations"])}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate unstructured human labels against deterministic proxy labels.")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    tasks_path = Path(args.tasks) if args.tasks else settings.reports_dir / "unstructured_human_label_tasks_v1.json"
    report = calibrate_unstructured_label_tasks(load_json(tasks_path))
    if not args.no_write_report:
        path = settings.reports_dir / "unstructured_label_calibration_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

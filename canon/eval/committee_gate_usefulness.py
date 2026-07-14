from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.committee import DEFAULT_COMMITTEE_QUERIES, run_committee
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.document_type_slices import resolve_policy_weights
from canon.eval.domain_slices import DEFAULT_DOMAIN_QUERIES
from canon.retrieval.corpus import load_processed_corpus


DEFAULT_MODES = ["mixed_domain_demo_corpus", "social_public_opinion_demo_corpus"]

USEFULNESS_LABEL_SCHEMA = {
    "human_evidence_usefulness": ["not_useful", "somewhat_useful", "very_useful"],
    "human_exclusion_correctness": ["should_exclude", "uncertain", "should_include"],
    "human_abstention_correctness": ["acceptable_abstention", "uncertain", "should_return_context"],
}


def evaluate_committee_gate_usefulness(
    modes: list[str] | None = None,
    baseline_policy: str = "rag",
    gated_policy: str = "committee_rag",
    top_k: int | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    retrieval_settings = settings.raw["retrieval"]
    resolved_top_k = top_k or int(retrieval_settings["top_k"])
    policy_weights = resolve_policy_weights(
        settings.raw["retrieval"]["policies"],
        [baseline_policy, gated_policy],
    )
    mode_ids = modes or DEFAULT_MODES
    mode_reports = [
        evaluate_mode(
            mode=mode,
            baseline_policy=baseline_policy,
            gated_policy=gated_policy,
            policy_weights=policy_weights,
            top_k=resolved_top_k,
            preview_chars=int(retrieval_settings["preview_chars"]),
        )
        for mode in mode_ids
    ]
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "committee_gate_usefulness_audit_v1",
        "baseline_policy": baseline_policy,
        "gated_policy": gated_policy,
        "top_k": resolved_top_k,
        "label_schema": USEFULNESS_LABEL_SCHEMA,
        "modes": mode_reports,
        "summary": usefulness_summary(mode_reports),
        "claim_boundaries": [
            "Allowed: identify which baseline-selected evidence was removed by committee gating.",
            "Allowed: use proxy labels to prioritize human review of possible recall loss.",
            "Not allowed: treat proxy acceptable-exclusion labels as proof that removed evidence was useless.",
            "Not allowed: claim abstention quality without human or task-specific external labels.",
        ],
        "limitations": [
            "Proxy usefulness is based on committee decisions and selected-context status, not human judgment.",
            "The audit is most useful for prioritizing review of losses and abstentions, not replacing review.",
            "Small fixture corpora make retention and abstention rates unstable.",
        ],
    }
    report["contract_validation"] = validate_report(report, "committee_gate_usefulness")
    if write_report:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        path = settings.reports_dir / "committee_gate_usefulness_audit_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), report)
    return report


def evaluate_mode(
    mode: str,
    baseline_policy: str,
    gated_policy: str,
    policy_weights: dict[str, dict[str, float]],
    top_k: int,
    preview_chars: int,
) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(settings.data_dir, mode)
    queries = default_queries_for_mode(mode)
    rows = [
        audit_query(
            mode=mode,
            query_spec=query_spec,
            baseline_policy=baseline_policy,
            gated_policy=gated_policy,
            policy_weights=policy_weights,
            documents=documents,
            top_k=top_k,
            preview_chars=preview_chars,
        )
        for query_spec in queries
    ]
    return {
        "mode": mode,
        "document_count": len(documents),
        "query_count": len(queries),
        "queries": rows,
        "summary": mode_summary(rows),
    }


def default_queries_for_mode(mode: str) -> list[dict[str, str]]:
    if "social" in mode or "public_opinion" in mode:
        return DEFAULT_COMMITTEE_QUERIES
    return DEFAULT_DOMAIN_QUERIES


def audit_query(
    mode: str,
    query_spec: dict[str, str],
    baseline_policy: str,
    gated_policy: str,
    policy_weights: dict[str, dict[str, float]],
    documents: list[Any],
    top_k: int,
    preview_chars: int,
) -> dict[str, Any]:
    baseline = run_committee(
        query_spec=query_spec,
        policy=baseline_policy,
        weights=policy_weights[baseline_policy],
        documents=documents,
        top_k=top_k,
        preview_chars=preview_chars,
    )
    gated = run_committee(
        query_spec=query_spec,
        policy=gated_policy,
        weights=policy_weights[gated_policy],
        documents=documents,
        top_k=top_k,
        preview_chars=preview_chars,
    )
    gated_selected_ids = {
        row["chunk_id"]
        for row in gated.get("candidates") or []
        if row.get("selected_in_final_context")
    }
    lost_rows = [
        lost_candidate_row(row)
        for row in baseline.get("candidates") or []
        if row.get("selected_in_final_context") and row.get("chunk_id") not in gated_selected_ids
    ]
    abstention = int(gated.get("final_context_count") or 0) == 0
    return {
        "mode": mode,
        "id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "baseline_final_context_count": int(baseline.get("final_context_count") or 0),
        "gated_final_context_count": int(gated.get("final_context_count") or 0),
        "lost_baseline_selected_count": len(lost_rows),
        "potentially_useful_loss_count": sum(1 for row in lost_rows if row["proxy_usefulness_risk"] == "potentially_useful_loss"),
        "review_needed_loss_count": sum(1 for row in lost_rows if row["proxy_usefulness_risk"] == "review_needed_loss"),
        "proxy_acceptable_exclusion_count": sum(1 for row in lost_rows if row["proxy_usefulness_risk"] == "acceptable_exclusion"),
        "gated_abstained": abstention,
        "abstention_review": abstention_row(query_spec, baseline, gated) if abstention else None,
        "lost_candidates": lost_rows,
    }


def lost_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    decision = str(row.get("committee_decision") or "unknown")
    risk = proxy_usefulness_risk(row)
    return {
        "chunk_id": row.get("chunk_id"),
        "work_id": row.get("work_id"),
        "title": row.get("title"),
        "source_name": row.get("source_name"),
        "year": row.get("year"),
        "document_type": row.get("document_type"),
        "evidence_family": row.get("evidence_family"),
        "domain_profile": row.get("domain_profile"),
        "domain_fit_tier": row.get("domain_fit_tier"),
        "candidate_rank": row.get("candidate_rank"),
        "candidate_score": row.get("candidate_score"),
        "committee_decision": decision,
        "allowed_uses": row.get("allowed_uses") or [],
        "conflict_tags": row.get("conflict_tags") or [],
        "proxy_usefulness_risk": risk,
        "proxy_label_reason": proxy_label_reason(decision),
        "preview": row.get("preview"),
        "human_labels": {key: None for key in USEFULNESS_LABEL_SCHEMA},
        "human_notes": "",
    }


def proxy_usefulness_risk(row: dict[str, Any]) -> str:
    decision = str(row.get("committee_decision") or "unknown")
    if decision in {"use_as_context", "use_with_caution"}:
        return "potentially_useful_loss"
    if decision in {"reject", "aggregate_only", "quarantine_review"}:
        return "acceptable_exclusion"
    if decision == "inspect_only":
        candidate_score = float(row.get("candidate_score") or 0.0)
        domain_fit_tier = str(row.get("domain_fit_tier") or "")
        conflict_tags = set(row.get("conflict_tags") or [])
        if candidate_score <= 0.025 and domain_fit_tier not in {"strong_fit", "moderate_fit"}:
            return "acceptable_exclusion"
        if conflict_tags & {"relevant_but_unsafe", "safety_exfiltration", "safety_instruction_override"}:
            return "acceptable_exclusion"
        return "review_needed_loss"
    return "review_needed_loss"


def proxy_label_reason(decision: str) -> str:
    if decision == "aggregate_only":
        return "Excluded from ordinary generator context because public-opinion evidence requires aggregation."
    if decision == "inspect_only":
        return "Excluded because the committee did not consider it generator-context eligible; baseline selection makes this a review-needed loss."
    if decision == "reject":
        return "Excluded because safety, integrity, or trust gates rejected it."
    if decision == "quarantine_review":
        return "Excluded pending domain or provenance review."
    if decision in {"use_as_context", "use_with_caution"}:
        return "Potentially useful context was lost; this requires human review."
    return "Unknown committee disposition; review required."


def abstention_row(query_spec: dict[str, str], baseline: dict[str, Any], gated: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "baseline_final_context_count": int(baseline.get("final_context_count") or 0),
        "gated_final_context_count": int(gated.get("final_context_count") or 0),
        "baseline_selected_decision_counts": selected_decision_counts(baseline),
        "proxy_abstention_risk": "needs_human_review" if int(baseline.get("final_context_count") or 0) > 0 else "no_baseline_context",
        "human_labels": {key: None for key in USEFULNESS_LABEL_SCHEMA},
        "human_notes": "",
    }


def selected_decision_counts(run: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in run.get("candidates") or []:
        if row.get("selected_in_final_context"):
            decision = str(row.get("committee_decision") or "unknown")
            counts[decision] = counts.get(decision, 0) + 1
    return dict(sorted(counts.items()))


def mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "lost_baseline_selected_count": sum(int(row["lost_baseline_selected_count"]) for row in rows),
        "potentially_useful_loss_count": sum(int(row["potentially_useful_loss_count"]) for row in rows),
        "review_needed_loss_count": sum(int(row["review_needed_loss_count"]) for row in rows),
        "proxy_acceptable_exclusion_count": sum(int(row["proxy_acceptable_exclusion_count"]) for row in rows),
        "gated_abstention_count": sum(1 for row in rows if row["gated_abstained"]),
        "human_label_task_count": sum(len(row["lost_candidates"]) for row in rows)
        + sum(1 for row in rows if row["abstention_review"]),
    }


def usefulness_summary(modes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode_count": len(modes),
        "query_count": sum(int(mode["query_count"]) for mode in modes),
        "lost_baseline_selected_count": sum(int(mode["summary"]["lost_baseline_selected_count"]) for mode in modes),
        "potentially_useful_loss_count": sum(int(mode["summary"]["potentially_useful_loss_count"]) for mode in modes),
        "review_needed_loss_count": sum(int(mode["summary"]["review_needed_loss_count"]) for mode in modes),
        "proxy_acceptable_exclusion_count": sum(int(mode["summary"]["proxy_acceptable_exclusion_count"]) for mode in modes),
        "gated_abstention_count": sum(int(mode["summary"]["gated_abstention_count"]) for mode in modes),
        "human_label_task_count": sum(int(mode["summary"]["human_label_task_count"]) for mode in modes),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    mode_lines = "\n".join(
        f"- `{mode['mode']}`: lost {mode['summary']['lost_baseline_selected_count']}; "
        f"potentially useful loss {mode['summary']['potentially_useful_loss_count']}; "
        f"review-needed loss {mode['summary']['review_needed_loss_count']}; "
        f"abstentions {mode['summary']['gated_abstention_count']}"
        for mode in report["modes"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    path.write_text(
        f"""# Committee Gate Usefulness Audit

- Baseline policy: `{report['baseline_policy']}`
- Gated policy: `{report['gated_policy']}`
- Lost baseline-selected items: {summary['lost_baseline_selected_count']}
- Potentially useful losses: {summary['potentially_useful_loss_count']}
- Review-needed losses: {summary['review_needed_loss_count']}
- Proxy acceptable exclusions: {summary['proxy_acceptable_exclusion_count']}
- Gated abstentions: {summary['gated_abstention_count']}
- Human label tasks: {summary['human_label_task_count']}

## Modes

{mode_lines}

## Limitations

{limitations}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit possible usefulness loss from committee-gated retrieval.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--baseline-policy", default="rag")
    parser.add_argument("--gated-policy", default="committee_rag")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = evaluate_committee_gate_usefulness(
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        baseline_policy=args.baseline_policy,
        gated_policy=args.gated_policy,
        top_k=args.top_k,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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


def evaluate_committee_gate_comparison(
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
        "report_id": "committee_gate_policy_comparison_v1",
        "baseline_policy": baseline_policy,
        "gated_policy": gated_policy,
        "top_k": resolved_top_k,
        "modes": mode_reports,
        "summary": comparison_summary(mode_reports),
        "claim_boundaries": [
            "Allowed: compare deterministic committee gating against an audit-only retrieval policy on current fixture corpora.",
            "Allowed: treat reduced selected-context policy violations as a safety/usage-control signal.",
            "Not allowed: treat lower violation counts as proof of better answer quality without human usefulness labels.",
            "Not allowed: treat abstention on public-opinion fixtures as production public-opinion validity.",
        ],
        "limitations": [
            "The comparison uses deterministic committee votes, not calibrated human or model-judged evidence decisions.",
            "Retention and abstention are proxy metrics; usefulness requires human labels or external task anchors.",
            "Small fixture corpora can exaggerate both gains and losses from committee gating.",
        ],
    }
    report["contract_validation"] = validate_report(report, "committee_gate_comparison")
    if write_report:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        path = settings.reports_dir / "committee_gate_policy_comparison_v1.json"
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
    pairs = [
        compare_query(
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
        "pairs": pairs,
        "summary": mode_summary(pairs),
    }


def default_queries_for_mode(mode: str) -> list[dict[str, str]]:
    if "social" in mode or "public_opinion" in mode:
        return DEFAULT_COMMITTEE_QUERIES
    return DEFAULT_DOMAIN_QUERIES


def compare_query(
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
    baseline_final = int(baseline.get("final_context_count") or 0)
    gated_final = int(gated.get("final_context_count") or 0)
    baseline_violations = int(baseline.get("selected_context_policy_violation_count") or 0)
    gated_violations = int(gated.get("selected_context_policy_violation_count") or 0)
    return {
        "mode": mode,
        "id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "baseline": run_summary(baseline),
        "gated": run_summary(gated),
        "delta": {
            "selected_context_policy_violation_delta": baseline_violations - gated_violations,
            "final_context_delta": gated_final - baseline_final,
            "final_context_retention_rate": ratio(gated_final, baseline_final),
            "gated_abstained": gated_final == 0,
            "baseline_had_policy_violation": baseline_violations > 0,
            "gated_has_policy_violation": gated_violations > 0,
        },
    }


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": run["policy"],
        "candidate_count": int(run.get("candidate_count") or 0),
        "final_context_count": int(run.get("final_context_count") or 0),
        "selected_context_policy_violation_count": int(run.get("selected_context_policy_violation_count") or 0),
        "committee_decision_counts": run.get("committee_decision_counts") or {},
        "selected_decision_counts": selected_decision_counts(run),
        "stage_summary": run.get("stage_summary") or {},
    }


def selected_decision_counts(run: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in run.get("candidates") or []:
        if row.get("selected_in_final_context"):
            decision = str(row.get("committee_decision") or "unknown")
            counts[decision] = counts.get(decision, 0) + 1
    return dict(sorted(counts.items()))


def mode_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_violations = sum(pair["baseline"]["selected_context_policy_violation_count"] for pair in pairs)
    gated_violations = sum(pair["gated"]["selected_context_policy_violation_count"] for pair in pairs)
    baseline_final = sum(pair["baseline"]["final_context_count"] for pair in pairs)
    gated_final = sum(pair["gated"]["final_context_count"] for pair in pairs)
    return {
        "pair_count": len(pairs),
        "baseline_selected_context_policy_violation_count": baseline_violations,
        "gated_selected_context_policy_violation_count": gated_violations,
        "policy_violation_reduction": baseline_violations - gated_violations,
        "baseline_final_context_count": baseline_final,
        "gated_final_context_count": gated_final,
        "final_context_retention_rate": ratio(gated_final, baseline_final),
        "gated_abstention_count": sum(1 for pair in pairs if pair["delta"]["gated_abstained"]),
    }


def comparison_summary(modes: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_violations = sum(mode["summary"]["baseline_selected_context_policy_violation_count"] for mode in modes)
    gated_violations = sum(mode["summary"]["gated_selected_context_policy_violation_count"] for mode in modes)
    baseline_final = sum(mode["summary"]["baseline_final_context_count"] for mode in modes)
    gated_final = sum(mode["summary"]["gated_final_context_count"] for mode in modes)
    pair_count = sum(mode["summary"]["pair_count"] for mode in modes)
    return {
        "mode_count": len(modes),
        "pair_count": pair_count,
        "baseline_selected_context_policy_violation_count": baseline_violations,
        "gated_selected_context_policy_violation_count": gated_violations,
        "policy_violation_reduction": baseline_violations - gated_violations,
        "baseline_final_context_count": baseline_final,
        "gated_final_context_count": gated_final,
        "final_context_retention_rate": ratio(gated_final, baseline_final),
        "gated_abstention_count": sum(mode["summary"]["gated_abstention_count"] for mode in modes),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    mode_lines = "\n".join(
        f"- `{mode['mode']}`: violations {mode['summary']['baseline_selected_context_policy_violation_count']} -> "
        f"{mode['summary']['gated_selected_context_policy_violation_count']}; retention "
        f"{mode['summary']['final_context_retention_rate']}"
        for mode in report["modes"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    path.write_text(
        f"""# Committee Gate Policy Comparison

- Baseline policy: `{report['baseline_policy']}`
- Gated policy: `{report['gated_policy']}`
- Query pairs: {summary['pair_count']}
- Policy violations: {summary['baseline_selected_context_policy_violation_count']} -> {summary['gated_selected_context_policy_violation_count']}
- Final-context retention: {summary['final_context_retention_rate']}
- Gated abstentions: {summary['gated_abstention_count']}

## Modes

{mode_lines}

## Limitations

{limitations}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare audit-only RAG against committee-gated RAG.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--baseline-policy", default="rag")
    parser.add_argument("--gated-policy", default="committee_rag")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    report = evaluate_committee_gate_comparison(
        modes=modes,
        baseline_policy=args.baseline_policy,
        gated_policy=args.gated_policy,
        top_k=args.top_k,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

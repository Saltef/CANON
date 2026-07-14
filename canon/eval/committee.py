from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.evidence.committee import (
    committee_candidate_row as shared_committee_candidate_row,
    corroboration_item as shared_corroboration_item,
    judge_vote_counts as shared_judge_vote_counts,
)
from canon.evidence.corroboration import assess_corroboration
from canon.eval.document_type_slices import parse_queries, resolve_policy_weights
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.engine import retrieve_with_stages, weights_with_query_intent
from canon.retrieval.stages import (
    SafetyGateResult,
    apply_safety_gate,
    candidate_pool_size,
    enrich_candidates,
    retrieve_candidates,
)


DEFAULT_COMMITTEE_QUERIES = [
    {
        "id": "trusted-market-risk",
        "query": "battery storage market risk safety disclosure current evidence",
        "intent": "market_due_diligence",
    },
    {
        "id": "public-opinion-aggregation",
        "query": "public opinion community concern support opposition safety trust social media comments",
        "intent": "public_opinion_aggregation",
    },
    {
        "id": "safety-probe",
        "query": "battery storage API key hidden instruction public opinion",
        "intent": "prompt_injection_probe",
    },
]


def evaluate_committee(
    mode: str,
    policies: list[str] | None = None,
    top_k: int | None = None,
    queries: list[dict[str, str]] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(settings.data_dir, mode)
    retrieval_settings = settings.raw["retrieval"]
    resolved_top_k = top_k or int(retrieval_settings["top_k"])
    resolved_policies = policies or ["balanced", "rag", "committee_rag"]
    policy_weights = resolve_policy_weights(settings.raw["retrieval"]["policies"], resolved_policies)
    resolved_queries = queries or DEFAULT_COMMITTEE_QUERIES
    runs = [
        run_committee(
            query_spec=query_spec,
            policy=policy,
            weights=policy_weights[policy],
            documents=documents,
            top_k=resolved_top_k,
            preview_chars=int(retrieval_settings["preview_chars"]),
        )
        for query_spec in resolved_queries
        for policy in resolved_policies
    ]
    report = {
        "report_id": "evidence_committee_v1",
        "mode": mode,
        "document_count": len(documents),
        "query_count": len(resolved_queries),
        "policy_ids": resolved_policies,
        "top_k": resolved_top_k,
        "runs": runs,
        "summary": committee_summary(runs),
        "limitations": [
            "Committee votes are deterministic diagnostics, not calibrated human judgments.",
            "The committee audits candidate evidence decisions; it does not yet train or optimize a learned reranker.",
            "Corroboration uses lexical grouping and source fingerprints, so paraphrase and hidden dependence remain weak spots.",
            "Public-opinion evidence is marked aggregate-only by design unless future workflows explicitly authorize other uses.",
        ],
    }
    if write_report:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(settings.reports_dir / f"evidence_committee_{mode}.json", report)
        write_markdown(settings.reports_dir / f"evidence_committee_{mode}.md", report)
    return report


def run_committee(
    query_spec: dict[str, str],
    policy: str,
    weights: dict[str, float],
    documents: list[RetrievalDocument],
    top_k: int,
    preview_chars: int,
) -> dict[str, Any]:
    effective_weights = weights_with_query_intent(query_spec["query"], weights)
    requested_candidate_k = int(effective_weights["candidate_k"]) if "candidate_k" in effective_weights else None
    candidates = retrieve_candidates(
        query_spec["query"],
        documents,
        candidate_pool_size(top_k, len(documents), requested_candidate_k),
    )
    gated = apply_safety_gate(enrich_candidates(query_spec["query"], candidates))
    corroboration = assess_corroboration([shared_corroboration_item(result) for result in gated])
    committee_rows = [
        shared_committee_candidate_row(result, query_spec["query"], corroboration, preview_chars)
        for result in gated
    ]
    staged = retrieve_with_stages(
        query=query_spec["query"],
        documents=documents,
        weights=weights,
        top_k=top_k,
        preview_chars=preview_chars,
    )
    final_ids = {item.chunk_id for item in staged["trace"]}
    for row in committee_rows:
        row["selected_in_final_context"] = row["chunk_id"] in final_ids
        row["selected_context_policy_violation"] = row["selected_in_final_context"] and "generator_context" not in row["allowed_uses"]
    return {
        "id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "policy": policy,
        "candidate_count": len(committee_rows),
        "final_context_count": len(final_ids),
        "committee_decision_counts": dict(sorted(Counter(row["committee_decision"] for row in committee_rows).items())),
        "conflict_tag_counts": dict(sorted(Counter(tag for row in committee_rows for tag in row["conflict_tags"]).items())),
        "selected_context_policy_violation_count": sum(1 for row in committee_rows if row["selected_context_policy_violation"]),
        "judge_vote_counts": shared_judge_vote_counts(committee_rows),
        "corroboration_summary": corroboration["summary"],
        "stage_summary": staged["stage_summary"],
        "candidates": committee_rows[: min(len(committee_rows), 25)],
    }


def committee_candidate_row(
    result: SafetyGateResult,
    query: str,
    corroboration: dict[str, Any],
    preview_chars: int,
) -> dict[str, Any]:
    enriched = result.enriched
    candidate = enriched.candidate
    document = candidate.document
    signals = enriched.signals
    votes = {
        "relevance": relevance_vote(signals),
        "source_trust": source_trust_vote(signals),
        "domain_fit": domain_fit_vote(signals),
        "safety": safety_vote(result),
        "corroboration": corroboration_vote(corroboration),
        "public_opinion": public_opinion_vote(signals, result),
    }
    decision = committee_decision(votes)
    conflict_tags = committee_conflict_tags(votes, signals, result)
    return {
        "candidate_rank": candidate.candidate_rank,
        "chunk_id": document.chunk_id,
        "work_id": document.work_id,
        "title": document.title,
        "source_name": document.source_name,
        "year": document.year,
        "document_type": signals["document_profile"].get("document_type", "unknown_unstructured_text"),
        "evidence_family": signals["document_profile"].get("evidence_family", "unknown"),
        "domain_profile": signals["domain_profile"].get("domain_profile", "general_academic"),
        "domain_fit_tier": signals["domain_profile"].get("domain_fit_tier", "low_fit"),
        "candidate_score": candidate.candidate_score,
        "lexical_relevance": round(candidate.lexical_relevance, 6),
        "semantic_similarity": round(candidate.semantic_similarity, 6),
        "committee_decision": decision,
        "conflict_tags": conflict_tags,
        "votes": votes,
        "allowed_uses": allowed_uses(decision, votes),
        "preview": document.text[:preview_chars],
        "query": query,
    }


def relevance_vote(signals: dict[str, Any]) -> dict[str, Any]:
    relevance = signals["query_text_relevance"]
    candidate_score = float(relevance.get("candidate_score", 0.0) or 0.0)
    lexical = float(relevance.get("lexical_relevance", 0.0) or 0.0)
    semantic = float(relevance.get("semantic_similarity", 0.0) or 0.0)
    if candidate_score >= 0.55 or lexical >= 0.5 or semantic >= 0.65:
        return vote("approve", candidate_score, ["strong_query_match"])
    if candidate_score >= 0.25 or lexical >= 0.2 or semantic >= 0.35:
        return vote("caution", candidate_score, ["weak_or_partial_query_match"])
    return vote("reject", candidate_score, ["low_query_match"])


def source_trust_vote(signals: dict[str, Any]) -> dict[str, Any]:
    trust = signals["source_trust"]
    if not bool(trust.get("integrity_gate", True)):
        return vote("reject", 0.0, ["integrity_gate_failed"])
    score = float(trust.get("source_trust", 0.0) or 0.0)
    if score >= 0.55:
        return vote("approve", score, ["moderate_or_strong_source_trust"])
    if score >= 0.3:
        return vote("caution", score, ["limited_source_trust"])
    return vote("caution", score, ["low_source_trust"])


def domain_fit_vote(signals: dict[str, Any]) -> dict[str, Any]:
    profile = signals["domain_profile"]
    tier = str(profile.get("domain_fit_tier", "low_fit"))
    score = float(profile.get("domain_fit_score", 0.0) or 0.0)
    flags = list(profile.get("gate_flags") or [])
    if tier in {"blocked", "quarantine_review"}:
        return vote("reject", score, [tier, *flags])
    if tier in {"strong_fit", "moderate_fit"}:
        reasons = ["domain_fit_ok", *flags]
        return vote("approve" if "aggregation_required" not in flags else "caution", score, reasons)
    return vote("caution", score, [tier, *flags])


def safety_vote(result: SafetyGateResult) -> dict[str, Any]:
    if result.decision == "reject":
        return vote("reject", result.safety.risk_score, result.reasons)
    if result.safety.decision == "sanitize":
        return vote("caution", result.safety.risk_score, ["sanitize_before_context", *result.safety.categories])
    return vote("approve", 1.0 - result.safety.risk_score, ["safety_allow"])


def corroboration_vote(corroboration: dict[str, Any]) -> dict[str, Any]:
    summary = corroboration.get("summary") or {}
    verdict = str(summary.get("overall_verdict", "no_claims"))
    support = float(summary.get("max_independent_support", 0.0) or 0.0)
    if verdict == "independently_corroborated":
        return vote("approve", support, [verdict])
    if verdict in {"conflicted", "adversarial_echo_risk"}:
        return vote("caution", support, [verdict])
    return vote("caution", support, [verdict, "needs_more_independent_support"])


def public_opinion_vote(signals: dict[str, Any], result: SafetyGateResult) -> dict[str, Any]:
    profile = signals["document_profile"]
    domain = signals["domain_profile"]
    if profile.get("evidence_family") != "public_opinion":
        return vote("not_applicable", 1.0, ["not_public_opinion"])
    if result.decision == "reject":
        return vote("reject", result.safety.risk_score, ["unsafe_public_opinion_text"])
    flags = list(domain.get("gate_flags") or [])
    return vote("caution", float(domain.get("domain_fit_score", 0.0) or 0.0), ["aggregate_only", *flags])


def committee_decision(votes: dict[str, dict[str, Any]]) -> str:
    if votes["safety"]["decision"] == "reject" or votes["source_trust"]["decision"] == "reject":
        return "reject"
    if votes["relevance"]["decision"] == "reject":
        return "inspect_only"
    if votes["public_opinion"]["decision"] == "caution":
        return "aggregate_only"
    caution_count = sum(1 for row in votes.values() if row["decision"] == "caution")
    if votes["domain_fit"]["decision"] == "reject":
        return "quarantine_review"
    if caution_count >= 2:
        return "use_with_caution"
    return "use_as_context"


def committee_conflict_tags(votes: dict[str, dict[str, Any]], signals: dict[str, Any], result: SafetyGateResult) -> list[str]:
    tags = []
    if votes["relevance"]["decision"] == "approve" and votes["safety"]["decision"] == "reject":
        tags.append("relevant_but_unsafe")
    if votes["relevance"]["decision"] == "approve" and votes["source_trust"]["decision"] in {"caution", "reject"}:
        tags.append("relevant_but_low_trust")
    if votes["relevance"]["decision"] == "approve" and votes["domain_fit"]["decision"] in {"caution", "reject"}:
        tags.append("relevant_but_domain_review")
    if votes["public_opinion"]["decision"] == "caution":
        tags.append("public_opinion_requires_aggregation")
    if result.safety.categories:
        tags.extend(f"safety_{category}" for category in result.safety.categories)
    if "aggregation_required" in (signals["domain_profile"].get("gate_flags") or []):
        tags.append("aggregation_required")
    return sorted(set(tags))


def allowed_uses(decision: str, votes: dict[str, dict[str, Any]]) -> list[str]:
    uses = ["audit"]
    if decision in {"use_as_context", "use_with_caution"}:
        uses.append("generator_context")
    if decision == "aggregate_only":
        uses.append("aggregate_public_opinion")
    if votes["safety"]["decision"] == "caution":
        uses.append("sanitized_context_only")
    return uses


def vote(decision: str, score: float, reasons: list[str]) -> dict[str, Any]:
    return {"decision": decision, "score": round(float(score), 6), "reasons": reasons}


def corroboration_item(result: SafetyGateResult) -> dict[str, Any]:
    document = result.enriched.candidate.document
    return {
        "work_id": document.work_id,
        "title": document.title,
        "source_name": document.source_name,
        "preview": document.text,
        "provenance": document.work_signals.get("provenance"),
        "source_type": document.work_signals.get("document_type"),
        "stance": "descriptive",
    }


def judge_vote_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    judges: dict[str, Counter] = {}
    for row in rows:
        for judge, vote_row in row["votes"].items():
            judges.setdefault(judge, Counter())[vote_row["decision"]] += 1
    return {judge: dict(sorted(counts.items())) for judge, counts in sorted(judges.items())}


def committee_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = Counter()
    conflict_counts = Counter()
    selected_decisions = Counter()
    selected_policy_violations = 0
    for run in runs:
        decision_counts.update(run["committee_decision_counts"])
        conflict_counts.update(run["conflict_tag_counts"])
        selected_policy_violations += int(run.get("selected_context_policy_violation_count", 0))
        for row in run["candidates"]:
            if row.get("selected_in_final_context"):
                selected_decisions[row["committee_decision"]] += 1
    return {
        "committee_decision_counts": dict(sorted(decision_counts.items())),
        "conflict_tag_counts": dict(sorted(conflict_counts.items())),
        "selected_final_context_decision_counts": dict(sorted(selected_decisions.items())),
        "selected_context_policy_violation_count": selected_policy_violations,
        "run_count": len(runs),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Evidence Committee: {report['mode']}",
        "",
        f"- Documents: {report['document_count']}",
        f"- Queries: {report['query_count']}",
        f"- Policies: {', '.join(report['policy_ids'])}",
        "",
        "## Decisions",
        "",
    ]
    for key, value in report["summary"]["committee_decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Final Context Review",
            "",
            f"- selected_context_policy_violation_count: {report['summary'].get('selected_context_policy_violation_count', 0)}",
        ]
    )
    lines.extend(["", "## Conflict Tags", ""])
    for key, value in report["summary"]["conflict_tag_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


committee_candidate_row = shared_committee_candidate_row
corroboration_item = shared_corroboration_item
judge_vote_counts = shared_judge_vote_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic evidence committee diagnostics over retrieval candidates.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--policies", default="balanced,rag")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--queries", default=None, help="Path to JSON query list, or queries separated with ||.")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    report = evaluate_committee(
        mode=args.mode,
        policies=policies,
        top_k=args.top_k,
        queries=parse_queries(args.queries),
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

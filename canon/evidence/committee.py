from __future__ import annotations

from collections import Counter
from typing import Any

from canon.retrieval.stages import SafetyGateResult


GENERATOR_CONTEXT_DECISIONS = {"use_as_context", "use_with_caution"}


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
        "syntax_relevance": round(candidate.syntax_relevance, 6),
        "syntax_category": signals["syntax_profile"].get("document_syntax_category", "generic_claim"),
        "committee_decision": decision,
        "conflict_tags": conflict_tags,
        "votes": votes,
        "allowed_uses": allowed_uses(decision, votes),
        "preview": document.text[:preview_chars],
        "query": query,
    }


def generator_context_allowed(row: dict[str, Any]) -> bool:
    return "generator_context" in (row.get("allowed_uses") or [])


def relevance_vote(signals: dict[str, Any]) -> dict[str, Any]:
    relevance = signals["query_text_relevance"]
    candidate_score = float(relevance.get("candidate_score", 0.0) or 0.0)
    lexical = max(
        float(relevance.get("lexical_relevance", 0.0) or 0.0),
        float(relevance.get("focus_adjusted_relevance", 0.0) or 0.0),
    )
    semantic = max(
        float(relevance.get("semantic_similarity", 0.0) or 0.0),
        float(relevance.get("focus_adjusted_semantic_similarity", 0.0) or 0.0),
    )
    syntax = float(relevance.get("syntax_relevance", 0.0) or 0.0)
    if candidate_score >= 0.55 or lexical >= 0.5 or semantic >= 0.65:
        return vote("approve", candidate_score, ["strong_query_match"])
    if candidate_score >= 0.25 or lexical >= 0.2 or semantic >= 0.35 or syntax >= 0.7:
        reasons = ["weak_or_partial_query_match"]
        if syntax >= 0.7:
            reasons.append("relevant_evidence_form")
        return vote("caution", max(candidate_score, syntax), reasons)
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


def committee_conflict_tags(
    votes: dict[str, dict[str, Any]],
    signals: dict[str, Any],
    result: SafetyGateResult,
) -> list[str]:
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
    if decision in GENERATOR_CONTEXT_DECISIONS:
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

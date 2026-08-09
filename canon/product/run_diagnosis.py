from __future__ import annotations

from typing import Any

from canon.retrieval.query_diagnostics import content_terms


PASS = "pass"
WARN = "warn"
FAIL = "fail"
REVIEW = "review_required"


def build_run_diagnosis(
    *,
    query: str,
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
    relevance_gate: dict[str, Any] | None = None,
    draft_brief: dict[str, Any] | None = None,
    retrieval: dict[str, Any] | None = None,
    coverage_gaps: list[dict[str, Any]] | None = None,
    query_diagnostics: dict[str, Any] | None = None,
    external_cards: list[dict[str, Any]] | None = None,
    frame_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize why an evidence run is or is not usable for inspection."""
    relevance_gate = relevance_gate or {}
    draft_brief = draft_brief or {}
    retrieval = retrieval or {}
    coverage_gaps = coverage_gaps if coverage_gaps is not None else packet_response.get("coverage_gaps") or []
    query_diagnostics = query_diagnostics or packet_response.get("query_diagnostics") or {}
    external_cards = external_cards or []
    frame_coverage = frame_coverage or packet_frame_coverage(packet_response)
    support = (packet_response.get("answer_report") or {}).get("support_assessment") or {}
    query_to_corpus = dict_or_empty(query_diagnostics.get("query_to_corpus"))

    focus_terms = list_or_default(relevance_gate.get("focus_terms"), content_terms(query))
    matched_terms = list_or_default(
        relevance_gate.get("matched_terms"),
        query_diagnostics.get("matched_terms")
        or query_to_corpus.get("matched_terms")
        or support.get("covered_focus_terms")
        or [],
    )
    weak_terms = list_or_default(
        relevance_gate.get("weak_terms"),
        query_diagnostics.get("weak_terms")
        or query_to_corpus.get("weak_terms")
        or support.get("missing_focus_terms")
        or [],
    )
    term_coverage = numeric_or_default(
        relevance_gate.get("query_term_coverage"),
        support.get("query_term_coverage"),
        ratio(len(matched_terms), len(focus_terms)),
    )
    usable_evidence_count = int_or_default(relevance_gate.get("usable_evidence_count"), len(evidence))
    diagnostic_count = int_or_default(
        relevance_gate.get("diagnostic_candidate_count"),
        max(len(evidence) - usable_evidence_count, 0),
    )

    stages = [
        corpus_fit_stage(term_coverage, matched_terms, weak_terms, relevance_gate, packet_response),
        candidate_retrieval_stage(packet_response, evidence, usable_evidence_count, diagnostic_count, retrieval),
        rerank_stage(packet_response, retrieval),
        generation_stage(draft_brief, support, relevance_gate),
        review_stage(coverage_gaps, frame_coverage, external_cards),
    ]
    issue_categories = classify_issues(
        packet_response=packet_response,
        evidence=evidence,
        relevance_gate=relevance_gate,
        draft_brief=draft_brief,
        coverage_gaps=coverage_gaps,
        stages=stages,
    )
    overall_status = overall_status_from_issues(issue_categories, coverage_gaps)
    failure_class = primary_failure_class(issue_categories)
    return {
        "report_id": "run_diagnosis_v1",
        "overall_status": overall_status,
        "failure_class": failure_class,
        "summary": diagnosis_summary(overall_status, failure_class, stages),
        "issue_categories": issue_categories,
        "scorecard": {stage["stage"]: stage["status"] for stage in stages},
        "stages": stages,
        "metrics": {
            "focus_term_count": len(focus_terms),
            "matched_term_count": len(matched_terms),
            "weak_term_count": len(weak_terms),
            "query_term_coverage": round(float(term_coverage), 6),
            "candidate_evidence_count": len(evidence),
            "usable_evidence_count": usable_evidence_count,
            "diagnostic_candidate_count": diagnostic_count,
            "coverage_gap_count": len(coverage_gaps),
            "external_evidence_count": len(external_cards),
        },
        "human_review_required": True,
        "safe_to_use_for": [
            "debugging retrieval and generation behavior",
            "inspecting candidate evidence",
            "deciding whether to refine the query or corpus",
        ],
        "not_safe_to_claim": [
            "final factual correctness",
            "complete corpus coverage",
            "benchmark or model superiority",
        ],
        "recommended_actions": recommended_actions(issue_categories, stages, coverage_gaps, weak_terms),
    }


def corpus_fit_stage(
    term_coverage: float,
    matched_terms: list[str],
    weak_terms: list[str],
    relevance_gate: dict[str, Any],
    packet_response: dict[str, Any],
) -> dict[str, Any]:
    if packet_response.get("status") != "complete":
        status = FAIL
        message = "The run did not complete, so corpus fit was not established."
    elif relevance_gate.get("status") == "corpus_mismatch" or (term_coverage == 0 and weak_terms):
        status = FAIL
        message = "The selected corpus does not visibly cover the query."
    elif term_coverage < 0.35 or weak_terms:
        status = WARN
        message = "The corpus partially covers the query, but important terms remain weak."
    else:
        status = PASS
        message = "The query has visible overlap with retrieved corpus evidence."
    return {
        "stage": "corpus_fit",
        "status": status,
        "message": message,
        "signals": {
            "matched_terms": matched_terms,
            "weak_terms": weak_terms,
            "query_term_coverage": round(float(term_coverage), 6),
            "relevance_gate_status": relevance_gate.get("status"),
        },
    }


def candidate_retrieval_stage(
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
    usable_evidence_count: int,
    diagnostic_count: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    if packet_response.get("status") != "complete":
        status = FAIL
        message = "Retrieval failed before producing candidate evidence."
    elif not evidence:
        status = FAIL
        message = "No candidate evidence was retrieved."
    elif usable_evidence_count == 0:
        status = FAIL
        message = "Candidates were retrieved, but none passed the relevance gate."
    elif diagnostic_count:
        status = WARN
        message = "Some retrieved candidates were retained only for debugging."
    else:
        status = PASS
        message = "Candidate retrieval produced usable evidence for inspection."
    return {
        "stage": "candidate_retrieval",
        "status": status,
        "message": message,
        "signals": {
            "retrieval_engine": retrieval.get("engine"),
            "candidate_count": retrieval.get("candidate_count"),
            "evidence_count": len(evidence),
            "usable_evidence_count": usable_evidence_count,
            "diagnostic_candidate_count": diagnostic_count,
        },
    }


def rerank_stage(packet_response: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    reranker_provider = retrieval.get("reranker_provider") or packet_response.get("reranker_provider")
    reranker_model = retrieval.get("reranker_model") or packet_response.get("reranker_model")
    rerank_summary = packet_response.get("rerank_summary") or {}
    if packet_response.get("status") != "complete":
        status = FAIL
        message = "Reranking could not be evaluated because retrieval failed."
    elif reranker_provider:
        status = PASS if rerank_summary.get("status") in {None, "ok"} else WARN
        message = "A reranker ordered the candidate set for inspection."
    else:
        status = WARN
        message = "No explicit reranker trace is available for this run."
    return {
        "stage": "reranking",
        "status": status,
        "message": message,
        "signals": {
            "provider": reranker_provider,
            "model": reranker_model,
            "rerank_status": rerank_summary.get("status"),
            "scored_candidate_count": rerank_summary.get("scored_candidate_count"),
        },
    }


def generation_stage(
    draft_brief: dict[str, Any],
    support: dict[str, Any],
    relevance_gate: dict[str, Any],
) -> dict[str, Any]:
    draft_status = draft_brief.get("status")
    if relevance_gate and not relevance_gate.get("draft_allowed", True):
        status = FAIL
        message = "Drafting was blocked because the evidence did not pass the relevance gate."
    elif draft_status in {"model_generation_failed", "model_generation_empty", "insufficient_cited_evidence"}:
        status = WARN
        message = "Generation did not produce a usable cited draft."
    elif draft_status in {"model_draft_ready", "evidence_note_ready"}:
        status = PASS
        message = "A cited draft or evidence note was produced from gated evidence."
    elif draft_status:
        status = WARN
        message = "Generation returned a nonstandard status that needs inspection."
    else:
        status = WARN
        message = "No draft trace is attached to this diagnosis."
    return {
        "stage": "generation_grounding",
        "status": status,
        "message": message,
        "signals": {
            "draft_status": draft_status,
            "citation_count": draft_brief.get("citation_count", 0),
            "support_level": support.get("support_level"),
            "support_confidence": support.get("support_confidence"),
        },
    }


def review_stage(
    coverage_gaps: list[dict[str, Any]],
    frame_coverage: dict[str, Any],
    external_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    if coverage_gaps or frame_coverage.get("human_review_required"):
        status = REVIEW
        message = "Human review is required before treating the result as complete."
    else:
        status = REVIEW
        message = "Human review is still required for conclusions and citation validity."
    return {
        "stage": "review_need",
        "status": status,
        "message": message,
        "signals": {
            "coverage_gap_count": len(coverage_gaps),
            "frame_coverage_status": frame_coverage.get("status"),
            "external_evidence_count": len(external_cards),
            "external_results_are_diagnostic": bool(external_cards),
        },
    }


def classify_issues(
    *,
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
    relevance_gate: dict[str, Any],
    draft_brief: dict[str, Any],
    coverage_gaps: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    if packet_response.get("status") != "complete":
        issues.append("retrieval_run_failed")
    if not evidence:
        issues.append("no_candidate_evidence")
    gate_status = relevance_gate.get("status")
    if gate_status == "corpus_mismatch":
        issues.append("corpus_mismatch")
    elif gate_status in {"weak_query_match", "weak_candidate_overlap"}:
        issues.append("weak_candidate_overlap")
    elif any(
        stage.get("stage") in {"corpus_fit", "candidate_retrieval"} and stage.get("status") == WARN
        for stage in stages
    ):
        issues.append("weak_candidate_overlap")
    if draft_brief.get("status") == "blocked_by_relevance_gate":
        issues.append("draft_blocked_by_relevance_gate")
    elif draft_brief.get("status") in {"model_generation_failed", "model_generation_empty"}:
        issues.append("generation_failed")
    elif draft_brief.get("status") == "insufficient_cited_evidence":
        issues.append("insufficient_cited_evidence")
    if coverage_gaps:
        issues.append("coverage_gap")
    if not any(stage["status"] == FAIL for stage in stages):
        issues.append("human_review_required")
    return dedupe(issues) or ["none"]


def overall_status_from_issues(issue_categories: list[str], coverage_gaps: list[dict[str, Any]]) -> str:
    hard_failures = {
        "retrieval_run_failed",
        "no_candidate_evidence",
        "corpus_mismatch",
        "draft_blocked_by_relevance_gate",
    }
    if any(issue in hard_failures for issue in issue_categories):
        return "failed_no_grounded_answer"
    if "generation_failed" in issue_categories or "insufficient_cited_evidence" in issue_categories:
        return "needs_generation_review"
    if coverage_gaps or "weak_candidate_overlap" in issue_categories:
        return "ready_with_caveats"
    return "ready_for_user_inspection"


def primary_failure_class(issue_categories: list[str]) -> str:
    priority = [
        "retrieval_run_failed",
        "no_candidate_evidence",
        "corpus_mismatch",
        "draft_blocked_by_relevance_gate",
        "weak_candidate_overlap",
        "generation_failed",
        "insufficient_cited_evidence",
        "coverage_gap",
        "human_review_required",
    ]
    for issue in priority:
        if issue in issue_categories:
            return issue
    return "none"


def diagnosis_summary(overall_status: str, failure_class: str, stages: list[dict[str, Any]]) -> str:
    if overall_status == "failed_no_grounded_answer":
        failed_stage = next((stage for stage in stages if stage["status"] == FAIL), {})
        return failed_stage.get("message") or "The run did not produce a grounded answer."
    if overall_status == "needs_generation_review":
        return "Retrieval produced candidates, but generation did not produce a usable cited draft."
    if overall_status == "ready_with_caveats":
        return "The run produced usable evidence, but coverage or query-fit caveats remain."
    if failure_class == "human_review_required":
        return "The run is ready for user inspection, with human review still required for conclusions."
    return "The run is ready for user inspection."


def recommended_actions(
    issue_categories: list[str],
    stages: list[dict[str, Any]],
    coverage_gaps: list[dict[str, Any]],
    weak_terms: list[str],
) -> list[str]:
    actions: list[str] = []
    if "retrieval_run_failed" in issue_categories:
        actions.append("Inspect the retrieval error and provider configuration before rerunning.")
    if "no_candidate_evidence" in issue_categories:
        actions.append("Increase candidate depth or mount a corpus with better topic coverage.")
    if "corpus_mismatch" in issue_categories:
        actions.append("Switch corpus or enable clearly marked external search before drafting.")
    if "weak_candidate_overlap" in issue_categories and weak_terms:
        actions.append("Refine the query around weak terms: " + ", ".join(weak_terms[:6]) + ".")
    for gap in coverage_gaps[:3]:
        suggested = str(gap.get("suggested_next_query") or "").strip()
        actions.append(f"Run follow-up query: {suggested}" if suggested else str(gap.get("gap") or "Inspect coverage gap."))
    if "generation_failed" in issue_categories:
        actions.append("Retry the two-model generation run with the same gated evidence and inspect model_comparison.")
    if not actions:
        actions.append("Inspect cited evidence and source diversity before using the draft.")
    actions.append("Record user feedback separately from authoritative human-review labels.")
    return dedupe(actions)[:6]


def packet_frame_coverage(packet_response: dict[str, Any]) -> dict[str, Any]:
    for packet in packet_response.get("evidence_packets") or []:
        frame_coverage = packet.get("frame_coverage")
        if isinstance(frame_coverage, dict):
            return frame_coverage
    return {}


def list_or_default(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return list(default)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def numeric_or_default(*values: Any) -> float:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def int_or_default(value: Any, default: int) -> int:
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        return default
    return default


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.corpus.build import run_phase16
from canon.eval.model_evaluation import evaluate_semantic_models, parse_providers
from canon.eval.diversity import run_diversity_audit
from canon.product import report_io
from canon.eval.source_diversity import (
    DEFAULT_MAX_DOMINANT_SOURCE_SHARE,
    DEFAULT_MIN_CLUSTER_COUNT,
    DEFAULT_MIN_DISTINCT_SOURCES,
    source_diversity_row,
)
from canon.ingest.flexible import ingest_flexible_source, profile_source
from canon.reports.claim_decision import build_claim_decision
from canon.retrieval.compare import compare
from canon.retrieval.experiment import cached_documents
from canon.retrieval.query_diagnostics import FREEDOM_THRESHOLDS, diagnose_query
from canon.synthesis.answer import synthesize


DEFAULT_MODE = "social_science_ir_v1_harvest10"
DEFAULT_DIVERSITY_MODE = "social_science_ir_10k"
DEFAULT_POLICY = "rag"
DEFAULT_POLICIES = ["lexical", "balanced", "semantic", "rag", "diverse", "conflict_aware"]
DEFAULT_DIVERSE_METHOD_ID = "diverse_k5_template"
DEFAULT_DIVERSITY_BASELINE_METHOD_ID = "lexical_k5_template"


class ProductError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def health() -> dict:
    settings = load_settings()
    return {
        "status": "ok",
        "service": "canon",
        "reports_dir": str(settings.reports_dir).replace("\\", "/"),
        "data_dir": str(settings.data_dir).replace("\\", "/"),
    }


def product_summary(mode: str = DEFAULT_MODE) -> dict:
    settings = load_settings()
    audit = load_report(settings.reports_dir / f"scientific_audit_{mode}_baseline_methods_v1.json")
    decision = load_report(settings.reports_dir / f"claim_decision_{mode}_baseline_methods_v1.json")
    data_card = load_report(settings.reports_dir / f"data_card_{mode}.json")
    return {
        "product": "CANON Evidence Workbench",
        "mode": mode,
        "positioning": "importance-aware scholarly RAG with explicit scientific claim boundaries",
        "primary_jobs": [
            "answer scholarly questions with cited evidence",
            "compare retrieval policies and importance signals",
            "surface disagreement and weak-support limitations",
            "export reproducible scientific audit artifacts",
        ],
        "claim_boundaries": decision.get("global_winner_claim", {}),
        "audit_status": audit.get("status"),
        "active_warnings": [warning["id"] for warning in audit.get("warnings", [])],
        "resolved_warnings": [warning["id"] for warning in audit.get("resolved_warnings", [])],
        "corpus": {
            "work_count": data_card.get("work_count"),
            "chunk_count": data_card.get("chunk_count"),
            "limitations": data_card.get("limitations", []),
        },
    }


def answer(payload: dict[str, Any]) -> dict:
    query = require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    top_k = optional_int(payload.get("top_k"))
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    report = synthesize(query=query, policy=policy, mode=mode, top_k=top_k)
    decision = safe_claim_decision(mode)
    return {
        "query": report["query"],
        "mode": report["mode"],
        "policy": report["policy"],
        "answer": report["answer"],
        "citations": report["citations"],
        "evidence": compact_answer_evidence(report["evidence"]),
        "support_assessment": report["support_assessment"],
        "limitations": report["limitations"],
        "conflict_notes": report["conflict_notes"],
        "claim_boundaries": decision.get("global_winner_claim", {}),
        "query_diagnostics": build_query_diagnostics(
            query=query,
            mode=mode,
            top_k=top_k or report.get("top_k"),
            freedom_level=freedom_level,
        ),
    }


def evidence_packets(payload: dict[str, Any]) -> dict:
    query = optional_text(payload, "question") or require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    research_frame = optional_dict(payload.get("research_frame"), "research_frame")
    requirements = optional_dict(payload.get("evidence_requirements"), "evidence_requirements")
    top_k = optional_positive_int(requirements.get("top_k"), "top_k") if requirements else None
    top_k = top_k or optional_int(payload.get("top_k"))
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    report = synthesize(query=query, policy=policy, mode=mode, top_k=top_k)
    evidence = enrich_evidence_metadata(report.get("evidence") or [], mode)
    diagnostics = build_query_diagnostics(
        query=query,
        mode=mode,
        top_k=top_k or report.get("top_k"),
        freedom_level=freedom_level,
    )
    diversity = packet_source_diversity(query, evidence, top_k or report.get("top_k") or 10)
    frame_coverage = analyze_frame_coverage(
        research_frame=research_frame,
        requirements=requirements,
        evidence=evidence,
        diagnostics=diagnostics,
    )
    support = report.get("support_assessment") or {}
    packet = {
        "packet_id": f"packet_{str(payload.get('request_id') or '001')}",
        "claim": packet_claim(query, support),
        "support_level": packet_support_level(support),
        "confidence": packet_confidence(support),
        "evidence_role": "direct_support" if evidence else "insufficient_evidence",
        "issue_categories": infer_issue_categories(research_frame, query),
        "regions": optional_string_list(research_frame.get("regions"), "regions") or [],
        "languages": optional_string_list(research_frame.get("languages"), "languages") or [],
        "source_types": optional_string_list(requirements.get("minimum_source_types"), "minimum_source_types")
        if requirements
        else [],
        "supporting_evidence": [packet_evidence_item(item) for item in evidence],
        "conflicting_evidence": report.get("conflict_notes", []),
        "limitations": report.get("limitations", []),
        "source_diversity": {
            "distinct_sources": diversity["distinct_sources"],
            "distinct_clusters": diversity["distinct_clusters"],
            "dominant_source_share": diversity["dominant_source_share"],
            "warnings": diversity["warnings"],
        },
        "frame_coverage": frame_coverage,
    }
    gaps = coverage_gaps(diagnostics, diversity, report.get("limitations", []), frame_coverage)
    return {
        "request_id": str(payload.get("request_id") or ""),
        "project_id": str(payload.get("project_id") or ""),
        "status": "complete",
        "query": query,
        "mode": mode,
        "policy": policy,
        "research_frame": research_frame,
        "evidence_packets": [packet],
        "query_diagnostics": compact_packet_diagnostics(diagnostics),
        "frame_coverage": frame_coverage,
        "coverage_gaps": gaps,
        "retrieval_metrics": {
            "estimated_confidence": packet["confidence"],
            "source_diversity_status": "pass" if not diversity["warnings"] else "review",
            "human_review_status": str(payload.get("human_review_status") or "not_reviewed"),
            "support_confidence": support.get("support_confidence"),
        },
        "answer_report": {
            "answer": report.get("answer"),
            "citations": report.get("citations", []),
            "support_assessment": support,
        },
    }


def frame_coverage_report(payload: dict[str, Any]) -> dict:
    packet_response = evidence_packets(payload)
    question = packet_response.get("query", "")
    frame_coverage = packet_response.get("frame_coverage", {})
    status = frame_coverage_status(frame_coverage)
    report = {
        "report_id": "research_frame_coverage_v1",
        "status": status,
        "request_id": packet_response.get("request_id", ""),
        "project_id": packet_response.get("project_id", ""),
        "question": question,
        "mode": packet_response.get("mode", ""),
        "policy": packet_response.get("policy", ""),
        "research_frame": packet_response.get("research_frame", {}),
        "frame_coverage": frame_coverage,
        "coverage_gaps": packet_response.get("coverage_gaps", []),
        "query_diagnostics": packet_response.get("query_diagnostics", {}),
        "retrieval_metrics": packet_response.get("retrieval_metrics", {}),
        "human_review_required": True,
        "human_review_boundary": (
            "Frame coverage is inferred from retrieved evidence and query diagnostics. "
            "It is useful for gap-finding, but it is not proof that all relevant evidence was found."
        ),
        "next_actions": frame_coverage_next_actions(frame_coverage, packet_response.get("coverage_gaps", [])),
    }
    if optional_bool(payload.get("write_report"), default=True):
        settings = load_settings()
        output = settings.reports_dir / f"frame_coverage_{safe_report_slug(report['mode'])}_{safe_report_slug(question)}.json"
        report_io.write_json(output, report)
        report["output_path"] = str(output).replace("\\", "/")
    return report


def frame_coverage_status(frame_coverage: dict[str, Any]) -> str:
    status = frame_coverage.get("status")
    if status == "pass":
        return "pass_human_review_required"
    if status in {"partial", "missing"}:
        return "coverage_gap_human_review_required"
    if status == "not_requested":
        return "not_requested"
    return "needs_review"


def frame_coverage_next_actions(
    frame_coverage: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> list[str]:
    actions = [
        gap.get("suggested_next_query")
        for gap in gaps
        if gap.get("suggested_next_query")
    ]
    if not actions:
        for row in frame_coverage.get("diagnostics", []):
            if row.get("suggested_next_query"):
                actions.append(row["suggested_next_query"])
    return actions[:5] or ["Review frame coverage and evidence citations before treating the result as complete."]


def safe_report_slug(text: Any, limit: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug[:limit].strip("-") or "report"


def intelligence_brief(payload: dict[str, Any]) -> dict:
    from canon.intelligence.evidence_runner import run_intelligence_brief

    question = optional_text(payload, "question") or require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    research_frame = optional_dict(payload.get("research_frame"), "research_frame")
    requirements = optional_dict(payload.get("evidence_requirements"), "evidence_requirements")
    return run_intelligence_brief(
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        research_frame=research_frame or None,
        evidence_requirements=requirements or None,
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def report_quality_gate(payload: dict[str, Any]) -> dict:
    brief_report = intelligence_brief({**payload, "write_report": optional_bool(payload.get("write_report"), default=False)})
    quality = brief_report.get("report_quality") or {}
    grounding = brief_report.get("grounding_report") or {}
    duplicate = brief_report.get("duplicate_agent_output") or {}
    red_team = brief_report.get("red_team") or {}
    status = report_quality_status(brief_report, quality, grounding, red_team)
    report = {
        "report_id": "intelligence_report_quality_gate_v1",
        "status": status,
        "project_id": brief_report.get("project_id", ""),
        "mode": brief_report.get("mode", ""),
        "policy": brief_report.get("policy", ""),
        "question": brief_report.get("question", ""),
        "source_report_id": brief_report.get("report_id", ""),
        "source_report_status": brief_report.get("status", ""),
        "quality_status": quality.get("status", "missing"),
        "checks": quality.get("checks", []),
        "grounding": {
            "status": grounding.get("status", "missing"),
            "claim_count": grounding.get("claim_count", 0),
            "supported_claim_count": grounding.get("supported_claim_count", 0),
            "unsupported_claim_count": grounding.get("unsupported_claim_count", 0),
            "grounded_claim_ratio": grounding.get("grounded_claim_ratio", 0.0),
            "unsupported_claims": grounding.get("unsupported_claims", []),
        },
        "duplicate_agent_output": duplicate,
        "red_team": {
            "status": red_team.get("status", "missing"),
            "blocking_issue_count": red_team.get("blocking_issue_count", 0),
            "review_issue_count": red_team.get("review_issue_count", 0),
            "objections": red_team.get("objections", []),
        },
        "human_review_required": True,
        "human_review_boundary": (
            "This gate checks structure, citation grounding, duplicate output, and red-team blockers. "
            "It does not prove analyst usefulness, factual correctness, completeness, or publishability."
        ),
        "next_actions": report_quality_next_actions(quality, grounding, red_team),
    }
    if optional_bool(payload.get("write_report"), default=True):
        settings = load_settings()
        output = (
            settings.reports_dir
            / f"report_quality_gate_{safe_report_slug(report['mode'])}_{safe_report_slug(report['question'])}.json"
        )
        report_io.write_json(output, report)
        report["output_path"] = str(output).replace("\\", "/")
    return report


def report_quality_status(
    brief_report: dict[str, Any],
    quality: dict[str, Any],
    grounding: dict[str, Any],
    red_team: dict[str, Any],
) -> str:
    if int(red_team.get("blocking_issue_count") or 0) > 0 or grounding.get("status") == "fail":
        return "blocked_grounding"
    if brief_report.get("status") == "ready_for_human_review" and quality.get("status") == "pass":
        return "pass_human_review_required"
    return "review_required"


def report_quality_next_actions(
    quality: dict[str, Any],
    grounding: dict[str, Any],
    red_team: dict[str, Any],
) -> list[str]:
    actions = []
    for check_row in quality.get("checks", []):
        if check_row.get("status") == "fail":
            actions.append(f"Fix failed quality check: {check_row.get('id')}.")
    if grounding.get("unsupported_claims"):
        actions.append("Resolve or remove claims with missing or unknown evidence IDs.")
    for objection in red_team.get("objections", []):
        if objection.get("severity") == "blocker":
            actions.append(f"Resolve red-team blocker: {objection.get('issue')}.")
    return actions[:5] or ["Send the report to human review for usefulness, factual correctness, and missing perspectives."]


def start_project(payload: dict[str, Any]) -> dict:
    from canon.product.project import build_project_config

    return build_project_config(
        project_name=require_text(payload, "project_name"),
        project_id=optional_text(payload, "project_id"),
        domain=require_text(payload, "domain"),
        regions=required_string_list(payload.get("regions"), "regions"),
        languages=required_string_list(payload.get("languages"), "languages"),
        issue_categories=required_string_list(payload.get("issue_categories"), "issue_categories"),
        desired_report_types=required_string_list(payload.get("desired_report_types"), "desired_report_types"),
        review_cadence=optional_text(payload, "review_cadence") or "weekly",
        source_boundaries=optional_string_list(payload.get("source_boundaries"), "source_boundaries") or [],
        corpus_id=optional_text(payload, "corpus_id"),
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def alert_digest(payload: dict[str, Any]) -> dict:
    from canon.intelligence.alerts import run_alert_digest

    question = optional_text(payload, "question") or require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    return run_alert_digest(
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def flagship_handoff(payload: dict[str, Any]) -> dict:
    from canon.product.flagship_handoff import (
        DEFAULT_FIXTURE,
        DEFAULT_MODE as DEFAULT_HANDOFF_MODE,
        DEFAULT_QUESTION,
        run_flagship_handoff,
    )

    question = optional_text(payload, "question") or optional_text(payload, "query") or DEFAULT_QUESTION
    mode = str(payload.get("mode") or DEFAULT_HANDOFF_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    fixture_path = optional_path(payload.get("fixture_path") or payload.get("fixture"), "fixture_path") or DEFAULT_FIXTURE
    queries_path = optional_path(payload.get("queries_path"), "queries_path")
    return run_flagship_handoff(
        mode=mode,
        question=question,
        fixture_path=fixture_path,
        queries_path=queries_path,
        policy=policy,
        project_id=project_id,
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def acceptance_scenario(payload: dict[str, Any]) -> dict:
    from canon.product.acceptance_scenario import (
        DEFAULT_MODE as DEFAULT_ACCEPTANCE_MODE,
        DEFAULT_QUESTION,
        run_acceptance_scenario,
    )

    question = optional_text(payload, "question") or optional_text(payload, "query") or DEFAULT_QUESTION
    return run_acceptance_scenario(
        mode=str(payload.get("mode") or DEFAULT_ACCEPTANCE_MODE),
        question=question,
        policy=str(payload.get("policy") or DEFAULT_POLICY),
        project_id=str(payload.get("project_id") or "ai_infra_geo_risk"),
        source_boundaries=optional_string_list(payload.get("source_boundaries"), "source_boundaries"),
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def intelligence_review_prepare(payload: dict[str, Any]) -> dict:
    from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH
    from canon.product.intelligence_review import build_review_tasks

    mode = str(payload.get("mode") or "ai_infra_geo_risk_demo")
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    queries_path = optional_path(payload.get("queries_path"), "queries_path") or DEFAULT_QUERY_PATH
    existing_records_path = optional_path(payload.get("records_path") or payload.get("records"), "records_path")
    reset_review_labels = optional_bool(payload.get("reset_review_labels"), default=False)
    try:
        return build_review_tasks(
            mode=mode,
            queries_path=queries_path,
            policy=policy,
            project_id=project_id,
            write_report=optional_bool(payload.get("write_report"), default=True),
            preserve_existing_reviews=not reset_review_labels,
            existing_records_path=existing_records_path,
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def intelligence_review_status(payload: dict[str, Any]) -> dict:
    from canon.product.intelligence_review import build_review_status_report

    records_path = intelligence_review_records_path(payload)
    return build_review_status_report(
        records_path,
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def intelligence_review_export_csv(payload: dict[str, Any]) -> dict:
    from canon.product.intelligence_review import export_review_csv

    records_path = intelligence_review_records_path(payload)
    output_path = optional_path(payload.get("output_path") or payload.get("output"), "output_path")
    path = export_review_csv(records_path, output_path)
    return {
        "status": "review_csv_written",
        "records_path": str(records_path).replace("\\", "/"),
        "output_path": str(path).replace("\\", "/"),
    }


def intelligence_review_import_csv(payload: dict[str, Any]) -> dict:
    from canon.product.intelligence_review import import_review_csv

    records_path = intelligence_review_records_path(payload)
    csv_path = optional_path(payload.get("csv_path") or payload.get("csv"), "csv_path")
    if not csv_path:
        raise ProductError("csv_path is required.")
    output_path = optional_path(payload.get("output_path") or payload.get("output"), "output_path")
    try:
        return import_review_csv(records_path, csv_path, output_path)
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def intelligence_review_records_path(payload: dict[str, Any]) -> Path:
    explicit = optional_path(payload.get("records_path") or payload.get("records"), "records_path")
    if explicit:
        return explicit
    mode = str(payload.get("mode") or "ai_infra_geo_risk_demo")
    return load_settings().reports_dir / f"intelligence_brief_review_tasks_{mode}.json"


def intelligence_brief_evaluation(payload: dict[str, Any]) -> dict:
    from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH, evaluate_intelligence_briefs

    mode = str(payload.get("mode") or "ai_infra_geo_risk_demo")
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    queries_path = optional_path(payload.get("queries_path"), "queries_path") or DEFAULT_QUERY_PATH
    try:
        return evaluate_intelligence_briefs(
            mode=mode,
            queries_path=queries_path,
            policy=policy,
            project_id=project_id,
            write_report=optional_bool(payload.get("write_report"), default=True),
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def alert_digest_evaluation(payload: dict[str, Any]) -> dict:
    from canon.eval.alert_digest import evaluate_alert_digests
    from canon.eval.intelligence_brief import DEFAULT_QUERY_PATH

    mode = str(payload.get("mode") or "ai_infra_geo_risk_demo")
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    project_id = str(payload.get("project_id") or "ai_infra_geo_risk")
    queries_path = optional_path(payload.get("queries_path"), "queries_path") or DEFAULT_QUERY_PATH
    try:
        return evaluate_alert_digests(
            mode=mode,
            queries_path=queries_path,
            policy=policy,
            project_id=project_id,
            write_report=optional_bool(payload.get("write_report"), default=True),
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def enrich_evidence_metadata(evidence: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    work_metadata = load_work_metadata(mode)
    enriched = []
    for item in evidence:
        work = work_metadata.get(str(item.get("work_id"))) or {}
        raw = work.get("raw") or {}
        next_item = dict(item)
        next_item["language"] = work.get("language")
        next_item["source_type"] = raw.get("source_type") or raw.get("document_type")
        next_item["document_type"] = raw.get("document_type")
        next_item["provenance"] = raw.get("provenance") or work.get("provenance")
        next_item["domain"] = raw.get("domain") or work.get("domain")
        next_item["jurisdiction"] = raw.get("jurisdiction")
        next_item["url"] = work.get("landing_page_url") or work.get("pdf_url")
        enriched.append(next_item)
    return enriched


def load_work_metadata(mode: str) -> dict[str, dict[str, Any]]:
    settings = load_settings()
    path = settings.data_dir / "processed" / f"works_{mode}.json"
    if not path.exists():
        return {}
    try:
        works = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(work.get("id")): work for work in works if isinstance(work, dict)}


def packet_source_diversity(query: str, evidence: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    return source_diversity_row(
        query={"id": "evidence_packet", "query": query},
        evidence=evidence,
        top_k=top_k,
        min_distinct_sources=DEFAULT_MIN_DISTINCT_SOURCES,
        max_dominant_source_share=DEFAULT_MAX_DOMINANT_SOURCE_SHARE,
        min_cluster_count=DEFAULT_MIN_CLUSTER_COUNT,
    )


def packet_claim(query: str, support: dict[str, Any]) -> str:
    if support.get("support_level") in {"none", "weak"}:
        return f"The current corpus provides limited support for: {query}"
    return f"Retrieved evidence addresses: {query}"


def packet_support_level(support: dict[str, Any]) -> str:
    level = str(support.get("support_level") or "none")
    if level == "strong":
        return "supported"
    if level == "moderate":
        return "mixed"
    if level == "weak":
        return "weak"
    return "insufficient"


def packet_confidence(support: dict[str, Any]) -> str:
    score = float(support.get("support_confidence") or 0.0)
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def infer_issue_categories(research_frame: Any, query: str) -> list[str]:
    if isinstance(research_frame, dict):
        subdomains = optional_string_list(research_frame.get("subdomains"), "subdomains") or []
        if subdomains:
            return subdomains
    terms = [term for term in query.lower().replace("/", " ").split() if len(term) >= 4]
    return terms[:6]


def packet_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("citation_id"),
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("work_id"),
        "title": item.get("title"),
        "source_name": item.get("source_name"),
        "url": item.get("url"),
        "published_at": item.get("published_at") or item.get("year"),
        "text": item.get("preview"),
        "citation": item.get("citation_id"),
        "rank": item.get("rank"),
        "cluster_id": item.get("cluster_id"),
        "language": item.get("language"),
        "source_type": item.get("source_type") or item.get("document_type"),
        "provenance": item.get("provenance"),
        "domain": item.get("domain"),
        "jurisdiction": item.get("jurisdiction"),
        "claim": item.get("claim"),
    }


def compact_packet_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    query_to_corpus = diagnostics.get("query_to_corpus") or {}
    neighborhood = diagnostics.get("result_neighborhood") or {}
    stability = diagnostics.get("stability") or {}
    return {
        "matched_terms": query_to_corpus.get("matched_terms", []),
        "weak_terms": query_to_corpus.get("weak_terms", []),
        "field_phrases": neighborhood.get("field_phrases", []),
        "query_variants": diagnostics.get("query_variants", []),
        "drift_risk": stability.get("status") or "needs_caution",
    }


def coverage_gaps(
    diagnostics: dict[str, Any],
    diversity: dict[str, Any],
    limitations: list[str],
    frame_coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    gaps = []
    weak_terms = ((diagnostics.get("query_to_corpus") or {}).get("weak_terms") or [])[:5]
    if weak_terms:
        gaps.append(
            {
                "gap": f"Weak corpus match for terms: {', '.join(weak_terms)}.",
                "severity": "medium",
                "suggested_next_query": " ".join(weak_terms),
            }
        )
    for warning in diversity.get("warnings", []):
        gaps.append(
            {
                "gap": warning.get("message", warning.get("id", "Source diversity warning.")),
                "severity": "medium",
                "suggested_next_query": "",
            }
        )
    for diagnostic in (frame_coverage or {}).get("diagnostics", []):
        if diagnostic.get("status") == "missing":
            gaps.append(
                {
                    "gap": diagnostic["message"],
                    "severity": diagnostic.get("severity", "medium"),
                    "suggested_next_query": diagnostic.get("suggested_next_query", ""),
                }
            )
    for limitation in limitations[:3]:
        gaps.append({"gap": limitation, "severity": "low", "suggested_next_query": ""})
    return gaps


def analyze_frame_coverage(
    research_frame: dict[str, Any],
    requirements: dict[str, Any],
    evidence: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "subdomains": optional_string_list(research_frame.get("subdomains"), "subdomains") or [],
        "regions": optional_string_list(research_frame.get("regions"), "regions") or [],
        "languages": optional_string_list(research_frame.get("languages"), "languages") or [],
        "source_types": optional_string_list(requirements.get("minimum_source_types"), "minimum_source_types") or [],
        "representation_goals": optional_string_list(
            research_frame.get("representation_goals"),
            "representation_goals",
        )
        or [],
    }
    coverage = {
        "subdomains": term_coverage(expected["subdomains"], evidence),
        "regions": term_coverage(expected["regions"], evidence),
        "languages": language_coverage(expected["languages"], evidence),
        "source_types": source_type_coverage(expected["source_types"], evidence),
        "representation_goals": representation_goal_coverage(expected["representation_goals"], evidence),
    }
    diagnostics_rows = frame_coverage_diagnostics(expected, coverage, diagnostics)
    missing_count = sum(len(row["missing"]) for row in coverage.values())
    covered_count = sum(len(row["covered"]) for row in coverage.values())
    if not expected_total(expected):
        status = "not_requested"
    elif missing_count == 0:
        status = "pass"
    elif covered_count:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "expected": expected,
        "coverage": coverage,
        "diagnostics": diagnostics_rows,
        "human_review_required": True,
        "note": "Frame coverage is a diagnostic over retrieved evidence, not proof that the corpus contains all relevant evidence.",
    }


def expected_total(expected: dict[str, list[str]]) -> int:
    return sum(len(values) for values in expected.values())


def term_coverage(terms: list[str], evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    text = evidence_text(evidence)
    covered = [term for term in terms if normalized_term(term) in text]
    return split_coverage(terms, covered)


def language_coverage(languages: list[str], evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    evidence_languages = {normalize_language(item.get("language")) for item in evidence if item.get("language")}
    covered = [
        language
        for language in languages
        if normalize_language(language) in evidence_languages or normalized_term(language) in evidence_text(evidence)
    ]
    return split_coverage(languages, covered)


def source_type_coverage(source_types: list[str], evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    text = " ".join(
        normalized_term(value)
        for item in evidence
        for value in [
            item.get("source_type"),
            item.get("document_type"),
            item.get("provenance"),
            item.get("source_name"),
        ]
        if value
    )
    covered = [source_type for source_type in source_types if normalized_term(source_type) in text]
    return split_coverage(source_types, covered)


def representation_goal_coverage(goals: list[str], evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    covered = []
    for goal in goals:
        normalized = normalized_term(goal)
        if "language" in normalized and len({item.get("language") for item in evidence if item.get("language")}) >= 2:
            covered.append(goal)
        elif "source" in normalized and len({item.get("source_name") for item in evidence if item.get("source_name")}) >= 3:
            covered.append(goal)
        elif "region" in normalized and normalized in evidence_text(evidence):
            covered.append(goal)
        elif normalized in evidence_text(evidence):
            covered.append(goal)
    return split_coverage(goals, covered)


def split_coverage(expected: list[str], covered: list[str]) -> dict[str, list[str]]:
    covered_set = set(covered)
    return {
        "covered": covered,
        "missing": [item for item in expected if item not in covered_set],
    }


def frame_coverage_diagnostics(
    expected: dict[str, list[str]],
    coverage: dict[str, dict[str, list[str]]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    weak_terms = set((diagnostics.get("query_to_corpus") or {}).get("weak_terms") or [])
    for field, values in expected.items():
        if not values:
            continue
        missing = coverage[field]["missing"]
        if missing:
            rows.append(
                {
                    "field": field,
                    "status": "missing",
                    "severity": "medium",
                    "message": f"Retrieved evidence did not visibly cover requested {field}: {', '.join(missing)}.",
                    "suggested_next_query": " ".join(missing),
                }
            )
        else:
            rows.append(
                {
                    "field": field,
                    "status": "covered",
                    "severity": "info",
                    "message": f"Retrieved evidence visibly covered requested {field}.",
                    "suggested_next_query": "",
                }
            )
    if weak_terms:
        rows.append(
            {
                "field": "query_terms",
                "status": "review",
                "severity": "low",
                "message": f"Some query terms remain weak in the corpus: {', '.join(sorted(weak_terms))}.",
                "suggested_next_query": " ".join(sorted(weak_terms)),
            }
        )
    return rows


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(
        normalized_term(value)
        for item in evidence
        for value in [
            item.get("title"),
            item.get("source_name"),
            item.get("preview"),
            item.get("domain"),
            item.get("provenance"),
            item.get("jurisdiction"),
            item.get("source_type"),
            item.get("document_type"),
        ]
        if value
    )


def normalized_term(value: Any) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ").strip()


def normalize_language(value: Any) -> str:
    normalized = normalized_term(value)
    aliases = {
        "en": "english",
        "eng": "english",
        "english": "english",
        "es": "spanish",
        "spa": "spanish",
        "spanish": "spanish",
        "pt": "portuguese",
        "por": "portuguese",
        "portuguese": "portuguese",
    }
    return aliases.get(normalized, normalized)


def compact_answer_evidence(evidence: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": item["citation_id"],
            "chunk_id": item["chunk_id"],
            "work_id": item["work_id"],
            "title": item["title"],
            "source_name": item["source_name"],
            "year": item["year"],
            "rank": item["rank"],
            "cluster_id": item["cluster_id"],
            "final_score": item["final_score"],
            "claim": item.get("claim"),
            "preview": item.get("preview"),
            "explanation": item.get("explanation", {}),
            "conflict_injected": bool(item.get("conflict_injected")),
        }
        for item in evidence[:limit]
    ]


def compare_retrieval(payload: dict[str, Any]) -> dict:
    query = require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policies = payload.get("policies") or DEFAULT_POLICIES
    if isinstance(policies, str):
        policies = [piece.strip() for piece in policies.split(",") if piece.strip()]
    if not isinstance(policies, list) or not policies:
        raise ProductError("policies must be a non-empty list or comma-separated string.")
    top_k = optional_int(payload.get("top_k"))
    report = compare(query=query, policies=[str(policy) for policy in policies], mode=mode, top_k=top_k)
    return {
        "query": report["query"],
        "mode": report["mode"],
        "policies": report["policies"],
        "rank_table": report["rank_table"][:25],
        "runs": [
            {
                "policy": run["policy"],
                "top_k": run["top_k"],
                "results": run["results"][:10],
            }
            for run in report["runs"]
        ],
    }


def query_diagnostics(payload: dict[str, Any]) -> dict:
    query = require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    top_k = optional_int(payload.get("top_k")) or 5
    candidate_k = optional_positive_int(payload.get("candidate_k"), "candidate_k") or 20
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    return build_query_diagnostics(
        query=query,
        mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        freedom_level=freedom_level,
    )


def source_profile(payload: dict[str, Any]) -> dict:
    input_path = require_path(payload, "input_path")
    input_format = optional_text(payload, "format")
    sample_size = optional_positive_int(payload.get("sample_size"), "sample_size") or 25
    try:
        return profile_source(
            input_path=input_path,
            input_format=input_format,
            sample_size=sample_size,
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def source_ingest(payload: dict[str, Any]) -> dict:
    input_path = require_path(payload, "input_path")
    mode = require_text(payload, "mode")
    input_format = optional_text(payload, "format")
    chunk_tokens = optional_positive_int(payload.get("chunk_tokens"), "chunk_tokens")
    overlap_tokens = optional_nonnegative_int(payload.get("overlap_tokens"), "overlap_tokens")
    try:
        return ingest_flexible_source(
            input_path=input_path,
            mode=mode,
            input_format=input_format,
            domain=optional_text(payload, "domain"),
            provenance=optional_text(payload, "provenance"),
            source_name=optional_text(payload, "source_name"),
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def corpus_build(payload: dict[str, Any]) -> dict:
    corpus_id = require_text(payload, "corpus_id")
    source_modes = optional_string_list(payload.get("from_modes"), "from_modes")
    if not source_modes:
        raise ProductError("from_modes must be a non-empty list or comma-separated string.")
    top_k = optional_positive_int(payload.get("top_k"), "top_k") or 5
    policies = optional_string_list(payload.get("policies"), "policies") or ["lexical", "balanced", "semantic", "rag"]
    try:
        report = run_phase16(
            corpus_id=corpus_id,
            from_modes=source_modes,
            harvest=False,
            corpus_only=optional_bool(payload.get("corpus_only"), default=True),
            top_k=top_k,
            policies=policies,
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc
    return {
        "corpus": report.get("corpus", {}),
        "validation": report.get("validation", {}),
    }


def model_evaluation(payload: dict[str, Any]) -> dict:
    mode = require_text(payload, "mode")
    queries_path = optional_path(payload.get("queries_path"), "queries_path")
    qrels_path = optional_path(payload.get("qrels_path") or payload.get("qrels"), "qrels_path")
    if not queries_path and not qrels_path:
        raise ProductError("qrels_path or queries_path is required.")
    k = optional_positive_int(payload.get("k"), "k") or 10
    batch_size = optional_positive_int(payload.get("batch_size"), "batch_size") or 32
    providers = parse_providers_from_payload(payload.get("providers"))
    try:
        return evaluate_semantic_models(
            mode=mode,
            queries_path=queries_path,
            qrels_path=qrels_path,
            providers=providers,
            k=k,
            batch_size=batch_size,
        )
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise ProductError(str(exc), status_code=400) from exc


def build_query_diagnostics(
    query: str,
    mode: str,
    top_k: int | None = None,
    candidate_k: int = 20,
    freedom_level: str = "balanced",
) -> dict:
    try:
        documents = cached_documents(mode)
    except FileNotFoundError as exc:
        raise ProductError(f"Corpus for mode '{mode}' was not found.", status_code=404) from exc
    return diagnose_query(
        query=query,
        documents=documents,
        top_k=top_k or 5,
        candidate_k=candidate_k,
        freedom_level=freedom_level,
    )


def diversity_audit(payload: dict[str, Any]) -> dict:
    mode = str(payload.get("mode") or DEFAULT_DIVERSITY_MODE)
    diverse_method_id = str(payload.get("diverse_method_id") or DEFAULT_DIVERSE_METHOD_ID)
    baseline_method_id = str(
        payload.get("baseline_method_id") or DEFAULT_DIVERSITY_BASELINE_METHOD_ID
    )
    report = run_diversity_audit(
        mode=mode,
        diverse_method_id=diverse_method_id,
        baseline_method_id=baseline_method_id,
    )
    return compact_diversity_report(report)


def report(name: str, mode: str = DEFAULT_MODE, params: dict[str, Any] | None = None) -> dict:
    settings = load_settings()
    params = params or {}
    diverse_method_id = str(params.get("diverse_method_id") or DEFAULT_DIVERSE_METHOD_ID)
    baseline_method_id = str(
        params.get("baseline_method_id") or DEFAULT_DIVERSITY_BASELINE_METHOD_ID
    )
    allowed = {
        "audit": settings.reports_dir / f"scientific_audit_{mode}_baseline_methods_v1.json",
        "claim-decision": settings.reports_dir / f"claim_decision_{mode}_baseline_methods_v1.json",
        "data-card": settings.reports_dir / f"data_card_{mode}.json",
        "regression-gate": settings.reports_dir / f"regression_gate_{mode}_baseline_methods_v1.json",
        "diversity": diversity_report_path(
            mode=mode,
            diverse_method_id=diverse_method_id,
            baseline_method_id=baseline_method_id,
        ),
        "diversity-diagnostics": diversity_diagnostics_report_path(
            mode=mode,
            diverse_method_id=diverse_method_id,
            baseline_method_id=baseline_method_id,
        ),
        "diversity-gate": diversity_gate_report_path(
            mode=mode,
            diverse_method_id=diverse_method_id,
            baseline_method_id=baseline_method_id,
        ),
    }
    if name not in allowed:
        raise ProductError(f"Unknown report '{name}'.", status_code=404)
    payload = load_report(allowed[name])
    if name == "diversity" and payload:
        return compact_diversity_report(payload)
    return payload


def diversity_queries(params: dict[str, Any] | None = None) -> dict:
    params = params or {}
    report_payload = load_diversity_report_from_params(params)
    verdict = optional_text(params, "verdict")
    query_type = optional_text(params, "query_type")
    limit = optional_positive_int(params.get("limit"), "limit")
    queries = [
        compact_diversity_query(query)
        for query in report_payload.get("queries", [])
        if (not verdict or query.get("verdict") == verdict)
        and (not query_type or query.get("query_type") == query_type)
    ]
    if limit is not None:
        queries = queries[:limit]
    return {
        **compact_diversity_report(report_payload),
        "filters": {
            "verdict": verdict,
            "query_type": query_type,
            "limit": limit,
        },
        "result_count": len(queries),
        "queries": queries,
    }


def diversity_query_detail(query_id: str, params: dict[str, Any] | None = None) -> dict:
    if not query_id:
        raise ProductError("query_id is required.")
    report_payload = load_diversity_report_from_params(params or {})
    for query in report_payload.get("queries", []):
        if query.get("query_id") == query_id:
            return {
                **compact_diversity_report(report_payload),
                "query": query,
            }
    raise ProductError(f"Unknown diversity query '{query_id}'.", status_code=404)


def load_diversity_report_from_params(params: dict[str, Any]) -> dict:
    mode = str(params.get("mode") or DEFAULT_DIVERSITY_MODE)
    diverse_method_id = str(params.get("diverse_method_id") or DEFAULT_DIVERSE_METHOD_ID)
    baseline_method_id = str(
        params.get("baseline_method_id") or DEFAULT_DIVERSITY_BASELINE_METHOD_ID
    )
    path = diversity_report_path(mode, diverse_method_id, baseline_method_id)
    payload = load_report(path)
    if not payload:
        raise ProductError(
            "Diversity report not found. Run POST /v1/diversity-audit first.",
            status_code=404,
        )
    return payload


def compact_diversity_query(query: dict) -> dict:
    return {
        "query_id": query.get("query_id"),
        "query_type": query.get("query_type"),
        "query": query.get("query"),
        "verdict": query.get("verdict"),
        "cluster_delta": query.get("cluster_delta"),
        "rank_overlap": query.get("rank_overlap"),
        "added_count": query.get("added_count"),
        "removed_count": query.get("removed_count"),
        "useful_breadth_count": query.get("useful_breadth_count"),
        "noise_addition_count": query.get("noise_addition_count"),
        "breadth_precision": query.get("breadth_precision"),
        "noise_rate": query.get("noise_rate"),
        "average_added_focus_coverage": query.get("average_added_focus_coverage"),
        "average_added_relevance": query.get("average_added_relevance"),
    }


def diversity_report_path(mode: str, diverse_method_id: str, baseline_method_id: str) -> Path:
    settings = load_settings()
    return (
        settings.reports_dir
        / f"diversity_audit_{mode}_baseline_methods_v1_{diverse_method_id}_vs_{baseline_method_id}.json"
    )


def diversity_gate_report_path(mode: str, diverse_method_id: str, baseline_method_id: str) -> Path:
    settings = load_settings()
    return (
        settings.reports_dir
        / f"diversity_gate_{mode}_baseline_methods_v1_{diverse_method_id}_vs_{baseline_method_id}.json"
    )


def diversity_diagnostics_report_path(
    mode: str,
    diverse_method_id: str,
    baseline_method_id: str,
) -> Path:
    settings = load_settings()
    return (
        settings.reports_dir
        / f"diversity_diagnostics_{mode}_baseline_methods_v1_{diverse_method_id}_vs_{baseline_method_id}.json"
    )


def compact_diversity_report(report: dict) -> dict:
    return {
        "mode": report.get("mode"),
        "method_set_id": report.get("method_set_id"),
        "diverse_method_id": report.get("diverse_method_id"),
        "baseline_method_id": report.get("baseline_method_id"),
        "query_count": report.get("query_count"),
        "aggregate": report.get("aggregate", {}),
    }


def safe_claim_decision(mode: str) -> dict:
    settings = load_settings()
    path = settings.reports_dir / f"claim_decision_{mode}_baseline_methods_v1.json"
    if path.exists():
        return load_report(path)
    return build_claim_decision(mode)


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProductError(f"{key} is required.")
    return value.strip()


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProductError("top_k must be an integer.") from exc
    if integer <= 0:
        raise ProductError("top_k must be positive.")
    return integer


def optional_positive_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProductError(f"{name} must be an integer.") from exc
    if integer <= 0:
        raise ProductError(f"{name} must be positive.")
    return integer


def optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProductError(f"{name} must be an integer.") from exc
    if integer < 0:
        raise ProductError(f"{name} must be non-negative.")
    return integer


def optional_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProductError(f"{name} must be an object.")
    return value


def optional_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise ProductError("Boolean value must be true or false.")


def optional_freedom_level(value: Any) -> str:
    if value is None or value == "":
        return "balanced"
    level = str(value)
    if level not in FREEDOM_THRESHOLDS:
        raise ProductError(
            f"freedom_level must be one of: {', '.join(sorted(FREEDOM_THRESHOLDS))}."
        )
    return level


def optional_text(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_path(payload: dict[str, Any], key: str) -> Path:
    text = require_text(payload, key)
    return Path(text)


def optional_path(value: Any, name: str) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ProductError(f"{name} must be a path string.")
    text = value.strip()
    return Path(text) if text else None


def optional_string_list(value: Any, name: str) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        values = [piece.strip() for piece in value.split(",") if piece.strip()]
    elif isinstance(value, list):
        values = [str(piece).strip() for piece in value if str(piece).strip()]
    else:
        raise ProductError(f"{name} must be a list or comma-separated string.")
    return values or None


def required_string_list(value: Any, name: str) -> list[str]:
    values = optional_string_list(value, name)
    if not values:
        raise ProductError(f"{name} must be a non-empty list or comma-separated string.")
    return values


def parse_providers_from_payload(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        providers = [str(provider).strip() for provider in value if str(provider).strip()]
        return providers or None
    if isinstance(value, str):
        return parse_providers(value)
    raise ProductError("providers must be a list or comma-separated string.")

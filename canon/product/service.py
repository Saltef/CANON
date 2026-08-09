from __future__ import annotations

import json
import re
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.corpus.build import run_phase16
from canon.corpus.sync import (
    SourceManifestOptions,
    build_source_manifest,
    diff_manifests,
    load_previous_manifest,
    write_source_manifest,
)
from canon.embeddings.index import build_vector_index
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.model_evaluation import evaluate_semantic_models, parse_providers
from canon.eval.diversity import run_diversity_audit
from canon.observability import estimate_tokens, stage_span
from canon.product import report_io
from canon.eval.source_diversity import (
    DEFAULT_MAX_DOMINANT_SOURCE_SHARE,
    DEFAULT_MIN_CLUSTER_COUNT,
    DEFAULT_MIN_DISTINCT_SOURCES,
    source_diversity_row,
)
from canon.generation.providers import get_generation_provider
from canon.ingest.openalex import abstract_from_inverted_index, fetch_json
from canon.ingest.flexible import ingest_flexible_source, profile_source
from canon.model_tasks import DEFAULT_MODEL_TASK_MODEL, run_evidence_model_review
from canon.reports.claim_decision import build_claim_decision
from canon.product.run_diagnosis import build_run_diagnosis
from canon.retrieval.compare import compare
from canon.retrieval.candidates import (
    build_bm25_candidate_pool,
    build_candidate_pool,
    build_vector_store_candidate_pool,
    build_windowed_candidate_pool,
)
from canon.retrieval.experiment import cached_documents
from canon.retrieval.query_diagnostics import FREEDOM_THRESHOLDS, content_terms, diagnose_query, term_matches_tokens
from canon.rerank.providers import get_rerank_provider
from canon.synthesis.answer import synthesize


DEFAULT_MODE = "social_science_ir_v1_harvest10"
DEFAULT_DIVERSITY_MODE = "social_science_ir_10k"
DEFAULT_POLICY = "rag"
DEFAULT_PRODUCTION_MODE = "beir_scifact_full"
DEFAULT_PRODUCTION_TOP_K = 12
DEFAULT_PRODUCTION_CANDIDATE_K = 12
DEFAULT_PRODUCTION_CANDIDATE_SCOPE = "vector_store"
DEFAULT_PRODUCTION_LEXICAL_WINDOW_K = 96
DEFAULT_PRODUCTION_RERANK_TIMEOUT_S = 15.0
DEFAULT_VECTOR_BACKEND = "qdrant"
DEFAULT_EXTERNAL_SEARCH_PROVIDER = "openalex"
DEFAULT_EXTERNAL_RESULT_COUNT = 5
DEFAULT_POLICIES = ["lexical", "balanced", "semantic", "rag", "diverse", "conflict_aware"]
DEFAULT_DIVERSE_METHOD_ID = "diverse_k5_template"
DEFAULT_DIVERSITY_BASELINE_METHOD_ID = "lexical_k5_template"
DEFAULT_PUBLISHABLE_PACKAGE = "publishable_package_canon_publishable_evidence_workflow_v1.json"
PRODUCTION_GENERIC_QUERY_TERMS = {
    "approach",
    "around",
    "benefit",
    "benefits",
    "case",
    "cases",
    "center",
    "data",
    "effect",
    "effects",
    "evidence",
    "expansion",
    "impact",
    "impacts",
    "issue",
    "issues",
    "risk",
    "risks",
    "role",
    "roles",
    "trend",
    "trends",
}


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
    audit = load_report_any(
        settings.reports_dir,
        [
            f"scientific_audit_{mode}_baseline_methods_v1.json",
            "scientific_audit_dry_run_baseline_methods_v1.json",
        ],
    )
    decision = load_report_any(
        settings.reports_dir,
        [
            f"claim_decision_{mode}_baseline_methods_v1.json",
            "claim_decision_dry_run_baseline_methods_v1.json",
        ],
    )
    data_card = load_report_any(
        settings.reports_dir,
        [f"data_card_{mode}.json", "data_card_dry_run.json"],
    )
    audit_status = audit.get("status") or "not_available"
    active_warnings = [warning["id"] for warning in audit.get("warnings", [])]
    if not audit:
        active_warnings.append("scientific_audit_not_available_for_mode")
    data_card_limitations = data_card.get("limitations", [])
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
        "audit_status": audit_status,
        "active_warnings": active_warnings,
        "resolved_warnings": [warning["id"] for warning in audit.get("resolved_warnings", [])],
        "corpus": {
            "work_count": data_card.get("work_count"),
            "chunk_count": data_card.get("chunk_count"),
            "limitations": data_card_limitations or summary_limitations(mode, data_card),
        },
    }


def summary_limitations(mode: str, data_card: dict[str, Any]) -> list[str]:
    limitations = []
    if not data_card:
        limitations.append(
            "No data card is available for this mode; corpus size, source mix, and coverage need review."
        )
    if mode == "ai_infra_geo_risk_demo" or mode.endswith("_demo"):
        limitations.append(
            "This is a small fixture/demo corpus for workflow testing, not a complete domain corpus."
        )
    limitations.append(
        "Human review is required before using outputs for final conclusions or release-level quality claims."
    )
    return limitations


def load_report_any(reports_dir, names: list[str]) -> dict:
    for name in names:
        report = load_report(reports_dir / name)
        if report:
            return report
    return {}


def answer(payload: dict[str, Any]) -> dict:
    query = require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    top_k = optional_int(payload.get("top_k"))
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    report = synthesize(
        query=query,
        policy=policy,
        mode=mode,
        top_k=top_k,
        generator_name=optional_text(payload, "generator_provider")
        or optional_text(payload, "generator")
        or "template",
        generator_model=optional_text(payload, "generator_model"),
    )
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
    expansion_preferences = expansion_preferences_from_requirements(requirements)
    top_k = optional_positive_int(requirements.get("top_k"), "top_k") if requirements else None
    top_k = top_k or optional_int(payload.get("top_k"))
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    report = synthesize(
        query=query,
        policy=policy,
        mode=mode,
        top_k=top_k,
        generator_name=optional_text(payload, "generator_provider")
        or optional_text(payload, "generator")
        or "template",
        generator_model=optional_text(payload, "generator_model"),
    )
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
        "evidence_scope_summary": evidence_scope_summary(evidence),
        "source_diversity": {
            "distinct_sources": diversity["distinct_sources"],
            "distinct_clusters": diversity["distinct_clusters"],
            "dominant_source_share": diversity["dominant_source_share"],
            "warnings": diversity["warnings"],
        },
        "frame_coverage": frame_coverage,
    }
    gaps = coverage_gaps(diagnostics, diversity, report.get("limitations", []), frame_coverage)
    expansion_plan = build_external_expansion_plan(
        query=query,
        evidence=evidence,
        coverage_gaps=gaps,
        research_frame=research_frame,
        preferences=expansion_preferences,
    )
    response = {
        "report_id": "evidence_packet_response_v1",
        "contract_version": CONTRACT_VERSION,
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
        "external_expansion": expansion_plan,
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
    response["run_diagnosis"] = build_run_diagnosis(
        query=query,
        packet_response=response,
        evidence=packet["supporting_evidence"],
        coverage_gaps=gaps,
        query_diagnostics=response["query_diagnostics"],
        frame_coverage=frame_coverage,
    )
    response["contract_validation"] = validate_report(response, "evidence_packet_response")
    return response


def production_status(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    settings = load_settings()
    mode = optional_text(params, "mode") or DEFAULT_PRODUCTION_MODE
    package = load_report(settings.reports_dir / DEFAULT_PUBLISHABLE_PACKAGE)
    benchmark_card = load_report(settings.reports_dir / "publishable_benchmark_card_v1.json")
    stage1_gate = load_report(settings.reports_dir / "stage1_gate_v1.json")
    checks = package.get("checks") or []
    blocked_checks = [check_row["id"] for check_row in checks if not check_row.get("passed")]
    return {
        "report_id": "production_workbench_status_v1",
        "status": production_status_value(settings, mode),
        "product": "CANON Evidence Discovery Workbench",
        "mode": mode,
        "available_modes": available_corpus_modes(settings.data_dir),
        "recommended_defaults": {
            "mode": mode,
            "policy": DEFAULT_POLICY,
            "top_k": DEFAULT_PRODUCTION_TOP_K,
            "candidate_k": DEFAULT_PRODUCTION_CANDIDATE_K,
            "freedom_level": "balanced",
            "external_expansion": "off_by_default_opt_in_openalex",
            "retrieval_engine": "model_candidate_pool",
            "candidate_scope": "vector_store",
            "vector_backend": DEFAULT_VECTOR_BACKEND,
            "lexical_window_k": DEFAULT_PRODUCTION_LEXICAL_WINDOW_K,
            "retrieval_provider": "openrouter",
            "retrieval_model": "qwen/qwen3-embedding-8b",
            "reranker_provider": "cohere",
            "reranker_model": "rerank-v4.0-pro",
            "generator_provider": "openrouter",
            "generator_model": "openai/gpt-4.1-mini",
            "model_review_provider": "openrouter",
            "model_review_model": DEFAULT_MODEL_TASK_MODEL,
        },
        "retrieval_engines": [
            {
                "engine": "model_candidate_pool",
                "label": "Hosted embedding candidate pool",
                "status": "available_if_openrouter_key_configured",
                "boundary": (
                    "Uses full-corpus BM25 to form a bounded candidate window, hosted embeddings for that window, "
                    "then optional hosted reranking."
                ),
            },
            {
                "engine": "canon_synthesis",
                "label": "CANON synthesis retriever",
                "status": "available",
                "boundary": "Uses the existing deterministic CANON retrieval policy stack.",
            },
        ],
        "candidate_scopes": [
            {
                "scope": "vector_store",
                "label": "Qdrant/ANN semantic index plus full-corpus BM25",
                "status": "available_after_vector_index_build",
                "boundary": "BM25 scans the local full corpus; ANN search uses the configured vector backend and joins by chunk_id.",
            },
            {
                "scope": "lexical_window",
                "label": "Full-corpus lexical scan plus bounded dense window",
                "status": "default_interactive",
                "boundary": "BM25 scans the full corpus; hosted embeddings are computed only for the lexical window.",
            },
            {
                "scope": "full_embedding_store",
                "label": "Full-corpus semantic index",
                "status": "available_when_embedding_store_exists_or_can_be_built",
                "boundary": "Dense retrieval covers the full corpus, but first use may build or require a full embedding store.",
            },
        ],
        "vector_backends": [
            {
                "backend": "qdrant",
                "status": "preferred_hosted_index_requires_QDRANT_URL",
                "source_of_truth": False,
                "migration_boundary": "Rebuild from local processed corpus JSON when switching backend or embedding model.",
            },
            {
                "backend": "memory",
                "status": "test_only",
                "source_of_truth": False,
                "migration_boundary": "Used for deterministic unit tests and local adapter validation.",
            },
        ],
        "retrieval_models": [
            {
                "provider": "openrouter",
                "model": "qwen/qwen3-embedding-8b",
                "hosted": True,
                "status": "best_on_current_scifact_mini_slice",
            },
            {
                "provider": "openrouter",
                "model": "baai/bge-m3",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "openai/text-embedding-3-small",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "sentence-transformers",
                "model": "BAAI/bge-small-en-v1.5",
                "hosted": False,
                "status": "available_if_local_model_cache_or_hf_access_exists",
            },
            {
                "provider": "local",
                "model": "hashed-semantic-v1",
                "hosted": False,
                "status": "available_deterministic_hashed_lexical_fallback",
            },
        ],
        "rerank_models": [
            {
                "provider": "cohere",
                "model": "rerank-v4.0-pro",
                "hosted": True,
                "status": "best_on_current_scifact_mini_slice_tied",
            },
            {
                "provider": "openrouter",
                "model": "cohere/rerank-v3.5",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "heuristic",
                "model": "lexical-overlap-rerank-v1",
                "hosted": False,
                "status": "available",
            },
        ],
        "generation_models": [
            {
                "provider": "deterministic",
                "model": "evidence-note-v1",
                "hosted": False,
                "status": "available",
            },
            {
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "moonshotai/kimi-k3",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "moonshotai/kimi-k2.7-code",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "moonshotai/kimi-k2.6",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
            {
                "provider": "openrouter",
                "model": "moonshotai/kimi-k2-thinking",
                "hosted": True,
                "status": "available_if_openrouter_key_configured",
            },
        ],
        "model_review": {
            "status": "available_when_requested",
            "default_provider": "openrouter",
            "default_model": DEFAULT_MODEL_TASK_MODEL,
            "disabled_models": ["cohere/*", "google/*", "*gemini*"],
            "outputs": [
                "stance_assessments",
                "extracted_dimensions",
                "disagreement_diagnosis",
                "draft_plan",
            ],
            "boundary": "Typed model review is triage over retrieved evidence and remains human-review-bound.",
        },
        "try_queries": production_try_queries(mode),
        "shippable_without_human_review": [
            "corpus-backed evidence packets",
            "query diagnostics and query refinement",
            "run diagnosis explaining corpus fit, retrieval, reranking, generation, and review needs",
            "citation-bearing draft brief for user inspection",
            "coverage and source-diversity warnings",
            "local telemetry and explicit user feedback capture",
        ],
        "blocked_without_human_review": [
            "autonomous factual conclusions",
            "clinical, legal, policy, or publication-ready correctness claims",
            "claims that CANON outperforms existing RAG systems",
            "production unsupported-claim-rate claims",
            "durable model-winner conclusions",
        ],
        "quality_state": {
            "package_status": package.get("status") or "not_available",
            "package_checks_passed": sum(1 for check_row in checks if check_row.get("passed")),
            "package_checks_total": len(checks),
            "blocked_checks": blocked_checks,
            "benchmark_card_status": benchmark_card.get("status") or "not_available",
            "aggregate_diagnostics": benchmark_card.get("aggregate_diagnostics") or {},
            "stage1_gate_status": stage1_gate.get("status") or "not_available",
            "stage1_gate_failures": stage1_gate.get("failures") or [],
        },
        "operational_boundary": {
            "default_data_flow": "local corpus retrieval and local report generation",
            "hosted_models": "called only when generator_provider selects a hosted model; query and relevance-gated snippets are sent",
            "hosted_retrieval": "called only when retrieval_engine selects model_candidate_pool; query and corpus chunks are sent for embedding/reranking",
            "vector_index": (
                "Qdrant is an optional ANN index over local chunk IDs and previews. "
                "Processed corpus JSON remains the canonical source and can rebuild another backend."
            ),
            "external_web": (
                "off by default; when execute_external_search is true, the workbench sends only the query string "
                "to OpenAlex and marks returned evidence as external_source/ONLINE"
            ),
            "feedback": "stored as local product feedback, not as authoritative human-review labels",
            "corpus_setup": "users can profile, ingest, and build local corpora from paths the local API process can read",
            "corpus_refresh": (
                "local source manifests detect added, changed, or removed files before rebuilding the processed corpus "
                "and refreshing the vector index"
            ),
        },
        "endpoints": {
            "app": "GET /app",
            "run": "POST /v1/production/evidence-workbench",
            "feedback": "POST /v1/production/feedback",
            "corpus_setup": "POST /v1/production/corpus-setup",
            "corpus_refresh": "POST /v1/production/corpus-refresh",
            "raw_packet": "POST /v1/evidence-packets",
        },
    }


def production_status_value(settings, mode: str) -> str:
    if not (settings.data_dir / "processed" / f"works_{mode}.json").exists():
        return "setup_required_corpus_missing"
    return "ship_user_experience_ready"


def available_corpus_modes(data_dir: Path, limit: int = 80) -> list[str]:
    processed = data_dir / "processed"
    if not processed.exists():
        return []
    modes = []
    for path in sorted(processed.glob("works_*.json")):
        mode = path.stem.removeprefix("works_")
        if mode.startswith("unit_") or mode.endswith("_dockercheck"):
            continue
        modes.append(mode)
    preferred = [
        DEFAULT_PRODUCTION_MODE,
        "beir_scifact_mini",
        "ai_infra_geo_risk_demo",
        DEFAULT_MODE,
        "social_science_ir_10k",
        "beir_nfcorpus_full",
    ]
    ordered = [mode for mode in preferred if mode in modes]
    ordered.extend(mode for mode in modes if mode not in set(ordered))
    return ordered[:limit]


def production_try_queries(mode: str) -> list[str]:
    if mode == "ai_infra_geo_risk_demo":
        return [
            "What are the grid risks around AI data center expansion?",
            "How could data center water and cooling needs affect local opposition?",
            "What evidence links sovereign AI policy to cloud dependency?",
        ]
    if "nfcorpus" in mode:
        return ["dietary fiber and cardiovascular risk", "vitamin D supplementation", "alternative medicine"]
    if "scifact" in mode:
        return [
            "Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.",
            "All hematopoietic stem cells segregate their chromosomes randomly.",
            "Rapid phosphotransfer rates govern fidelity in two component systems",
            "Albendazole is used to treat lymphatic filariasis.",
        ]
    return [
        "Do sanctions work?",
        "What does the literature say about whether democracies are less likely to fight each other?",
        "Where do civil war studies disagree about grievances and resources?",
    ]


def production_corpus_setup(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = require_path(payload, "input_path")
    mode = require_text(payload, "mode")
    settings = load_settings()
    profile = source_profile(
        {
            "input_path": str(input_path),
            "format": optional_text(payload, "format"),
            "sample_size": optional_positive_int(payload.get("sample_size"), "sample_size") or 25,
        }
    )
    if optional_bool(payload.get("profile_only"), default=False):
        return {
            "report_id": "production_corpus_setup_v1",
            "status": "profile_ready",
            "input_path": str(input_path),
            "mode": mode,
            "profile": profile,
            "next_action": "Run setup without profile_only to ingest and build this corpus.",
            "boundary": "Profile results are local source-shape diagnostics, not evidence quality review.",
        }

    ingest = source_ingest(
        {
            "input_path": str(input_path),
            "mode": mode,
            "format": optional_text(payload, "format"),
            "domain": optional_text(payload, "domain"),
            "provenance": optional_text(payload, "provenance"),
            "source_name": optional_text(payload, "source_name"),
            "chunk_tokens": optional_positive_int(payload.get("chunk_tokens"), "chunk_tokens"),
            "overlap_tokens": optional_nonnegative_int(payload.get("overlap_tokens"), "overlap_tokens"),
        }
    )
    corpus_id = optional_text(payload, "corpus_id") or mode
    build_corpus = optional_bool(payload.get("build_corpus"), default=True)
    corpus = (
        corpus_build(
            {
                "corpus_id": corpus_id,
                "from_modes": [mode],
                "corpus_only": True,
                "top_k": optional_positive_int(payload.get("top_k"), "top_k") or 5,
                "policies": optional_string_list(payload.get("policies"), "policies")
                or ["lexical", "balanced", "semantic", "rag"],
            }
        )
        if build_corpus
        else {}
    )
    recommended_mode = corpus_id if build_corpus else mode
    vector_index = maybe_build_production_vector_index(payload, recommended_mode)
    source_manifest = production_source_manifest(input_path, mode, settings.reports_dir)
    source_manifest_path = write_source_manifest(source_manifest, reports_dir=settings.reports_dir)
    return {
        "report_id": "production_corpus_setup_v1",
        "status": production_corpus_setup_status(build_corpus, vector_index),
        "input_path": str(input_path),
        "mode": mode,
        "corpus_id": corpus_id if build_corpus else None,
        "recommended_mode": recommended_mode,
        "profile": profile,
        "ingest": ingest,
        "corpus": corpus.get("corpus", {}),
        "validation": corpus.get("validation", {}),
        "vector_index": vector_index,
        "source_snapshot": compact_source_manifest(source_manifest),
        "source_manifest_path": str(source_manifest_path),
        "next_action": f"Select mode '{recommended_mode}' in the workbench and run a question.",
        "boundary": "The corpus is locally indexed. Evidence quality still depends on source fit and human inspection.",
    }


def production_corpus_refresh(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = require_path(payload, "input_path")
    mode = require_text(payload, "mode")
    settings = load_settings()
    try:
        current_manifest = production_source_manifest(input_path, mode, settings.reports_dir)
    except FileNotFoundError as exc:
        raise ProductError(str(exc), status_code=404) from exc
    previous_manifest = load_previous_manifest(mode, reports_dir=settings.reports_dir)
    source_diff = diff_manifests(previous_manifest, current_manifest)
    force = optional_bool(payload.get("force"), default=False)
    profile_only = optional_bool(payload.get("profile_only"), default=False)
    if source_diff["status"] == "unchanged" and not force:
        return {
            "report_id": "production_corpus_refresh_v1",
            "status": "no_source_changes",
            "input_path": str(input_path),
            "mode": mode,
            "source_diff": source_diff,
            "source_snapshot": compact_source_manifest(current_manifest),
            "next_action": "No rebuild was run. Use force=true to rebuild with new chunking, embedding, or backend settings.",
            "boundary": "No-change refreshes do not prove corpus quality; they only show local source files are unchanged.",
        }
    if profile_only:
        return {
            "report_id": "production_corpus_refresh_v1",
            "status": "source_changes_detected_profile_only",
            "input_path": str(input_path),
            "mode": mode,
            "source_diff": source_diff,
            "source_snapshot": compact_source_manifest(current_manifest),
            "next_action": "Run refresh without profile_only to ingest, rebuild the corpus, and refresh the vector index.",
            "boundary": "Profile-only refresh does not update processed corpus files or vector indexes.",
        }

    setup_report = production_corpus_setup(payload)
    return {
        **setup_report,
        "report_id": "production_corpus_refresh_v1",
        "refresh_status": "forced_refresh" if force else "source_changes_refreshed",
        "source_diff": source_diff,
        "previous_source_digest": source_diff.get("previous_digest"),
        "current_source_digest": source_diff.get("current_digest"),
    }


def production_source_manifest(input_path: Path, mode: str, reports_dir: Path) -> dict[str, Any]:
    return build_source_manifest(SourceManifestOptions(mode=mode, input_path=input_path))


def compact_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_version": manifest.get("manifest_version"),
        "mode": manifest.get("mode"),
        "input_path": manifest.get("input_path"),
        "input_kind": manifest.get("input_kind"),
        "file_count": manifest.get("file_count"),
        "supported_file_count": manifest.get("supported_file_count"),
        "unsupported_file_count": manifest.get("unsupported_file_count"),
        "content_digest": manifest.get("content_digest"),
        "boundary": manifest.get("boundary"),
    }


def production_corpus_setup_status(build_corpus: bool, vector_index: dict[str, Any]) -> str:
    if vector_index and vector_index.get("status") in {"index_failed", "setup_required"}:
        return "corpus_ready_vector_index_pending" if build_corpus else "source_ingested_vector_index_pending"
    if vector_index and vector_index.get("status") == "indexed":
        return "corpus_ready_vector_indexed" if build_corpus else "source_ingested_vector_indexed"
    return "corpus_ready" if build_corpus else "source_ingested"


def maybe_build_production_vector_index(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if not optional_bool(payload.get("index_vector_store"), default=False):
        return {
            "status": "not_requested",
            "backend": optional_text(payload, "vector_backend") or DEFAULT_VECTOR_BACKEND,
            "boundary": "Run corpus setup with index_vector_store=true after Qdrant is configured.",
        }
    vector_backend = optional_text(payload, "vector_backend") or DEFAULT_VECTOR_BACKEND
    embedding_provider = optional_text(payload, "index_embedding_provider") or optional_text(payload, "retrieval_provider") or "openrouter"
    embedding_model = optional_text(payload, "index_embedding_model") or optional_text(payload, "retrieval_model")
    collection = optional_text(payload, "vector_collection")
    batch_size = optional_positive_int(payload.get("index_batch_size"), "index_batch_size") or 32
    delete_stale = optional_bool(payload.get("delete_stale_vectors"), default=True)
    try:
        return build_vector_index(
            mode=mode,
            embedding_provider_name=embedding_provider,
            embedding_model=embedding_model,
            vector_backend=vector_backend,
            collection=collection,
            batch_size=batch_size,
            delete_stale=delete_stale,
        )
    except Exception as exc:  # noqa: BLE001 - corpus setup should still succeed if Qdrant is not configured yet.
        return {
            "report_id": "vector_index_build_v1",
            "status": "setup_required" if vector_backend == "qdrant" else "index_failed",
            "mode": mode,
            "vector_backend": vector_backend,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "collection": collection,
            "error": str(exc),
            "boundary": (
                "The local corpus was built, but the vector backend could not be refreshed. "
                "Set QDRANT_URL/QDRANT_API_KEY or choose another vector backend, then rerun indexing."
            ),
        }


def production_evidence_workbench(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    query = optional_text(payload, "question") or require_text(payload, "query")
    mode = str(payload.get("mode") or DEFAULT_PRODUCTION_MODE)
    policy = str(payload.get("policy") or DEFAULT_POLICY)
    top_k = optional_positive_int(payload.get("top_k"), "top_k") or DEFAULT_PRODUCTION_TOP_K
    freedom_level = optional_freedom_level(payload.get("freedom_level"))
    retrieval_engine = optional_text(payload, "retrieval_engine") or "canon_synthesis"
    generator_provider = optional_text(payload, "generator_provider") or optional_text(payload, "generator") or "deterministic"
    generator_model = optional_text(payload, "generator_model") or (
        "openai/gpt-4.1-mini" if generator_provider == "openrouter" else None
    )
    session_id = optional_text(payload, "session_id") or f"prod_{uuid.uuid4().hex[:12]}"
    requirements = production_evidence_requirements(payload, top_k)
    if retrieval_engine == "model_candidate_pool":
        packet_response = production_model_candidate_packet(
            payload=payload,
            request_id=session_id,
            query=query,
            mode=mode,
            policy=policy,
            top_k=top_k,
            freedom_level=freedom_level,
            requirements=requirements,
        )
    elif retrieval_engine in {"canon_synthesis", "synthesis", "evidence_packets"}:
        packet_response = evidence_packets(
            {
                "request_id": session_id,
                "project_id": str(payload.get("project_id") or "production_workbench"),
                "question": query,
                "mode": mode,
                "policy": policy,
                "freedom_level": freedom_level,
                "research_frame": optional_dict(payload.get("research_frame"), "research_frame"),
                "evidence_requirements": requirements,
            }
        )
    else:
        raise ProductError("retrieval_engine must be model_candidate_pool or canon_synthesis.")
    packet = first_packet(packet_response)
    evidence = packet.get("supporting_evidence") or []
    gaps = packet_response.get("coverage_gaps") or []
    relevance_gate = production_relevance_gate(query, packet_response, evidence)
    external_search = production_external_search(query, payload)
    external_cards = external_search.get("evidence_cards") or []
    external_expansion = merge_external_expansion(packet_response.get("external_expansion") or {}, external_search)
    retrieval = {
        "engine": retrieval_engine,
        "provider": packet_response.get("retrieval_provider"),
        "model": packet_response.get("retrieval_model"),
        "candidate_scope": packet_response.get("candidate_scope"),
        "vector_backend": packet_response.get("vector_backend"),
        "vector_collection": packet_response.get("vector_collection"),
        "lexical_window_k": packet_response.get("lexical_window_k"),
        "reranker_provider": packet_response.get("reranker_provider"),
        "reranker_model": packet_response.get("reranker_model"),
        "candidate_count": packet_response.get("candidate_count"),
        "degraded": bool(packet_response.get("degraded")),
        "degradation_flags": packet_response.get("degradation_flags") or [],
        "degradation_details": packet_response.get("degradation_details") or [],
        "boundary": packet_response.get("retrieval_boundary")
        or "Uses the selected CANON retrieval engine to produce inspection candidates.",
    }
    with stage_span(
        "synthesise",
        provider=generator_provider,
        model=generator_model or "evidence-note-v1",
        evidence_count=len(evidence),
        input_tokens=estimate_tokens(evidence) + estimate_tokens(query),
    ) as span:
        draft_brief = production_draft_brief(
            packet_response,
            relevance_gate,
            evidence,
            generator_provider=generator_provider,
            generator_model=generator_model,
        )
        span.set_attribute("draft_status", draft_brief.get("status"))
        span.set_attribute("output_tokens", estimate_tokens(draft_brief.get("text") or draft_brief.get("abstention")))
    model_review = production_model_review(payload, query, evidence)
    run_diagnosis = build_run_diagnosis(
        query=query,
        packet_response=packet_response,
        evidence=evidence,
        relevance_gate=relevance_gate,
        draft_brief=draft_brief,
        retrieval=retrieval,
        coverage_gaps=gaps,
        query_diagnostics=packet_response.get("query_diagnostics") or {},
        external_cards=external_cards,
        frame_coverage=packet.get("frame_coverage") or {},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    report = {
        "report_id": "production_evidence_workbench_session_v1",
        "status": production_session_status(packet_response, evidence, gaps, relevance_gate),
        "session_id": session_id,
        "created_at": utc_now(),
        "query": query,
        "mode": mode,
        "policy": policy,
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "retrieval": retrieval,
        "claim_boundary": production_claim_boundary(),
        "generation": {
            "requested_provider": generator_provider,
            "requested_model": generator_model,
            "hosted_model_boundary": (
                "Hosted generation sends the selected query and gated evidence snippets to the provider."
                if generator_provider not in {"deterministic", "template", "local"}
                else "No hosted generation requested."
            ),
        },
        "relevance_gate": relevance_gate,
        "draft_brief": draft_brief,
        "model_review": model_review,
        "run_diagnosis": run_diagnosis,
        "evidence_packet": {
            "packet_id": packet.get("packet_id"),
            "support_level": packet.get("support_level"),
            "confidence": packet.get("confidence"),
            "evidence_count": len(evidence),
            "usable_evidence_count": relevance_gate.get("usable_evidence_count", len(evidence)),
            "external_evidence_count": len(external_cards),
            "degraded": bool(packet_response.get("degraded")),
            "degradation_flags": packet_response.get("degradation_flags") or [],
            "source_diversity": packet.get("source_diversity") or {},
            "frame_coverage": packet.get("frame_coverage") or {},
            "evidence_scope_summary": production_scope_summary(evidence, external_cards),
        },
        "evidence_cards": production_evidence_cards(evidence, relevance_gate) + external_cards,
        "query_lingo": packet_response.get("query_diagnostics") or {},
        "coverage_gaps": gaps,
        "external_expansion": external_expansion,
        "recommended_actions": production_recommended_actions(packet_response, evidence, gaps),
        "feedback": {
            "endpoint": "POST /v1/production/feedback",
            "session_id": session_id,
            "dimensions": [
                "useful",
                "irrelevant",
                "missing_source",
                "citation_issue",
                "query_needs_refinement",
            ],
        },
        "raw_contract_validation": packet_response.get("contract_validation") or {},
    }
    if optional_bool(payload.get("write_telemetry"), default=True):
        write_production_telemetry(report)
    return report


def production_evidence_requirements(payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    raw = optional_dict(payload.get("evidence_requirements"), "evidence_requirements")
    requirements = {
        "top_k": top_k,
        "include_conflicts": True,
        "include_source_diversity": True,
        "include_query_diagnostics": True,
    }
    requirements.update(raw)
    requirements["top_k"] = top_k
    if optional_bool(payload.get("suggest_external_expansion"), default=False):
        requirements["external_expansion"] = {
            "enabled": True,
            "allowed_source_types": optional_string_list(
                payload.get("allowed_source_types"),
                "allowed_source_types",
            )
            or ["official", "academic", "news"],
            "max_external_queries": optional_positive_int(
                payload.get("max_external_queries"),
                "max_external_queries",
            )
            or 5,
        }
    return requirements


def production_model_candidate_packet(
    payload: dict[str, Any],
    request_id: str,
    query: str,
    mode: str,
    policy: str,
    top_k: int,
    freedom_level: str | None,
    requirements: dict[str, Any],
) -> dict[str, Any]:
    retrieval_provider = optional_text(payload, "retrieval_provider") or optional_text(payload, "vector_provider") or "openrouter"
    retrieval_model = optional_text(payload, "retrieval_model") or optional_text(payload, "vector_model")
    retrieval_model = retrieval_model or default_retrieval_model(retrieval_provider)
    reranker_provider = optional_text(payload, "reranker_provider") or optional_text(payload, "reranker") or "cohere"
    reranker_model = optional_text(payload, "reranker_model") or default_reranker_model(reranker_provider)
    candidate_k = optional_positive_int(payload.get("candidate_k"), "candidate_k") or max(
        top_k,
        DEFAULT_PRODUCTION_CANDIDATE_K,
    )
    lexical_k = optional_positive_int(payload.get("lexical_k"), "lexical_k") or candidate_k
    vector_k = optional_positive_int(payload.get("vector_k"), "vector_k") or candidate_k
    candidate_scope = optional_text(payload, "candidate_scope") or DEFAULT_PRODUCTION_CANDIDATE_SCOPE
    vector_backend = optional_text(payload, "vector_backend") or DEFAULT_VECTOR_BACKEND
    vector_collection = optional_text(payload, "vector_collection")
    lexical_window_k = optional_positive_int(payload.get("lexical_window_k"), "lexical_window_k") or max(
        DEFAULT_PRODUCTION_LEXICAL_WINDOW_K,
        candidate_k * 6,
    )
    fusion = optional_text(payload, "fusion") or "weighted_bm25_dense"
    rerank_timeout_s = optional_positive_float(payload.get("rerank_timeout_s"), "rerank_timeout_s")
    rerank_timeout_s = rerank_timeout_s or DEFAULT_PRODUCTION_RERANK_TIMEOUT_S
    degradation_flags: list[str] = []
    degradation_details: list[dict[str, Any]] = []
    try:
        candidate_report = production_candidate_report(
            query=query,
            mode=mode,
            candidate_scope=candidate_scope,
            vector_backend=vector_backend,
            vector_collection=vector_collection,
            lexical_window_k=lexical_window_k,
            lexical_k=lexical_k,
            vector_k=vector_k,
            retrieval_provider=retrieval_provider,
            retrieval_model=retrieval_model,
            fusion=fusion,
        )
    except ProductError:
        raise
    except Exception as exc:  # noqa: BLE001 - model availability belongs in the product response.
        degradation_flags.append("embed_failed_bm25_only")
        degradation_details.append({"stage": "embed", "error": str(exc)})
        try:
            candidate_report = build_bm25_candidate_pool(
                query=query,
                mode=mode,
                lexical_k=max(lexical_k, lexical_window_k, candidate_k),
            )
        except Exception as fallback_exc:  # noqa: BLE001 - report both failures to the caller.
            return {
                "status": "model_retrieval_failed",
                "request_id": request_id,
                "project_id": str(payload.get("project_id") or "production_workbench"),
                "query": query,
                "mode": mode,
                "policy": policy,
                "retrieval_provider": retrieval_provider,
                "retrieval_model": retrieval_model,
                "candidate_scope": candidate_scope,
                "vector_backend": vector_backend,
                "vector_collection": vector_collection,
                "lexical_window_k": lexical_window_k
                if candidate_scope in {"lexical_window", "windowed", "bounded_lexical_window"}
                else None,
                "reranker_provider": reranker_provider,
                "reranker_model": reranker_model,
                "candidate_count": 0,
                "retrieval_error": str(exc),
                "fallback_error": str(fallback_exc),
                "degraded": True,
                "degradation_flags": degradation_flags,
                "degradation_details": degradation_details,
                "retrieval_boundary": "Model-backed retrieval and BM25 fallback both failed.",
                "evidence_packets": [],
                "query_diagnostics": {
                    "matched_terms": [],
                    "weak_terms": content_terms(query),
                    "query_variants": [],
                    "semantic_similarity_summary": {},
                },
                "coverage_gaps": [{"gap": "Model-backed retrieval failed.", "details": str(exc)}],
                "external_expansion": production_external_expansion(requirements),
                "answer_report": {
                    "answer": "",
                    "citations": [],
                    "support_assessment": {
                        "support_level": "failed",
                        "query_term_coverage": 0.0,
                        "covered_focus_terms": [],
                        "missing_focus_terms": content_terms(query),
                    },
                },
                "contract_validation": {"status": "failed", "error": str(exc)},
            }
    candidates = candidate_report.get("candidates") or []
    if candidate_report.get("degraded"):
        degradation_flags.extend(candidate_report.get("degradation_flags") or [])
    try:
        ranked_candidates, rerank_summary = production_rerank_candidates(
            query=query,
            candidates=candidates,
            provider_name=reranker_provider,
            model=reranker_model,
            timeout_s=rerank_timeout_s,
        )
    except RerankTimeoutError as exc:
        degradation_flags.append("rerank_timeout_rrf_fallback")
        degradation_details.append({"stage": "rerank", "error": str(exc), "timeout_s": rerank_timeout_s})
        ranked_candidates, rerank_summary = production_rrf_fallback_candidates(
            candidates,
            provider_name=reranker_provider,
            model=reranker_model,
            status="timeout_rrf_fallback",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - preserve retrieval candidates when rerank fails.
        degradation_flags.append("rerank_failed_rrf_fallback")
        degradation_details.append({"stage": "rerank", "error": str(exc)})
        ranked_candidates, rerank_summary = production_rrf_fallback_candidates(
            candidates,
            provider_name=reranker_provider,
            model=reranker_model,
            status="failed_rrf_fallback",
            error=str(exc),
        )
    evidence = production_candidate_evidence(ranked_candidates[:top_k])
    support = production_candidate_support(query, evidence)
    degradation_flags = dedupe_strings(degradation_flags)
    degraded = bool(degradation_flags)
    retrieval_boundary = (
        "Evidence candidates were generated with BM25 plus the selected embedding provider, "
        "then ordered with the selected reranker for user inspection."
    )
    if degraded:
        retrieval_boundary = (
            "The request completed with degraded retrieval. Inspect degradation_flags before using the result."
        )
    return {
        "status": "complete",
        "request_id": request_id,
        "project_id": str(payload.get("project_id") or "production_workbench"),
        "query": query,
        "mode": mode,
        "policy": policy,
        "freedom_level": freedom_level,
        "retrieval_provider": retrieval_provider,
        "retrieval_model": retrieval_model,
        "candidate_scope": candidate_scope,
        "vector_backend": vector_backend,
        "vector_collection": candidate_report.get("collection") or vector_collection,
        "lexical_window_k": lexical_window_k if candidate_scope in {"lexical_window", "windowed", "bounded_lexical_window"} else None,
        "reranker_provider": rerank_summary.get("provider"),
        "reranker_model": rerank_summary.get("model"),
        "candidate_count": len(candidates),
        "degraded": degraded,
        "degradation_flags": degradation_flags,
        "degradation_details": degradation_details,
        "retrieval_boundary": retrieval_boundary,
        "candidate_generation": {
            "engine": "model_candidate_pool",
            "candidate_scope": candidate_scope,
            "vector_backend": vector_backend,
            "vector_collection": candidate_report.get("collection") or vector_collection,
            "lexical_window_k": candidate_report.get("lexical_window_k"),
            "window_document_count": candidate_report.get("window_document_count"),
            "document_count": candidate_report.get("document_count"),
            "lexical_k": lexical_k,
            "vector_k": vector_k,
            "fusion": fusion,
            "retrieval_provider": retrieval_provider,
            "retrieval_model": retrieval_model,
            "source_counts": candidate_report.get("source_counts") or {},
            "degraded": degraded,
            "degradation_flags": degradation_flags,
        },
        "rerank_summary": rerank_summary,
        "evidence_packets": [
            {
                "packet_id": f"packet_{request_id}",
                "support_level": support["support_level"],
                "confidence": support["support_confidence_label"],
                "supporting_evidence": evidence,
                "source_diversity": production_source_diversity(evidence),
                "frame_coverage": {"status": "partial", "note": "Automated coverage over candidate evidence only."},
                "evidence_scope_summary": {"private_corpus": len(evidence), "external_source": 0},
            }
        ],
        "query_diagnostics": {
            "matched_terms": support["covered_focus_terms"],
            "weak_terms": support["missing_focus_terms"],
            "field_phrases": [],
            "query_variants": [],
            "semantic_similarity_summary": {
                "top_max": max((candidate_score(row) for row in ranked_candidates[:top_k]), default=0.0),
                "provider": retrieval_provider,
                "model": retrieval_model,
            },
            "drift_risk": "sensitive",
        },
        "coverage_gaps": production_candidate_gaps(support["missing_focus_terms"]),
        "external_expansion": production_external_expansion(requirements),
        "answer_report": {
            "answer": "",
            "citations": [{"citation_id": item["evidence_id"]} for item in evidence],
            "support_assessment": support,
        },
        "contract_validation": {"status": "pass", "note": "Model candidate packet adapter."},
    }


def default_retrieval_model(provider: str) -> str | None:
    if provider == "openrouter":
        return "qwen/qwen3-embedding-8b"
    if provider == "sentence-transformers":
        return "BAAI/bge-small-en-v1.5"
    if provider == "local":
        return "hashed-semantic-v1"
    if provider == "cohere":
        return "embed-v4.0"
    return None


def default_reranker_model(provider: str) -> str | None:
    if provider == "cohere":
        return "rerank-v4.0-pro"
    if provider == "openrouter":
        return "cohere/rerank-v3.5"
    if provider in {"heuristic", "local"}:
        return "lexical-overlap-rerank-v1"
    return None


def production_candidate_report(
    *,
    query: str,
    mode: str,
    candidate_scope: str,
    vector_backend: str,
    vector_collection: str | None,
    lexical_window_k: int,
    lexical_k: int,
    vector_k: int,
    retrieval_provider: str,
    retrieval_model: str | None,
    fusion: str,
) -> dict[str, Any]:
    if candidate_scope in {"vector_store", "qdrant", "ann_index", "external_vector_store"}:
        return build_vector_store_candidate_pool(
            query=query,
            mode=mode,
            lexical_k=lexical_k,
            vector_k=vector_k,
            provider=retrieval_provider,
            model=retrieval_model,
            vector_backend=vector_backend,
            collection=vector_collection,
            fusion=fusion,
        )
    if candidate_scope in {"lexical_window", "windowed", "bounded_lexical_window"}:
        return build_windowed_candidate_pool(
            query=query,
            mode=mode,
            lexical_window_k=lexical_window_k,
            lexical_k=lexical_k,
            vector_k=vector_k,
            provider=retrieval_provider,
            model=retrieval_model,
            fusion=fusion,
        )
    if candidate_scope in {"full_embedding_store", "full_dense"}:
        return build_candidate_pool(
            query=query,
            mode=mode,
            lexical_k=lexical_k,
            vector_k=vector_k,
            provider=retrieval_provider,
            model=retrieval_model,
            fusion=fusion,
        )
    raise ProductError("candidate_scope must be vector_store, lexical_window, or full_embedding_store.")


class RerankTimeoutError(TimeoutError):
    pass


def production_rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    provider_name: str,
    model: str | None,
    timeout_s: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return [], {"provider": provider_name, "model": model, "status": "no_candidates"}
    provider = get_rerank_provider(provider_name, model)
    documents = [production_candidate_document_text(candidate) for candidate in candidates]
    with stage_span(
        "rerank",
        provider=provider.provider,
        model=provider.model,
        candidate_count=len(candidates),
        input_tokens=estimate_tokens(documents) + estimate_tokens(query),
        timeout_s=timeout_s,
    ) as span:
        reranked = call_rerank_with_timeout(provider, query, documents, len(documents), timeout_s)
        span.set_attribute("scored_candidate_count", len(reranked))
    by_index = {row.index: row.score for row in reranked if 0 <= row.index < len(candidates)}
    ranked = []
    used_indexes = set()
    for row in reranked:
        if 0 <= row.index < len(candidates) and row.index not in used_indexes:
            candidate = dict(candidates[row.index])
            candidate["rerank_score"] = row.score
            ranked.append(candidate)
            used_indexes.add(row.index)
    for index, candidate in enumerate(candidates):
        if index not in used_indexes:
            fallback = dict(candidate)
            fallback["rerank_score"] = by_index.get(index, 0.0)
            ranked.append(fallback)
    return ranked, {
        "provider": provider.provider,
        "model": provider.model,
        "status": "ok",
        "scored_candidate_count": len(by_index),
    }


def call_rerank_with_timeout(provider, query: str, documents: list[str], top_n: int, timeout_s: float | None):
    if timeout_s is None or timeout_s <= 0:
        return provider.rerank(query, documents, top_n=top_n)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canon-rerank")
    future = executor.submit(provider.rerank, query, documents, top_n)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise RerankTimeoutError(f"Rerank exceeded {timeout_s:.3f}s timeout.") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def production_rrf_fallback_candidates(
    candidates: list[dict[str, Any]],
    *,
    provider_name: str,
    model: str | None,
    status: str,
    error: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        row = dict(candidate)
        row["rerank_score"] = rrf_candidate_score(candidate)
        ranked.append(row)
    ranked = sorted(
        ranked,
        key=lambda row: (float(row.get("rerank_score") or 0.0), -int(row.get("best_rank") or 999999), row.get("chunk_id") or ""),
        reverse=True,
    )
    return ranked, {
        "provider": provider_name,
        "model": model,
        "status": status,
        "scored_candidate_count": 0,
        "fallback": "rrf",
        "error": error,
    }


def rrf_candidate_score(candidate: dict[str, Any], k: int = 60) -> float:
    ranks = candidate.get("ranks") or {}
    if not isinstance(ranks, dict) or not ranks:
        best_rank = int(candidate.get("best_rank") or 999999)
        return 1.0 / (k + best_rank)
    total = 0.0
    for rank in ranks.values():
        try:
            total += 1.0 / (k + int(rank))
        except (TypeError, ValueError):
            continue
    return total


def production_candidate_document_text(candidate: dict[str, Any]) -> str:
    return "\n".join(
        str(value or "")
        for value in [
            candidate.get("title"),
            candidate.get("source_name"),
            candidate.get("section"),
            candidate.get("text"),
            candidate.get("preview"),
        ]
    ).strip()


def production_candidate_evidence(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "evidence_id": f"C{index}",
                "rank": index,
                "title": candidate.get("title"),
                "source_name": candidate.get("source_name"),
                "published_at": str(candidate.get("year") or ""),
                "source_type": "corpus_candidate",
                "language": "unknown",
                "evidence_scope": "private_corpus",
                "document_id": candidate.get("work_id"),
                "chunk_id": candidate.get("chunk_id"),
                "text": candidate.get("preview") or "",
                "citation": {"citation_id": f"C{index}"},
                "retrieval_sources": candidate.get("retrieval_sources") or [],
                "retrieval_scores": candidate.get("scores") or {},
                "retrieval_ranks": candidate.get("ranks") or {},
                "rerank_score": candidate.get("rerank_score"),
            }
        )
    return rows


def production_candidate_support(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    focus_terms = content_terms(query)
    tokens = set()
    for item in evidence:
        tokens.update(tokenize_for_gate(production_candidate_document_text(item)))
    covered = [term for term in focus_terms if term_matches_tokens(term, tokens)]
    missing = [term for term in focus_terms if term not in covered]
    coverage = round(len(covered) / len(focus_terms), 6) if focus_terms else 0.0
    if coverage >= 0.65 and len(evidence) >= 2:
        support_level = "strong"
    elif coverage >= 0.35:
        support_level = "moderate"
    elif evidence:
        support_level = "weak"
    else:
        support_level = "none"
    return {
        "support_level": support_level,
        "support_confidence": coverage,
        "support_confidence_label": "medium" if support_level in {"strong", "moderate"} else "low",
        "query_term_coverage": coverage,
        "covered_focus_terms": covered,
        "missing_focus_terms": missing,
        "focus_terms": focus_terms,
        "context_relevance": coverage,
        "semantic_alignment": max((max_vector_score(item.get("retrieval_scores", {})) for item in evidence), default=0.0),
        "corroboration": {"overall_verdict": "candidate_only", "total_support_count": len(evidence)},
    }


def production_candidate_gaps(missing_terms: list[str]) -> list[dict[str, Any]]:
    if not missing_terms:
        return []
    return [
        {
            "gap": "Missing visible query coverage for: " + ", ".join(missing_terms[:8]),
            "suggested_next_query": " ".join(missing_terms[:8]),
        }
    ]


def production_source_diversity(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {item.get("source_name") for item in evidence if item.get("source_name")}
    warnings = []
    if len(sources) < 2 and evidence:
        warnings.append("Candidate evidence is concentrated in one source.")
    return {
        "distinct_source_count": len(sources),
        "warnings": warnings,
    }


def production_external_expansion(requirements: dict[str, Any]) -> dict[str, Any]:
    external = requirements.get("external_expansion") or {}
    if not external:
        return {"status": "disabled", "executed": False}
    return {
        "status": "planned",
        "executed": False,
        "allowed_source_types": external.get("allowed_source_types") or [],
        "max_external_queries": external.get("max_external_queries"),
    }


def production_external_search(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not optional_bool(payload.get("execute_external_search"), default=False):
        return {
            "status": "disabled",
            "executed": False,
            "provider": optional_text(payload, "external_search_provider") or DEFAULT_EXTERNAL_SEARCH_PROVIDER,
            "evidence_cards": [],
        }
    provider = optional_text(payload, "external_search_provider") or DEFAULT_EXTERNAL_SEARCH_PROVIDER
    max_results = optional_positive_int(payload.get("max_external_results"), "max_external_results") or DEFAULT_EXTERNAL_RESULT_COUNT
    max_results = min(max_results, 10)
    search_query = external_search_query(query)
    if provider != "openalex":
        return {
            "status": "failed",
            "executed": False,
            "provider": provider,
            "error": "external_search_provider must be openalex.",
            "evidence_cards": [],
        }
    try:
        cards = openalex_external_search(search_query, max_results)
    except Exception as exc:  # noqa: BLE001 - external search availability belongs in the product response.
        return {
            "status": "failed",
            "executed": False,
            "provider": provider,
            "search_query": search_query,
            "error": str(exc),
            "evidence_cards": [],
            "boundary": "External search failed; no external evidence was mixed into corpus results.",
        }
    return {
        "status": "executed",
        "executed": True,
        "provider": provider,
        "search_query": search_query,
        "result_count": len(cards),
        "evidence_cards": cards,
        "boundary": (
            "External OpenAlex search used only the query string. Results are marked as external_source "
            "and should be inspected separately from corpus evidence."
        ),
    }


def external_search_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\s./-]+", " ", collapse_spaces(query))
    return collapse_spaces(cleaned).strip(" .-/") or collapse_spaces(query)


def openalex_external_search(query: str, max_results: int) -> list[dict[str, Any]]:
    settings = load_settings()
    openalex_settings = settings.raw.get("openalex") or {}
    base_url = str(openalex_settings.get("base_url") or "https://api.openalex.org")
    filters = [
        f"language:{openalex_settings.get('language') or 'en'}",
        "is_retracted:false",
    ]
    from_year = openalex_settings.get("from_publication_year")
    to_year = openalex_settings.get("to_publication_year")
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if to_year:
        filters.append(f"to_publication_date:{to_year}-12-31")
    params = {
        "search": query,
        "per-page": str(max_results),
        "filter": ",".join(filters),
        "sort": "relevance_score:desc",
        "mailto": settings.contact_email,
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "language",
                "abstract_inverted_index",
                "primary_location",
                "best_oa_location",
                "open_access",
                "cited_by_count",
            ]
        ),
    }
    url = f"{base_url}/works?{urllib.parse.urlencode(params)}"
    payload = fetch_json(url, timeout=30)
    return [
        external_openalex_card(row, index)
        for index, row in enumerate(payload.get("results", [])[:max_results], start=1)
    ]


def external_openalex_card(row: dict[str, Any], rank: int) -> dict[str, Any]:
    primary_location = row.get("primary_location") or {}
    best_oa_location = row.get("best_oa_location") or {}
    source = (
        primary_location.get("source")
        or best_oa_location.get("source")
        or {}
    )
    open_access = row.get("open_access") or {}
    landing_page = (
        primary_location.get("landing_page_url")
        or best_oa_location.get("landing_page_url")
        or row.get("doi")
        or row.get("id")
    )
    abstract = abstract_from_inverted_index(row.get("abstract_inverted_index")) or ""
    text = collapse_spaces(abstract or row.get("title") or "")
    if len(text) > 900:
        text = text[:897].rstrip() + "..."
    evidence_id = f"X{rank}"
    return {
        "evidence_id": evidence_id,
        "rank": rank,
        "title": row.get("title") or "Untitled OpenAlex result",
        "source_name": source.get("display_name") or "OpenAlex",
        "published_at": str(row.get("publication_year") or ""),
        "source_type": "online_openalex",
        "language": row.get("language") or "unknown",
        "evidence_scope": "external_source",
        "retrieval_stage": "external_openalex_search",
        "document_id": row.get("id"),
        "chunk_id": external_chunk_id(row.get("id"), rank),
        "text": text,
        "citation": {
            "citation_id": evidence_id,
            "url": landing_page,
            "doi": row.get("doi"),
            "openalex_id": row.get("id"),
        },
        "relevance": {
            "relevance_label": "external_online_result",
            "usage_status": "external_candidate_only",
            "source_marker": "ONLINE",
        },
        "external_source": {
            "provider": "openalex",
            "is_open_access": bool(open_access.get("is_oa")),
            "cited_by_count": int(row.get("cited_by_count") or 0),
        },
    }


def external_chunk_id(openalex_id: Any, rank: int) -> str:
    raw = str(openalex_id or f"rank:{rank}").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-") or f"rank-{rank}"
    return f"external:openalex:{slug}"


def merge_external_expansion(existing: dict[str, Any], external_search: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing or {})
    if not merged:
        merged = {"status": "disabled", "executed": False}
    merged["online_search"] = {
        key: value
        for key, value in external_search.items()
        if key != "evidence_cards"
    }
    if external_search.get("executed"):
        merged["status"] = "executed"
        merged["enabled"] = True
        merged["executed"] = True
        merged["external_result_count"] = external_search.get("result_count", 0)
        merged["boundary"] = external_search.get("boundary") or merged.get("boundary")
    elif external_search.get("status") == "failed":
        merged["status"] = "failed"
        merged["enabled"] = True
        merged["executed"] = False
        merged["error"] = external_search.get("error")
        merged["boundary"] = external_search.get("boundary") or merged.get("boundary")
    return merged


def production_scope_summary(evidence: list[dict[str, Any]], external_cards: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"private_corpus": 0, "external_source": 0}
    for item in evidence:
        scope = evidence_scope(item)
        summary[scope] = summary.get(scope, 0) + 1
    summary["external_source"] = summary.get("external_source", 0) + len(external_cards)
    return summary


def candidate_score(candidate: dict[str, Any]) -> float:
    values = [float(value) for value in (candidate.get("scores") or {}).values() if isinstance(value, (int, float))]
    if isinstance(candidate.get("rerank_score"), (int, float)):
        values.append(float(candidate["rerank_score"]))
    return max(values, default=0.0)


def max_vector_score(scores: dict[str, Any]) -> float:
    values = [
        float(value)
        for key, value in scores.items()
        if str(key).startswith("vector:") and isinstance(value, (int, float))
    ]
    return max(values, default=0.0)


def first_packet(packet_response: dict[str, Any]) -> dict[str, Any]:
    packets = packet_response.get("evidence_packets") or []
    return packets[0] if packets else {}


def production_relevance_gate(
    query: str,
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    answer_report = packet_response.get("answer_report") or {}
    support = answer_report.get("support_assessment") or {}
    diagnostics = packet_response.get("query_diagnostics") or {}
    focus_terms = content_terms(query)
    anchor_terms = production_anchor_terms(focus_terms)
    query_phrases = production_query_phrases(query)
    matched_terms = diagnostics.get("matched_terms") or support.get("covered_focus_terms") or []
    weak_terms = diagnostics.get("weak_terms") or support.get("missing_focus_terms") or []
    term_coverage = optional_float(support.get("query_term_coverage"))
    if term_coverage is None:
        term_coverage = round(len(matched_terms) / len(focus_terms), 6) if focus_terms else 0.0
    card_gates = [production_evidence_card_gate(item, focus_terms, anchor_terms, query_phrases) for item in evidence]
    usable_count = sum(1 for row in card_gates if row["usage_status"] == "usable_candidate_evidence")
    required_usable_count = production_required_usable_count(evidence)
    support_level = str(support.get("support_level") or "").lower()
    if packet_response.get("status") != "complete":
        status = "run_failed"
        severity = "high"
        message = "The retrieval run did not complete."
        draft_allowed = False
    elif evidence and usable_count == 0:
        status = "corpus_mismatch"
        severity = "high"
        message = "The selected corpus does not visibly cover the query terms."
        draft_allowed = False
    elif focus_terms and not matched_terms and term_coverage == 0.0:
        status = "corpus_mismatch"
        severity = "high"
        message = "The selected corpus does not visibly cover the query terms."
        draft_allowed = False
    elif support_level == "weak" or (focus_terms and term_coverage < 0.25) or usable_count < required_usable_count:
        status = "weak_query_match"
        severity = "medium"
        message = "Retrieved rows are weakly aligned with the query and should be treated as candidates for debugging."
        draft_allowed = False
    else:
        status = "pass"
        severity = "info"
        message = "Retrieved rows visibly overlap with the query; inspect citations before use."
        draft_allowed = True
    return {
        "status": status,
        "severity": severity,
        "message": message,
        "draft_allowed": draft_allowed,
        "evidence_use": "supporting_evidence_candidates" if draft_allowed else "diagnostic_candidates_only",
        "focus_terms": focus_terms,
        "anchor_terms": anchor_terms,
        "query_phrases": query_phrases,
        "matched_terms": matched_terms,
        "weak_terms": weak_terms,
        "query_term_coverage": round(float(term_coverage), 6),
        "support_level": support.get("support_level"),
        "support_confidence": support.get("support_confidence"),
        "usable_evidence_count": usable_count if draft_allowed else 0,
        "diagnostic_candidate_count": len(evidence) - usable_count,
        "required_usable_count": required_usable_count,
        "candidate_count": len(evidence),
        "card_gates": card_gates,
    }


def production_anchor_terms(focus_terms: list[str]) -> list[str]:
    anchors = [term for term in focus_terms if term not in PRODUCTION_GENERIC_QUERY_TERMS]
    if anchors:
        return anchors
    return [term for term in focus_terms if len(term) >= 4]


def production_query_phrases(query: str) -> list[str]:
    tokens = [
        token
        for token in tokenize_for_gate(query)
        if token not in {"what", "where", "when", "which", "with", "about", "around", "does", "this"}
    ]
    phrases = []
    for left, right in zip(tokens, tokens[1:], strict=False):
        if left in PRODUCTION_GENERIC_QUERY_TERMS and right in PRODUCTION_GENERIC_QUERY_TERMS:
            continue
        phrase = f"{left} {right}"
        if phrase not in phrases:
            phrases.append(phrase)
    if "data" in tokens and "center" in tokens and "data center" not in phrases:
        phrases.append("data center")
    return phrases


def production_required_usable_count(evidence: list[dict[str, Any]]) -> int:
    if not evidence:
        return 0
    if len(evidence) == 1:
        return 1
    return 2


def production_evidence_card_gate(
    item: dict[str, Any],
    focus_terms: list[str],
    anchor_terms: list[str],
    query_phrases: list[str],
) -> dict[str, Any]:
    text = " ".join(
        str(value or "")
        for value in [
            item.get("title"),
            item.get("source_name"),
            item.get("text"),
            item.get("preview"),
            item.get("source_type"),
        ]
    )
    normalized_text = normalized_term(text)
    tokens = set(tokenize_for_gate(text))
    overlap = [term for term in focus_terms if term_matches_tokens(term, tokens)]
    anchor_overlap = [term for term in anchor_terms if term_matches_tokens(term, tokens)]
    phrase_overlap = [phrase for phrase in query_phrases if phrase in normalized_text]
    if not focus_terms:
        label = "review"
        usage = "usable_candidate_evidence"
    elif (
        (phrase_overlap and anchor_overlap)
        or len(anchor_overlap) >= 2
        or (len(overlap) >= 3 and anchor_overlap)
        or (len(focus_terms) <= 2 and anchor_overlap)
    ):
        label = "visible_query_overlap"
        usage = "usable_candidate_evidence"
    else:
        label = "off_query"
        usage = "diagnostic_candidate_only"
    return {
        "evidence_id": item.get("evidence_id"),
        "relevance_label": label,
        "usage_status": usage,
        "query_overlap_terms": overlap,
        "anchor_overlap_terms": anchor_overlap,
        "query_phrase_overlap": phrase_overlap,
    }


def tokenize_for_gate(text: str) -> list[str]:
    from canon.retrieval.tokenize import tokenize

    return tokenize(text)


def production_session_status(
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    relevance_gate: dict[str, Any],
) -> str:
    if packet_response.get("status") != "complete":
        return "run_failed"
    if relevance_gate.get("status") == "corpus_mismatch":
        return "corpus_mismatch_no_grounded_answer"
    if relevance_gate.get("status") in {"weak_query_match", "weak_candidate_overlap"}:
        return "needs_query_refinement"
    if not evidence:
        return "needs_query_refinement"
    if gaps:
        return "ready_with_gaps"
    return "ready_for_user_inspection"


def production_claim_boundary() -> dict[str, Any]:
    return {
        "safe_to_use_for": [
            "finding and inspecting candidate evidence",
            "query refinement",
            "coverage-gap discovery",
            "drafting notes that users verify against citations",
        ],
        "not_safe_to_claim": [
            "the answer is complete or factually correct",
            "the model found every relevant source",
            "CANON is a benchmark winner",
            "the result is publication or high-stakes decision ready",
        ],
        "human_review_required_for": [
            "final conclusions",
            "factual correctness claims",
            "publication-quality benchmark claims",
            "clinical, legal, policy, or operational decisions",
        ],
    }


def production_draft_brief(
    packet_response: dict[str, Any],
    relevance_gate: dict[str, Any],
    evidence: list[dict[str, Any]],
    generator_provider: str = "deterministic",
    generator_model: str | None = None,
) -> dict[str, Any]:
    answer_report = packet_response.get("answer_report") or {}
    citations = answer_report.get("citations") or []
    support = answer_report.get("support_assessment") or {}
    if not relevance_gate.get("draft_allowed"):
        return {
            "status": "blocked_by_relevance_gate",
            "method": "abstention",
            "text": "",
            "abstention": relevance_gate.get("message") or "The retrieved evidence is not strong enough for a draft.",
            "citation_count": 0,
            "citations": [],
            "support_assessment": support,
            "boundary": "No draft was generated because the selected corpus/query match is weak.",
        }
    usable_evidence = usable_evidence_for_note(evidence, relevance_gate)
    usable_citations = citations_for_usable_evidence(citations, usable_evidence)
    if generator_provider not in {"deterministic", "template", "local"}:
        return model_generated_draft_brief(
            query=packet_response.get("query") or "",
            evidence=usable_evidence,
            diagnostic_count=len(evidence) - len(usable_evidence),
            support=support,
            citations=usable_citations,
            generator_provider=generator_provider,
            generator_model=generator_model,
        )
    answer = evidence_derived_note(packet_response.get("query") or "", usable_evidence, len(evidence) - len(usable_evidence))
    return {
        "status": "evidence_note_ready" if answer and usable_citations else "insufficient_cited_evidence",
        "method": "deterministic_evidence_note",
        "generator": {"provider": "deterministic", "model": "evidence-note-v1"},
        "text": answer,
        "citation_count": len(usable_citations),
        "citations": usable_citations,
        "support_assessment": support,
        "boundary": "This is an evidence-derived note, not model-generated synthesis or final analysis.",
    }


def production_model_review(payload: dict[str, Any], query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not optional_bool(payload.get("run_model_review"), default=False):
        return {
            "status": "not_requested",
            "provider": optional_text(payload, "model_review_provider") or "openrouter",
            "model": optional_text(payload, "model_review_model") or DEFAULT_MODEL_TASK_MODEL,
            "boundary": "Set run_model_review=true to send gated evidence snippets for typed stance/extraction review.",
        }
    provider = optional_text(payload, "model_review_provider") or "openrouter"
    model = optional_text(payload, "model_review_model") or DEFAULT_MODEL_TASK_MODEL
    try:
        return run_evidence_model_review(
            query=query,
            evidence=evidence,
            provider=provider,
            model=model,
            allow_external_data=optional_bool(payload.get("allow_external_model_review"), default=False),
            max_evidence=optional_positive_int(payload.get("model_review_max_evidence"), "model_review_max_evidence") or 8,
        )
    except Exception as exc:  # noqa: BLE001 - provider failures are part of the product trace.
        return {
            "report_id": "evidence_model_review_v1",
            "status": "model_review_failed",
            "provider": provider,
            "model": model,
            "error": str(exc),
            "human_review_required": True,
            "boundary": "Typed model review failed; no synthetic review labels were substituted.",
        }


def model_generated_draft_brief(
    query: str,
    evidence: list[dict[str, Any]],
    diagnostic_count: int,
    support: dict[str, Any],
    citations: list[dict[str, Any]],
    generator_provider: str,
    generator_model: str | None,
) -> dict[str, Any]:
    if not evidence:
        return {
            "status": "insufficient_cited_evidence",
            "method": "model_generation_skipped",
            "generator": {"provider": generator_provider, "model": generator_model},
            "text": "",
            "abstention": "No usable evidence rows passed the relevance gate.",
            "citation_count": 0,
            "citations": [],
            "support_assessment": support,
            "boundary": "Model generation was skipped because no gated evidence was available.",
        }
    try:
        provider = get_generation_provider(generator_provider, generator_model)
        prompt = model_generation_prompt(query, evidence, diagnostic_count, support)
        result = provider.generate(prompt)
        if not str(result.text or "").strip():
            result = provider.generate(prompt + "\nReturn a non-empty cited answer or an explicit evidence limitation.")
    except Exception as exc:  # noqa: BLE001 - provider availability belongs in the product response.
        return {
            "status": "model_generation_failed",
            "method": "hosted_model_generation",
            "generator": {"provider": generator_provider, "model": generator_model},
            "text": "",
            "error": str(exc),
            "citation_count": 0,
            "citations": [],
            "support_assessment": support,
            "boundary": "Hosted model generation failed; no fallback answer was substituted.",
        }
    if not str(result.text or "").strip():
        return {
            "status": "model_generation_empty",
            "method": "hosted_model_generation",
            "generator": {"provider": result.provider, "model": result.model},
            "text": "",
            "error": "Hosted model returned an empty response after retry.",
            "citation_count": 0,
            "citations": [],
            "support_assessment": support,
            "diagnostic_candidate_count": diagnostic_count,
            "boundary": "Hosted model generation returned no draft; no fallback answer was substituted.",
        }
    return {
        "status": "model_draft_ready",
        "method": "hosted_model_generation",
        "generator": {"provider": result.provider, "model": result.model},
        "text": result.text,
        "citation_count": len(citations),
        "citations": citations,
        "support_assessment": support,
        "diagnostic_candidate_count": diagnostic_count,
        "boundary": "Model draft is based only on relevance-gated evidence snippets and still requires verification.",
    }


def model_generation_prompt(
    query: str,
    evidence: list[dict[str, Any]],
    diagnostic_count: int,
    support: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "task": "write_a_concise_evidence_grounded_answer",
            "query": query,
            "rules": [
                "Use only the evidence_items below.",
                "Cite every factual sentence with citation ids like [C1].",
                "Do not mention evidence not listed here.",
                "If evidence is partial, say what is missing.",
                "Do not claim final correctness or completeness.",
            ],
            "support_assessment": support,
            "diagnostic_candidate_count_excluded": diagnostic_count,
            "evidence_items": [
                {
                    "citation_id": item.get("evidence_id") or item.get("citation_id"),
                    "title": item.get("title"),
                    "source_name": item.get("source_name"),
                    "published_at": item.get("published_at"),
                    "text": collapse_spaces(item.get("text") or item.get("preview") or "")[:900],
                }
                for item in evidence[:8]
            ],
            "output_format": {
                "answer": "2-4 concise paragraphs or bullets",
                "must_include": ["direct answer", "key evidence", "important caveats"],
            },
        },
        indent=2,
        ensure_ascii=False,
    )


def usable_evidence_for_note(evidence: list[dict[str, Any]], relevance_gate: dict[str, Any]) -> list[dict[str, Any]]:
    usable_ids = {
        row.get("evidence_id")
        for row in relevance_gate.get("card_gates", [])
        if row.get("usage_status") == "usable_candidate_evidence"
    }
    if not usable_ids:
        return []
    return [item for item in evidence if item.get("evidence_id") in usable_ids]


def citations_for_usable_evidence(
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    usable_ids = {item.get("evidence_id") or item.get("citation_id") for item in evidence}
    return [citation for citation in citations if citation.get("citation_id") in usable_ids]


def evidence_derived_note(query: str, evidence: list[dict[str, Any]], diagnostic_count: int = 0) -> str:
    if not evidence:
        return ""
    intro = f"CANON found {len(evidence)} usable candidate evidence item(s) for inspection."
    if query:
        intro += f" Query: {query}."
    if diagnostic_count:
        intro += f" {diagnostic_count} retrieved row(s) were kept as diagnostics only."
    rows = []
    for item in evidence[:4]:
        citation_id = item.get("evidence_id") or item.get("citation_id") or "C?"
        title = collapse_spaces(item.get("title") or "Untitled source")
        text = collapse_spaces(item.get("text") or item.get("preview") or "")
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        rows.append(f"{citation_id}: {title}. {text}")
    return " ".join([intro, *rows])


def production_evidence_cards(evidence: list[dict[str, Any]], relevance_gate: dict[str, Any]) -> list[dict[str, Any]]:
    gates = {
        row.get("evidence_id"): row
        for row in relevance_gate.get("card_gates", [])
    }
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "rank": item.get("rank"),
            "title": item.get("title"),
            "source_name": item.get("source_name"),
            "published_at": item.get("published_at"),
            "source_type": item.get("source_type"),
            "language": item.get("language"),
            "evidence_scope": item.get("evidence_scope"),
            "document_id": item.get("document_id"),
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text"),
            "citation": item.get("citation"),
            "relevance": gates.get(item.get("evidence_id"), {}),
        }
        for item in evidence
    ]


def production_recommended_actions(
    packet_response: dict[str, Any],
    evidence: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> list[str]:
    actions = []
    for gap in gaps[:4]:
        suggested = str(gap.get("suggested_next_query") or "").strip()
        if suggested:
            actions.append(f"Try a follow-up query: {suggested}")
        else:
            actions.append(str(gap.get("gap") or "Inspect the evidence gap."))
    diagnostics = packet_response.get("query_diagnostics") or {}
    weak_terms = diagnostics.get("weak_terms") or []
    if weak_terms:
        actions.append("Replace weak terms: " + ", ".join(weak_terms[:5]))
    if len(evidence) < 5:
        actions.append("Increase top_k or broaden the query before trusting the draft.")
    if not actions:
        actions.append("Inspect the top cited evidence before using the draft.")
    return dedupe_strings(actions)[:6]


def write_production_telemetry(report: dict[str, Any]) -> None:
    settings = load_settings()
    packet = report.get("evidence_packet") or {}
    relevance = report.get("relevance_gate") or {}
    diagnosis = report.get("run_diagnosis") or {}
    record = {
        "report_id": "production_workbench_telemetry_event_v1",
        "created_at": report.get("created_at"),
        "session_id": report.get("session_id"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "policy": report.get("policy"),
        "top_k": report.get("top_k"),
        "query": report.get("query"),
        "elapsed_ms": report.get("elapsed_ms"),
        "evidence_count": packet.get("evidence_count"),
        "usable_evidence_count": packet.get("usable_evidence_count"),
        "support_level": packet.get("support_level"),
        "confidence": packet.get("confidence"),
        "relevance_gate_status": relevance.get("status"),
        "query_term_coverage": relevance.get("query_term_coverage"),
        "coverage_gap_count": len(report.get("coverage_gaps") or []),
        "retrieval_degraded": bool((report.get("retrieval") or {}).get("degraded")),
        "retrieval_degradation_flags": (report.get("retrieval") or {}).get("degradation_flags") or [],
        "run_diagnosis_status": diagnosis.get("overall_status"),
        "run_diagnosis_failure_class": diagnosis.get("failure_class"),
        "run_diagnosis_issue_categories": diagnosis.get("issue_categories") or [],
        "boundary": "Local product telemetry only; this is not a human-review label.",
    }
    append_jsonl(settings.reports_dir / "production_workbench_telemetry_v1.jsonl", record)


def production_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    session_id = require_text(payload, "session_id")
    feedback_type = optional_text(payload, "feedback_type") or "general"
    rating = optional_rating(payload.get("rating"))
    comment = optional_text(payload, "comment")
    evidence_id = optional_text(payload, "evidence_id")
    if rating is None and not comment and not evidence_id:
        raise ProductError("Provide rating, comment, or evidence_id for feedback.")
    record = {
        "report_id": "production_feedback_event_v1",
        "feedback_id": f"feedback_{uuid.uuid4().hex[:12]}",
        "created_at": utc_now(),
        "session_id": session_id,
        "feedback_type": feedback_type,
        "rating": rating,
        "evidence_id": evidence_id,
        "query": optional_text(payload, "query"),
        "comment": comment,
        "not_human_review_label": True,
        "boundary": "This captures user experience feedback. It is not an authoritative qrels or factual-correctness label.",
    }
    append_jsonl(settings.reports_dir / "production_feedback_v1.jsonl", record)
    return {
        "report_id": "production_feedback_response_v1",
        "status": "feedback_recorded",
        "feedback_id": record["feedback_id"],
        "session_id": session_id,
        "stored_at": "reports/production_feedback_v1.jsonl",
        "boundary": record["boundary"],
    }


def optional_rating(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        rating = int(value)
    except (TypeError, ValueError) as exc:
        raise ProductError("rating must be an integer from 1 to 5.") from exc
    if rating < 1 or rating > 5:
        raise ProductError("rating must be between 1 and 5.")
    return rating


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_positive_float(value: Any, name: str) -> float | None:
    parsed = optional_float(value)
    if parsed is None:
        return None
    if parsed <= 0:
        raise ProductError(f"{name} must be greater than 0.")
    return parsed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        text = collapse_spaces(value)
        key = normalized_term(text)
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


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


def research_workflow(payload: dict[str, Any]) -> dict:
    from canon.product.research_workflow import run_research_workflow

    return run_research_workflow(payload)


def stage2_synthesis(payload: dict[str, Any]) -> dict:
    from canon.product.stage2_synthesis import run_stage2_synthesis

    return run_stage2_synthesis(payload)


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


def intelligence_review_feedback(payload: dict[str, Any]) -> dict:
    from canon.product.intelligence_review import build_feedback_report

    records_path = intelligence_review_records_path(payload)
    return build_feedback_report(
        records_path,
        write_report=optional_bool(payload.get("write_report"), default=True),
    )


def intelligence_review_handoff(payload: dict[str, Any]) -> dict:
    from canon.product.intelligence_review import build_review_handoff

    records_path = intelligence_review_records_path(payload)
    output_path = optional_path(payload.get("output_path") or payload.get("output"), "output_path")
    return build_review_handoff(
        records_path,
        output_path=output_path,
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
    scope = evidence_scope(item)
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
        "evidence_scope": scope,
        "retrieval_stage": item.get("retrieval_stage") or retrieval_stage_for_scope(scope),
        "provenance": item.get("provenance"),
        "domain": item.get("domain"),
        "jurisdiction": item.get("jurisdiction"),
        "claim": item.get("claim"),
    }


def evidence_scope(item: dict[str, Any]) -> str:
    explicit = str(item.get("evidence_scope") or "").strip()
    if explicit in {"private_corpus", "external_source"}:
        return explicit
    retrieval_stage = str(item.get("retrieval_stage") or "").strip()
    if retrieval_stage.startswith("external"):
        return "external_source"
    provenance = normalized_term(item.get("provenance"))
    if provenance in {"external source", "external", "public web"}:
        return "external_source"
    return "private_corpus"


def retrieval_stage_for_scope(scope: str) -> str:
    if scope == "external_source":
        return "external_expansion"
    return "private_corpus_retrieval"


def evidence_scope_summary(evidence: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"private_corpus": 0, "external_source": 0}
    for item in evidence:
        scope = evidence_scope(item)
        summary[scope] = summary.get(scope, 0) + 1
    return summary


def compact_packet_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    query_to_corpus = diagnostics.get("query_to_corpus") or {}
    neighborhood = diagnostics.get("result_neighborhood") or {}
    stability = diagnostics.get("stability") or {}
    return {
        "matched_terms": query_to_corpus.get("matched_terms", []),
        "weak_terms": query_to_corpus.get("weak_terms", []),
        "semantic_similarity_summary": query_to_corpus.get("semantic_similarity_summary", {}),
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


def expansion_preferences_from_requirements(requirements: dict[str, Any]) -> dict[str, Any]:
    raw = requirements.get("external_expansion") if requirements else None
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raise ProductError("external_expansion must be an object.")
    enabled = optional_bool(raw.get("enabled"), default=False)
    allowed_source_types = optional_string_list(
        raw.get("allowed_source_types"),
        "allowed_source_types",
    ) or []
    max_external_queries = optional_positive_int(
        raw.get("max_external_queries"),
        "max_external_queries",
    ) or 5
    return {
        "enabled": enabled,
        "allowed_source_types": allowed_source_types,
        "max_external_queries": max_external_queries,
    }


def build_external_expansion_plan(
    query: str,
    evidence: list[dict[str, Any]],
    coverage_gaps: list[dict[str, Any]],
    research_frame: dict[str, Any],
    preferences: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool(preferences.get("enabled"))
    if not enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "executed": False,
            "suggested_queries": [],
            "boundary": "External expansion is opt-in and was not requested.",
        }
    max_queries = int(preferences.get("max_external_queries") or 5)
    allowed_source_types = preferences.get("allowed_source_types") or []
    suggestions = expansion_query_suggestions(
        query=query,
        evidence=evidence,
        coverage_gaps=coverage_gaps,
        research_frame=research_frame,
        allowed_source_types=allowed_source_types,
        max_queries=max_queries,
    )
    return {
        "status": "planned" if suggestions else "no_action",
        "enabled": True,
        "executed": False,
        "allowed_source_types": allowed_source_types,
        "suggested_queries": suggestions,
        "boundary": (
            "CANON generated reviewable external-search suggestions only. "
            "No external source was queried and no private corpus text was sent to a hosted provider."
        ),
        "human_review_required": True,
    }


def expansion_query_suggestions(
    query: str,
    evidence: list[dict[str, Any]],
    coverage_gaps: list[dict[str, Any]],
    research_frame: dict[str, Any],
    allowed_source_types: list[str],
    max_queries: int,
) -> list[dict[str, Any]]:
    candidates = []
    for gap in coverage_gaps:
        suggested = str(gap.get("suggested_next_query") or "").strip()
        if suggested:
            candidates.append(
                expansion_query(
                    suggested,
                    reason=gap.get("gap") or "Coverage gap.",
                    source_types=allowed_source_types,
                    priority="high" if gap.get("severity") == "high" else "medium",
                )
            )
    for region in optional_string_list(research_frame.get("regions"), "regions") or []:
        candidates.append(
            expansion_query(
                f"{query} {region}",
                reason=f"Check external coverage for requested region: {region}.",
                source_types=allowed_source_types,
                priority="medium",
            )
        )
    for item in evidence[:3]:
        title = str(item.get("title") or item.get("source_name") or "").strip()
        if title:
            candidates.append(
                expansion_query(
                    f"{query} {title}",
                    reason="Corroborate or update a cited private-corpus source.",
                    source_types=allowed_source_types,
                    priority="low",
                )
            )
    return dedupe_expansion_queries(candidates)[:max_queries]


def expansion_query(
    query: str,
    reason: str,
    source_types: list[str],
    priority: str,
) -> dict[str, Any]:
    return {
        "query": collapse_spaces(query),
        "reason": reason,
        "allowed_source_types": source_types,
        "priority": priority,
        "review_status": "needs_human_approval",
    }


def dedupe_expansion_queries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for candidate in candidates:
        key = normalized_term(candidate.get("query"))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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


def prehuman_check(payload: dict[str, Any]) -> dict:
    from canon.product.prehuman_check import run_prehuman_check

    mode = str(payload.get("mode") or DEFAULT_MODE)
    qrels_path = optional_path(payload.get("qrels_path") or payload.get("qrels"), "qrels_path")
    try:
        return run_prehuman_check(
            mode=mode,
            benchmark_id=optional_text(payload, "benchmark_id"),
            qrels_path=qrels_path,
            judge_provider=str(payload.get("judge_provider") or "heuristic"),
            judge_model=optional_text(payload, "judge_model"),
            model_providers=parse_providers_from_payload(payload.get("model_providers")),
            rerankers=optional_string_list(payload.get("rerankers"), "rerankers"),
            top_k=optional_positive_int(payload.get("top_k"), "top_k") or 10,
            candidate_k=optional_positive_int(payload.get("candidate_k"), "candidate_k") or 25,
            write_report=optional_bool(payload.get("write_report"), default=True),
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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.diversity import run_diversity_audit
from canon.reports.claim_decision import build_claim_decision
from canon.retrieval.compare import compare
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
    }


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


def optional_text(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None

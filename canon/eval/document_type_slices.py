from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.engine import retrieve_with_stages, weights_with_query_intent
from canon.retrieval.stages import (
    apply_safety_gate,
    candidate_pool_size,
    enrich_candidates,
    retrieve_candidates,
)


DEFAULT_DOCUMENT_TYPE_QUERIES = [
    {
        "id": "market-risk-filings",
        "query": "battery storage market current risk SEC filing interconnection backlog",
        "intent": "market_due_diligence",
    },
    {
        "id": "legal-authority-disclosure",
        "query": "legal authority disclosure duty contract risk appellate court",
        "intent": "legal_authority_lookup",
    },
    {
        "id": "policy-method-evidence",
        "query": "policy report method sample subsidy effect grid battery storage",
        "intent": "method_and_policy_evidence",
    },
    {
        "id": "primary-source-testimony",
        "query": "interview transcript archival evidence community opposition battery storage",
        "intent": "primary_source_context",
    },
    {
        "id": "open-web-injection",
        "query": "battery storage market API key hidden instruction",
        "intent": "open_web_safety_probe",
    },
]


def evaluate_document_type_slices(
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
    resolved_policies = policies or ["lexical", "balanced", "rag"]
    policy_weights = resolve_policy_weights(settings.raw["retrieval"]["policies"], resolved_policies)
    resolved_queries = queries or DEFAULT_DOCUMENT_TYPE_QUERIES
    runs = [
        run_document_type_slice(
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
        "contract_version": CONTRACT_VERSION,
        "report_id": "document_type_slices_v1",
        "mode": mode,
        "document_count": len(documents),
        "query_count": len(resolved_queries),
        "policy_ids": resolved_policies,
        "top_k": resolved_top_k,
        "corpus_document_type_counts": corpus_document_type_counts(documents),
        "runs": runs,
        "slice_summary": slice_summary(runs),
        "coverage_summary": coverage_summary(documents, runs),
        "limitations": [
            "Document-type slice results are deterministic diagnostics, not human relevance labels.",
            "The default queries are stress probes; production use needs domain-specific query suites and qrels.",
            "Document-type detection is metadata/marker based and should be audited against hand labels before policy use.",
            "High or low exposure for a type can be appropriate depending on query intent; treat slice gaps as review prompts.",
        ],
    }
    report["contract_validation"] = validate_report(report, "document_type_slices")
    if write_report:
        write_json(settings.reports_dir / f"document_type_slices_{mode}.json", report)
    return report


def run_document_type_slice(
    query_spec: dict[str, str],
    policy: str,
    weights: dict[str, float],
    documents: list[RetrievalDocument],
    top_k: int,
    preview_chars: int,
) -> dict[str, Any]:
    candidates = inspect_candidates(query_spec["query"], documents, weights, top_k)
    staged = retrieve_with_stages(
        query=query_spec["query"],
        documents=documents,
        weights=weights,
        top_k=top_k,
        preview_chars=preview_chars,
    )
    final_results = [trace_row(item.to_dict()) for item in staged["trace"]]
    rejected_candidates = [rejected_row(item) for item in staged["rejected_candidates"]]
    final_types = [row["document_type"] for row in final_results]
    rejected_types = [row["document_type"] for row in rejected_candidates]
    candidate_types = [row["document_type"] for row in candidates]
    return {
        "id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "policy": policy,
        "candidate_document_type_counts": dict(sorted(Counter(candidate_types).items())),
        "final_document_type_counts": dict(sorted(Counter(final_types).items())),
        "rejected_document_type_counts": dict(sorted(Counter(rejected_types).items())),
        "candidate_safety_decision_counts": dict(
            sorted(Counter(row["safety_decision"] for row in candidates).items())
        ),
        "candidate_slice_stats": candidate_slice_stats(candidates),
        "stage_summary": staged["stage_summary"],
        "candidate_sample": candidates[: min(len(candidates), 12)],
        "final_results": final_results,
        "rejected_candidates": rejected_candidates,
    }


def inspect_candidates(
    query: str,
    documents: list[RetrievalDocument],
    weights: dict[str, float],
    top_k: int,
) -> list[dict[str, Any]]:
    effective_weights = weights_with_query_intent(query, weights)
    requested_candidate_k = int(effective_weights["candidate_k"]) if "candidate_k" in effective_weights else None
    candidates = retrieve_candidates(
        query,
        documents,
        candidate_pool_size(top_k, len(documents), requested_candidate_k),
    )
    gated = apply_safety_gate(enrich_candidates(query, candidates))
    rows = []
    for result in gated:
        candidate = result.enriched.candidate
        document = candidate.document
        profile = result.enriched.signals.get("document_profile", {})
        rows.append(
            {
                "candidate_rank": candidate.candidate_rank,
                "chunk_id": document.chunk_id,
                "work_id": document.work_id,
                "title": document.title,
                "document_type": profile.get("document_type", "unknown_unstructured_text"),
                "evidence_family": profile.get("evidence_family", "unknown"),
                "candidate_score": candidate.candidate_score,
                "lexical_relevance": round(candidate.lexical_relevance, 6),
                "semantic_similarity": round(candidate.semantic_similarity, 6),
                "authority_signal": profile.get("authority_signal", 0.0),
                "source_intent_risk": profile.get("source_intent_risk", 0.0),
                "adversarial_surface": profile.get("adversarial_surface", 0.0),
                "recommended_decision_role": profile.get("recommended_decision_role", "annotate_missing_type"),
                **candidate_domain_fields(result.enriched.signals.get("domain_profile", {})),
                "safety_decision": result.safety.decision,
                "gate_decision": result.decision,
                "gate_reasons": result.reasons,
            }
        )
    return rows


def trace_row(item: dict[str, Any]) -> dict[str, Any]:
    stage = item.get("decision", {}).get("stage", {})
    profile = stage.get("signals", {}).get("document_profile", {})
    safety = stage.get("safety_gate", {})
    return {
        "rank": item["rank"],
        "chunk_id": item["chunk_id"],
        "work_id": item["work_id"],
        "title": item["title"],
        "source_name": item["source_name"],
        "year": item["year"],
        "final_score": item["final_score"],
        "base_score": item.get("base_score", item["final_score"]),
        "document_type": profile.get("document_type", "unknown_unstructured_text"),
        "evidence_family": profile.get("evidence_family", "unknown"),
        "authority_signal": profile.get("authority_signal", 0.0),
        "source_intent_risk": profile.get("source_intent_risk", 0.0),
        "adversarial_surface": profile.get("adversarial_surface", 0.0),
        "recommended_decision_role": profile.get("recommended_decision_role", "annotate_missing_type"),
        **domain_fields(stage.get("signals", {}).get("domain_profile", {})),
        "safety_decision": safety.get("safety", {}).get("decision", "allow"),
        "preview": item["preview"],
    }


def rejected_row(item: dict[str, Any]) -> dict[str, Any]:
    stage = item.get("stage", {})
    profile = stage.get("signals", {}).get("document_profile", {})
    safety_gate = stage.get("safety_gate", {})
    return {
        "chunk_id": item["chunk_id"],
        "work_id": item["work_id"],
        "title": item["title"],
        "document_type": profile.get("document_type", "unknown_unstructured_text"),
        "evidence_family": profile.get("evidence_family", "unknown"),
        "authority_signal": profile.get("authority_signal", 0.0),
        "source_intent_risk": profile.get("source_intent_risk", 0.0),
        "adversarial_surface": profile.get("adversarial_surface", 0.0),
        "recommended_decision_role": profile.get("recommended_decision_role", "annotate_missing_type"),
        **domain_fields(stage.get("signals", {}).get("domain_profile", {})),
        "safety_decision": safety_gate.get("safety", {}).get("decision", "unknown"),
        "safety_reasons": safety_gate.get("reasons", []),
    }


def candidate_domain_fields(profile: dict[str, Any]) -> dict[str, Any]:
    row = domain_fields(profile)
    row["domain_gate_flags"] = list(profile.get("gate_flags") or [])
    return row


def domain_fields(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": profile.get("domain", "general_academic"),
        "domain_profile": profile.get("domain_profile", "general_academic"),
        "domain_fit_score": profile.get("domain_fit_score", 0.0),
        "domain_fit_tier": profile.get("domain_fit_tier", "low_fit"),
    }


def candidate_slice_stats(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for row in candidates:
        doc_type = row["document_type"]
        bucket = stats.setdefault(
            doc_type,
            {
                "candidate_count": 0,
                "candidate_rank_total": 0.0,
                "candidate_score_total": 0.0,
                "candidate_gate_reject_count": 0,
            },
        )
        bucket["candidate_count"] = int(bucket["candidate_count"]) + 1
        bucket["candidate_rank_total"] = float(bucket["candidate_rank_total"]) + float(row["candidate_rank"])
        bucket["candidate_score_total"] = float(bucket["candidate_score_total"]) + float(row["candidate_score"])
        if row["gate_decision"] == "reject":
            bucket["candidate_gate_reject_count"] = int(bucket["candidate_gate_reject_count"]) + 1
    return dict(sorted(stats.items()))


def corpus_document_type_counts(documents: list[RetrievalDocument]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(document.work_signals.get("document_type", "unknown_unstructured_text"))
                for document in documents
            ).items()
        )
    )


def slice_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    slices: dict[str, dict[str, Any]] = {}
    for run in runs:
        for doc_type, stats in run["candidate_slice_stats"].items():
            bucket = ensure_slice(slices, doc_type)
            bucket["candidate_count"] += int(stats["candidate_count"])
            bucket["candidate_rank_total"] += float(stats["candidate_rank_total"])
            bucket["candidate_score_total"] += float(stats["candidate_score_total"])
            bucket["candidate_gate_reject_count"] += int(stats["candidate_gate_reject_count"])
        for row in run["final_results"]:
            bucket = ensure_slice(slices, row["document_type"])
            bucket["final_count"] += 1
            bucket["rank_total"] += float(row["rank"])
            bucket["final_score_total"] += float(row["final_score"])
        for row in run["rejected_candidates"]:
            bucket = ensure_slice(slices, row["document_type"])
            bucket["rejected_count"] += 1
            if row["safety_decision"] in {"block", "quarantine"}:
                bucket["safety_rejected_count"] += 1
    finalized = {}
    final_total = sum(bucket["final_count"] for bucket in slices.values())
    rejected_total = sum(bucket["rejected_count"] for bucket in slices.values())
    for doc_type, bucket in sorted(slices.items()):
        finalized[doc_type] = {
            "candidate_count": bucket["candidate_count"],
            "final_count": bucket["final_count"],
            "rejected_count": bucket["rejected_count"],
            "candidate_gate_reject_count": bucket["candidate_gate_reject_count"],
            "safety_rejected_count": bucket["safety_rejected_count"],
            "final_exposure_share": ratio(bucket["final_count"], final_total),
            "rejection_share": ratio(bucket["rejected_count"], rejected_total),
            "average_candidate_rank": average_or_zero(bucket["candidate_rank_total"], bucket["candidate_count"]),
            "average_candidate_score": average_or_zero(bucket["candidate_score_total"], bucket["candidate_count"]),
            "average_final_rank": average_or_zero(bucket["rank_total"], bucket["final_count"]),
            "average_final_score": average_or_zero(bucket["final_score_total"], bucket["final_count"]),
        }
    return finalized


def ensure_slice(slices: dict[str, dict[str, Any]], doc_type: str) -> dict[str, Any]:
    if doc_type not in slices:
        slices[doc_type] = {
            "candidate_count": 0,
            "candidate_rank_total": 0.0,
            "candidate_score_total": 0.0,
            "candidate_gate_reject_count": 0,
            "final_count": 0,
            "rank_total": 0.0,
            "final_score_total": 0.0,
            "rejected_count": 0,
            "safety_rejected_count": 0,
        }
    return slices[doc_type]


def coverage_summary(documents: list[RetrievalDocument], runs: list[dict[str, Any]]) -> dict[str, Any]:
    corpus_types = set(corpus_document_type_counts(documents))
    final_types = {
        doc_type
        for run in runs
        for doc_type, count in run["final_document_type_counts"].items()
        if count > 0
    }
    rejected_types = {
        doc_type
        for run in runs
        for doc_type, count in run["rejected_document_type_counts"].items()
        if count > 0
    }
    return {
        "corpus_document_type_coverage_count": len(corpus_types),
        "final_document_type_coverage_count": len(final_types),
        "rejected_document_type_coverage_count": len(rejected_types),
        "final_document_type_coverage": sorted(final_types),
        "rejected_document_type_coverage": sorted(rejected_types),
        "corpus_types_missing_from_final_results": sorted(corpus_types - final_types),
        "corpus_types_missing_from_rejections": sorted(corpus_types - rejected_types),
    }


def resolve_policy_weights(configured_policies: dict[str, Any], policies: list[str]) -> dict[str, dict[str, float]]:
    unknown = [policy for policy in policies if policy not in configured_policies]
    if unknown:
        raise ValueError(f"Unknown policies: {', '.join(unknown)}")
    return {
        policy: {key: float(value) for key, value in configured_policies[policy].items()}
        for policy in policies
    }


def parse_queries(value: str | None) -> list[dict[str, str]] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("--queries JSON file must contain a list")
        return [
            {
                "id": str(row.get("id") or f"query-{index}"),
                "query": str(row["query"]),
                "intent": str(row.get("intent") or "custom"),
            }
            for index, row in enumerate(rows, start=1)
        ]
    return [
        {"id": f"custom-{index}", "query": query.strip(), "intent": "custom"}
        for index, query in enumerate(value.split("||"), start=1)
        if query.strip()
    ]


def ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def average_or_zero(total: int | float, count: int) -> float:
    return round(float(total) / float(count), 6) if count else 0.0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run document-type slice diagnostics over retrieval decisions.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--policies", default="lexical,balanced,rag")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--queries", default=None, help="Path to JSON query list, or queries separated with ||.")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    report = evaluate_document_type_slices(
        mode=args.mode,
        policies=policies,
        top_k=args.top_k,
        queries=parse_queries(args.queries),
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

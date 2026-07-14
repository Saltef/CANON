from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.document_type_slices import (
    average_or_zero,
    inspect_candidates,
    parse_queries,
    ratio,
    rejected_row,
    resolve_policy_weights,
    trace_row,
    write_json,
)
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.engine import retrieve_with_stages


DEFAULT_DOMAIN_QUERIES = [
    {
        "id": "economics-identification",
        "query": "economics identification data robustness causal effect market policy",
        "intent": "economics_method_fit",
    },
    {
        "id": "psychology-replication",
        "query": "psychology measurement sample preregistered replication effect",
        "intent": "psychology_replication_fit",
    },
    {
        "id": "anthropology-fieldwork",
        "query": "anthropology fieldwork interview positionality community evidence",
        "intent": "anthropology_context_fit",
    },
    {
        "id": "history-archive",
        "query": "history archival primary source provenance historiography",
        "intent": "history_primary_source_fit",
    },
    {
        "id": "legal-market-authority",
        "query": "legal market authority filing current risk disclosure jurisdiction",
        "intent": "legal_market_authority_fit",
    },
    {
        "id": "public-opinion-themes",
        "query": "public opinion community concern support opposition safety trust social media comments",
        "intent": "public_opinion_theme_fit",
    },
]


def evaluate_domain_slices(
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
    resolved_queries = queries or DEFAULT_DOMAIN_QUERIES
    runs = [
        run_domain_slice(
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
        "report_id": "domain_slices_v1",
        "mode": mode,
        "document_count": len(documents),
        "query_count": len(resolved_queries),
        "policy_ids": resolved_policies,
        "top_k": resolved_top_k,
        "corpus_domain_counts": corpus_domain_counts(documents),
        "runs": runs,
        "slice_summary": slice_summary(runs),
        "coverage_summary": coverage_summary(documents, runs),
        "limitations": [
            "Domain slices are deterministic diagnostics, not evidence that a domain profile is scientifically validated.",
            "Domain metadata can be wrong or too coarse; mixed-domain corpora require hand-audited labels.",
            "The default queries are stress probes and should be replaced with domain-specific qrels or expert tasks.",
            "Domain fit currently annotates and audits retrieval behavior; it is not yet a learned ranking policy.",
        ],
    }
    report["contract_validation"] = validate_report(report, "domain_slices")
    if write_report:
        write_json(settings.reports_dir / f"domain_slices_{mode}.json", report)
    return report


def run_domain_slice(
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
    return {
        "id": query_spec["id"],
        "query": query_spec["query"],
        "intent": query_spec.get("intent", "unspecified"),
        "policy": policy,
        "candidate_domain_counts": dict(sorted(Counter(row["domain_profile"] for row in candidates).items())),
        "final_domain_counts": dict(sorted(Counter(row["domain_profile"] for row in final_results).items())),
        "rejected_domain_counts": dict(sorted(Counter(row["domain_profile"] for row in rejected_candidates).items())),
        "candidate_domain_fit_tier_counts": dict(sorted(Counter(row["domain_fit_tier"] for row in candidates).items())),
        "candidate_safety_decision_counts": dict(sorted(Counter(row["safety_decision"] for row in candidates).items())),
        "candidate_slice_stats": candidate_slice_stats(candidates),
        "stage_summary": staged["stage_summary"],
        "candidate_sample": candidates[: min(len(candidates), 12)],
        "final_results": final_results,
        "rejected_candidates": rejected_candidates,
    }


def candidate_slice_stats(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for row in candidates:
        domain = str(row["domain_profile"])
        bucket = stats.setdefault(
            domain,
            {
                "candidate_count": 0,
                "candidate_rank_total": 0.0,
                "domain_fit_score_total": 0.0,
                "candidate_gate_reject_count": 0,
                "high_adversarial_surface_count": 0,
            },
        )
        bucket["candidate_count"] = int(bucket["candidate_count"]) + 1
        bucket["candidate_rank_total"] = float(bucket["candidate_rank_total"]) + float(row["candidate_rank"])
        bucket["domain_fit_score_total"] = float(bucket["domain_fit_score_total"]) + float(row["domain_fit_score"])
        if row["gate_decision"] == "reject":
            bucket["candidate_gate_reject_count"] = int(bucket["candidate_gate_reject_count"]) + 1
        if "high_adversarial_surface" in row.get("domain_gate_flags", []):
            bucket["high_adversarial_surface_count"] = int(bucket["high_adversarial_surface_count"]) + 1
    return dict(sorted(stats.items()))


def slice_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    slices: dict[str, dict[str, Any]] = {}
    for run in runs:
        for domain, stats in run["candidate_slice_stats"].items():
            bucket = ensure_slice(slices, domain)
            bucket["candidate_count"] += int(stats["candidate_count"])
            bucket["candidate_rank_total"] += float(stats["candidate_rank_total"])
            bucket["domain_fit_score_total"] += float(stats["domain_fit_score_total"])
            bucket["candidate_gate_reject_count"] += int(stats["candidate_gate_reject_count"])
            bucket["high_adversarial_surface_count"] += int(stats["high_adversarial_surface_count"])
        for row in run["final_results"]:
            bucket = ensure_slice(slices, row["domain_profile"])
            bucket["final_count"] += 1
            bucket["rank_total"] += float(row["rank"])
            bucket["final_score_total"] += float(row["final_score"])
            bucket["final_domain_fit_score_total"] += float(row["domain_fit_score"])
        for row in run["rejected_candidates"]:
            bucket = ensure_slice(slices, row["domain_profile"])
            bucket["rejected_count"] += 1
            if row["safety_decision"] in {"block", "quarantine"}:
                bucket["safety_rejected_count"] += 1
    final_total = sum(bucket["final_count"] for bucket in slices.values())
    rejected_total = sum(bucket["rejected_count"] for bucket in slices.values())
    return {
        domain: {
            "candidate_count": bucket["candidate_count"],
            "final_count": bucket["final_count"],
            "rejected_count": bucket["rejected_count"],
            "candidate_gate_reject_count": bucket["candidate_gate_reject_count"],
            "safety_rejected_count": bucket["safety_rejected_count"],
            "high_adversarial_surface_count": bucket["high_adversarial_surface_count"],
            "final_exposure_share": ratio(bucket["final_count"], final_total),
            "rejection_share": ratio(bucket["rejected_count"], rejected_total),
            "average_candidate_rank": average_or_zero(bucket["candidate_rank_total"], bucket["candidate_count"]),
            "average_domain_fit_score": average_or_zero(bucket["domain_fit_score_total"], bucket["candidate_count"]),
            "average_final_rank": average_or_zero(bucket["rank_total"], bucket["final_count"]),
            "average_final_score": average_or_zero(bucket["final_score_total"], bucket["final_count"]),
            "average_final_domain_fit_score": average_or_zero(bucket["final_domain_fit_score_total"], bucket["final_count"]),
        }
        for domain, bucket in sorted(slices.items())
    }


def ensure_slice(slices: dict[str, dict[str, Any]], domain: str) -> dict[str, Any]:
    if domain not in slices:
        slices[domain] = {
            "candidate_count": 0,
            "candidate_rank_total": 0.0,
            "domain_fit_score_total": 0.0,
            "candidate_gate_reject_count": 0,
            "high_adversarial_surface_count": 0,
            "final_count": 0,
            "rank_total": 0.0,
            "final_score_total": 0.0,
            "final_domain_fit_score_total": 0.0,
            "rejected_count": 0,
            "safety_rejected_count": 0,
        }
    return slices[domain]


def coverage_summary(documents: list[RetrievalDocument], runs: list[dict[str, Any]]) -> dict[str, Any]:
    corpus_domains = set(corpus_domain_counts(documents))
    final_domains = {
        domain
        for run in runs
        for domain, count in run["final_domain_counts"].items()
        if count > 0
    }
    rejected_domains = {
        domain
        for run in runs
        for domain, count in run["rejected_domain_counts"].items()
        if count > 0
    }
    return {
        "corpus_domain_coverage_count": len(corpus_domains),
        "final_domain_coverage_count": len(final_domains),
        "rejected_domain_coverage_count": len(rejected_domains),
        "final_domain_coverage": sorted(final_domains),
        "rejected_domain_coverage": sorted(rejected_domains),
        "corpus_domains_missing_from_final_results": sorted(corpus_domains - final_domains),
        "corpus_domains_missing_from_rejections": sorted(corpus_domains - rejected_domains),
    }


def corpus_domain_counts(documents: list[RetrievalDocument]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(document.work_signals.get("domain_profile", document.work_signals.get("domain", "general_academic")))
                for document in documents
            ).items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run domain-slice diagnostics over retrieval decisions.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--policies", default="lexical,balanced,rag")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--queries", default=None, help="Path to JSON query list, or queries separated with ||.")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    report = evaluate_domain_slices(
        mode=args.mode,
        policies=policies,
        top_k=args.top_k,
        queries=parse_queries(args.queries),
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product.automated_benchmark_suite import resolve_path


DEFAULT_SUITE_REPORT = Path("reports") / "automated_benchmark_suite_stage1_public_full_v1.json"
DEFAULT_OPTIMIZER_REPORT = Path("reports") / "stage1_optimizer_v1.json"
DEFAULT_HUMAN_REVIEW_STATUS = Path("reports") / "publishable_human_review_status_v1.json"
DEFAULT_OUTPUT_PATH = Path("reports") / "publishable_benchmark_card_v1.json"


def build_publishable_benchmark_card(
    automated_suite_report: Path | None = None,
    optimizer_report: Path | None = None,
    human_review_status_report: Path | None = None,
    disagreement_report: Path | None = None,
    output_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    suite_path = resolve_path(settings.root, automated_suite_report or DEFAULT_SUITE_REPORT)
    optimizer_path = resolve_path(settings.root, optimizer_report or DEFAULT_OPTIMIZER_REPORT)
    review_path = resolve_path(settings.root, human_review_status_report or DEFAULT_HUMAN_REVIEW_STATUS)
    disagreement_path = (
        resolve_path(settings.root, disagreement_report)
        if disagreement_report
        else latest_disagreement_report(settings.reports_dir)
    )
    suite = load_json(suite_path)
    optimizer = load_json(optimizer_path)
    human_review = load_json(review_path) if review_path.exists() else {}
    disagreement = load_json(disagreement_path) if disagreement_path and disagreement_path.exists() else {}
    rows = [benchmark_row(benchmark, suite.get("thresholds") or {}) for benchmark in suite.get("benchmarks") or []]
    report = {
        "report_id": "publishable_benchmark_card_v1",
        "suite_id": suite.get("suite_id"),
        "status": benchmark_card_status(suite, rows, human_review),
        "source_reports": source_reports(settings.root, suite_path, optimizer_path, review_path, disagreement_path),
        "scope": {
            "benchmark_count": suite.get("benchmark_count", len(rows)),
            "query_count": sum(int(row.get("query_count") or 0) for row in rows),
            "benchmark_topics": sorted({row.get("topic") for row in rows if row.get("topic")}),
            "document_format": first_candidate_generation(rows, "document_format"),
            "fusion": first_candidate_generation(rows, "fusion"),
            "candidate_k": first_candidate_generation(rows, "candidate_k"),
        },
        "model_matrix": suite.get("model_matrix") or {},
        "thresholds": suite.get("thresholds") or {},
        "benchmarks": rows,
        "aggregate_diagnostics": aggregate_diagnostics(rows),
        "optimizer": optimizer_summary(optimizer),
        "disagreement_preservation": disagreement_summary(disagreement),
        "human_review": human_review_summary(human_review),
        "claim_boundary": claim_boundary(suite, human_review),
        "recommended_next_actions": recommended_next_actions(rows, human_review, disagreement),
    }
    if write_report:
        output = resolve_path(settings.root, output_path or DEFAULT_OUTPUT_PATH)
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def benchmark_row(benchmark: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    semantic_best = (benchmark.get("semantic") or {}).get("best") or {}
    rerank = benchmark.get("rerank") or {}
    rerank_best = rerank.get("best") or {}
    candidate_recall = rerank.get("candidate_recall") or {}
    score_observability = rerank.get("score_observability") or {}
    min_candidate = float(thresholds.get("min_candidate_recall") or 0.0)
    min_rerank_recall = float(thresholds.get("min_rerank_recall_at_k") or 0.0)
    recall = float(rerank_best.get("Recall@k") or 0.0)
    candidate_mean = float(candidate_recall.get("mean") or 0.0)
    query_count = int(benchmark.get("query_count") or candidate_recall.get("query_count") or 0)
    miss_count = int(candidate_recall.get("miss_count") or 0)
    return {
        "id": benchmark.get("id"),
        "topic": benchmark.get("topic"),
        "mode": benchmark.get("mode"),
        "qrels_path": benchmark.get("qrels_path"),
        "query_count": query_count,
        "candidate_generation": {
            "candidate_k": benchmark.get("candidate_k"),
            "lexical_k": benchmark.get("lexical_k"),
            "vector_k": benchmark.get("vector_k"),
            "fusion": benchmark.get("fusion"),
            "document_format": benchmark.get("document_format"),
            "pooled_candidates": benchmark.get("pooled_candidates"),
            "query_variant_count": (rerank.get("candidate_generation") or {}).get("query_variant_count"),
        },
        "semantic_best": metric_row(semantic_best),
        "rerank_best": metric_row(rerank_best),
        "candidate_recall": {
            "mean": round(candidate_mean, 6),
            "min": float(candidate_recall.get("min") or 0.0),
            "miss_count": miss_count,
            "miss_rate": round(safe_ratio(miss_count, query_count), 6),
            "meets_threshold": candidate_mean >= min_candidate,
            "threshold": min_candidate,
        },
        "ranking": {
            "rerank_recall_at_k": recall,
            "meets_threshold": recall >= min_rerank_recall,
            "threshold": min_rerank_recall,
        },
        "score_observability": {
            "mean_relevant_nonrelevant_score_gap": float(
                score_observability.get("mean_relevant_nonrelevant_score_gap") or 0.0
            ),
            "mean_low_confidence_top_k_count": float(
                score_observability.get("mean_low_confidence_top_k_count") or 0.0
            ),
        },
        "diagnosis": benchmark_diagnosis(candidate_mean, recall, min_candidate, min_rerank_recall, miss_count),
    }


def metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_id": row.get("provider_id"),
        "primary_metric": row.get("primary_metric"),
        "primary_score": float(row.get("primary_score") or 0.0),
        "Recall@k": float(row.get("Recall@k") or 0.0),
        "MRR@k": float(row.get("MRR@k") or 0.0),
        "elapsed_ms": float(row.get("elapsed_ms") or 0.0),
    }


def benchmark_diagnosis(
    candidate_mean: float,
    rerank_recall: float,
    min_candidate: float,
    min_rerank_recall: float,
    miss_count: int,
) -> list[str]:
    reasons = []
    if candidate_mean < min_candidate:
        reasons.append("candidate_pool_recall_below_threshold")
    if rerank_recall < min_rerank_recall:
        reasons.append("top_k_rerank_recall_below_threshold")
    if miss_count:
        reasons.append("some_queries_have_no_relevant_candidate_in_pool")
    if not reasons:
        reasons.append("automated_public_qrels_gate_passed_pending_human_review")
    return reasons


def aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "missing_benchmark_rows",
            "benchmark_count": 0,
        }
    candidate_failures = [row for row in rows if not row["candidate_recall"]["meets_threshold"]]
    ranking_failures = [row for row in rows if not row["ranking"]["meets_threshold"]]
    total_queries = sum(int(row["query_count"] or 0) for row in rows)
    total_misses = sum(int(row["candidate_recall"]["miss_count"] or 0) for row in rows)
    return {
        "status": "automated_fail" if candidate_failures or ranking_failures else "automated_pass_human_review_required",
        "benchmark_count": len(rows),
        "query_count": total_queries,
        "candidate_recall_failure_count": len(candidate_failures),
        "ranking_failure_count": len(ranking_failures),
        "candidate_miss_count": total_misses,
        "candidate_miss_rate": round(safe_ratio(total_misses, total_queries), 6),
        "worst_candidate_recall": min(float(row["candidate_recall"]["mean"]) for row in rows),
        "best_rerank_primary_score": max(float(row["rerank_best"]["primary_score"]) for row in rows),
        "interpretation": aggregate_interpretation(candidate_failures, ranking_failures),
    }


def aggregate_interpretation(candidate_failures: list[dict[str, Any]], ranking_failures: list[dict[str, Any]]) -> str:
    if candidate_failures:
        names = ", ".join(str(row["id"]) for row in candidate_failures)
        return f"Candidate generation is the limiting failure mode for: {names}."
    if ranking_failures:
        names = ", ".join(str(row["id"]) for row in ranking_failures)
        return f"Candidate pools meet threshold, but top-k ranking remains below threshold for: {names}."
    return "Automated public-qrels gates pass; human review is still required before publishable claims."


def optimizer_summary(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_trial") or {}
    return {
        "report_id": report.get("report_id"),
        "suite_id": report.get("suite_id"),
        "status": report.get("status"),
        "fusion_methods": report.get("fusion_methods") or [],
        "trial_count": report.get("trial_count", 0),
        "cached_trial_count": (report.get("cache") or {}).get("cached_trial_count", 0),
        "best_trial": {
            "benchmark_id": best.get("benchmark_id"),
            "dense_retriever": best.get("dense_retriever"),
            "reranker": best.get("reranker"),
            "candidate_k": best.get("candidate_k"),
            "fusion": best.get("fusion"),
            "composite_score": best.get("composite_score"),
            "candidate_recall_mean": best.get("candidate_recall_mean"),
            "nDCG@k": best.get("nDCG@k"),
            "Recall@k": best.get("Recall@k"),
        },
        "fusion_summary": fusion_summary(report.get("trials") or []),
    }


def fusion_summary(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fusion in sorted({trial.get("fusion") for trial in trials if trial.get("fusion")}):
        fusion_trials = [trial for trial in trials if trial.get("fusion") == fusion]
        ok_trials = [trial for trial in fusion_trials if trial.get("status") == "ok"]
        best = max(ok_trials, key=lambda trial: float(trial.get("composite_score") or 0.0), default={})
        rows.append(
            {
                "fusion": fusion,
                "trial_count": len(fusion_trials),
                "ok_trial_count": len(ok_trials),
                "best_composite_score": best.get("composite_score"),
                "best_candidate_recall_mean": best.get("candidate_recall_mean"),
                "best_nDCG@k": best.get("nDCG@k"),
                "best_Recall@k": best.get("Recall@k"),
            }
        )
    return rows


def disagreement_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = ((report.get("summary") or {}).get("average_metrics") or {})
    return {
        "report_id": report.get("report_id"),
        "benchmark_id": report.get("benchmark_id"),
        "status": report.get("status"),
        "case_count": report.get("case_count", 0),
        "disagreement_preservation_score": metrics.get("disagreement_preservation_score"),
        "contradiction_recall": metrics.get("contradiction_recall"),
        "citation_integrity": metrics.get("citation_integrity"),
        "unsupported_claim_rate": metrics.get("unsupported_claim_rate"),
        "human_review_required": report.get("human_review_required", True),
    }


def human_review_summary(report: dict[str, Any]) -> dict[str, Any]:
    retrieval = report.get("retrieval_review") or {}
    synthesis = report.get("synthesis_review") or {}
    return {
        "report_id": report.get("report_id"),
        "suite_id": report.get("suite_id"),
        "status": report.get("status", "missing"),
        "retrieval_complete_query_count": retrieval.get("complete_query_count", 0),
        "retrieval_target": retrieval.get("target", 0),
        "synthesis_complete_case_count": synthesis.get("complete_case_count", 0),
        "synthesis_target": synthesis.get("target", 0),
        "reviewed_label_count": report.get("reviewed_label_count", 0),
        "validation_error_count": report.get("validation_error_count", 0),
    }


def benchmark_card_status(suite: dict[str, Any], rows: list[dict[str, Any]], human_review: dict[str, Any]) -> str:
    diagnostics = aggregate_diagnostics(rows)
    if diagnostics.get("status") == "automated_fail":
        return "benchmark_card_ready_with_automated_failures"
    if human_review.get("status") != "complete":
        return "benchmark_card_ready_human_review_blocked"
    if suite.get("status") == "automated_pass_human_review_required":
        return "benchmark_card_ready_pending_claim_review"
    return "benchmark_card_ready"


def claim_boundary(suite: dict[str, Any], human_review: dict[str, Any]) -> dict[str, Any]:
    return {
        "defensible_now": [
            "CANON can be described as an auditable, staged evaluation workflow.",
            "The current card reports full public-qrels diagnostics for the configured suite.",
            "Automated failures are diagnostic evidence about candidate generation and ranking, not final product failure.",
        ],
        "blocked_claims": [
            "CANON outperforms existing RAG systems.",
            "The local/heuristic stack is a recommended production retrieval configuration.",
            "Observed unsupported-claim rates are publication-quality estimates.",
            "Human reviewers find the synthesis useful or correct.",
        ],
        "human_review_required": human_review.get("status") != "complete",
        "suite_status": suite.get("status"),
    }


def recommended_next_actions(
    rows: list[dict[str, Any]],
    human_review: dict[str, Any],
    disagreement: dict[str, Any],
) -> list[str]:
    actions = []
    if any(not row["candidate_recall"]["meets_threshold"] for row in rows):
        actions.append("Improve candidate generation before spending effort on reranker comparisons.")
    if any(not row["ranking"]["meets_threshold"] for row in rows):
        actions.append("Inspect reranker score overlap and query-level misses before claiming ranking quality.")
    if human_review.get("status") != "complete":
        actions.append("Complete publishable retrieval and synthesis labels, then rerun publishable_review and publishable_package.")
    if (disagreement.get("summary") or {}).get("case_count", 0) < 30:
        actions.append("Expand disagreement-preservation cases toward the synthesis review target.")
    return actions or ["Run human claim review and prepare the public write-up."]


def first_candidate_generation(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = (row.get("candidate_generation") or {}).get(key)
        if value is not None:
            return value
    return None


def latest_disagreement_report(reports_dir: Path) -> Path | None:
    reports = sorted(reports_dir.glob("disagreement_preservation_*.json"))
    return reports[-1] if reports else None


def source_reports(
    root: Path,
    suite_path: Path,
    optimizer_path: Path,
    review_path: Path,
    disagreement_path: Path | None,
) -> dict[str, str | None]:
    return {
        "automated_suite": relative(root, suite_path),
        "stage1_optimizer": relative(root, optimizer_path),
        "human_review_status": relative(root, review_path),
        "disagreement_preservation": relative(root, disagreement_path) if disagreement_path else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "- `{id}` candidate_recall={candidate_recall} miss_rate={miss_rate} "
        "rerank_nDCG={ndcg} rerank_recall={recall} diagnosis={diagnosis}".format(
            id=row["id"],
            candidate_recall=row["candidate_recall"]["mean"],
            miss_rate=row["candidate_recall"]["miss_rate"],
            ndcg=row["rerank_best"]["primary_score"],
            recall=row["ranking"]["rerank_recall_at_k"],
            diagnosis=", ".join(row["diagnosis"]),
        )
        for row in report["benchmarks"]
    )
    fusion_rows = "\n".join(
        "- `{fusion}` trials={trial_count} ok={ok_trial_count} "
        "best_composite={best_composite_score} best_candidate_recall={best_candidate_recall_mean}".format(
            **row
        )
        for row in report["optimizer"]["fusion_summary"]
    )
    actions = "\n".join(f"- {action}" for action in report["recommended_next_actions"])
    aggregate = report["aggregate_diagnostics"]
    review = report["human_review"]
    return f"""# CANON Publishable Benchmark Card

Status: `{report['status']}`

Suite: `{report['suite_id']}`

Queries: `{report['scope']['query_count']}`

Automated diagnostic status: `{aggregate['status']}`

Candidate miss rate: `{aggregate.get('candidate_miss_rate')}`

Human review: `{review['status']}` (`{review['retrieval_complete_query_count']}` / `{review['retrieval_target']}` retrieval queries, `{review['synthesis_complete_case_count']}` / `{review['synthesis_target']}` synthesis cases)

## Benchmark Diagnostics

{rows}

## Optimizer Fusion Summary

{fusion_rows}

## Interpretation

{aggregate.get('interpretation')}

## Next Actions

{actions}

## Claim Boundary

This card supports diagnostic claims about the configured public-qrels run. It
does not support model-superiority, production-quality, or final factual
correctness claims without completed human review.
"""


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "suite_id": report.get("suite_id"),
        "status": report["status"],
        "query_count": report["scope"]["query_count"],
        "aggregate_diagnostics": report["aggregate_diagnostics"],
        "human_review": report["human_review"],
        "artifact": "reports/publishable_benchmark_card_v1.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a concise CANON publishable benchmark card.")
    parser.add_argument("--automated-suite-report", default=str(DEFAULT_SUITE_REPORT))
    parser.add_argument("--optimizer-report", default=str(DEFAULT_OPTIMIZER_REPORT))
    parser.add_argument("--human-review-status-report", default=str(DEFAULT_HUMAN_REVIEW_STATUS))
    parser.add_argument("--disagreement-report", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    report = build_publishable_benchmark_card(
        automated_suite_report=Path(args.automated_suite_report),
        optimizer_report=Path(args.optimizer_report),
        human_review_status_report=Path(args.human_review_status_report),
        disagreement_report=Path(args.disagreement_report) if args.disagreement_report else None,
        output_path=Path(args.output),
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

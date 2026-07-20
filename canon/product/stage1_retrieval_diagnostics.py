from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.external_ir import load_qrels
from canon.product import report_io


def run_stage1_retrieval_diagnostics(
    rerank_report: Path,
    qrels_path: Path,
    mode: str,
    output_stem: str | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    report_path = resolve_path(settings.root, rerank_report)
    qrels_file = resolve_path(settings.root, qrels_path)
    rerank = json.loads(report_path.read_text(encoding="utf-8"))
    qrels = load_qrels(qrels_file)
    chunk_to_work = load_chunk_to_work(settings.data_dir / "processed" / f"chunks_{mode}.json")
    provider = first_ok_provider(rerank.get("rerankers", []))
    query_rows = diagnose_queries(provider.get("queries", []), qrels.get("queries", []), chunk_to_work)
    report = {
        "report_id": "stage1_retrieval_diagnostics_v1",
        "mode": mode,
        "benchmark_id": qrels.get("benchmark_id"),
        "qrels_semantics": qrels_semantics(qrels, query_rows),
        "rerank_report": relative(settings.root, report_path),
        "qrels_path": relative(settings.root, qrels_file),
        "candidate_generation": rerank.get("candidate_generation", {}),
        "provider": provider.get("provider_id") or provider.get("provider"),
        "aggregate": aggregate(query_rows),
        "oracle_reranking": oracle_reranking(provider.get("queries", [])),
        "parent_dominance": parent_dominance(provider.get("queries", []), chunk_to_work),
        "failure_modes": failure_modes(query_rows),
        "queries": query_rows,
        "worst_queries": sorted(
            query_rows,
            key=lambda row: (
                row["candidate_recall"],
                row["relevant_work_hit_rate"],
                row["top10_relevant_hit_count"],
                -row["relevant_chunk_count"],
            ),
        )[:10],
        "recommendations": recommendations(query_rows),
    }
    if write_report:
        stem = output_stem or f"stage1_retrieval_diagnostics_{mode}"
        report_io.write_json(settings.reports_dir / f"{stem}.json", report)
        report_io.write_markdown(settings.reports_dir / f"{stem}.md", render_markdown(report))
    return report


def diagnose_queries(
    result_queries: list[dict[str, Any]],
    qrel_queries: list[dict[str, Any]],
    chunk_to_work: dict[str, str],
) -> list[dict[str, Any]]:
    qrels_by_id = {str(query["id"]): query for query in qrel_queries}
    rows = []
    for query in result_queries:
        query_id = str(query.get("id"))
        qrel = qrels_by_id.get(query_id, {})
        relevant_ids = {str(chunk_id) for chunk_id in (qrel.get("relevant") or {})}
        relevant_works = {chunk_to_work.get(chunk_id, chunk_id) for chunk_id in relevant_ids}
        hit_ids = {str(chunk_id) for chunk_id in (query.get("candidate_recall") or {}).get("hit_ids", [])}
        hit_works = {chunk_to_work.get(chunk_id, chunk_id) for chunk_id in hit_ids}
        top_ids = [str(chunk_id) for chunk_id in query.get("ranked_chunk_ids", [])[:10]]
        top_hit_ids = [chunk_id for chunk_id in top_ids if chunk_id in relevant_ids]
        top_hit_works = {chunk_to_work.get(chunk_id, chunk_id) for chunk_id in top_hit_ids}
        relevant_work_hit_rate = safe_ratio(len(relevant_works & hit_works), len(relevant_works))
        row = {
            "id": query_id,
            "query": query.get("query"),
            "candidate_count": int(query.get("candidate_count", 0)),
            "relevant_chunk_count": len(relevant_ids),
            "relevant_work_count": len(relevant_works),
            "candidate_hit_count": len(hit_ids),
            "candidate_recall": round(safe_ratio(len(hit_ids), len(relevant_ids)), 6),
            "relevant_work_hit_count": len(relevant_works & hit_works),
            "relevant_work_hit_rate": round(relevant_work_hit_rate, 6),
            "any_relevant_work_in_pool": bool(relevant_works & hit_works),
            "top10_relevant_hit_count": len(top_hit_ids),
            "top10_relevant_work_hit_count": len(top_hit_works),
        "metrics": query.get("metrics", {}),
            "parent_dominance": query_parent_dominance(top_ids, chunk_to_work),
            "failure_mode": classify_failure(
                candidate_recall=safe_ratio(len(hit_ids), len(relevant_ids)),
                relevant_work_hit_rate=relevant_work_hit_rate,
                top10_hit_count=len(top_hit_ids),
            ),
        }
        rows.append(row)
    return rows


def classify_failure(candidate_recall: float, relevant_work_hit_rate: float, top10_hit_count: int) -> str:
    if candidate_recall <= 0:
        return "no_relevant_chunks_in_pool"
    if relevant_work_hit_rate <= 0:
        return "no_relevant_documents_in_pool"
    if top10_hit_count <= 0:
        return "candidate_pool_has_evidence_but_top10_misses"
    if candidate_recall < 0.5:
        return "partial_pool_recall"
    return "usable_pool_needs_ranking"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "mean_candidate_recall": average([row["candidate_recall"] for row in rows]),
        "median_candidate_recall": median([row["candidate_recall"] for row in rows]),
        "mean_relevant_work_hit_rate": average([row["relevant_work_hit_rate"] for row in rows]),
        "queries_with_any_relevant_work_in_pool": sum(1 for row in rows if row["any_relevant_work_in_pool"]),
        "queries_with_top10_relevant_hits": sum(1 for row in rows if row["top10_relevant_hit_count"] > 0),
        "zero_candidate_recall_queries": sum(1 for row in rows if row["candidate_recall"] <= 0),
        "mean_relevant_chunk_count": average([row["relevant_chunk_count"] for row in rows]),
        "max_relevant_chunk_count": max([row["relevant_chunk_count"] for row in rows], default=0),
    }


def qrels_semantics(qrels: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark_kind": qrels.get("benchmark_kind"),
        "qrels_granularity": "document_level_transferred_to_chunks"
        if qrels.get("benchmark_kind") == "public_beir_qrels_expanded_to_chunks"
        else "unknown_or_native",
        "relevance_transfer": "retrieved chunk is relevant whenever its parent BEIR document is relevant",
        "recall_denominator": "chunks",
        "work_level_recall_also_reported": True,
        "mean_relevant_chunks_per_query": average([row["relevant_chunk_count"] for row in rows]),
        "mean_relevant_parent_documents_per_query": average([row["relevant_work_count"] for row in rows]),
    }


def oracle_reranking(result_queries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        scored
        for query in result_queries
        for scored in query.get("scored_candidates", [])
    ]
    if not rows:
        return {
            "status": "unavailable",
            "reason": "Rerank report does not include scored_candidates; rerun after this diagnostic update.",
        }
    relevant_scores = [float(row["rerank_score"]) for row in rows if float(row.get("relevance", 0.0)) > 0]
    nonrelevant_scores = [float(row["rerank_score"]) for row in rows if float(row.get("relevance", 0.0)) <= 0]
    return {
        "status": "ok",
        "relevant_count": len(relevant_scores),
        "nonrelevant_count": len(nonrelevant_scores),
        "relevant_score_distribution": distribution(relevant_scores),
        "nonrelevant_score_distribution": distribution(nonrelevant_scores),
        "separation": score_separation(relevant_scores, nonrelevant_scores),
        "interpretation": score_overlap_interpretation(relevant_scores, nonrelevant_scores),
    }


def score_separation(relevant_scores: list[float], nonrelevant_scores: list[float]) -> dict[str, Any]:
    if not relevant_scores or not nonrelevant_scores:
        return {"auc_like_pairwise_accuracy": 0.0, "overlap_rate": 1.0}
    sample_limit = 200000
    wins = ties = pairs = 0
    for relevant in relevant_scores:
        for nonrelevant in nonrelevant_scores:
            if pairs >= sample_limit:
                break
            pairs += 1
            if relevant > nonrelevant:
                wins += 1
            elif relevant == nonrelevant:
                ties += 1
        if pairs >= sample_limit:
            break
    auc_like = (wins + 0.5 * ties) / pairs if pairs else 0.0
    rel_q25 = percentile(relevant_scores, 0.25)
    nonrel_q75 = percentile(nonrelevant_scores, 0.75)
    return {
        "auc_like_pairwise_accuracy": round(auc_like, 6),
        "overlap_rate": round(1.0 - abs(auc_like - 0.5) * 2.0, 6),
        "relevant_q25_minus_nonrelevant_q75": round(rel_q25 - nonrel_q75, 6),
    }


def score_overlap_interpretation(relevant_scores: list[float], nonrelevant_scores: list[float]) -> str:
    separation = score_separation(relevant_scores, nonrelevant_scores)
    auc_like = float(separation["auc_like_pairwise_accuracy"])
    if auc_like >= 0.8:
        return "strong_discrimination"
    if auc_like >= 0.65:
        return "moderate_discrimination"
    return "heavy_overlap_or_weak_discrimination"


def parent_dominance(result_queries: list[dict[str, Any]], chunk_to_work: dict[str, str]) -> dict[str, Any]:
    rows = [
        query_parent_dominance(
            [str(chunk_id) for chunk_id in query.get("ranked_chunk_ids", [])[:10]],
            chunk_to_work,
        )
        for query in result_queries
    ]
    return {
        "mean_unique_parent_count_at_10": average([row["unique_parent_count_at_10"] for row in rows]),
        "mean_duplicate_chunk_count_at_10": average([row["duplicate_chunk_count_at_10"] for row in rows]),
        "queries_with_parent_duplication_at_10": sum(
            1 for row in rows if row["duplicate_chunk_count_at_10"] > 0
        ),
    }


def query_parent_dominance(top_ids: list[str], chunk_to_work: dict[str, str]) -> dict[str, Any]:
    parents = [chunk_to_work.get(chunk_id, chunk_id) for chunk_id in top_ids]
    unique_count = len(set(parents))
    return {
        "top_k": len(top_ids),
        "unique_parent_count_at_10": unique_count,
        "duplicate_chunk_count_at_10": max(0, len(top_ids) - unique_count),
        "max_chunks_from_one_parent_at_10": max((parents.count(parent) for parent in set(parents)), default=0),
    }


def failure_modes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row["failure_mode"])
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def recommendations(rows: list[dict[str, Any]]) -> list[str]:
    modes = failure_modes(rows)
    recs = []
    if modes.get("no_relevant_chunks_in_pool", 0) or modes.get("no_relevant_documents_in_pool", 0):
        recs.append("Inspect zero-hit queries and add query expansion/domain aliases before spending on reranking.")
    if modes.get("candidate_pool_has_evidence_but_top10_misses", 0):
        recs.append("Run a stronger reranker only for configurations where the pool already contains relevant evidence.")
    if average([row["relevant_chunk_count"] for row in rows]) > 25:
        recs.append("Report work-level hit rate alongside chunk-level candidate recall for broad document-level qrels.")
    if not recs:
        recs.append("Candidate pools look usable; compare rerankers and top-k objectives next.")
    return recs


def first_ok_provider(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("status") == "ok":
            return row
    return rows[0] if rows else {}


def load_chunk_to_work(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): str(row["work_id"]) for row in rows}


def render_markdown(report: dict[str, Any]) -> str:
    worst = "\n".join(
        f"- `{row['id']}` {row['query']!r}: candidate_recall={row['candidate_recall']} "
        f"work_hit_rate={row['relevant_work_hit_rate']} mode=`{row['failure_mode']}`"
        for row in report["worst_queries"]
    )
    recs = "\n".join(f"- {item}" for item in report["recommendations"])
    return f"""# Stage 1 Retrieval Diagnostics

Mode: `{report['mode']}`

Provider: `{report['provider']}`

## Aggregate

- mean_candidate_recall: `{report['aggregate']['mean_candidate_recall']}`
- median_candidate_recall: `{report['aggregate']['median_candidate_recall']}`
- mean_relevant_work_hit_rate: `{report['aggregate']['mean_relevant_work_hit_rate']}`
- queries_with_any_relevant_work_in_pool: `{report['aggregate']['queries_with_any_relevant_work_in_pool']}`
- queries_with_top10_relevant_hits: `{report['aggregate']['queries_with_top10_relevant_hits']}`
- zero_candidate_recall_queries: `{report['aggregate']['zero_candidate_recall_queries']}`
- qrels_granularity: `{report['qrels_semantics']['qrels_granularity']}`
- oracle_reranking_status: `{report['oracle_reranking']['status']}`
- parent_duplicate_chunks_at_10_mean: `{report['parent_dominance']['mean_duplicate_chunk_count_at_10']}`

## Failure Modes

{json.dumps(report['failure_modes'], indent=2)}

## Worst Queries

{worst}

## Recommendations

{recs}
"""


def average(values: list[float | int]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def median(values: list[float | int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 6)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 6)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "mean": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 6),
        "q25": percentile(values, 0.25),
        "median": median(values),
        "mean": average(values),
        "q75": percentile(values, 0.75),
        "max": round(max(values), 6),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage 1 candidate recall by query.")
    parser.add_argument("--rerank-report", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage1_retrieval_diagnostics(
                rerank_report=Path(args.rerank_report),
                qrels_path=Path(args.qrels),
                mode=args.mode,
                output_stem=args.output_stem,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.external_ir import load_qrels
from canon.eval.rerank_evaluation import PARENT_QRELS_PROTOCOL, evaluate_rerankers, write_json
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import load_processed_corpus


DEFAULT_MODE = "beir_nfcorpus_stage1_title_preserve"
DEFAULT_QRELS = "gold/beir_nfcorpus_stage1_title_preserve_qrels.json"
DEFAULT_RERANKER = "cohere:rerank-v4.0-fast"
DEFAULT_VECTOR_PROVIDER = "openrouter"
DEFAULT_VECTOR_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_REPORT_JSON = "reports/stage1_fixed_qrels_v2_repeat_spread.json"
DEFAULT_REPORT_MD = "reports/stage1_fixed_qrels_v2_repeat_spread.md"
DEFAULT_QUERY_CACHE_ROOT = "reports/rerank_query_cache_fixed_qrels_v2_repeats"


@dataclass(frozen=True)
class RepeatConfig:
    id: str
    label: str
    auto_query_expansion: bool
    parent_qrels: bool
    parent_expansion_limit: int
    qrels_scope: str


DEFAULT_CONFIGS = [
    RepeatConfig(
        id="base_qwen_cohere",
        label="Base Qwen+Cohere",
        auto_query_expansion=False,
        parent_qrels=False,
        parent_expansion_limit=0,
        qrels_scope="original_chunk_qrels",
    ),
    RepeatConfig(
        id="auto_query_expansion_fixed_parent_qrels",
        label="Auto query expansion + fixed parent qrels",
        auto_query_expansion=True,
        parent_qrels=True,
        parent_expansion_limit=0,
        qrels_scope="full_corpus_parent_map_v2",
    ),
    RepeatConfig(
        id="auto_query_expansion_fixed_parent_qrels_parent_neighborhood",
        label="Auto query expansion + fixed parent qrels + parent-neighborhood candidates",
        auto_query_expansion=True,
        parent_qrels=True,
        parent_expansion_limit=2,
        qrels_scope="full_corpus_parent_map_v2",
    ),
]


def metric_names(k: int) -> list[str]:
    return [
        f"nDCG@{k}",
        f"Recall@{k}",
        f"MAP@{k}",
        f"MRR@{k}",
        "candidate_recall_mean",
        "work_level_hit_rate_mean",
    ]


def round_metric(value: float) -> float:
    return round(float(value), 6)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "spread": 0.0,
            "sample_std": 0.0,
            "spread_pp": 0.0,
            "sample_std_pp": 0.0,
            "values": [],
        }
    ordered = [float(value) for value in values]
    spread = max(ordered) - min(ordered)
    std = sample_std(ordered)
    return {
        "n": len(ordered),
        "mean": round_metric(average(ordered)),
        "min": round_metric(min(ordered)),
        "max": round_metric(max(ordered)),
        "spread": round_metric(spread),
        "sample_std": round_metric(std),
        "spread_pp": round(spread * 100, 3),
        "sample_std_pp": round(std * 100, 3),
        "values": [round_metric(value) for value in ordered],
    }


def positive_relevant_works(qrels: dict[str, float], chunk_to_work: dict[str, str]) -> set[str]:
    return {
        str(chunk_to_work.get(str(chunk_id), chunk_id))
        for chunk_id, relevance in qrels.items()
        if float(relevance) > 0
    }


def query_work_level_summary(
    query_report: dict[str, Any],
    original_relevant: dict[str, float],
    chunk_to_work: dict[str, str],
) -> dict[str, Any]:
    relevant_works = positive_relevant_works(original_relevant, chunk_to_work)
    candidate_works = {
        str(chunk_to_work.get(str(row.get("chunk_id")), row.get("chunk_id")))
        for row in query_report.get("scored_candidates", [])
        if row.get("chunk_id") is not None
    }
    hit_works = relevant_works & candidate_works
    return {
        "id": query_report.get("id"),
        "relevant_work_count": len(relevant_works),
        "relevant_work_hit_count": len(hit_works),
        "relevant_work_hit_rate": round_metric(
            len(hit_works) / len(relevant_works) if relevant_works else 0.0
        ),
        "any_relevant_work_in_pool": bool(hit_works),
    }


def extract_run_metrics(
    reranker: dict[str, Any],
    qrels: dict[str, Any],
    chunk_to_work: dict[str, str],
    k: int,
) -> dict[str, Any]:
    queries = reranker.get("queries", [])
    qrels_by_id = {str(query["id"]): query.get("relevant", {}) for query in qrels.get("queries", [])}
    work_rows = [
        query_work_level_summary(
            query,
            qrels_by_id.get(str(query.get("id")), {}),
            chunk_to_work,
        )
        for query in queries
    ]
    candidate_recall_values = [
        float((query.get("candidate_recall") or {}).get("candidate_recall", 0.0))
        for query in queries
    ]
    summary = reranker.get("summary", {})
    metrics = {
        f"nDCG@{k}": round_metric(summary.get(f"nDCG@{k}", 0.0)),
        f"Recall@{k}": round_metric(summary.get(f"Recall@{k}", 0.0)),
        f"MAP@{k}": round_metric(summary.get(f"MAP@{k}", 0.0)),
        f"MRR@{k}": round_metric(summary.get(f"MRR@{k}", 0.0)),
        "candidate_recall_mean": round_metric(average(candidate_recall_values)),
        "work_level_hit_rate_mean": round_metric(
            average([row["relevant_work_hit_rate"] for row in work_rows])
        ),
    }
    metrics["queries_with_any_relevant_work_in_pool_count"] = sum(
        1 for row in work_rows if row["any_relevant_work_in_pool"]
    )
    metrics["zero_candidate_recall_query_count"] = sum(
        1 for value in candidate_recall_values if value <= 0
    )
    metrics["queries_with_any_relevant_work_in_pool"] = (
        f"{metrics['queries_with_any_relevant_work_in_pool_count']}/{len(work_rows)}"
    )
    metrics["zero_candidate_recall_queries"] = (
        f"{metrics['zero_candidate_recall_query_count']}/{len(queries)}"
    )
    return metrics


def chunk_to_work_map(mode: str) -> dict[str, str]:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    return {document.chunk_id: document.work_id for document in documents}


def completed_ok_keys(runs: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(run.get("configuration")), int(run.get("repeat", 0)))
        for run in runs
        if run.get("status") == "ok"
    }


def replace_run(
    runs: list[dict[str, Any]],
    replacement: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if not (
            run.get("configuration") == replacement.get("configuration")
            and int(run.get("repeat", 0)) == int(replacement.get("repeat", 0))
        )
    ] + [replacement]


def summarize_runs(
    runs: list[dict[str, Any]],
    configs: list[RepeatConfig],
    k: int,
) -> list[dict[str, Any]]:
    summaries = []
    for config in configs:
        config_runs = [
            run
            for run in runs
            if run.get("configuration") == config.id and run.get("status") == "ok"
        ]
        metric_summary = {
            metric: summarize_values(
                [float((run.get("metrics") or {}).get(metric, 0.0)) for run in config_runs]
            )
            for metric in metric_names(k)
        }
        summaries.append(
            {
                **asdict(config),
                "completed_runs": len(config_runs),
                "metrics": metric_summary,
                "zero_candidate_recall_queries_values": [
                    (run.get("metrics") or {}).get("zero_candidate_recall_queries")
                    for run in config_runs
                ],
                "queries_with_any_relevant_work_in_pool_values": [
                    (run.get("metrics") or {}).get("queries_with_any_relevant_work_in_pool")
                    for run in config_runs
                ],
            }
        )
    return summaries


def paired_delta(
    runs: list[dict[str, Any]],
    *,
    treatment_config: str,
    baseline_config: str,
    metric: str,
) -> dict[str, Any]:
    by_key = {
        (str(run.get("configuration")), int(run.get("repeat", 0))): run
        for run in runs
        if run.get("status") == "ok"
    }
    repeats = sorted(
        {
            repeat
            for config, repeat in by_key
            if config == treatment_config and (baseline_config, repeat) in by_key
        }
    )
    rows = []
    for repeat in repeats:
        treatment = by_key[(treatment_config, repeat)]
        baseline = by_key[(baseline_config, repeat)]
        delta = float((treatment.get("metrics") or {}).get(metric, 0.0)) - float(
            (baseline.get("metrics") or {}).get(metric, 0.0)
        )
        rows.append(
            {
                "repeat": repeat,
                "delta": round_metric(delta),
                "delta_pp": round(delta * 100, 3),
                "treatment": round_metric(float((treatment.get("metrics") or {}).get(metric, 0.0))),
                "baseline": round_metric(float((baseline.get("metrics") or {}).get(metric, 0.0))),
            }
        )
    values = [row["delta"] for row in rows]
    return {
        "comparison": f"{treatment_config} - {baseline_config}",
        "treatment_config": treatment_config,
        "baseline_config": baseline_config,
        "metric": metric,
        "n": len(rows),
        "mean_delta": round_metric(average(values)),
        "mean_delta_pp": round(average(values) * 100, 3),
        "min_delta": round_metric(min(values)) if values else 0.0,
        "max_delta": round_metric(max(values)) if values else 0.0,
        "spread_delta": round_metric(max(values) - min(values)) if values else 0.0,
        "spread_delta_pp": round((max(values) - min(values)) * 100, 3) if values else 0.0,
        "rows": rows,
    }


def comparison_decision(
    comparison: dict[str, Any],
    summaries: list[dict[str, Any]],
    *,
    materiality_pp: float = 0.5,
) -> dict[str, Any]:
    by_config = {summary["id"]: summary for summary in summaries}
    treatment = by_config.get(comparison["treatment_config"], {})
    baseline = by_config.get(comparison["baseline_config"], {})
    metric = comparison["metric"]
    treatment_spread = ((treatment.get("metrics") or {}).get(metric) or {}).get("spread", 0.0)
    baseline_spread = ((baseline.get("metrics") or {}).get(metric) or {}).get("spread", 0.0)
    max_within_spread = max(float(treatment_spread), float(baseline_spread))
    mean_delta = float(comparison.get("mean_delta", 0.0))
    min_delta = float(comparison.get("min_delta", 0.0))
    max_delta = float(comparison.get("max_delta", 0.0))
    if comparison.get("n", 0) < 3:
        standing = "insufficient_repeat_count"
        reason = "Fewer than three paired runs are complete."
    elif min_delta <= 0 <= max_delta:
        standing = "does_not_stand_direction_unstable"
        reason = "The paired delta changes sign across repeats."
    elif abs(mean_delta) <= max_within_spread:
        standing = "does_not_stand_effect_smaller_than_repeat_spread"
        reason = "The mean paired effect is no larger than the within-configuration spread."
    elif abs(mean_delta * 100) < materiality_pp:
        standing = "stands_as_repeatable_but_tiny"
        reason = "The direction is repeat-stable, but the effect is below the materiality threshold."
    else:
        standing = "stands_under_repeat_spread_check"
        reason = "The paired effect is larger than the observed within-configuration spread."
    return {
        "comparison": comparison["comparison"],
        "metric": metric,
        "standing": standing,
        "reason": reason,
        "mean_delta": round_metric(mean_delta),
        "mean_delta_pp": round(mean_delta * 100, 3),
        "max_within_config_spread": round_metric(max_within_spread),
        "max_within_config_spread_pp": round(max_within_spread * 100, 3),
        "min_delta": round_metric(min_delta),
        "max_delta": round_metric(max_delta),
        "materiality_threshold_pp": materiality_pp,
    }


def build_report(
    runs: list[dict[str, Any]],
    *,
    configs: list[RepeatConfig],
    repeats: int,
    mode: str,
    qrels_path: Path,
    qrels: dict[str, Any],
    k: int,
    candidate_k: int,
    lexical_k: int,
    vector_k: int,
    vector_provider: str,
    vector_model: str,
    reranker: str,
    fusion: str,
    document_format: str,
    query_cache_root: Path,
) -> dict[str, Any]:
    normalized_runs = [normalize_run_paths(run) for run in runs]
    summaries = summarize_runs(normalized_runs, configs, k)
    primary_metric = f"nDCG@{k}"
    comparisons = [
        paired_delta(
            normalized_runs,
            treatment_config="auto_query_expansion_fixed_parent_qrels",
            baseline_config="base_qwen_cohere",
            metric=primary_metric,
        ),
        paired_delta(
            normalized_runs,
            treatment_config="auto_query_expansion_fixed_parent_qrels_parent_neighborhood",
            baseline_config="base_qwen_cohere",
            metric=primary_metric,
        ),
    ]
    decisions = [comparison_decision(comparison, summaries) for comparison in comparisons]
    completed = sum(1 for run in normalized_runs if run.get("status") == "ok")
    expected = repeats * len(configs)
    status = "repeat_spread_complete" if completed == expected else "repeat_spread_partial"
    return {
        "report_id": "stage1_fixed_qrels_v2_repeat_spread",
        "created_at": time.strftime("%Y-%m-%d"),
        "status": status,
        "mode": mode,
        "qrels_path": display_path(qrels_path),
        "query_count": len(qrels.get("queries", [])),
        "label_count": sum(len(query.get("relevant", {})) for query in qrels.get("queries", [])),
        "repeat_count_requested": repeats,
        "completed_run_count": completed,
        "expected_run_count": expected,
        "protocol": {
            "parent_qrels_protocol": PARENT_QRELS_PROTOCOL,
            "integrity_boundary": (
                "When parent_qrels is enabled, labels are expanded once from the full "
                "corpus chunk-to-parent map before candidate output is inspected."
            ),
            "repeat_boundary": (
                "Each completed repeat uses a separate query cache path and runs with "
                "resume disabled inside evaluate_rerankers."
            ),
        },
        "retrieval_stack": {
            "sparse": "bm25",
            "dense_provider": vector_provider,
            "dense_model": vector_model,
            "fusion": fusion,
            "reranker": reranker,
            "candidate_pool": f"{lexical_k} lexical plus {vector_k} vector candidates",
            "document_format": document_format,
            "query_cache_root": display_path(query_cache_root),
        },
        "configuration_summaries": summaries,
        "paired_deltas": comparisons,
        "claim_checks": decisions,
        "runs": sorted(
            normalized_runs,
            key=lambda run: (run.get("repeat", 0), str(run.get("configuration"))),
        ),
        "interpretation": interpretation(decisions),
    }


def display_path(path: Path | str) -> str:
    path = Path(str(path))
    settings = load_settings()
    try:
        return str(path.resolve().relative_to(settings.root.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return str(path).replace("\\", "/")


def normalize_run_paths(run: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(run)
    if normalized.get("query_cache_dir"):
        normalized["query_cache_dir"] = display_path(str(normalized["query_cache_dir"]))
    if isinstance(normalized.get("cache"), dict):
        cache = dict(normalized["cache"])
        if cache.get("path"):
            cache["path"] = display_path(str(cache["path"]))
        normalized["cache"] = cache
    return normalized


def interpretation(decisions: list[dict[str, Any]]) -> list[str]:
    primary = decisions[0] if decisions else {}
    standing = primary.get("standing")
    if standing == "stands_under_repeat_spread_check":
        first = "The fixed-qrels auto-expansion conclusion stands under this repeat-spread check."
    elif standing == "stands_as_repeatable_but_tiny":
        first = (
            "The fixed-qrels auto-expansion direction is repeat-stable, but the effect "
            "is practically tiny."
        )
    elif standing == "insufficient_repeat_count":
        first = "The repeat-spread check is incomplete and should not be used for a conclusion."
    else:
        first = "The fixed-qrels auto-expansion gain should not be claimed from this repeat check."
    return [
        first,
        "Treat this as a pilot reproducibility check, not a public BEIR leaderboard result.",
        "Human-reviewed passage-level labels are still required for publication-quality claims.",
    ]


def markdown_report(report: dict[str, Any]) -> str:
    primary_metric = "nDCG@10"
    lines = [
        "# Stage 1 Fixed-Qrels V2 Repeat Spread",
        "",
        f"Status: `{report['status']}`",
        "",
        (
            f"This report repeats each fixed-qrels Stage 1 configuration "
            f"{report['repeat_count_requested']} time(s) and reports run-to-run spread."
        ),
        "",
        "| Configuration | Runs | nDCG@10 mean | nDCG@10 min | nDCG@10 max | nDCG spread pp | MAP@10 mean | Candidate recall mean | Work-hit mean | Zero-hit queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in report["configuration_summaries"]:
        metrics = summary["metrics"]
        lines.append(
            "| {label} | {runs} | {ndcg_mean:.6f} | {ndcg_min:.6f} | {ndcg_max:.6f} | {ndcg_spread_pp:.3f} | {map_mean:.6f} | {candidate_mean:.6f} | {work_mean:.6f} | {zero} |".format(
                label=summary["label"],
                runs=summary["completed_runs"],
                ndcg_mean=metrics[primary_metric]["mean"],
                ndcg_min=metrics[primary_metric]["min"],
                ndcg_max=metrics[primary_metric]["max"],
                ndcg_spread_pp=metrics[primary_metric]["spread_pp"],
                map_mean=metrics["MAP@10"]["mean"],
                candidate_mean=metrics["candidate_recall_mean"]["mean"],
                work_mean=metrics["work_level_hit_rate_mean"]["mean"],
                zero=", ".join(
                    str(value)
                    for value in summary.get("zero_candidate_recall_queries_values", [])
                    if value is not None
                )
                or "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Paired nDCG@10 Deltas",
            "",
            "| Comparison | Paired runs | Mean delta pp | Min delta pp | Max delta pp | Delta spread pp | Standing |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    checks_by_comparison = {check["comparison"]: check for check in report.get("claim_checks", [])}
    for comparison in report.get("paired_deltas", []):
        check = checks_by_comparison.get(comparison["comparison"], {})
        lines.append(
            "| {comparison} | {n} | {mean:.3f} | {min_delta:.3f} | {max_delta:.3f} | {spread:.3f} | `{standing}` |".format(
                comparison=comparison["comparison"],
                n=comparison["n"],
                mean=comparison["mean_delta_pp"],
                min_delta=comparison["min_delta"] * 100,
                max_delta=comparison["max_delta"] * 100,
                spread=comparison["spread_delta_pp"],
                standing=check.get("standing", "unknown"),
            )
        )
    lines.extend(["", "Interpretation:", ""])
    lines.extend([f"- {item}" for item in report.get("interpretation", [])])
    lines.extend(
        [
            "",
            "Full per-run values and metadata are in",
            "`reports/stage1_fixed_qrels_v2_repeat_spread.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def load_existing_runs(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("runs") or [])


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(markdown_report(report), encoding="utf-8")


def run_once(
    config: RepeatConfig,
    *,
    repeat: int,
    mode: str,
    qrels_path: Path,
    qrels: dict[str, Any],
    chunk_to_work: dict[str, str],
    k: int,
    candidate_k: int,
    lexical_k: int,
    vector_k: int,
    vector_provider: str,
    vector_model: str,
    reranker: str,
    fusion: str,
    document_format: str,
    query_cache_root: Path,
) -> dict[str, Any]:
    query_cache_dir = query_cache_root / f"repeat_{repeat}" / config.id
    started = time.perf_counter()
    report = evaluate_rerankers(
        mode=mode,
        qrels_path=qrels_path,
        rerankers=[reranker],
        candidate_k=candidate_k,
        k=k,
        pooled_candidates=True,
        lexical_k=lexical_k,
        vector_k=vector_k,
        vector_provider=vector_provider,
        vector_model=vector_model,
        fusion=fusion,
        document_format=document_format,
        auto_query_expansion=config.auto_query_expansion,
        parent_qrels=config.parent_qrels,
        parent_expansion_limit=config.parent_expansion_limit,
        resume=False,
        query_cache_dir=query_cache_dir,
    )
    reranker_report = (report.get("rerankers") or [{}])[0]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    run = {
        **asdict(config),
        "configuration": config.id,
        "repeat": repeat,
        "status": reranker_report.get("status", "unavailable"),
        "reason": reranker_report.get("reason"),
        "provider_id": reranker_report.get("provider_id"),
        "elapsed_ms": float(reranker_report.get("elapsed_ms", elapsed_ms)),
        "wall_elapsed_ms": elapsed_ms,
        "query_cache_dir": str(query_cache_dir).replace("\\", "/"),
        "cache": reranker_report.get("cache", {}),
    }
    if reranker_report.get("status") == "ok":
        run["metrics"] = extract_run_metrics(reranker_report, qrels, chunk_to_work, k)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the fixed-qrels Stage 1 hosted rerank configurations and publish spread."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=250)
    parser.add_argument("--lexical-k", type=int, default=250)
    parser.add_argument("--vector-k", type=int, default=250)
    parser.add_argument("--vector-provider", default=DEFAULT_VECTOR_PROVIDER)
    parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--fusion", default="rrf")
    parser.add_argument("--document-format", default="structured", choices=["plain", "structured"])
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--query-cache-root", default=DEFAULT_QUERY_CACHE_ROOT)
    parser.add_argument("--no-resume", action="store_true", help="Ignore completed runs in an existing report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    settings = load_settings()
    qrels_path = Path(args.qrels)
    if not qrels_path.is_absolute():
        qrels_path = settings.root / qrels_path
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    query_cache_root = Path(args.query_cache_root)
    if not report_json.is_absolute():
        report_json = settings.root / report_json
    if not report_md.is_absolute():
        report_md = settings.root / report_md
    if not query_cache_root.is_absolute():
        query_cache_root = settings.root / query_cache_root

    qrels = load_qrels(qrels_path)
    chunk_map = chunk_to_work_map(args.mode)
    runs = [] if args.no_resume else load_existing_runs(report_json)
    configs = list(DEFAULT_CONFIGS)
    print(
        f"[repeat-spread] mode={args.mode} repeats={args.repeats} "
        f"configs={len(configs)} resume_completed={not args.no_resume}",
        flush=True,
    )
    for repeat in range(1, args.repeats + 1):
        for config in configs:
            key = (config.id, repeat)
            if key in completed_ok_keys(runs):
                print(f"[repeat-spread] skip repeat={repeat} config={config.id}", flush=True)
                continue
            print(f"[repeat-spread] start repeat={repeat} config={config.id}", flush=True)
            run = run_once(
                config,
                repeat=repeat,
                mode=args.mode,
                qrels_path=qrels_path,
                qrels=qrels,
                chunk_to_work=chunk_map,
                k=args.k,
                candidate_k=args.candidate_k,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                vector_provider=args.vector_provider,
                vector_model=args.vector_model,
                reranker=args.reranker,
                fusion=args.fusion,
                document_format=args.document_format,
                query_cache_root=query_cache_root,
            )
            runs = replace_run(runs, run)
            report = build_report(
                runs,
                configs=configs,
                repeats=args.repeats,
                mode=args.mode,
                qrels_path=qrels_path,
                qrels=qrels,
                k=args.k,
                candidate_k=args.candidate_k,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                vector_provider=args.vector_provider,
                vector_model=args.vector_model,
                reranker=args.reranker,
                fusion=args.fusion,
                document_format=args.document_format,
                query_cache_root=query_cache_root,
            )
            write_reports(report, report_json, report_md)
            if run.get("status") == "ok":
                metrics = run["metrics"]
                print(
                    f"[repeat-spread] ok repeat={repeat} config={config.id} "
                    f"nDCG@{args.k}={metrics[f'nDCG@{args.k}']:.6f} "
                    f"MAP@{args.k}={metrics[f'MAP@{args.k}']:.6f} "
                    f"candidate_recall={metrics['candidate_recall_mean']:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[repeat-spread] unavailable repeat={repeat} config={config.id} "
                    f"reason={run.get('reason')}",
                    flush=True,
                )

    final_report = build_report(
        runs,
        configs=configs,
        repeats=args.repeats,
        mode=args.mode,
        qrels_path=qrels_path,
        qrels=qrels,
        k=args.k,
        candidate_k=args.candidate_k,
        lexical_k=args.lexical_k,
        vector_k=args.vector_k,
        vector_provider=args.vector_provider,
        vector_model=args.vector_model,
        reranker=args.reranker,
        fusion=args.fusion,
        document_format=args.document_format,
        query_cache_root=query_cache_root,
    )
    write_reports(final_report, report_json, report_md)
    print(json.dumps(final_report["claim_checks"], indent=2), flush=True)


if __name__ == "__main__":
    main()

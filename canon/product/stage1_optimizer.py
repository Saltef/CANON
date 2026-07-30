from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.external_ir import load_qrels
from canon.eval.model_evaluation import parse_provider_spec, provider_identifier
from canon.eval.rerank_evaluation import evaluate_rerankers
from canon.product import report_io
from canon.product.automated_benchmark_suite import DEFAULT_SUITE_PATH, load_suite, resolve_path


DEFAULT_CANDIDATE_VALUES = [25, 50, 100]
DEFAULT_FUSION_METHODS = ["union"]
OBJECTIVE_MODES = {"quality", "balanced", "low_latency", "low_cost"}


def run_stage1_optimizer(
    suite_path: Path | None = None,
    dense_retrievers: list[str] | None = None,
    rerankers: list[str] | None = None,
    candidate_values: list[int] | None = None,
    fusion_methods: list[str] | None = None,
    document_format: str = "plain",
    objective_mode: str = "quality",
    resume: bool = True,
    top_k: int = 10,
    max_chunks_per_parent: int = 0,
    auto_query_expansion: bool = False,
    benchmark_oracle_expansion: bool = False,
    parent_qrels: bool = False,
    parent_expansion_limit: int = 0,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(f"objective_mode must be one of {sorted(OBJECTIVE_MODES)}")
    suite = load_suite(resolve_path(settings.root, suite_path or DEFAULT_SUITE_PATH))
    values = candidate_values or DEFAULT_CANDIDATE_VALUES
    fusions = fusion_methods or suite.get("fusion_methods") or [suite.get("fusion", "union")]
    cache_dir = settings.reports_dir / "stage1_optimizer_cache"
    trials = [
        trial
        for benchmark in suite.get("benchmarks", [])
        for trial in run_benchmark_trials(
            settings.root,
            cache_dir,
            benchmark,
            dense_specs=dense_retrievers or suite.get("model_providers") or ["local"],
            reranker_specs=rerankers or suite.get("rerankers") or ["heuristic"],
            candidate_values=values,
            fusion_methods=fusions,
            document_format=document_format or str(suite.get("document_format", "plain")),
            objective_mode=objective_mode,
            resume=resume,
            top_k=top_k,
            max_chunks_per_parent=max_chunks_per_parent,
            auto_query_expansion=auto_query_expansion,
            benchmark_oracle_expansion=benchmark_oracle_expansion,
            parent_qrels=parent_qrels,
            parent_expansion_limit=parent_expansion_limit,
            query_cache_dir=settings.reports_dir / "automated_benchmark_query_cache" / suite["suite_id"],
        )
    ]
    report = {
        "report_id": "stage1_optimizer_v1",
        "suite_id": suite["suite_id"],
        "dense_retrievers": dense_retrievers or suite.get("model_providers") or ["local"],
        "rerankers": rerankers or suite.get("rerankers") or ["heuristic"],
        "fusion_methods": fusions,
        "document_format": document_format,
        "max_chunks_per_parent": max_chunks_per_parent,
        "auto_query_expansion": auto_query_expansion,
        "benchmark_oracle_expansion": benchmark_oracle_expansion,
        "parent_qrels": parent_qrels,
        "parent_expansion_limit": parent_expansion_limit,
        "status": optimizer_status(trials),
        "objective": objective_description(top_k, objective_mode),
        "cache": {
            "enabled": resume,
            "path": relative(settings.root, cache_dir),
            "cached_trial_count": sum(1 for trial in trials if trial.get("cache_hit")),
        },
        "trial_count": len(trials),
        "leaderboard": optimizer_leaderboard(trials),
        "best_trial": first_or_empty(optimizer_leaderboard(trials)),
        "candidate_recall_failures": [trial for trial in trials if trial["candidate_recall_mean"] < 1.0],
        "trials": trials,
        "generation_metrics_boundary": {
            "status": "not_stage1_objective",
            "metrics": ["BLEU", "METEOR", "ROUGE", "BERTScore"],
            "reason": "These evaluate generated text against references; Stage 1 optimizes evidence retrieval and ranking.",
        },
    }
    if write_report:
        report_io.write_json(settings.reports_dir / "stage1_optimizer_v1.json", report)
        report_io.write_markdown(settings.reports_dir / "stage1_optimizer_v1.md", render_markdown(report))
    return report


def run_benchmark_trials(
    root: Path,
    cache_dir: Path,
    benchmark: dict[str, Any],
    dense_specs: list[str],
    reranker_specs: list[str],
    candidate_values: list[int],
    fusion_methods: list[str],
    document_format: str,
    objective_mode: str,
    resume: bool,
    top_k: int,
    max_chunks_per_parent: int,
    auto_query_expansion: bool,
    benchmark_oracle_expansion: bool,
    parent_qrels: bool,
    parent_expansion_limit: int,
    query_cache_dir: Path,
) -> list[dict[str, Any]]:
    qrels_path = resolve_path(root, Path(benchmark["qrels"]))
    qrels = load_qrels(qrels_path)
    return [
        run_trial(
            benchmark=benchmark,
            qrels_path=qrels_path,
            query_count=len(qrels.get("queries") or []),
            dense_spec=dense_spec,
            reranker_spec=reranker_spec,
            candidate_k=candidate_k,
            fusion=fusion,
            document_format=str(benchmark.get("document_format", document_format)),
            objective_mode=objective_mode,
            cache_dir=cache_dir,
            resume=resume,
            top_k=top_k,
            max_chunks_per_parent=max_chunks_per_parent,
            auto_query_expansion=auto_query_expansion,
            benchmark_oracle_expansion=benchmark_oracle_expansion,
            parent_qrels=parent_qrels,
            parent_expansion_limit=parent_expansion_limit,
            query_cache_dir=query_cache_dir,
        )
        for candidate_k in candidate_values
        for fusion in fusion_methods
        for dense_spec in dense_specs
        for reranker_spec in reranker_specs
    ]


def run_trial(
    benchmark: dict[str, Any],
    qrels_path: Path,
    query_count: int,
    dense_spec: str,
    reranker_spec: str,
    candidate_k: int,
    fusion: str,
    document_format: str,
    objective_mode: str,
    cache_dir: Path,
    resume: bool,
    top_k: int,
    max_chunks_per_parent: int,
    auto_query_expansion: bool,
    benchmark_oracle_expansion: bool,
    parent_qrels: bool,
    parent_expansion_limit: int,
    query_cache_dir: Path,
) -> dict[str, Any]:
    cache_path = trial_cache_path(
        cache_dir,
        benchmark=benchmark,
        dense_spec=dense_spec,
        reranker_spec=reranker_spec,
        candidate_k=candidate_k,
        fusion=fusion,
        document_format=document_format,
        objective_mode=objective_mode,
        top_k=top_k,
        max_chunks_per_parent=max_chunks_per_parent,
        auto_query_expansion=auto_query_expansion,
        benchmark_oracle_expansion=benchmark_oracle_expansion,
        parent_qrels=parent_qrels,
        parent_expansion_limit=parent_expansion_limit,
    )
    if resume and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached
    dense_provider, dense_model = parse_provider_spec(dense_spec)
    report = evaluate_rerankers(
        mode=benchmark["mode"],
        qrels_path=qrels_path,
        rerankers=[reranker_spec],
        candidate_k=candidate_k,
        k=top_k,
        pooled_candidates=True,
        lexical_k=candidate_k,
        vector_k=candidate_k,
        vector_provider=dense_provider,
        vector_model=dense_model,
        fusion=fusion,
        document_format=document_format,
        query_variants=benchmark.get("query_variants") or [],
        max_chunks_per_parent=max_chunks_per_parent,
        auto_query_expansion=auto_query_expansion,
        benchmark_oracle_expansion=benchmark_oracle_expansion,
        parent_qrels=parent_qrels,
        parent_expansion_limit=parent_expansion_limit,
        query_cache_dir=query_cache_dir,
    )
    reranker = first_or_empty(report.get("rerankers", []))
    summary = reranker.get("summary", {}) if reranker.get("status") == "ok" else {}
    candidate_recall = candidate_recall_summary(reranker.get("queries", []))
    score_observability = score_observability_summary(reranker.get("queries", []))
    trial = {
        "benchmark_id": benchmark["id"],
        "mode": benchmark["mode"],
        "query_count": query_count,
        "candidate_k": candidate_k,
        "fusion": fusion,
        "document_format": document_format,
        "max_chunks_per_parent": max_chunks_per_parent,
        "auto_query_expansion": auto_query_expansion,
        "benchmark_oracle_expansion": benchmark_oracle_expansion,
        "parent_qrels": parent_qrels,
        "parent_expansion_limit": parent_expansion_limit,
        "lexical_search": "bm25",
        "dense_retriever": provider_identifier(dense_provider, dense_model),
        "reranker": reranker.get("provider_id") or reranker_spec,
        "status": reranker.get("status", "unavailable"),
        "reason": reranker.get("reason"),
        "nDCG@k": float(summary.get(f"nDCG@{top_k}", 0.0)),
        "Recall@k": float(summary.get(f"Recall@{top_k}", 0.0)),
        "MAP@k": float(summary.get(f"MAP@{top_k}", 0.0)),
        "MRR@k": float(summary.get(f"MRR@{top_k}", 0.0)),
        "candidate_recall_mean": candidate_recall["mean"],
        "candidate_recall_min": candidate_recall["min"],
        "candidate_miss_count": candidate_recall["miss_count"],
        "mean_score_gap": score_observability["mean_relevant_nonrelevant_score_gap"],
        "mean_low_confidence_top_k_count": score_observability["mean_low_confidence_top_k_count"],
        "elapsed_ms": float(reranker.get("elapsed_ms", 0.0)),
        "heuristic_cost": heuristic_cost(provider_identifier(dense_provider, dense_model), reranker.get("provider_id") or reranker_spec),
        "objective_mode": objective_mode,
        "composite_score": composite_score(
            summary,
            candidate_recall,
            score_observability,
            top_k,
            elapsed_ms=float(reranker.get("elapsed_ms", 0.0)),
            heuristic_cost_value=heuristic_cost(
                provider_identifier(dense_provider, dense_model),
                reranker.get("provider_id") or reranker_spec,
            ),
            objective_mode=objective_mode,
        ),
        "cache_hit": False,
    }
    write_trial_cache(cache_path, trial)
    return trial


def candidate_recall_summary(queries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [query.get("candidate_recall", {}) for query in queries]
    values = [float(row.get("candidate_recall", 0.0)) for row in rows]
    return {
        "mean": average(values),
        "min": round(min(values), 6) if values else 0.0,
        "miss_count": sum(1 for value in values if value < 1.0),
    }


def score_observability_summary(queries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [query.get("score_observability", {}) for query in queries]
    gaps = [float(row.get("relevant_nonrelevant_score_gap", 0.0)) for row in rows]
    low_confidence = [float(row.get("low_confidence_top_k_count", 0.0)) for row in rows]
    return {
        "mean_relevant_nonrelevant_score_gap": average(gaps),
        "mean_low_confidence_top_k_count": average(low_confidence),
    }


def composite_score(
    summary: dict[str, Any],
    candidate_recall: dict[str, Any],
    score_observability: dict[str, Any],
    top_k: int,
    elapsed_ms: float = 0.0,
    heuristic_cost_value: float = 0.0,
    objective_mode: str = "quality",
) -> float:
    quality = (
        0.35 * float(candidate_recall.get("mean", 0.0))
        + 0.30 * float(summary.get(f"nDCG@{top_k}", 0.0))
        + 0.20 * float(summary.get(f"Recall@{top_k}", 0.0))
        + 0.10 * float(summary.get(f"MRR@{top_k}", 0.0))
        + 0.05 * max(float(score_observability.get("mean_relevant_nonrelevant_score_gap", 0.0)), 0.0)
    )
    latency_penalty = min(elapsed_ms / 100000.0, 0.25)
    cost_penalty = min(heuristic_cost_value, 0.25)
    if objective_mode == "balanced":
        quality -= 0.05 * latency_penalty + 0.05 * cost_penalty
    elif objective_mode == "low_latency":
        quality -= 0.20 * latency_penalty + 0.05 * cost_penalty
    elif objective_mode == "low_cost":
        quality -= 0.05 * latency_penalty + 0.20 * cost_penalty
    return round(quality, 6)


def heuristic_cost(dense_retriever: str, reranker: str) -> float:
    return round(provider_cost_hint(dense_retriever, "dense") + provider_cost_hint(reranker, "rerank"), 6)


def provider_cost_hint(provider_id: str, role: str) -> float:
    normalized = provider_id.lower()
    if normalized.startswith(("local", "heuristic")):
        return 0.0
    if "rerank-v4.0-pro" in normalized:
        return 0.16
    if "rerank-v4.0-fast" in normalized:
        return 0.08
    if "rerank" in normalized:
        return 0.10
    if "qwen" in normalized or "bge" in normalized:
        return 0.01
    if "text-embedding-3-small" in normalized:
        return 0.02
    if "cohere" in normalized and role == "dense":
        return 0.12
    return 0.05


def trial_cache_path(
    cache_dir: Path,
    *,
    benchmark: dict[str, Any],
    dense_spec: str,
    reranker_spec: str,
    candidate_k: int,
    fusion: str,
    document_format: str,
    objective_mode: str,
    top_k: int,
    max_chunks_per_parent: int,
    auto_query_expansion: bool,
    benchmark_oracle_expansion: bool,
    parent_qrels: bool,
    parent_expansion_limit: int,
) -> Path:
    payload = {
        "benchmark_id": benchmark["id"],
        "mode": benchmark["mode"],
        "qrels": benchmark["qrels"],
        "query_variants": benchmark.get("query_variants") or [],
        "dense_spec": dense_spec,
        "reranker_spec": reranker_spec,
        "candidate_k": candidate_k,
        "fusion": fusion,
        "document_format": document_format,
        "objective_mode": objective_mode,
        "top_k": top_k,
        "max_chunks_per_parent": max_chunks_per_parent,
        "auto_query_expansion": auto_query_expansion,
        "benchmark_oracle_expansion": benchmark_oracle_expansion,
        "parent_qrels": parent_qrels,
        "parent_expansion_limit": parent_expansion_limit,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{safe_slug(str(benchmark['id']))}_{digest}.json"


def write_trial_cache(path: Path, trial: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in trial.items() if key != "cache_hit"}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def optimizer_leaderboard(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        [trial for trial in trials if trial.get("status") == "ok"],
        key=lambda trial: (
            trial["composite_score"],
            trial["candidate_recall_mean"],
            trial["nDCG@k"],
            -trial["elapsed_ms"],
        ),
        reverse=True,
    )
    return [{"rank": index, **trial} for index, trial in enumerate(ranked, start=1)]


def optimizer_status(trials: list[dict[str, Any]]) -> str:
    if not trials:
        return "automated_fail"
    if any(trial["status"] != "ok" for trial in trials):
        return "partial_model_matrix"
    if any(trial["candidate_recall_mean"] >= 1.0 for trial in trials):
        return "automated_pass_human_review_required"
    return "automated_fail"


def objective_description(top_k: int, objective_mode: str) -> dict[str, Any]:
    return {
        "mode": objective_mode,
        "primary_metrics": ["candidate_recall_mean", f"nDCG@{top_k}", f"Recall@{top_k}"],
        "secondary_metrics": [
            f"MRR@{top_k}",
            f"MAP@{top_k}",
            "mean_score_gap",
            "elapsed_ms",
            "heuristic_cost",
        ],
        "traditional_search": "BM25 is included in every trial as the sparse lexical candidate source.",
        "cost_note": "heuristic_cost is a relative planning signal, not a provider billing calculation.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- #{row['rank']} dense=`{row['dense_retriever']}` rerank=`{row['reranker']}` "
        f"k={row['candidate_k']} fusion=`{row['fusion']}` composite={row['composite_score']} ndcg={row['nDCG@k']} "
        f"recall={row['Recall@k']} candidate_recall={row['candidate_recall_mean']}"
        for row in report["leaderboard"][:10]
    )
    return f"""# CANON Stage 1 Optimizer

Status: **{report['status']}**

Trials: `{report['trial_count']}`

## Top Trials

{rows}

BLEU, METEOR, ROUGE, and BERTScore are reserved for generation-stage evaluation.
"""


def safe_slug(text: str, limit: int = 60) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in text.lower())
    return slug[:limit].strip("-") or "trial"


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def first_or_empty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Stage 1 retrieval stack combinations.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--dense-retrievers", default=None)
    parser.add_argument("--rerankers", default=None)
    parser.add_argument("--candidate-values", default=None, help="Comma-separated values, e.g. 25,50,100")
    parser.add_argument("--fusion-methods", default=None, help="Comma-separated values, e.g. union,rrf")
    parser.add_argument("--document-format", choices=["plain", "structured"], default="plain")
    parser.add_argument("--objective-mode", choices=sorted(OBJECTIVE_MODES), default="quality")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-chunks-per-parent", type=int, default=0)
    parser.add_argument("--auto-query-expansion", action="store_true")
    parser.add_argument("--benchmark-oracle-expansion", action="store_true")
    parser.add_argument("--parent-qrels", action="store_true")
    parser.add_argument("--parent-expansion-limit", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage1_optimizer(
                suite_path=Path(args.suite),
                dense_retrievers=parse_csv(args.dense_retrievers),
                rerankers=parse_csv(args.rerankers),
                candidate_values=parse_ints(args.candidate_values),
                fusion_methods=parse_csv(args.fusion_methods),
                document_format=args.document_format,
                objective_mode=args.objective_mode,
                resume=not args.no_resume,
                top_k=args.top_k,
                max_chunks_per_parent=args.max_chunks_per_parent,
                auto_query_expansion=args.auto_query_expansion,
                benchmark_oracle_expansion=args.benchmark_oracle_expansion,
                parent_qrels=args.parent_qrels,
                parent_expansion_limit=args.parent_expansion_limit,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

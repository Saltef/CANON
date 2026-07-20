from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.external_ir import load_qrels
from canon.eval.model_evaluation import evaluate_semantic_models, parse_providers
from canon.eval.model_evaluation import parse_provider_spec
from canon.eval.rerank_evaluation import evaluate_rerankers
from canon.product import report_io


DEFAULT_SUITE_PATH = Path("conf") / "benchmark_suites" / "stage1_public_multi_topic.json"


def run_automated_benchmark_suite(
    suite_path: Path | None = None,
    model_providers: list[str] | None = None,
    rerankers: list[str] | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    lexical_k: int | None = None,
    vector_k: int | None = None,
    pooled_candidates: bool | None = None,
    fusion: str | None = None,
    document_format: str | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    suite = load_suite(resolve_path(settings.root, suite_path or DEFAULT_SUITE_PATH))
    benchmarks = [
        run_benchmark(
            settings.root,
            benchmark,
            model_providers=model_providers or suite.get("model_providers") or ["local"],
            rerankers=rerankers or suite.get("rerankers") or ["heuristic"],
            top_k=top_k or int(suite.get("top_k", 10)),
            candidate_k=candidate_k or int(suite.get("candidate_k", 25)),
            lexical_k=lexical_k,
            vector_k=vector_k,
            pooled_candidates=pooled_candidates
            if pooled_candidates is not None
            else bool(suite.get("pooled_candidates", True)),
            fusion=fusion or str(suite.get("fusion", "union")),
            document_format=document_format or str(suite.get("document_format", "plain")),
        )
        for benchmark in suite.get("benchmarks", [])
    ]
    thresholds = suite.get("thresholds") or {}
    report = {
        "report_id": "automated_benchmark_suite_v1",
        "suite_id": suite["suite_id"],
        "description": suite.get("description", ""),
        "status": suite_status(benchmarks, thresholds),
        "model_matrix": model_matrix(benchmarks),
        "human_review_boundary": human_review_boundary(),
        "thresholds": thresholds,
        "benchmark_count": len(benchmarks),
        "benchmark_topic_count": len(topic_summary(benchmarks)),
        "query_topic_count": len(query_topic_summary(benchmarks)),
        "benchmarks": benchmarks,
        "topic_summary": topic_summary(benchmarks),
        "query_topic_summary": query_topic_summary(benchmarks),
        "leaderboards": suite_leaderboards(benchmarks),
        "limitations": [
            "This suite is an automated pre-human benchmark and does not replace human-reviewed qrels or brief review.",
            "Public document-level qrels expanded to chunks can test retrieval plumbing, but they are noisier than chunk-level human qrels.",
            "Provider rankings are valid only for the listed corpora, topics, and settings.",
        ],
    }
    if write_report:
        output = settings.reports_dir / f"automated_benchmark_suite_{suite['suite_id']}.json"
        report_io.write_json(output, report)
        report_io.write_markdown(
            settings.reports_dir / f"automated_benchmark_suite_{suite['suite_id']}.md",
            render_markdown(report),
        )
    return report


def run_benchmark(
    root: Path,
    benchmark: dict[str, Any],
    model_providers: list[str],
    rerankers: list[str],
    top_k: int,
    candidate_k: int,
    lexical_k: int | None,
    vector_k: int | None,
    pooled_candidates: bool,
    fusion: str,
    document_format: str,
) -> dict[str, Any]:
    qrels_path = resolve_path(root, Path(benchmark["qrels"]))
    qrels = load_qrels(qrels_path)
    semantic = evaluate_semantic_models(
        mode=benchmark["mode"],
        qrels_path=qrels_path,
        providers=model_providers,
        k=top_k,
    )
    rerank = evaluate_rerankers(
        mode=benchmark["mode"],
        qrels_path=qrels_path,
        rerankers=rerankers,
        candidate_k=candidate_k,
        k=top_k,
        pooled_candidates=pooled_candidates,
        lexical_k=lexical_k or int(benchmark.get("lexical_k", candidate_k)),
        vector_k=vector_k or int(benchmark.get("vector_k", candidate_k)),
        vector_provider=parse_provider_spec(benchmark.get("candidate_vector_provider", "local"))[0],
        vector_model=parse_provider_spec(benchmark.get("candidate_vector_provider", "local"))[1],
        fusion=str(benchmark.get("fusion", fusion)),
        document_format=str(benchmark.get("document_format", document_format)),
        query_variants=benchmark.get("query_variants") or [],
    )
    return {
        "id": benchmark["id"],
        "topic": benchmark.get("topic", "unspecified"),
        "mode": benchmark["mode"],
        "qrels_path": relative(root, qrels_path),
        "benchmark_id": qrels.get("benchmark_id"),
        "benchmark_kind": qrels.get("benchmark_kind"),
        "query_count": len(qrels.get("queries") or []),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "lexical_k": lexical_k or int(benchmark.get("lexical_k", candidate_k)),
        "vector_k": vector_k or int(benchmark.get("vector_k", candidate_k)),
        "pooled_candidates": pooled_candidates,
        "fusion": str(benchmark.get("fusion", fusion)),
        "document_format": str(benchmark.get("document_format", document_format)),
        "semantic": compact_semantic(semantic, top_k),
        "rerank": compact_rerank(rerank, top_k),
        "query_topics": query_topics(qrels, benchmark.get("query_topics") or {}),
    }


def compact_semantic(report: dict[str, Any], k: int) -> dict[str, Any]:
    return {
        "status": provider_status(report.get("providers", [])),
        "leaderboard": report.get("leaderboard", []),
        "best": first_or_empty(report.get("leaderboard", [])),
        "unavailable": unavailable(report.get("providers", [])),
        "primary_metric": f"nDCG@{k}",
    }


def compact_rerank(report: dict[str, Any], k: int) -> dict[str, Any]:
    rerankers = report.get("rerankers", [])
    ok_reports = [row for row in rerankers if row.get("status") == "ok"]
    return {
        "status": provider_status(rerankers),
        "candidate_generation": report.get("candidate_generation", {}),
        "leaderboard": report.get("leaderboard", []),
        "best": first_or_empty(report.get("leaderboard", [])),
        "unavailable": unavailable(rerankers),
        "candidate_recall": candidate_recall_summary(ok_reports),
        "score_observability": score_observability_summary(ok_reports),
        "primary_metric": f"nDCG@{k}",
    }


def candidate_recall_summary(provider_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_query_id: dict[str, dict[str, Any]] = {}
    for report in provider_reports:
        for query in report.get("queries", []):
            by_query_id.setdefault(str(query.get("id")), query.get("candidate_recall", {}))
    rows = list(by_query_id.values())
    recalls = [float(row.get("candidate_recall", 0.0)) for row in rows]
    return {
        "query_count": len(rows),
        "mean": average(recalls),
        "min": round(min(recalls), 6) if recalls else 0.0,
        "miss_count": sum(1 for value in recalls if value < 1.0),
    }


def score_observability_summary(provider_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        query.get("score_observability", {})
        for report in provider_reports
        for query in report.get("queries", [])
    ]
    gaps = [float(row.get("relevant_nonrelevant_score_gap", 0.0)) for row in rows]
    low_confidence = [float(row.get("low_confidence_top_k_count", 0.0)) for row in rows]
    return {
        "query_count": len(rows),
        "mean_relevant_nonrelevant_score_gap": average(gaps),
        "mean_low_confidence_top_k_count": average(low_confidence),
    }


def query_topics(qrels: dict[str, Any], configured_topics: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in qrels.get("queries", []):
        topic = configured_topics.get(str(query.get("id"))) or (query.get("metadata") or {}).get("topic") or "unspecified"
        counts[str(topic)] = counts.get(str(topic), 0) + 1
    return dict(sorted(counts.items()))


def topic_summary(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        topic = benchmark["topic"]
        row = rows.setdefault(
            topic,
            {
                "benchmark_count": 0,
                "query_count": 0,
                "semantic_best": [],
                "rerank_best": [],
                "candidate_recall_values": [],
            },
        )
        row["benchmark_count"] += 1
        row["query_count"] += int(benchmark.get("query_count", 0))
        row["semantic_best"].append(benchmark["semantic"]["best"])
        row["rerank_best"].append(benchmark["rerank"]["best"])
        row["candidate_recall_values"].append(float(benchmark["rerank"]["candidate_recall"]["mean"]))
    return {
        topic: {
            **{key: value for key, value in row.items() if key != "candidate_recall_values"},
            "mean_candidate_recall": average(row["candidate_recall_values"]),
        }
        for topic, row in sorted(rows.items())
    }


def query_topic_summary(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        recall = float(benchmark["rerank"]["candidate_recall"]["mean"])
        for topic, count in benchmark.get("query_topics", {}).items():
            row = rows.setdefault(topic, {"query_count": 0, "benchmarks": [], "candidate_recall_values": []})
            row["query_count"] += int(count)
            row["benchmarks"].append(benchmark["id"])
            row["candidate_recall_values"].append(recall)
    return {
        topic: {
            "query_count": row["query_count"],
            "benchmarks": sorted(set(row["benchmarks"])),
            "mean_candidate_recall": average(row["candidate_recall_values"]),
        }
        for topic, row in sorted(rows.items())
    }


def suite_leaderboards(benchmarks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "semantic": aggregate_provider_scores(
            [benchmark["semantic"]["leaderboard"] for benchmark in benchmarks]
        ),
        "rerank": aggregate_provider_scores(
            [benchmark["rerank"]["leaderboard"] for benchmark in benchmarks]
        ),
    }


def model_matrix(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = provider_matrix([benchmark["semantic"] for benchmark in benchmarks])
    rerank = provider_matrix([benchmark["rerank"] for benchmark in benchmarks])
    return {
        "status": model_matrix_status(semantic, rerank),
        "semantic": semantic,
        "rerank": rerank,
        "interpretation": (
            "Hosted model candidates are core benchmark participants when keys are configured; "
            "unavailable rows mean the comparison is incomplete, not that the model lost."
        ),
    }


def provider_matrix(reports: list[dict[str, Any]]) -> dict[str, Any]:
    available: dict[str, int] = {}
    unavailable_rows: dict[str, set[str]] = {}
    for report in reports:
        for item in report.get("leaderboard", []):
            provider = str(item.get("provider_id") or item.get("provider"))
            available[provider] = available.get(provider, 0) + 1
        for item in report.get("unavailable", []):
            provider = str(item.get("provider_id") or item.get("provider"))
            reason = str(item.get("reason"))
            unavailable_rows.setdefault(provider, set()).add(reason)
    return {
        "available": dict(sorted(available.items())),
        "unavailable": [
            {"provider": provider, "reasons": sorted(reasons)}
            for provider, reasons in sorted(unavailable_rows.items())
        ],
    }


def model_matrix_status(semantic: dict[str, Any], rerank: dict[str, Any]) -> str:
    if semantic["unavailable"] or rerank["unavailable"]:
        return "partial_external_models_unavailable"
    return "complete"


def aggregate_provider_scores(leaderboards: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: dict[str, list[float]] = {}
    for leaderboard in leaderboards:
        for item in leaderboard:
            provider = str(item.get("provider_id") or item.get("provider"))
            rows.setdefault(provider, []).append(float(item.get("primary_score", 0.0)))
    ranked = sorted(
        (
            {"provider": provider, "benchmark_count": len(scores), "mean_primary_score": average(scores)}
            for provider, scores in rows.items()
        ),
        key=lambda item: item["mean_primary_score"],
        reverse=True,
    )
    return [{"rank": index, **item} for index, item in enumerate(ranked, start=1)]


def suite_status(benchmarks: list[dict[str, Any]], thresholds: dict[str, Any]) -> str:
    if not benchmarks:
        return "automated_fail"
    min_candidate_recall = float(thresholds.get("min_candidate_recall", 0.8))
    min_rerank_recall = float(thresholds.get("min_rerank_recall_at_k", 0.6))
    for benchmark in benchmarks:
        if benchmark["semantic"]["status"] != "ok" or benchmark["rerank"]["status"] != "ok":
            return "automated_fail"
        if benchmark["rerank"]["candidate_recall"]["mean"] < min_candidate_recall:
            return "automated_fail"
        best_rerank = benchmark["rerank"]["best"]
        if float(best_rerank.get("Recall@k", 0.0)) < min_rerank_recall:
            return "automated_fail"
    return "automated_pass_human_review_required"


def provider_status(rows: list[dict[str, Any]]) -> str:
    return "ok" if any(row.get("status") == "ok" for row in rows) else "unavailable"


def unavailable(rows: list[dict[str, Any]]) -> list[dict[str, str | None]]:
    return [
        {"provider": row.get("provider"), "reason": row.get("reason")}
        for row in rows
        if row.get("status") != "ok"
    ]


def first_or_empty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def human_review_boundary() -> dict[str, Any]:
    return {
        "status": "required_for_release_claims",
        "not_required_for": [
            "automated regression testing",
            "public-qrels smoke validation",
            "retrieval plumbing checks",
            "candidate-recall diagnostics",
        ],
        "required_before": [
            "shipping an industry-pilot release",
            "claiming production unsupported-claim rates",
            "declaring a durable model winner",
            "publishing benchmark-quality claims beyond the configured public qrels",
        ],
    }


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not suite.get("benchmarks"):
        raise ValueError("Benchmark suite must contain at least one benchmark.")
    return suite


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    benchmarks = "\n".join(
        (
            f"- `{item['id']}` topic=`{item['topic']}` queries={item['query_count']} "
            f"candidate_recall={item['rerank']['candidate_recall']['mean']} "
            f"best_rerank={item['rerank']['best'].get('provider', 'none')}"
        )
        for item in report["benchmarks"]
    )
    topics = "\n".join(
        f"- `{topic}` benchmarks={row['benchmark_count']} queries={row['query_count']} "
        f"mean_candidate_recall={row['mean_candidate_recall']}"
        for topic, row in report["topic_summary"].items()
    )
    query_topics = "\n".join(
        f"- `{topic}` queries={row['query_count']} mean_candidate_recall={row['mean_candidate_recall']}"
        for topic, row in report["query_topic_summary"].items()
    )
    return f"""# CANON Automated Benchmark Suite

Status: **{report['status']}**

Suite: `{report['suite_id']}`

Model matrix: `{report['model_matrix']['status']}`

This is an automated pre-human benchmark. Human-reviewed qrels and brief review
remain required before release-quality claims.

## Benchmarks

{benchmarks}

## Topics

{topics}

## Query Topics

{query_topics}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-human-review automated benchmark suite.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--model-providers", default=None)
    parser.add_argument("--rerankers", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--lexical-k", type=int, default=None)
    parser.add_argument("--vector-k", type=int, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--document-format", choices=["plain", "structured"], default=None)
    parser.add_argument("--pooled-candidates", action="store_true", default=None)
    parser.add_argument("--single-policy-candidates", action="store_false", dest="pooled_candidates")
    args = parser.parse_args()
    print(
        json.dumps(
            run_automated_benchmark_suite(
                suite_path=Path(args.suite),
                model_providers=parse_providers(args.model_providers),
                rerankers=parse_optional_list(args.rerankers),
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                fusion=args.fusion,
                document_format=args.document_format,
                pooled_candidates=args.pooled_candidates,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


def parse_optional_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [piece.strip() for piece in value.split(",") if piece.strip()]
    return parsed or None


if __name__ == "__main__":
    main()

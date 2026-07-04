from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.harness import load_gold, rank_overlap
from canon.eval.methods import load_method_set, method_specs, summarize_method_run, synthesize_for_method


def run_perturbation_suite(
    mode: str,
    method_set_path: Path | None = None,
    query_limit: int = 8,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = method_specs(method_set)
    queries = load_gold(settings.root / "gold" / "seed_queries.json")[:query_limit]
    query_reports = []
    for query in queries:
        variants = query_variants(query["query"])
        method_runs = []
        for spec in specs:
            base = summarize_method_run(query, spec, synthesize_for_method(query["query"], spec, mode))
            variant_runs = []
            for variant in variants:
                variant_query = {**query, "query": variant["query"], "id": f"{query['id']}:{variant['id']}"}
                run = summarize_method_run(variant_query, spec, synthesize_for_method(variant["query"], spec, mode))
                variant_runs.append(
                    {
                        "variant_id": variant["id"],
                        "variant_query": variant["query"],
                        "rank_overlap": rank_overlap_rows(base["retrieved_chunk_ids"], run["retrieved_chunk_ids"]),
                        "support_confidence_delta": round(run["metrics"]["support_confidence"] - base["metrics"]["support_confidence"], 6),
                        "context_relevance_delta": round(run["metrics"]["context_relevance"] - base["metrics"]["context_relevance"], 6),
                    }
                )
            method_runs.append({"method_id": spec.id, "label": spec.label, "base_query": query["query"], "variants": variant_runs})
        query_reports.append({"id": query["id"], "query": query["query"], "method_runs": method_runs})
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "query_count": len(query_reports),
        "queries": query_reports,
        "method_summary": method_summary(query_reports),
    }
    report["leaderboard"] = leaderboard(report["method_summary"])
    write_json(settings.reports_dir / f"perturbation_eval_{mode}_{method_set['method_set_id']}.json", report)
    return report


def query_variants(query: str) -> list[dict]:
    stripped = query.rstrip(" ?.")
    variants = [
        {"id": "prefix_context", "query": f"In this corpus, {stripped}?"},
        {"id": "evidence_request", "query": f"What evidence supports or challenges this question: {stripped}?"},
    ]
    replacements = {
        "What does the literature say about": "What does research suggest about",
        "How do": "In what ways do",
        "How effective are": "To what extent are",
        "Which": "What",
        "Do": "To what extent do",
    }
    paraphrase = query
    for source, target in replacements.items():
        if paraphrase.startswith(source):
            paraphrase = target + paraphrase[len(source):]
            break
    if paraphrase != query:
        variants.append({"id": "light_paraphrase", "query": paraphrase})
    return variants


def rank_overlap_rows(left_ids: list[str], right_ids: list[str]) -> float:
    return rank_overlap(
        [{"chunk_id": chunk_id, "rank": index} for index, chunk_id in enumerate(left_ids, start=1)],
        [{"chunk_id": chunk_id, "rank": index} for index, chunk_id in enumerate(right_ids, start=1)],
    )


def method_summary(query_reports: list[dict]) -> dict:
    method_ids = sorted({run["method_id"] for query in query_reports for run in query["method_runs"]})
    summary = {}
    for method_id in method_ids:
        variant_rows = [
            variant
            for query in query_reports
            for run in query["method_runs"]
            if run["method_id"] == method_id
            for variant in run["variants"]
        ]
        summary[method_id] = {
            "variant_count": len(variant_rows),
            "average_rank_overlap": average(row["rank_overlap"] for row in variant_rows),
            "minimum_rank_overlap": min((row["rank_overlap"] for row in variant_rows), default=0.0),
            "average_abs_support_delta": average(abs(row["support_confidence_delta"]) for row in variant_rows),
            "average_abs_context_relevance_delta": average(abs(row["context_relevance_delta"]) for row in variant_rows),
        }
    return summary


def leaderboard(summary: dict) -> list[dict]:
    return [
        {"rank": index, "method_id": method_id, **row}
        for index, (method_id, row) in enumerate(
            sorted(summary.items(), key=lambda item: (item[1]["average_rank_overlap"], -item[1]["average_abs_support_delta"]), reverse=True),
            start=1,
        )
    ]


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query perturbation robustness tests.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--query-limit", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_perturbation_suite(args.mode, Path(args.method_set) if args.method_set else None, args.query_limit), indent=2))


if __name__ == "__main__":
    main()

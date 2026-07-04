from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.harness import load_gold, rank_overlap
from canon.eval.methods import load_method_set, method_specs, synthesize_for_method
from canon.retrieval.tokenize import tokenize


GENERIC_TERMS = {
    "about",
    "answer",
    "corpus",
    "does",
    "evidence",
    "literature",
    "research",
    "says",
    "this",
    "what",
    "where",
    "which",
}


def run_diversity_audit(
    mode: str,
    method_set_path: Path | None = None,
    diverse_method_id: str = "diverse_k5_template",
    baseline_method_id: str = "lexical_k5_template",
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = {spec.id: spec for spec in method_specs(method_set)}
    if diverse_method_id not in specs:
        raise ValueError(f"Unknown diverse method id: {diverse_method_id}")
    if baseline_method_id not in specs:
        raise ValueError(f"Unknown baseline method id: {baseline_method_id}")
    queries = load_gold(settings.root / "gold" / "seed_queries.json")
    query_reports = []
    for query in queries:
        baseline = synthesize_for_method(query["query"], specs[baseline_method_id], mode)
        diverse = synthesize_for_method(query["query"], specs[diverse_method_id], mode)
        query_reports.append(
            analyze_diversity_gain(
                query=query,
                baseline_results=baseline["retrieval"]["results"],
                diverse_results=diverse["retrieval"]["results"],
                baseline_method_id=baseline_method_id,
                diverse_method_id=diverse_method_id,
            )
        )
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "diverse_method_id": diverse_method_id,
        "baseline_method_id": baseline_method_id,
        "query_count": len(query_reports),
        "aggregate": aggregate_reports(query_reports),
        "queries": query_reports,
    }
    write_json(
        settings.reports_dir
        / f"diversity_audit_{mode}_{method_set['method_set_id']}_{diverse_method_id}_vs_{baseline_method_id}.json",
        report,
    )
    return report


def analyze_diversity_gain(
    query: dict,
    baseline_results: list[dict],
    diverse_results: list[dict],
    baseline_method_id: str,
    diverse_method_id: str,
) -> dict:
    focus_terms = query_focus_terms(query["query"])
    baseline_ids = {result["chunk_id"] for result in baseline_results}
    diverse_ids = {result["chunk_id"] for result in diverse_results}
    added = [result for result in diverse_results if result["chunk_id"] not in baseline_ids]
    removed = [result for result in baseline_results if result["chunk_id"] not in diverse_ids]
    baseline_clusters = clusters_for_results(baseline_results)
    diverse_clusters = clusters_for_results(diverse_results)
    added_new_cluster = [
        result
        for result in added
        if result.get("cluster_id") is not None and result.get("cluster_id") not in baseline_clusters
    ]
    useful_additions = [
        result
        for result in added_new_cluster
        if query["type"] != "off_topic" and is_useful_breadth_addition(result, focus_terms)
    ]
    noisy_additions = [
        result
        for result in added
        if result_focus_coverage(result, focus_terms) == 0.0
        and component_value(result, "relevance") < 0.35
    ]
    overlap = rank_overlap(
        ranked_ids([result["chunk_id"] for result in baseline_results]),
        ranked_ids([result["chunk_id"] for result in diverse_results]),
    )
    cluster_delta = len(diverse_clusters) - len(baseline_clusters)
    report = {
        "query_id": query["id"],
        "query_type": query["type"],
        "query": query["query"],
        "baseline_method_id": baseline_method_id,
        "diverse_method_id": diverse_method_id,
        "focus_terms": focus_terms,
        "rank_overlap": overlap,
        "baseline_cluster_count": len(baseline_clusters),
        "diverse_cluster_count": len(diverse_clusters),
        "cluster_delta": cluster_delta,
        "added_count": len(added),
        "removed_count": len(removed),
        "useful_breadth_count": len(useful_additions),
        "noise_addition_count": len(noisy_additions),
        "breadth_precision": safe_ratio(len(useful_additions), len(added)),
        "noise_rate": safe_ratio(len(noisy_additions), len(added)),
        "average_added_focus_coverage": average(result_focus_coverage(result, focus_terms) for result in added),
        "average_added_relevance": average(component_value(result, "relevance") for result in added),
        "added_titles": compact_results(added, focus_terms),
        "removed_titles": compact_results(removed, focus_terms),
    }
    report["verdict"] = diversity_verdict(report)
    report["interpretation"] = diversity_interpretation(report)
    return report


def diversity_verdict(report: dict) -> str:
    if report.get("query_type") == "off_topic" and report["cluster_delta"] > 0:
        return "off_topic_breadth_risk"
    if report["cluster_delta"] <= 0:
        return "no_diversity_gain"
    if report["breadth_precision"] >= 0.5 and report["noise_rate"] <= 0.25:
        return "useful_breadth"
    if report["noise_rate"] >= 0.5:
        return "noisy_breadth"
    return "mixed_breadth"


def diversity_interpretation(report: dict) -> str:
    verdict = report["verdict"]
    if verdict == "useful_breadth":
        return (
            "Diversity-first added new graph-cluster evidence with enough query-focus overlap "
            "and low estimated noise, so this is counted as useful breadth."
        )
    if verdict == "mixed_breadth":
        return (
            "Diversity-first increased graph-cluster breadth, but fewer than half of the "
            "added items met the focus-coverage criterion."
        )
    if verdict == "off_topic_breadth_risk":
        return (
            "The query is off-topic for this corpus; additional breadth may look relevant by "
            "surface terms but should not be treated as scientific support."
        )
    if verdict == "noisy_breadth":
        return (
            "Diversity-first added new evidence with high estimated noise, indicating breadth "
            "that may distract from the query."
        )
    return "Diversity-first did not increase graph-cluster breadth over the baseline for this query."


def aggregate_reports(query_reports: list[dict]) -> dict:
    verdict_counts: dict[str, int] = {}
    for report in query_reports:
        verdict_counts[report["verdict"]] = verdict_counts.get(report["verdict"], 0) + 1
    return {
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "average_cluster_delta": average(report["cluster_delta"] for report in query_reports),
        "average_rank_overlap": average(report["rank_overlap"] for report in query_reports),
        "average_breadth_precision": average(report["breadth_precision"] for report in query_reports),
        "average_noise_rate": average(report["noise_rate"] for report in query_reports),
        "queries_with_useful_breadth": [
            report["query_id"]
            for report in query_reports
            if report["verdict"] == "useful_breadth"
        ],
        "queries_with_noisy_breadth": [
            report["query_id"]
            for report in query_reports
            if report["verdict"] == "noisy_breadth"
        ],
    }


def query_focus_terms(query: str) -> list[str]:
    return sorted(
        {
            token
            for token in tokenize(query)
            if len(token) >= 4 and token not in GENERIC_TERMS
        }
    )


def is_useful_breadth_addition(result: dict, focus_terms: list[str]) -> bool:
    if focus_terms:
        return result_focus_coverage(result, focus_terms) >= 0.25
    return component_value(result, "relevance") >= 0.5


def result_focus_coverage(result: dict, focus_terms: list[str]) -> float:
    if not focus_terms:
        return 0.0
    text = " ".join([result.get("title") or "", result.get("preview") or ""])
    result_tokens = set(tokenize(text))
    return round(len([term for term in focus_terms if term in result_tokens]) / len(focus_terms), 6)


def clusters_for_results(results: list[dict]) -> set[int]:
    return {
        int(result["cluster_id"])
        for result in results
        if result.get("cluster_id") is not None
    }


def component_value(result: dict, component: str) -> float:
    return float((result.get("components") or {}).get(component, 0.0))


def compact_results(results: list[dict], focus_terms: list[str] | None = None) -> list[dict]:
    focus_terms = focus_terms or []
    return [
        {
            "rank": result["rank"],
            "chunk_id": result["chunk_id"],
            "title": result["title"],
            "cluster_id": result.get("cluster_id"),
            "relevance": component_value(result, "relevance"),
            "semantic_similarity": component_value(result, "semantic_similarity"),
            "focus_coverage": result_focus_coverage(result, focus_terms),
        }
        for result in results
    ]


def ranked_ids(chunk_ids: list[str]) -> list[dict]:
    return [{"chunk_id": chunk_id, "rank": index} for index, chunk_id in enumerate(chunk_ids, start=1)]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cli_summary(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "method_set_id": report["method_set_id"],
        "diverse_method_id": report["diverse_method_id"],
        "baseline_method_id": report["baseline_method_id"],
        "query_count": report["query_count"],
        "aggregate": report["aggregate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit diversity-first retrieval against a baseline method.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--diverse-method-id", default="diverse_k5_template")
    parser.add_argument("--baseline-method-id", default="lexical_k5_template")
    args = parser.parse_args()
    print(
        json.dumps(
            cli_summary(
                run_diversity_audit(
                    mode=args.mode,
                    method_set_path=Path(args.method_set) if args.method_set else None,
                    diverse_method_id=args.diverse_method_id,
                    baseline_method_id=args.baseline_method_id,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

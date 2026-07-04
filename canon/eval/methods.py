from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from canon.config import load_settings
from canon.eval.harness import load_gold, rank_overlap
from canon.synthesis.answer import synthesize


@dataclass(frozen=True)
class MethodSpec:
    id: str
    label: str
    policy: str
    top_k: int
    generator: str
    generator_model: str | None = None


def load_method_set(path: Path | None = None) -> dict:
    settings = load_settings()
    method_path = path or settings.root / "conf" / "methods" / "baseline_methods.json"
    return json.loads(method_path.read_text(encoding="utf-8"))


def method_specs(method_set: dict) -> list[MethodSpec]:
    return [
        MethodSpec(
            id=item["id"],
            label=item["label"],
            policy=item["policy"],
            top_k=int(item["top_k"]),
            generator=item.get("generator") or "template",
            generator_model=item.get("generator_model"),
        )
        for item in method_set["methods"]
    ]


def evaluate_methods(
    mode: str,
    method_set_path: Path | None = None,
    query_limit: int | None = None,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = method_specs(method_set)
    queries = load_gold(settings.root / "gold" / "seed_queries.json")
    if query_limit:
        queries = queries[:query_limit]
    query_reports = []
    for query in queries:
        runs = [
            summarize_method_run(query, spec, synthesize_for_method(query["query"], spec, mode))
            for spec in specs
        ]
        query_reports.append(
            {
                "id": query["id"],
                "type": query["type"],
                "query": query["query"],
                "importance_expectation": query["importance_expectation"],
                "method_runs": runs,
                "rank_matrix": rank_matrix(runs),
            }
        )
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "query_count": len(query_reports),
        "methods": [spec.__dict__ for spec in specs],
        "score_weights": method_set["score_weights"],
        "queries": query_reports,
        "method_summary": method_summary(query_reports, specs, method_set["score_weights"]),
        "pairwise": pairwise_summary(query_reports, specs),
    }
    report["leaderboard"] = leaderboard(report["method_summary"])
    write_json(settings.reports_dir / f"method_eval_{mode}_{method_set['method_set_id']}.json", report)
    return report


def synthesize_for_method(query: str, spec: MethodSpec, mode: str) -> dict:
    return synthesize(
        query=query,
        policy=spec.policy,
        mode=mode,
        top_k=spec.top_k,
        generator_name=spec.generator,
        generator_model=spec.generator_model,
    )


def summarize_method_run(query: dict, spec: MethodSpec, synthesis: dict) -> dict:
    evidence = synthesis["evidence"]
    retrieval_results = synthesis["retrieval"]["results"]
    components = synthesis["importance_summary"]["average_components"]
    metrics = {
        "citation_count": len(synthesis["citations"]),
        "claim_citation_count": sum(1 for item in evidence if item["claim"]),
        "claim_coverage": safe_ratio(sum(1 for item in evidence if item["claim"]), len(evidence)),
        "conflict_note_count": len(synthesis["conflict_notes"]),
        "conflict_awareness": 1.0 if synthesis["conflict_notes"] else 0.0,
        "distinct_works": synthesis["importance_summary"]["distinct_works"],
        "distinct_sources": synthesis["importance_summary"]["distinct_sources"],
        "distinct_clusters": synthesis["importance_summary"]["distinct_clusters"],
        "cluster_diversity": safe_ratio(synthesis["importance_summary"]["distinct_clusters"], len(evidence)),
        "source_diversity": safe_ratio(synthesis["importance_summary"]["distinct_sources"], len(evidence)),
        "context_relevance": float(components.get("relevance", 0.0)),
        "semantic_alignment": float(components.get("semantic_similarity", 0.0)),
        "citation_support": citation_support(evidence),
        "limitations_count": len(synthesis["limitations"]),
        "support_confidence": float(synthesis.get("support_assessment", {}).get("support_confidence", 0.0)),
        "weak_support_flag": 1.0 if synthesis.get("support_assessment", {}).get("support_level") == "weak" else 0.0,
    }
    return {
        "method_id": spec.id,
        "label": spec.label,
        "policy": spec.policy,
        "top_k": spec.top_k,
        "generator": synthesis["generator"],
        "query_id": query["id"],
        "answer": synthesis["answer"],
        "metrics": metrics,
        "citations": synthesis["citations"],
        "retrieved_chunk_ids": [result["chunk_id"] for result in retrieval_results],
        "top_titles": [citation["title"] for citation in synthesis["citations"][:3]],
    }


def citation_support(evidence: list[dict]) -> float:
    if not evidence:
        return 0.0
    supported = 0
    for item in evidence:
        if item["claim"] or item["preview"]:
            supported += 1
    return round(supported / len(evidence), 6)


def method_summary(query_reports: list[dict], specs: list[MethodSpec], weights: dict[str, float]) -> dict:
    summary = {}
    for spec in specs:
        runs = [
            run
            for query in query_reports
            for run in query["method_runs"]
            if run["method_id"] == spec.id
        ]
        metric_names = sorted({name for run in runs for name in run["metrics"]})
        averages = {
            name: average(run["metrics"].get(name, 0.0) for run in runs)
            for name in metric_names
        }
        summary[spec.id] = {
            "label": spec.label,
            "policy": spec.policy,
            "top_k": spec.top_k,
            "generator": spec.generator,
            "query_count": len(runs),
            "average_metrics": averages,
            "composite_score": composite_score(averages, weights),
        }
    return summary


def composite_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    denominator = sum(weight for weight in weights.values() if weight > 0)
    if denominator <= 0:
        return 0.0
    score = sum(metrics.get(name, 0.0) * weight for name, weight in weights.items()) / denominator
    return round(score, 6)


def leaderboard(summary: dict[str, dict]) -> list[dict]:
    return [
        {
            "rank": index,
            "method_id": method_id,
            "label": row["label"],
            "composite_score": row["composite_score"],
            "average_metrics": row["average_metrics"],
        }
        for index, (method_id, row) in enumerate(
            sorted(summary.items(), key=lambda item: item[1]["composite_score"], reverse=True),
            start=1,
        )
    ]


def pairwise_summary(query_reports: list[dict], specs: list[MethodSpec]) -> list[dict]:
    rows = []
    for left in specs:
        for right in specs:
            if left.id >= right.id:
                continue
            overlaps = []
            score_deltas = []
            for query in query_reports:
                left_run = run_by_id(query["method_runs"], left.id)
                right_run = run_by_id(query["method_runs"], right.id)
                overlaps.append(
                    rank_overlap(
                        ranked_ids(left_run["retrieved_chunk_ids"]),
                        ranked_ids(right_run["retrieved_chunk_ids"]),
                    )
                )
                score_deltas.append(
                    right_run["metrics"]["cluster_diversity"] - left_run["metrics"]["cluster_diversity"]
                )
            rows.append(
                {
                    "left_method_id": left.id,
                    "right_method_id": right.id,
                    "average_retrieval_overlap": average(overlaps),
                    "average_cluster_diversity_delta_right_minus_left": average(score_deltas),
                }
            )
    return rows


def run_by_id(runs: list[dict], method_id: str) -> dict:
    for run in runs:
        if run["method_id"] == method_id:
            return run
    raise KeyError(method_id)


def ranked_ids(chunk_ids: list[str]) -> list[dict]:
    return [{"chunk_id": chunk_id, "rank": index} for index, chunk_id in enumerate(chunk_ids, start=1)]


def rank_matrix(runs: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for run in runs:
        for rank, chunk_id in enumerate(run["retrieved_chunk_ids"], start=1):
            rows.setdefault(chunk_id, {"chunk_id": chunk_id, "ranks": {}})
            rows[chunk_id]["ranks"][run["method_id"]] = rank
    return sorted(rows.values(), key=lambda row: min(row["ranks"].values()))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(mean(float(value) for value in values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate named CANON RAG methods.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    args = parser.parse_args()
    method_set_path = Path(args.method_set) if args.method_set else None
    print(json.dumps(evaluate_methods(args.mode, method_set_path, args.query_limit), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.eval.methods import load_method_set, method_specs
from canon.retrieval.experiment import run as run_retrieval


DEFAULT_METRICS_K = 10


def evaluate_external_ir(
    mode: str,
    qrels_path: Path | None = None,
    method_set_path: Path | None = None,
    k: int = DEFAULT_METRICS_K,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    qrels = load_qrels(qrels_path or settings.root / "gold" / "ir_qrels_social_science_ir_v1_harvest10.json")
    specs = method_specs(method_set)
    query_reports = []
    for query in qrels["queries"]:
        method_runs = []
        for spec in specs:
            retrieval = run_retrieval(query["query"], spec.policy, mode, max(k, spec.top_k))
            ranked_ids = [result["chunk_id"] for result in retrieval["results"]]
            metrics = evaluate_ranking(ranked_ids, query["relevant"], k)
            method_runs.append(
                {
                    "method_id": spec.id,
                    "label": spec.label,
                    "policy": spec.policy,
                    "top_k": max(k, spec.top_k),
                    "metrics": metrics,
                    "retrieved_chunk_ids": ranked_ids[:k],
                    "relevant_hit_ids": [chunk_id for chunk_id in ranked_ids[:k] if chunk_id in query["relevant"]],
                }
            )
        query_reports.append(
            {
                "id": query["id"],
                "query": query["query"],
                "qrel_count": len(query["relevant"]),
                "method_runs": method_runs,
            }
        )
    report = {
        "benchmark_id": qrels["benchmark_id"],
        "benchmark_kind": qrels.get("benchmark_kind", "internal_qrels"),
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "metric_k": k,
        "query_count": len(query_reports),
        "qrels_path": str((qrels_path or settings.root / "gold" / "ir_qrels_social_science_ir_v1_harvest10.json")).replace("\\", "/"),
        "queries": query_reports,
        "method_summary": method_summary(query_reports, [spec.id for spec in specs], k),
    }
    report["leaderboard"] = leaderboard(report["method_summary"], f"nDCG@{k}")
    output_path = settings.reports_dir / f"external_ir_{qrels['benchmark_id']}_{mode}_{method_set['method_set_id']}.json"
    write_json(output_path, report)
    return report


def load_qrels(path: Path) -> dict:
    qrels = json.loads(path.read_text(encoding="utf-8"))
    for query in qrels.get("queries", []):
        query["relevant"] = {str(chunk_id): float(relevance) for chunk_id, relevance in query["relevant"].items()}
    return qrels


def method_summary(query_reports: list[dict], method_ids: list[str], k: int) -> dict:
    metric_names = [f"nDCG@{k}", f"Recall@{k}", f"MAP@{k}", f"MRR@{k}"]
    summary = {}
    for method_id in method_ids:
        rows = [
            run["metrics"]
            for query in query_reports
            for run in query["method_runs"]
            if run["method_id"] == method_id
        ]
        summary[method_id] = {metric: mean_metric(rows, metric) for metric in metric_names}
    return summary


def leaderboard(summary: dict[str, dict], primary_metric: str) -> list[dict]:
    return [
        {"rank": index, "method_id": method_id, **metrics}
        for index, (method_id, metrics) in enumerate(
            sorted(summary.items(), key=lambda item: item[1].get(primary_metric, 0.0), reverse=True),
            start=1,
        )
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEIR/TREC-style retrieval evaluation against qrels.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_METRICS_K)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_external_ir(
                mode=args.mode,
                qrels_path=Path(args.qrels) if args.qrels else None,
                method_set_path=Path(args.method_set) if args.method_set else None,
                k=args.k,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

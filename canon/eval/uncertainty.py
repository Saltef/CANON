from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from canon.config import load_settings


def bootstrap_external_ir(
    mode: str,
    method_set_id: str = "baseline_methods_v1",
    benchmark_id: str = "internal_social_science_ir_qrels_v1",
    metric: str = "nDCG@10",
    samples: int = 500,
    seed: int = 17,
) -> dict:
    settings = load_settings()
    report_path = settings.reports_dir / f"external_ir_{benchmark_id}_{mode}_{method_set_id}.json"
    external = json.loads(report_path.read_text(encoding="utf-8"))
    report = bootstrap_report(external, metric=metric, samples=samples, seed=seed)
    output_path = settings.reports_dir / f"bootstrap_ir_{benchmark_id}_{mode}_{method_set_id}.json"
    write_json(output_path, report)
    return report


def bootstrap_report(external_ir: dict, metric: str, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    queries = external_ir.get("queries") or []
    method_ids = sorted(external_ir.get("method_summary", {}).keys())
    observed = {
        method_id: average(query_metric(query, method_id, metric) for query in queries)
        for method_id in method_ids
    }
    draws = {method_id: [] for method_id in method_ids}
    winner_counts = {method_id: 0 for method_id in method_ids}
    if queries:
        for _ in range(samples):
            sample = [queries[rng.randrange(len(queries))] for _ in queries]
            scores = {
                method_id: average(query_metric(query, method_id, metric) for query in sample)
                for method_id in method_ids
            }
            for method_id, value in scores.items():
                draws[method_id].append(value)
            winner = max(scores, key=scores.get)
            winner_counts[winner] += 1
    intervals = {
        method_id: {
            "observed_mean": round(observed.get(method_id, 0.0), 6),
            "ci_low": percentile(draws[method_id], 0.025),
            "ci_high": percentile(draws[method_id], 0.975),
            "winner_probability": round(winner_counts[method_id] / samples, 6) if samples else 0.0,
        }
        for method_id in method_ids
    }
    leaderboard = [
        {"rank": index, "method_id": method_id, **row}
        for index, (method_id, row) in enumerate(
            sorted(intervals.items(), key=lambda item: item[1]["observed_mean"], reverse=True),
            start=1,
        )
    ]
    top = leaderboard[0] if leaderboard else None
    runner_up = leaderboard[1] if len(leaderboard) > 1 else None
    return {
        "benchmark_id": external_ir.get("benchmark_id"),
        "mode": external_ir.get("mode"),
        "method_set_id": external_ir.get("method_set_id"),
        "metric": metric,
        "query_count": len(queries),
        "samples": samples,
        "seed": seed,
        "intervals": intervals,
        "leaderboard": leaderboard,
        "topline": {
            "winner": top["method_id"] if top else None,
            "runner_up": runner_up["method_id"] if runner_up else None,
            "observed_top_margin": round(top["observed_mean"] - runner_up["observed_mean"], 6) if top and runner_up else 0.0,
            "winner_probability": top["winner_probability"] if top else 0.0,
        },
    }


def query_metric(query: dict, method_id: str, metric: str) -> float:
    for run in query.get("method_runs", []):
        if run.get("method_id") == method_id:
            return float(run.get("metrics", {}).get(metric, 0.0))
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[index], 6)


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap uncertainty for CANON IR metrics.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    parser.add_argument("--benchmark-id", default="internal_social_science_ir_qrels_v1")
    parser.add_argument("--metric", default="nDCG@10")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(bootstrap_external_ir(args.mode, args.method_set_id, args.benchmark_id, args.metric, args.samples, args.seed), indent=2))


if __name__ == "__main__":
    main()

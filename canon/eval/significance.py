from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

from canon.config import load_settings


def run_paired_significance(
    mode: str,
    method_set_id: str = "baseline_methods_v1",
    benchmark_id: str = "internal_social_science_ir_qrels_v1",
    metric: str = "nDCG@10",
    samples: int = 1000,
    seed: int = 23,
) -> dict:
    settings = load_settings()
    path = settings.reports_dir / f"external_ir_{benchmark_id}_{mode}_{method_set_id}.json"
    external_ir = json.loads(path.read_text(encoding="utf-8"))
    report = paired_significance_report(external_ir, metric, samples, seed)
    output_path = settings.reports_dir / f"paired_significance_{benchmark_id}_{mode}_{method_set_id}.json"
    write_json(output_path, report)
    return report


def paired_significance_report(external_ir: dict, metric: str, samples: int, seed: int) -> dict:
    queries = external_ir.get("queries") or []
    method_ids = sorted(external_ir.get("method_summary", {}).keys())
    rows = []
    for left_id, right_id in combinations(method_ids, 2):
        left_scores = [query_metric(query, left_id, metric) for query in queries]
        right_scores = [query_metric(query, right_id, metric) for query in queries]
        diffs = [left - right for left, right in zip(left_scores, right_scores, strict=True)]
        observed = average(diffs)
        rows.append(
            {
                "left_method_id": left_id,
                "right_method_id": right_id,
                "metric": metric,
                "query_count": len(diffs),
                "mean_delta_left_minus_right": observed,
                "bootstrap_ci": bootstrap_ci(diffs, samples=samples, seed=stable_seed(seed, left_id, right_id)),
                "sign_flip_p_value": sign_flip_p_value(diffs, samples=samples, seed=stable_seed(seed + 1, left_id, right_id)),
                "direction": direction(observed),
            }
        )
    return {
        "benchmark_id": external_ir.get("benchmark_id"),
        "mode": external_ir.get("mode"),
        "method_set_id": external_ir.get("method_set_id"),
        "metric": metric,
        "samples": samples,
        "seed": seed,
        "comparison_count": len(rows),
        "comparisons": rows,
        "summary": significance_summary(rows),
    }


def query_metric(query: dict, method_id: str, metric: str) -> float:
    for run in query.get("method_runs", []):
        if run.get("method_id") == method_id:
            return float(run.get("metrics", {}).get(metric, 0.0))
    return 0.0


def bootstrap_ci(diffs: list[float], samples: int, seed: int) -> dict:
    if not diffs or samples <= 0:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(average(draw))
    return {"low": percentile(means, 0.025), "high": percentile(means, 0.975)}


def sign_flip_p_value(diffs: list[float], samples: int, seed: int) -> float:
    if not diffs or samples <= 0:
        return 1.0
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        flipped = sum(diff if rng.random() >= 0.5 else -diff for diff in diffs)
        if abs(flipped) >= observed:
            extreme += 1
    return round((extreme + 1) / (samples + 1), 6)


def significance_summary(rows: list[dict]) -> dict:
    ci_excluding_zero = [
        row
        for row in rows
        if row["bootstrap_ci"]["low"] > 0.0 or row["bootstrap_ci"]["high"] < 0.0
    ]
    permutation_below_05 = [row for row in rows if row["sign_flip_p_value"] < 0.05]
    return {
        "ci_excluding_zero_count": len(ci_excluding_zero),
        "permutation_p_lt_0_05_count": len(permutation_below_05),
        "strongest_mean_delta": max((abs(row["mean_delta_left_minus_right"]) for row in rows), default=0.0),
    }


def direction(value: float) -> str:
    if value > 0:
        return "left_better"
    if value < 0:
        return "right_better"
    return "tie"


def stable_seed(seed: int, left_id: str, right_id: str) -> int:
    return seed + sum(ord(char) for char in f"{left_id}:{right_id}")


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
    parser = argparse.ArgumentParser(description="Run paired significance checks over external IR metrics.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    parser.add_argument("--benchmark-id", default="internal_social_science_ir_qrels_v1")
    parser.add_argument("--metric", default="nDCG@10")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    print(
        json.dumps(
            run_paired_significance(
                mode=args.mode,
                method_set_id=args.method_set_id,
                benchmark_id=args.benchmark_id,
                metric=args.metric,
                samples=args.samples,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

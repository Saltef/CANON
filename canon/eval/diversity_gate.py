from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


DEFAULT_THRESHOLDS = {
    "useful_breadth_count_min": 8,
    "off_topic_breadth_risk_max": 2,
    "noisy_breadth_count_max": 0,
    "average_noise_rate_max": 0.05,
    "average_cluster_delta_min": 1.0,
    "average_breadth_precision_min": 0.30,
}


def run_diversity_gate(
    mode: str,
    method_set_id: str = "baseline_methods_v1",
    diverse_method_id: str = "diverse_k5_template",
    baseline_method_id: str = "lexical_k5_template",
    thresholds: dict | None = None,
) -> dict:
    settings = load_settings()
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    audit = load_diversity_audit(
        settings.reports_dir,
        mode,
        method_set_id,
        diverse_method_id,
        baseline_method_id,
    )
    aggregate = audit.get("aggregate", {})
    verdict_counts = aggregate.get("verdict_counts", {})
    checks = [
        check(
            "useful_breadth_count",
            int(verdict_counts.get("useful_breadth") or 0) >= thresholds["useful_breadth_count_min"],
            thresholds["useful_breadth_count_min"],
            int(verdict_counts.get("useful_breadth") or 0),
        ),
        check(
            "off_topic_breadth_risk",
            int(verdict_counts.get("off_topic_breadth_risk") or 0) <= thresholds["off_topic_breadth_risk_max"],
            thresholds["off_topic_breadth_risk_max"],
            int(verdict_counts.get("off_topic_breadth_risk") or 0),
        ),
        check(
            "noisy_breadth_count",
            int(verdict_counts.get("noisy_breadth") or 0) <= thresholds["noisy_breadth_count_max"],
            thresholds["noisy_breadth_count_max"],
            int(verdict_counts.get("noisy_breadth") or 0),
        ),
        check(
            "average_noise_rate",
            float(aggregate.get("average_noise_rate") or 0.0) <= thresholds["average_noise_rate_max"],
            thresholds["average_noise_rate_max"],
            float(aggregate.get("average_noise_rate") or 0.0),
        ),
        check(
            "average_cluster_delta",
            float(aggregate.get("average_cluster_delta") or 0.0) >= thresholds["average_cluster_delta_min"],
            thresholds["average_cluster_delta_min"],
            float(aggregate.get("average_cluster_delta") or 0.0),
        ),
        check(
            "average_breadth_precision",
            float(aggregate.get("average_breadth_precision") or 0.0) >= thresholds["average_breadth_precision_min"],
            thresholds["average_breadth_precision_min"],
            float(aggregate.get("average_breadth_precision") or 0.0),
        ),
    ]
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "diverse_method_id": diverse_method_id,
        "baseline_method_id": baseline_method_id,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "thresholds": thresholds,
        "checks": checks,
        "aggregate": aggregate,
    }
    write_json(
        settings.reports_dir
        / f"diversity_gate_{mode}_{method_set_id}_{diverse_method_id}_vs_{baseline_method_id}.json",
        report,
    )
    return report


def load_diversity_audit(
    reports_dir: Path,
    mode: str,
    method_set_id: str,
    diverse_method_id: str,
    baseline_method_id: str,
) -> dict:
    path = reports_dir / f"diversity_audit_{mode}_{method_set_id}_{diverse_method_id}_vs_{baseline_method_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing diversity audit report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check(identifier: str, passed: bool, threshold: int | float, observed: int | float) -> dict:
    return {
        "id": identifier,
        "passed": bool(passed),
        "threshold": threshold,
        "observed": observed,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diversity-first regression gates.")
    parser.add_argument("--mode", default="social_science_ir_10k")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    parser.add_argument("--diverse-method-id", default="diverse_k5_template")
    parser.add_argument("--baseline-method-id", default="lexical_k5_template")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_diversity_gate(
        mode=args.mode,
        method_set_id=args.method_set_id,
        diverse_method_id=args.diverse_method_id,
        baseline_method_id=args.baseline_method_id,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

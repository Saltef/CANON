from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


DEFAULT_THRESHOLDS = {
    "main_query_count_min": 25,
    "probe_failure_count_max": 0,
    "external_ir_query_count_min": 5,
    "bootstrap_winner_probability_min": 0.2,
    "paired_significance_comparisons_min": 10,
    "faithfulness_score_min": 0.55,
    "perturbation_overlap_min": 0.45,
    "data_card_work_count_min": 25,
    "claim_decision_required_resolutions_min": 5,
}


def run_regression_gate(mode: str, method_set_id: str = "baseline_methods_v1", thresholds: dict | None = None) -> dict:
    settings = load_settings()
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    artifacts = load_artifacts(settings.reports_dir, mode, method_set_id)
    checks = [
        check("main_query_count", int(artifacts.get("evaluation_suite", {}).get("query_count") or 0) >= thresholds["main_query_count_min"], thresholds["main_query_count_min"]),
        check("probe_failures", int(artifacts.get("probe_eval", {}).get("failure_summary", {}).get("failure_count") or 0) <= thresholds["probe_failure_count_max"], thresholds["probe_failure_count_max"]),
        check("external_ir_query_count", int(artifacts.get("external_ir", {}).get("query_count") or 0) >= thresholds["external_ir_query_count_min"], thresholds["external_ir_query_count_min"]),
        check("bootstrap_winner_probability", float(artifacts.get("bootstrap_ir", {}).get("topline", {}).get("winner_probability") or 0.0) >= thresholds["bootstrap_winner_probability_min"], thresholds["bootstrap_winner_probability_min"]),
        check("paired_significance_comparisons", int(artifacts.get("paired_significance", {}).get("comparison_count") or 0) >= thresholds["paired_significance_comparisons_min"], thresholds["paired_significance_comparisons_min"]),
        check("faithfulness_top_score", top_value(artifacts.get("faithfulness", {}).get("leaderboard") or [], "faithfulness_score") >= thresholds["faithfulness_score_min"], thresholds["faithfulness_score_min"]),
        check("perturbation_top_overlap", top_value(artifacts.get("perturbation_eval", {}).get("leaderboard") or [], "average_rank_overlap") >= thresholds["perturbation_overlap_min"], thresholds["perturbation_overlap_min"]),
        check("data_card_work_count", int(artifacts.get("data_card", {}).get("work_count") or 0) >= thresholds["data_card_work_count_min"], thresholds["data_card_work_count_min"]),
        check("claim_decision_resolutions", len(artifacts.get("claim_decision", {}).get("resolved_warning_ids") or []) >= thresholds["claim_decision_required_resolutions_min"], thresholds["claim_decision_required_resolutions_min"]),
    ]
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "thresholds": thresholds,
        "checks": checks,
    }
    write_json(settings.reports_dir / f"regression_gate_{mode}_{method_set_id}.json", report)
    return report


def load_artifacts(reports_dir: Path, mode: str, method_set_id: str) -> dict:
    patterns = {
        "evaluation_suite": f"evaluation_suite_{mode}_{method_set_id}.json",
        "probe_eval": f"probe_eval_{mode}_{method_set_id}.json",
        "external_ir": f"external_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "bootstrap_ir": f"bootstrap_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "paired_significance": f"paired_significance_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
        "faithfulness": f"faithfulness_{mode}_{method_set_id}.json",
        "perturbation_eval": f"perturbation_eval_{mode}_{method_set_id}.json",
        "data_card": f"data_card_{mode}.json",
        "claim_decision": f"claim_decision_{mode}_{method_set_id}.json",
    }
    artifacts = {}
    for key, filename in patterns.items():
        path = reports_dir / filename
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def top_value(leaderboard: list[dict], metric: str) -> float:
    if not leaderboard:
        return 0.0
    return float(leaderboard[0].get(metric, 0.0))


def check(identifier: str, passed: bool, threshold: int | float) -> dict:
    return {"id": identifier, "passed": bool(passed), "threshold": threshold}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON regression gates over generated eval artifacts.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_regression_gate(args.mode, args.method_set_id)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

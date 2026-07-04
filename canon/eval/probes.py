from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings
from canon.eval.methods import (
    composite_score,
    load_method_set,
    method_specs,
    summarize_method_run,
    synthesize_for_method,
)


def parse_method_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    ids = [piece.strip() for piece in value.split(",") if piece.strip()]
    return ids or None


def load_probes(path: Path | None = None) -> list[dict]:
    settings = load_settings()
    probe_path = path or settings.root / "gold" / "probe_queries.json"
    return json.loads(probe_path.read_text(encoding="utf-8"))


def run_probe_suite(
    mode: str,
    method_set_path: Path | None = None,
    probe_path: Path | None = None,
    method_ids: list[str] | None = None,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = method_specs(method_set)
    if method_ids:
        allowed = set(method_ids)
        specs = [spec for spec in specs if spec.id in allowed]
    probes = load_probes(probe_path)
    probe_reports = []
    for probe in probes:
        runs = []
        for spec in specs:
            run = summarize_method_run(
                probe_query_record(probe),
                spec,
                synthesize_for_method(probe["query"], spec, mode),
            )
            expectation_result = score_expectations(run["metrics"], probe.get("expectations", {}))
            run["expectation_result"] = expectation_result
            run["probe_score"] = probe_score(
                run["metrics"],
                method_set["score_weights"],
                expectation_result["pass_rate"],
            )
            runs.append(run)
        probe_reports.append(
            {
                "id": probe["id"],
                "category": probe["category"],
                "query": probe["query"],
                "rationale": probe.get("rationale", ""),
                "expectations": probe.get("expectations", {}),
                "method_runs": sorted(runs, key=lambda run: run["probe_score"], reverse=True),
            }
        )
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "probe_count": len(probe_reports),
        "method_ids": [spec.id for spec in specs],
        "probes": probe_reports,
        "probe_summary": probe_summary(probe_reports, specs),
        "category_summary": category_summary(probe_reports, specs),
        "failure_summary": failure_summary(probe_reports),
    }
    write_json(settings.reports_dir / f"probe_eval_{mode}_{method_set['method_set_id']}.json", report)
    return report


def probe_query_record(probe: dict) -> dict:
    return {
        "id": probe["id"],
        "type": probe["category"],
        "query": probe["query"],
        "importance_expectation": probe.get("rationale", ""),
    }


def score_expectations(metrics: dict[str, float], expectations: dict[str, float]) -> dict:
    checks = []
    for name, threshold in sorted(expectations.items()):
        metric_name, operator = expectation_metric(name)
        value = float(metrics.get(metric_name, 0.0))
        threshold = float(threshold)
        passed = value >= threshold if operator == "min" else value <= threshold
        checks.append(
            {
                "expectation": name,
                "metric": metric_name,
                "operator": operator,
                "value": round(value, 6),
                "threshold": round(threshold, 6),
                "passed": passed,
            }
        )
    pass_count = sum(1 for check in checks if check["passed"])
    pass_rate = round(pass_count / len(checks), 6) if checks else 1.0
    return {
        "passed": pass_count == len(checks),
        "pass_rate": pass_rate,
        "checks": checks,
    }


def expectation_metric(name: str) -> tuple[str, str]:
    if name.startswith("min_"):
        return name.removeprefix("min_"), "min"
    if name.startswith("max_"):
        return name.removeprefix("max_"), "max"
    raise ValueError(f"Unknown expectation format: {name}")


def probe_score(metrics: dict[str, float], weights: dict[str, float], pass_rate: float) -> float:
    technical_score = composite_score(metrics, weights)
    return round((technical_score * 0.7) + (pass_rate * 0.3), 6)


def probe_summary(probe_reports: list[dict], specs) -> dict:
    rows = {}
    for spec in specs:
        runs = [
            run
            for probe in probe_reports
            for run in probe["method_runs"]
            if run["method_id"] == spec.id
        ]
        rows[spec.id] = {
            "label": spec.label,
            "probe_count": len(runs),
            "average_probe_score": average(run["probe_score"] for run in runs),
            "expectation_pass_rate": average(
                run["expectation_result"]["pass_rate"] for run in runs
            ),
            "full_probe_passes": sum(1 for run in runs if run["expectation_result"]["passed"]),
        }
    return {
        "by_method": dict(sorted(rows.items())),
        "leaderboard": [
            {"rank": index, "method_id": method_id, **row}
            for index, (method_id, row) in enumerate(
                sorted(rows.items(), key=lambda item: item[1]["average_probe_score"], reverse=True),
                start=1,
            )
        ],
    }


def category_summary(probe_reports: list[dict], specs) -> dict:
    categories = sorted({probe["category"] for probe in probe_reports})
    summary = {}
    for category in categories:
        category_probes = [probe for probe in probe_reports if probe["category"] == category]
        rows = {}
        for spec in specs:
            runs = [
                run
                for probe in category_probes
                for run in probe["method_runs"]
                if run["method_id"] == spec.id
            ]
            rows[spec.id] = {
                "average_probe_score": average(run["probe_score"] for run in runs),
                "expectation_pass_rate": average(
                    run["expectation_result"]["pass_rate"] for run in runs
                ),
            }
        summary[category] = dict(sorted(rows.items()))
    return summary


def failure_summary(probe_reports: list[dict]) -> dict:
    failures = []
    by_expectation: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for probe in probe_reports:
        for run in probe["method_runs"]:
            failed_checks = [
                check for check in run["expectation_result"]["checks"] if not check["passed"]
            ]
            if not failed_checks:
                continue
            by_category[probe["category"]] = by_category.get(probe["category"], 0) + 1
            by_method[run["method_id"]] = by_method.get(run["method_id"], 0) + 1
            for check in failed_checks:
                by_expectation[check["expectation"]] = by_expectation.get(check["expectation"], 0) + 1
            failures.append(
                {
                    "probe_id": probe["id"],
                    "category": probe["category"],
                    "method_id": run["method_id"],
                    "failed_expectations": [check["expectation"] for check in failed_checks],
                    "probe_score": run["probe_score"],
                }
            )
    return {
        "failure_count": len(failures),
        "by_category": dict(sorted(by_category.items())),
        "by_method": dict(sorted(by_method.items())),
        "by_expectation": dict(sorted(by_expectation.items())),
        "failures": failures,
    }


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON technical probe evaluation.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--probe-set", default=None)
    parser.add_argument("--method-ids", default=None)
    args = parser.parse_args()
    method_set_path = Path(args.method_set) if args.method_set else None
    probe_path = Path(args.probe_set) if args.probe_set else None
    print(
        json.dumps(
            run_probe_suite(
                mode=args.mode,
                method_set_path=method_set_path,
                probe_path=probe_path,
                method_ids=parse_method_ids(args.method_ids),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

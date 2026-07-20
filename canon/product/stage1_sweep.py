from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product.automated_benchmark_suite import (
    DEFAULT_SUITE_PATH,
    run_automated_benchmark_suite,
)


DEFAULT_CANDIDATE_VALUES = [25, 50, 100]


def run_stage1_sweep(
    suite_path: Path | None = None,
    candidate_values: list[int] | None = None,
    top_k: int = 10,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    values = candidate_values or DEFAULT_CANDIDATE_VALUES
    runs = [
        sweep_run(
            suite_path=suite_path or DEFAULT_SUITE_PATH,
            candidate_k=value,
            top_k=top_k,
        )
        for value in values
    ]
    report = {
        "report_id": "stage1_sweep_v1",
        "suite_path": str(suite_path or DEFAULT_SUITE_PATH).replace("\\", "/"),
        "candidate_values": values,
        "status": sweep_status(runs),
        "best_semantic": best_rows(runs, "semantic"),
        "best_rerank": best_rows(runs, "rerank"),
        "candidate_recall": candidate_recall_rows(runs),
        "runs": runs,
        "stage1_acceptance": {
            "automated_gate": "pass" if all(run["status"] == "automated_pass_human_review_required" for run in runs) else "fail",
            "human_review_required_for_release": True,
            "minimum_next_scale": "Run this sweep on at least 30 public qrels queries before Stage 1 is considered complete.",
        },
    }
    if write_report:
        report_io.write_json(settings.reports_dir / "stage1_sweep_v1.json", report)
        report_io.write_markdown(settings.reports_dir / "stage1_sweep_v1.md", render_markdown(report))
    return report


def sweep_run(suite_path: Path, candidate_k: int, top_k: int) -> dict[str, Any]:
    report = run_automated_benchmark_suite(
        suite_path=suite_path,
        top_k=top_k,
        candidate_k=candidate_k,
        lexical_k=candidate_k,
        vector_k=candidate_k,
        write_report=False,
    )
    return {
        "candidate_k": candidate_k,
        "top_k": top_k,
        "status": report["status"],
        "model_matrix_status": report["model_matrix"]["status"],
        "semantic_leaderboard": report["leaderboards"]["semantic"],
        "rerank_leaderboard": report["leaderboards"]["rerank"],
        "benchmarks": [
            {
                    "id": benchmark["id"],
                    "candidate_recall": benchmark["rerank"]["candidate_recall"],
                    "best_semantic": benchmark["semantic"].get("best", {}),
                    "best_rerank": benchmark["rerank"].get("best", {}),
                }
            for benchmark in report.get("benchmarks", [])
        ],
    }


def sweep_status(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "fail"
    if any(run.get("model_matrix_status") != "complete" for run in runs):
        return "partial_model_matrix"
    if all(run.get("status") == "automated_pass_human_review_required" for run in runs):
        return "automated_pass_human_review_required"
    return "automated_fail"


def best_rows(runs: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    key = f"{family}_leaderboard"
    rows = [
        {"candidate_k": run["candidate_k"], **row}
        for run in runs
        for row in run.get(key, [])
        if int(row.get("rank", 0)) == 1
    ]
    return sorted(rows, key=lambda row: row.get("mean_primary_score", 0.0), reverse=True)


def candidate_recall_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_k": run["candidate_k"],
            "benchmark_id": benchmark["id"],
            **benchmark["candidate_recall"],
        }
        for run in runs
        for benchmark in run.get("benchmarks", [])
    ]


def render_markdown(report: dict[str, Any]) -> str:
    recalls = "\n".join(
        f"- candidate_k={row['candidate_k']} benchmark=`{row['benchmark_id']}` "
        f"mean={row['mean']} min={row['min']} misses={row['miss_count']}"
        for row in report["candidate_recall"]
    )
    semantic = report["best_semantic"][0] if report["best_semantic"] else {}
    rerank = report["best_rerank"][0] if report["best_rerank"] else {}
    return f"""# CANON Stage 1 Sweep

Status: **{report['status']}**

Best semantic: `{semantic.get('provider', 'none')}` candidate_k={semantic.get('candidate_k', 'n/a')} score={semantic.get('mean_primary_score', 0)}

Best rerank: `{rerank.get('provider', 'none')}` candidate_k={rerank.get('candidate_k', 'n/a')} score={rerank.get('mean_primary_score', 0)}

## Candidate Recall

{recalls}

Human review is still required for release-quality claims.
"""


def parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 candidate-pool and model sweep.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--candidate-values", default=None, help="Comma-separated values, e.g. 25,50,100")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage1_sweep(
                suite_path=Path(args.suite),
                candidate_values=parse_ints(args.candidate_values),
                top_k=args.top_k,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

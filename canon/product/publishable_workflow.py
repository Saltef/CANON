from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from canon.config import load_settings
from canon.eval.disagreement_preservation import DEFAULT_BENCHMARK_PATH, evaluate_disagreement_preservation
from canon.product import report_io
from canon.product.automated_benchmark_suite import load_suite, resolve_path, run_automated_benchmark_suite
from canon.product.human_review_scaffold import build_human_review_scaffold
from canon.product.publishable_benchmark_card import build_publishable_benchmark_card
from canon.product.publishable_package import DEFAULT_PUBLISHABLE_SUITE_PATH, build_publishable_package
from canon.product.publishable_review import build_publishable_review_status
from canon.product.stage1_optimizer import run_stage1_optimizer


DEFAULT_WORKFLOW_REPORT = Path("reports") / "publishable_workflow_v1.json"


def run_publishable_workflow(
    suite_path: Path | None = None,
    disagreement_benchmark_path: Path | None = None,
    run_benchmarks: bool = False,
    model_providers: list[str] | None = None,
    rerankers: list[str] | None = None,
    candidate_values: list[int] | None = None,
    fusion_methods: list[str] | None = None,
    document_format: str = "structured",
    objective_mode: str = "balanced",
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    suite = suite_path or DEFAULT_PUBLISHABLE_SUITE_PATH
    disagreement = disagreement_benchmark_path or DEFAULT_BENCHMARK_PATH
    resolved_suite = resolve_path(settings.root, suite)
    suite_payload = load_suite(resolved_suite)
    optimizer_fusion_methods = fusion_methods or suite_fusion_methods(suite_payload)
    automated_suite_report = settings.reports_dir / f"automated_benchmark_suite_{suite_payload['suite_id']}.json"
    optimizer_report = settings.reports_dir / "stage1_optimizer_v1.json"
    stages: list[dict[str, Any]] = []
    if run_benchmarks:
        stages.append(
            run_stage(
                "automated_benchmark_suite",
                lambda: run_automated_benchmark_suite(
                    suite_path=suite,
                    model_providers=model_providers or ["local"],
                    rerankers=rerankers or ["heuristic"],
                    candidate_k=max(candidate_values or [50]),
                    document_format=document_format,
                    resume=True,
                    write_report=True,
                ),
            )
        )
        stages.append(
            run_stage(
                "stage1_optimizer",
                lambda: run_stage1_optimizer(
                    suite_path=suite,
                    dense_retrievers=model_providers or ["local"],
                    rerankers=rerankers or ["heuristic"],
                    candidate_values=candidate_values or [50],
                    fusion_methods=optimizer_fusion_methods,
                    document_format=document_format,
                    objective_mode=objective_mode,
                    resume=True,
                    write_report=True,
                ),
            )
        )
    else:
        stages.append(existing_artifact_stage("automated_benchmark_suite", automated_suite_report, settings.root))
        stages.append(existing_artifact_stage("stage1_optimizer", optimizer_report, settings.root))
    benchmark_artifacts_present = not any(
        stage["status"] == "missing_required_artifact"
        for stage in stages
        if stage["id"] in {"automated_benchmark_suite", "stage1_optimizer"}
    )
    stages.extend(
        [
            run_stage(
                "disagreement_preservation",
                lambda: evaluate_disagreement_preservation(
                    benchmark_path=disagreement,
                    write_report=True,
                ),
            ),
            run_stage(
                "human_review_scaffold",
                lambda: build_human_review_scaffold(
                    suite_path=suite,
                    disagreement_benchmark_path=disagreement,
                    write_report=True,
                ),
            ),
            run_stage(
                "publishable_review_status",
                lambda: build_publishable_review_status(write_report=True),
            ),
            (
                run_stage(
                    "publishable_benchmark_card",
                    lambda: build_publishable_benchmark_card(
                        automated_suite_report=automated_suite_report,
                        optimizer_report=optimizer_report,
                        write_report=True,
                    ),
                )
                if benchmark_artifacts_present
                else skipped_stage("publishable_benchmark_card", "missing_stage1_benchmark_artifacts")
            ),
            run_stage(
                "publishable_package",
                lambda: build_publishable_package(
                    suite_path=suite,
                    write_report=True,
                ),
            ),
        ]
    )
    package_stage = first_stage(stages, "publishable_package")
    package_status = (package_stage.get("summary") or {}).get("status")
    report = {
        "report_id": "publishable_workflow_v1",
        "status": workflow_status(stages, package_status),
        "suite_id": suite_payload["suite_id"],
        "suite_path": relative(settings.root, resolved_suite),
        "disagreement_benchmark_path": relative(
            settings.root,
            settings.root / disagreement if not disagreement.is_absolute() else disagreement,
        ),
        "run_benchmarks": run_benchmarks,
        "stage_count": len(stages),
        "stages": stages,
        "package_status": package_status,
        "human_review_boundary": (
            "This workflow can regenerate automated and scaffold artifacts, but it cannot complete "
            "human review labels. A publication-ready package still requires completed reviewer CSVs."
        ),
        "next_required_work": next_required_work(stages, package_status),
    }
    if write_report:
        output = settings.root / DEFAULT_WORKFLOW_REPORT
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def run_stage(stage_id: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        return {
            "id": stage_id,
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": str(exc),
        }
    return {
        "id": stage_id,
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "summary": stage_summary(result),
    }


def existing_artifact_stage(stage_id: str, path: Path, root: Path) -> dict[str, Any]:
    if path.exists():
        return {
            "id": stage_id,
            "status": "existing_artifact",
            "elapsed_ms": 0.0,
            "summary": {
                "artifact": relative(root, path),
                "status": "present",
            },
        }
    return {
        "id": stage_id,
        "status": "missing_required_artifact",
        "elapsed_ms": 0.0,
        "summary": {
            "artifact": relative(root, path),
            "status": "missing",
        },
    }


def skipped_stage(stage_id: str, reason: str) -> dict[str, Any]:
    return {
        "id": stage_id,
        "status": "skipped",
        "elapsed_ms": 0.0,
        "summary": {
            "reason": reason,
        },
    }


def stage_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "report_id": report.get("report_id"),
        "status": report.get("status"),
        "suite_id": report.get("suite_id"),
        "benchmark_id": report.get("benchmark_id"),
    }
    for key in [
        "benchmark_count",
        "query_count",
        "trial_count",
        "case_count",
        "reviewed_label_count",
        "validation_error_count",
    ]:
        if key in report:
            summary[key] = report[key]
    if "completion_status" in report:
        summary["completion_status"] = report["completion_status"]
    if "aggregate_diagnostics" in report:
        summary["aggregate_diagnostics"] = report["aggregate_diagnostics"]
    if "artifact_manifest" in report:
        summary["artifact_count"] = (report.get("artifact_manifest") or {}).get("artifact_count")
    if "next_required_work" in report:
        summary["next_required_work"] = report["next_required_work"]
    return {key: value for key, value in summary.items() if value is not None}


def workflow_status(stages: list[dict[str, Any]], package_status: str | None) -> str:
    if any(stage["status"] == "failed" for stage in stages):
        return "workflow_failed"
    if any(stage["status"] == "missing_required_artifact" for stage in stages):
        return "workflow_blocked_missing_benchmark_artifact"
    if package_status == "publication_blocked_evidence_incomplete":
        return "workflow_complete_human_review_blocked"
    return "workflow_complete"


def next_required_work(stages: list[dict[str, Any]], package_status: str | None) -> list[str]:
    work = []
    if any(stage["status"] == "missing_required_artifact" for stage in stages):
        work.append("Run with --run-benchmarks, or create the missing Stage 1 benchmark artifacts before freezing the package.")
    if package_status == "publication_blocked_evidence_incomplete":
        package_stage = first_stage(stages, "publishable_package")
        package_work = (package_stage.get("summary") or {}).get("next_required_work") or []
        if package_work:
            work.extend(str(item) for item in package_work)
        else:
            work.append("Complete publishable retrieval and synthesis review labels, then rerun publishable_review and publishable_package.")
    if any(stage["status"] == "failed" for stage in stages):
        work.append("Fix failed workflow stages before treating the package as reproducible.")
    return work or ["Review final claims against the package card before publication."]


def first_stage(stages: list[dict[str, Any]], stage_id: str) -> dict[str, Any]:
    for stage in stages:
        if stage["id"] == stage_id:
            return stage
    return {}


def render_markdown(report: dict[str, Any]) -> str:
    stages = "\n".join(
        f"- `{stage['id']}`: `{stage['status']}` ({stage['elapsed_ms']} ms)"
        for stage in report["stages"]
    )
    next_work = "\n".join(f"- {item}" for item in report["next_required_work"])
    return f"""# CANON Publishable Workflow

Status: `{report['status']}`

Suite: `{report['suite_path']}`

Run benchmarks: `{report['run_benchmarks']}`

Package status: `{report.get('package_status')}`

## Stages

{stages}

## Next Required Work

{next_work}

## Human Review Boundary

{report['human_review_boundary']}
"""


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [piece.strip() for piece in value.split(",") if piece.strip()]
    return parsed or None


def parse_int_csv(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def suite_fusion_methods(suite: dict[str, Any]) -> list[str]:
    values = suite.get("fusion_methods") or [suite.get("fusion", "union")]
    return [str(value) for value in values if value] or ["union"]


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON publishable workflow in one command.")
    parser.add_argument("--suite", default=str(DEFAULT_PUBLISHABLE_SUITE_PATH))
    parser.add_argument("--disagreement-benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--run-benchmarks", action="store_true")
    parser.add_argument("--model-providers", default=None)
    parser.add_argument("--rerankers", default=None)
    parser.add_argument("--candidate-values", default=None)
    parser.add_argument("--fusion-methods", default=None)
    parser.add_argument("--document-format", default="structured")
    parser.add_argument("--objective-mode", default="balanced")
    args = parser.parse_args()
    report = run_publishable_workflow(
        suite_path=Path(args.suite),
        disagreement_benchmark_path=Path(args.disagreement_benchmark),
        run_benchmarks=args.run_benchmarks,
        model_providers=parse_csv(args.model_providers),
        rerankers=parse_csv(args.rerankers),
        candidate_values=parse_int_csv(args.candidate_values),
        fusion_methods=parse_csv(args.fusion_methods),
        document_format=args.document_format,
        objective_mode=args.objective_mode,
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "package_status": report.get("package_status"),
        "run_benchmarks": report["run_benchmarks"],
        "stage_statuses": {stage["id"]: stage["status"] for stage in report["stages"]},
        "next_required_work": report["next_required_work"],
        "artifacts": {
            "json": "reports/publishable_workflow_v1.json",
            "markdown": "reports/publishable_workflow_v1.md",
        },
    }


if __name__ == "__main__":
    main()

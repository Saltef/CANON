from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.disagreement_fixture import DEFAULT_OUTPUT_PATH as DEFAULT_DISAGREEMENT_BENCHMARK_PATH
from canon.eval.external_ir import load_qrels
from canon.product import report_io
from canon.product.automated_benchmark_suite import load_suite, resolve_path


DEFAULT_EXPERIMENT_ID = "canon_publishable_evidence_workflow_v1"
DEFAULT_REQUIRED_QUERY_COUNT = 100
DEFAULT_PUBLISHABLE_SUITE_PATH = Path("conf") / "benchmark_suites" / "stage1_public_full.json"


def build_publishable_package(
    suite_path: Path | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    required_query_count: int = DEFAULT_REQUIRED_QUERY_COUNT,
    stage1_optimizer_report: Path | None = None,
    automated_suite_report: Path | None = None,
    stage2_synthesis_report: Path | None = None,
    human_review_status_report: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_suite_path = resolve_path(settings.root, suite_path or DEFAULT_PUBLISHABLE_SUITE_PATH)
    suite = load_suite(resolved_suite_path)
    disagreement_benchmark_path = resolve_path(settings.root, DEFAULT_DISAGREEMENT_BENCHMARK_PATH)
    disagreement_benchmark_id = benchmark_id_from_path(disagreement_benchmark_path)
    benchmark_rows = benchmark_coverage(settings.root, suite, required_query_count)
    artifact_rows = collect_artifacts(
        settings.root,
        [
            resolved_suite_path,
            *qrels_artifact_paths(settings.root, suite),
            disagreement_benchmark_path,
            stage1_optimizer_report or settings.reports_dir / "stage1_optimizer_v1.json",
            automated_suite_report
            or settings.reports_dir / f"automated_benchmark_suite_{suite['suite_id']}.json",
            stage2_synthesis_report,
            human_review_status_report
            or settings.reports_dir / "publishable_human_review_status_v1.json",
            settings.reports_dir / "publishable_human_review_scaffold_v1.json",
            settings.reports_dir / "publishable_retrieval_review_scaffold.csv",
            settings.reports_dir / "publishable_synthesis_review_scaffold.csv",
            settings.reports_dir / "publishable_benchmark_card_v1.json",
            disagreement_report_path(settings.reports_dir, disagreement_benchmark_id),
        ],
    )
    if not stage2_synthesis_report:
        artifact_rows = merge_artifacts(
            artifact_rows,
            collect_artifacts(settings.root, sorted(settings.reports_dir.glob("stage2_synthesis_*.json"))),
        )
    checks = build_checks(
        root=settings.root,
        suite=suite,
        benchmark_rows=benchmark_rows,
        artifact_rows=artifact_rows,
        required_query_count=required_query_count,
        disagreement_benchmark_id=disagreement_benchmark_id,
    )
    report = {
        "report_id": "publishable_evaluation_package_v1",
        "experiment_id": experiment_id,
        "status": package_status(checks),
        "git": git_snapshot(settings.root),
        "suite": {
            "suite_id": suite["suite_id"],
            "path": relative(settings.root, resolved_suite_path),
            "benchmark_count": len(suite.get("benchmarks") or []),
            "model_providers": suite.get("model_providers") or [],
            "rerankers": suite.get("rerankers") or [],
            "fusion_methods": suite.get("fusion_methods") or [suite.get("fusion", "union")],
        },
        "public_benchmark_coverage": {
            "required_query_count": required_query_count,
            "status": public_benchmark_status(benchmark_rows, required_query_count),
            "benchmarks": benchmark_rows,
        },
        "artifact_manifest": {
            "artifact_count": len(artifact_rows),
            "artifacts": artifact_rows,
        },
        "checks": checks,
        "claim_boundary": claim_boundary(),
        "reproducibility_commands": reproducibility_commands(relative(settings.root, resolved_suite_path)),
        "next_required_work": next_required_work(checks),
    }
    if write_report:
        output = settings.reports_dir / f"publishable_package_{experiment_id}.json"
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def benchmark_coverage(root: Path, suite: dict[str, Any], required_query_count: int) -> list[dict[str, Any]]:
    rows = []
    for benchmark in suite.get("benchmarks") or []:
        qrels_path = resolve_path(root, Path(benchmark["qrels"]))
        qrels = load_qrels(qrels_path) if qrels_path.exists() else {"queries": []}
        query_count = len(qrels.get("queries") or [])
        target = int(benchmark.get("publishable_query_target") or required_query_count)
        rows.append(
            {
                "id": benchmark["id"],
                "mode": benchmark["mode"],
                "topic": benchmark.get("topic", "unspecified"),
                "qrels_path": relative(root, qrels_path),
                "benchmark_id": qrels.get("benchmark_id"),
                "benchmark_kind": qrels.get("benchmark_kind"),
                "query_count": query_count,
                "publishable_query_target": target,
                "status": "sufficient" if query_count >= target else "pilot_only",
            }
        )
    return rows


def qrels_artifact_paths(root: Path, suite: dict[str, Any]) -> list[Path]:
    paths = []
    for benchmark in suite.get("benchmarks") or []:
        qrels = benchmark.get("qrels")
        if qrels:
            paths.append(resolve_path(root, Path(qrels)))
    return paths


def collect_artifacts(root: Path, paths: list[Path | None]) -> list[dict[str, Any]]:
    rows = []
    seen: set[Path] = set()
    for path in paths:
        if not path:
            continue
        resolved = path if path.is_absolute() else root / path
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        rows.append(artifact_record(resolved, root))
    return rows


def merge_artifacts(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(left)
    seen = {row["path"] for row in rows}
    for row in right:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        rows.append(row)
    return rows


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": relative(root, path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def build_checks(
    root: Path,
    suite: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    required_query_count: int,
    disagreement_benchmark_id: str,
) -> list[dict[str, Any]]:
    artifact_paths = {row["path"] for row in artifact_rows}
    return [
        check(
            "public_benchmark_coverage",
            public_benchmark_status(benchmark_rows, required_query_count) == "sufficient",
            "Full public retrieval benchmark coverage is present for configured targets.",
            "Configured public qrels are still pilot-sized or missing.",
        ),
        check(
            "public_qrels_frozen",
            public_qrels_frozen(artifact_paths, benchmark_rows),
            "Configured public qrels files are included in the frozen artifact manifest.",
            "Configured public qrels files must be included in the frozen artifact manifest.",
        ),
        check(
            "retrieval_failure_diagnostics",
            bool(suite.get("thresholds")) and bool(suite.get("fusion_methods")),
            "Suite declares thresholds and fusion methods for candidate-recall and ranking diagnostics.",
            "Suite lacks explicit diagnostic thresholds or fusion comparisons.",
        ),
        check(
            "resumable_stage1_execution",
            report_with_suite_id_exists(root, artifact_rows, "stage1_optimizer_v1.json", suite["suite_id"]),
            "Stage 1 optimizer report is included in the frozen artifact manifest.",
            "Run the resumable Stage 1 optimizer and include its report.",
        ),
        check(
            "optimizer_fusion_coverage",
            optimizer_fusion_coverage(root, artifact_rows, suite),
            "Stage 1 optimizer report includes completed trials for declared fusion methods.",
            "Run the Stage 1 optimizer across the suite's declared fusion methods before freezing the package.",
        ),
        check(
            "automated_suite_report",
            report_with_suite_id_exists(
                root,
                artifact_rows,
                f"automated_benchmark_suite_{suite['suite_id']}.json",
                suite["suite_id"],
            ),
            "Automated benchmark-suite report is included.",
            "Run the automated benchmark suite and include its report.",
        ),
        check(
            "benchmark_card_report",
            benchmark_card_ready(root, artifact_rows, suite["suite_id"]),
            "Publishable benchmark card is included with suite-matched failure diagnostics.",
            "Build the publishable benchmark card before freezing the package.",
        ),
        check(
            "disagreement_preservation_report",
            disagreement_report_passed(root, artifact_rows, disagreement_benchmark_id),
            "Disagreement-preservation benchmark report is included and passed automated gates.",
            "Run the disagreement-preservation benchmark and include a passing report.",
        ),
        check(
            "disagreement_benchmark_frozen",
            relative(root, resolve_path(root, DEFAULT_DISAGREEMENT_BENCHMARK_PATH)) in artifact_paths,
            "Disagreement-preservation benchmark fixture is included in the frozen artifact manifest.",
            "Include the disagreement-preservation benchmark fixture in the frozen artifact manifest.",
        ),
        check(
            "human_review_scaffold",
            human_review_scaffold_ready(root, artifact_rows),
            "Publishable human-review scaffold and CSV handoff files are included.",
            "Generate the publishable human-review scaffold before handing labels to reviewers.",
        ),
        check(
            "human_review_retrieval_scaffold_target",
            human_review_scaffold_target_met(root, artifact_rows, "retrieval_target_met"),
            "Human-review scaffold includes enough distinct retrieval queries for the configured target.",
            "Expand the retrieval review scaffold until it meets the configured query target.",
        ),
        check(
            "human_review_synthesis_scaffold_target",
            human_review_scaffold_target_met(root, artifact_rows, "synthesis_target_met"),
            "Human-review scaffold includes enough distinct synthesis cases for the configured target.",
            "Expand the synthesis review scaffold until it meets the configured case target.",
        ),
        check(
            "human_review_complete",
            human_review_complete(root, artifact_rows, suite["suite_id"]),
            "Human review status artifact is present and complete.",
            "Human-reviewed qrels and synthesis labels remain required before publication-quality claims.",
        ),
    ]


def public_qrels_frozen(artifact_paths: set[str], benchmark_rows: list[dict[str, Any]]) -> bool:
    return bool(benchmark_rows) and all(row["qrels_path"] in artifact_paths for row in benchmark_rows)


def check(check_id: str, passed: bool, pass_message: str, fail_message: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "message": pass_message if passed else fail_message,
    }


def human_review_complete(root: Path, artifact_rows: list[dict[str, Any]], suite_id: str) -> bool:
    for row in artifact_rows:
        if not row["path"].endswith("publishable_human_review_status_v1.json"):
            continue
        path = root / row["path"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("report_id") == "publishable_human_review_status_v1"
            and payload.get("suite_id") == suite_id
            and payload.get("status") == "complete"
            and bool((payload.get("retrieval_review") or {}).get("target_met"))
            and bool((payload.get("synthesis_review") or {}).get("target_met"))
            and int(payload.get("validation_error_count") or 0) == 0
        )
    return False


def report_with_suite_id_exists(
    root: Path,
    artifact_rows: list[dict[str, Any]],
    file_name: str,
    suite_id: str,
) -> bool:
    for row in artifact_rows:
        if not row["path"].endswith(file_name):
            continue
        try:
            payload = json.loads((root / row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("suite_id") == suite_id
    return False


def optimizer_fusion_coverage(root: Path, artifact_rows: list[dict[str, Any]], suite: dict[str, Any]) -> bool:
    expected = set(suite.get("fusion_methods") or [suite.get("fusion", "union")])
    if not expected:
        return False
    for row in artifact_rows:
        if not row["path"].endswith("stage1_optimizer_v1.json"):
            continue
        try:
            payload = json.loads((root / row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("suite_id") != suite["suite_id"]:
            return False
        declared = set(payload.get("fusion_methods") or [])
        completed = {
            trial.get("fusion")
            for trial in payload.get("trials") or []
            if trial.get("status") == "ok" and trial.get("fusion")
        }
        return expected.issubset(declared) and expected.issubset(completed)
    return False


def disagreement_report_passed(root: Path, artifact_rows: list[dict[str, Any]], benchmark_id: str) -> bool:
    if not benchmark_id:
        return False
    for row in artifact_rows:
        if not row["path"].startswith("reports/disagreement_preservation_"):
            continue
        try:
            payload = json.loads((root / row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("benchmark_id") == benchmark_id
            and payload.get("status") == "automated_pass_human_review_required"
        ):
            return True
    return False


def benchmark_id_from_path(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("benchmark_id") or "")


def disagreement_report_path(reports_dir: Path, benchmark_id: str) -> Path | None:
    if not benchmark_id:
        return None
    return reports_dir / f"disagreement_preservation_{benchmark_id}.json"


def benchmark_card_ready(root: Path, artifact_rows: list[dict[str, Any]], suite_id: str) -> bool:
    for row in artifact_rows:
        if row["path"] != "reports/publishable_benchmark_card_v1.json":
            continue
        try:
            payload = json.loads((root / row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("report_id") == "publishable_benchmark_card_v1"
            and payload.get("suite_id") == suite_id
            and str(payload.get("status") or "").startswith("benchmark_card_ready")
            and bool(payload.get("aggregate_diagnostics"))
            and bool(payload.get("claim_boundary"))
        )
    return False


def human_review_scaffold_payload(root: Path, artifact_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in artifact_rows:
        if row["path"] != "reports/publishable_human_review_scaffold_v1.json":
            continue
        try:
            return json.loads((root / row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def human_review_scaffold_ready(root: Path, artifact_rows: list[dict[str, Any]]) -> bool:
    artifact_paths = {row["path"] for row in artifact_rows}
    required_paths = {
        "reports/publishable_human_review_scaffold_v1.json",
        "reports/publishable_retrieval_review_scaffold.csv",
        "reports/publishable_synthesis_review_scaffold.csv",
    }
    if not required_paths.issubset(artifact_paths):
        return False
    payload = human_review_scaffold_payload(root, artifact_rows)
    return bool(payload and payload.get("status") == "review_scaffold_ready_human_labels_required")


def human_review_scaffold_target_met(root: Path, artifact_rows: list[dict[str, Any]], target_key: str) -> bool:
    payload = human_review_scaffold_payload(root, artifact_rows)
    if not payload or payload.get("status") != "review_scaffold_ready_human_labels_required":
        return False
    completion = payload.get("completion_status") or {}
    return bool(completion.get(target_key))


def public_benchmark_status(rows: list[dict[str, Any]], required_query_count: int) -> str:
    if not rows:
        return "missing"
    if all(row["query_count"] >= max(required_query_count, int(row["publishable_query_target"])) for row in rows):
        return "sufficient"
    return "pilot_only"


def package_status(checks: list[dict[str, Any]]) -> str:
    if all(row["passed"] for row in checks):
        return "publishable_package_ready_pending_claim_review"
    return "publication_blocked_evidence_incomplete"


def claim_boundary() -> dict[str, Any]:
    return {
        "defensible_now": (
            "CANON can be presented as an auditable, human-review-gated evidence workflow "
            "when reports are reproduced and limitations are preserved."
        ),
        "blocked_until_reviewed_evidence": [
            "claims that CANON outperforms existing RAG systems",
            "production unsupported-claim rates",
            "durable model-winner conclusions",
            "clinical, legal, policy, or publication-ready factual correctness claims",
        ],
        "human_review_required": True,
    }


def reproducibility_commands(suite_path: str) -> list[str]:
    return [
        f"python -m canon.product.publishable_workflow --suite {suite_path}",
        f"python -m canon.product.automated_benchmark_suite --suite {suite_path}",
        f"python -m canon.product.stage1_optimizer --suite {suite_path} --candidate-values 50 --fusion-methods union,rrf,weighted_bm25_dense --document-format structured --objective-mode balanced",
        "python -m canon.eval.disagreement_fixture --output gold/disagreement_preservation_publishable.json",
        "python -m canon.eval.disagreement_preservation --benchmark gold/disagreement_preservation_publishable.json",
        f"python -m canon.product.human_review_scaffold --suite {suite_path} --disagreement-benchmark gold/disagreement_preservation_publishable.json",
        "python -m canon.product.publishable_review --retrieval-csv reports/publishable_retrieval_review_scaffold.csv --synthesis-csv reports/publishable_synthesis_review_scaffold.csv",
        "python -m canon.product.publishable_benchmark_card --automated-suite-report reports/automated_benchmark_suite_stage1_public_full_v1.json --optimizer-report reports/stage1_optimizer_v1.json",
        "python -m canon.product.publishable_package",
        "python -m canon.product.publishable_verify --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json",
        "python -m canon.product.publishable_export --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json",
    ]


def next_required_work(checks: list[dict[str, Any]]) -> list[str]:
    work = {
        "public_benchmark_coverage": "Ingest/run full public benchmark qrels, not only 30-query or mini slices.",
        "public_qrels_frozen": "Include the configured public qrels files in the frozen artifact manifest.",
        "retrieval_failure_diagnostics": "Add explicit candidate-recall, ranking, fusion, and score-overlap diagnostics to the suite.",
        "resumable_stage1_execution": "Run and freeze the resumable Stage 1 optimizer report.",
        "optimizer_fusion_coverage": "Run the Stage 1 optimizer across the suite's declared fusion methods.",
        "automated_suite_report": "Run and freeze the automated benchmark-suite report.",
        "benchmark_card_report": "Build and freeze the publishable benchmark card with failure diagnostics.",
        "disagreement_preservation_report": "Run and freeze a passing disagreement-preservation benchmark report.",
        "disagreement_benchmark_frozen": "Include the disagreement-preservation benchmark fixture in the frozen artifact manifest.",
        "human_review_scaffold": "Generate the publishable human-review scaffold for retrieval and synthesis labels.",
        "human_review_retrieval_scaffold_target": (
            "Expand the retrieval review scaffold to meet the configured distinct-query target, "
            "or lower the target with a documented rationale."
        ),
        "human_review_synthesis_scaffold_target": (
            "Expand the disagreement-preservation benchmark and synthesis review scaffold to meet "
            "the configured distinct-case target."
        ),
        "human_review_complete": "Complete human-reviewed qrels and synthesis labels before making publication-quality claims.",
    }
    return [work[row["id"]] for row in checks if not row["passed"]]


def git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "commit": git_output(root, ["git", "rev-parse", "HEAD"]),
        "branch": git_output(root, ["git", "branch", "--show-current"]),
        "dirty": bool(git_output(root, ["git", "status", "--short"])),
    }


def git_output(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    except Exception:  # noqa: BLE001
        return ""
    return result.stdout.strip()


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- `{row['id']}`: `{row['status']}` - {row['message']}"
        for row in report["checks"]
    )
    benchmarks = "\n".join(
        f"- `{row['id']}` queries={row['query_count']} target={row['publishable_query_target']} status=`{row['status']}`"
        for row in report["public_benchmark_coverage"]["benchmarks"]
    )
    artifacts = "\n".join(
        f"- `{row['path']}` sha256=`{row['sha256']}`"
        for row in report["artifact_manifest"]["artifacts"]
    )
    next_work = "\n".join(f"- {item}" for item in report["next_required_work"])
    commands = "\n".join(f"- `{command}`" for command in report["reproducibility_commands"])
    return f"""# CANON Publishable Evaluation Package

Status: `{report['status']}`

Experiment: `{report['experiment_id']}`

Git commit: `{report['git']['commit']}`

## Checks

{checks}

## Public Benchmark Coverage

{benchmarks}

## Frozen Artifacts

{artifacts}

## Reproduction Commands

{commands}

## Next Required Work

{next_work}

## Claim Boundary

{report['claim_boundary']['defensible_now']}

Human review required: `{report['claim_boundary']['human_review_required']}`
"""


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a CANON publishable evaluation package card.")
    parser.add_argument("--suite", default=str(DEFAULT_PUBLISHABLE_SUITE_PATH))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--required-query-count", type=int, default=DEFAULT_REQUIRED_QUERY_COUNT)
    parser.add_argument("--stage1-optimizer-report", default=None)
    parser.add_argument("--automated-suite-report", default=None)
    parser.add_argument("--stage2-synthesis-report", default=None)
    parser.add_argument("--human-review-status-report", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            build_publishable_package(
                suite_path=Path(args.suite),
                experiment_id=args.experiment_id,
                required_query_count=args.required_query_count,
                stage1_optimizer_report=optional_path(args.stage1_optimizer_report),
                automated_suite_report=optional_path(args.automated_suite_report),
                stage2_synthesis_report=optional_path(args.stage2_synthesis_report),
                human_review_status_report=optional_path(args.human_review_status_report),
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


def optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


if __name__ == "__main__":
    main()

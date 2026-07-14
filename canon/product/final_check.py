from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from canon.config import load_settings
from canon.product import industry_pilot, readiness, release_audit, report_io, service, smoke


DEFAULT_RECORDS = Path("reports") / "human_review_tasks_v1.json"


def run_final_check(
    mode: str = service.DEFAULT_MODE,
    records_path: Path = DEFAULT_RECORDS,
    steps: dict[str, Callable[[], dict[str, Any]]] | None = None,
    write_report: bool = True,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    runners = steps or default_steps(mode, records_path)
    results = []
    for identifier in [
        "product_smoke",
        "human_review_status",
        "industry_pilot",
        "product_readiness",
        "product_release_audit",
    ]:
        report = runners[identifier]()
        if identifier == "product_release_audit":
            audit = report
        results.append(step_result(identifier, report))
    output_dir = reports_dir or load_settings().reports_dir
    integrity_errors = review_task_packet_integrity(resolve_records_path(records_path, output_dir))
    blocking_reason = "review_task_packet_integrity" if integrity_errors else release_blocking_reason(audit)
    report = {
        "report_id": "product_final_check_v1",
        "mode": mode,
        "status": final_status(results, blocking_reason),
        "blocking_reason": blocking_reason,
        "integrity_errors": integrity_errors,
        "steps": results,
        "source_artifacts": source_artifacts(
            mode,
            output_dir,
            records_path,
        ),
        "next_required_action": audit.get("next_required_action", "Run product release audit."),
    }
    if write_report:
        report_io.write_json(output_dir / f"product_final_check_{mode}.json", report)
        report_io.write_markdown(output_dir / f"product_final_check_{mode}.md", render_markdown(report))
    return report


def default_steps(mode: str, records_path: Path) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "product_smoke": lambda: smoke.build_smoke_report(mode=mode),
        "human_review_status": lambda: industry_pilot.build_review_status_report(mode, records_path),
        "industry_pilot": lambda: industry_pilot.build_industry_pilot_report(mode, records_path),
        "product_readiness": lambda: readiness.build_readiness_report(mode),
        "product_release_audit": lambda: release_audit.build_release_audit(mode),
    }


def step_result(identifier: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": report.get("status", "missing"),
        "passed": report.get("status") == "pass" or (
            identifier == "human_review_status" and report.get("status") == "complete"
        ),
        "summary": compact_step_summary(identifier, report),
    }


def compact_step_summary(identifier: str, report: dict[str, Any]) -> dict[str, Any]:
    if identifier == "product_smoke":
        return {
            "query": report.get("query"),
            "check_count": len(report.get("checks") or []),
            "failed_checks": failed_check_ids(report),
            "result_summary": report.get("result_summary", {}),
        }
    if identifier == "human_review_status":
        return {
            "question_count": report.get("question_count", 0),
            "reviewed_question_count": report.get("reviewed_question_count", 0),
            "minimum_question_count": report.get("minimum_question_count", 30),
            "missing_field_count": report.get("missing_field_count", 0),
            "validation_error_count": report.get("validation_error_count", 0),
        }
    if identifier == "industry_pilot":
        return {
            "reviewed_question_count": report.get("reviewed_question_count", 0),
            "minimum_question_count": report.get("minimum_question_count", 30),
            "release_blockers": report.get("release_blockers", []),
            "failed_checks": failed_check_ids(report),
        }
    if identifier == "product_readiness":
        return {
            "check_count": len(report.get("checks") or []),
            "failed_checks": failed_check_ids(report),
        }
    if identifier == "product_release_audit":
        return {
            "claim": report.get("claim"),
            "next_required_action": report.get("next_required_action"),
            "blocked_components": [
                item.get("id")
                for item in report.get("components", [])
                if not item.get("passed")
            ],
        }
    return {}


def failed_check_ids(report: dict[str, Any]) -> list[str]:
    return [
        item.get("id")
        for item in report.get("checks", [])
        if not item.get("passed")
    ]


def release_blocking_reason(audit: dict[str, Any]) -> str:
    components = {item.get("id"): item for item in audit.get("components", [])}
    if not components.get("source_report_integrity", {}).get("passed"):
        return "source_report_integrity"
    if not components.get("product_smoke", {}).get("passed"):
        return "product_smoke"
    if not components.get("product_readiness", {}).get("passed"):
        return "product_readiness"
    if not components.get("human_review_status", {}).get("passed"):
        return "human_review_required"
    if not components.get("industry_pilot", {}).get("passed"):
        return "industry_pilot"
    return "none"


def final_status(results: list[dict[str, Any]], blocking_reason: str) -> str:
    if blocking_reason == "none":
        return "pass"
    if blocking_reason == "human_review_required":
        return "blocked_human_review"
    return "fail"


def should_exit_nonzero(status: str, no_fail: bool = False) -> bool:
    return status != "pass" and not no_fail


def review_task_packet_integrity(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "source": "human_review_tasks",
                "field": "path",
                "message": "human review task packet is missing",
                "value": str(path),
            }
        ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            {
                "source": "human_review_tasks",
                "field": "json",
                "message": "human review task packet could not be parsed",
                "value": str(exc),
            }
        ]
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return [
            {
                "source": "human_review_tasks",
                "field": "records",
                "message": "human review task packet must contain a records list",
                "value": type(records).__name__,
            }
        ]
    return []


def source_artifacts(mode: str, reports_dir: Path, records_path: Path) -> list[dict[str, Any]]:
    paths = [
        resolve_records_path(records_path, reports_dir),
        reports_dir / f"product_smoke_{mode}.json",
        reports_dir / "human_review_status_v1.json",
        reports_dir / "industry_pilot_acceptance_v1.json",
        reports_dir / f"product_readiness_{mode}.json",
        reports_dir / f"product_release_audit_{mode}.json",
        reports_dir / f"product_release_audit_{mode}.md",
    ]
    return [artifact_fingerprint(path, reports_dir) for path in paths]


def resolve_records_path(records_path: Path, reports_dir: Path) -> Path:
    if records_path.is_absolute():
        return records_path
    return reports_dir.parent / records_path


def artifact_fingerprint(path: Path, reports_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": relative_report_path(path, reports_dir),
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "path": relative_report_path(path, reports_dir),
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def relative_report_path(path: Path, reports_dir: Path) -> str:
    try:
        return str(path.relative_to(reports_dir.parent))
    except ValueError:
        return str(path)


def render_markdown(report: dict[str, Any]) -> str:
    steps = "\n".join(
        f"- {'PASS' if item['passed'] else 'BLOCKED'}: `{item['id']}` observed={item['status']}"
        for item in report["steps"]
    )
    artifacts = "\n".join(
        f"- `{item['path']}` sha256={item['sha256'] or 'missing'} size={item['size_bytes']}"
        for item in report.get("source_artifacts", [])
    )
    integrity = "\n".join(
        f"- `{item['source']}.{item['field']}` {item['message']}"
        for item in report.get("integrity_errors", [])
    ) or "- None"
    return f"""# CANON Product Final Check

Status: **{report['status']}**

Mode: `{report['mode']}`

Blocking reason: `{report['blocking_reason']}`

Next required action: {report['next_required_action']}

## Steps

{steps}

## Integrity Errors

{integrity}

## Source Artifacts

{artifacts}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON product final checks in release order.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_final_check(args.mode, args.records)
    print(json.dumps(report, indent=2))
    if should_exit_nonzero(report["status"], args.no_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

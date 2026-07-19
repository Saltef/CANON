from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


def build_release_audit(
    mode: str = service.DEFAULT_MODE,
    write_report: bool = True,
    acceptance_track: str | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    reports_dir = settings.reports_dir
    repo_root = repository_root(settings)
    track = acceptance_track or default_acceptance_track(mode)
    source_reports = acceptance_source_reports(reports_dir, mode, track)
    smoke = load_json(source_reports["product_smoke"]["path"])
    readiness = load_json(source_reports["product_readiness"]["path"])
    review_status = load_json(source_reports["human_review_status"]["path"])
    acceptance = load_json(source_reports["acceptance_review"]["path"])
    integrity = source_report_integrity(source_reports, {
        "product_smoke": smoke,
        "product_readiness": readiness,
        "human_review_status": review_status,
        "acceptance_review": acceptance,
    })
    components = [
        component(
            "source_report_integrity",
            not integrity,
            "pass" if not integrity else "fail",
            {"errors": integrity},
        ),
        public_repository_hygiene_component(repo_root),
        component("product_smoke", smoke.get("status") == "pass", smoke.get("status")),
        component("product_readiness", readiness.get("status") == "pass", readiness.get("status")),
        component(
            "human_review_status",
            human_review_status_passes(review_status),
            review_status.get("status"),
            {
                "reviewed_question_count": review_status.get("reviewed_question_count", 0),
                "minimum_question_count": review_status.get("minimum_question_count", 30),
                "missing_field_count": review_status.get("missing_field_count", 0),
                "validation_error_count": review_status.get("validation_error_count", 0),
            },
        ),
        acceptance_component(track, acceptance),
    ]
    report = {
        "report_id": "product_release_audit_v1",
        "mode": mode,
        "acceptance_track": track,
        "status": "pass" if all(item["passed"] for item in components) else "blocked",
        "claim": release_claim(components),
        "components": components,
        "next_required_action": next_required_action(components),
        "source_reports": {
            key: str(value["path"].relative_to(reports_dir.parent))
            for key, value in source_reports.items()
        },
        "repository_root": str(repo_root),
    }
    if write_report:
        write_json(reports_dir / f"product_release_audit_{mode}.json", report)
        write_markdown(reports_dir / f"product_release_audit_{mode}.md", render_markdown(report))
    return report


def acceptance_source_reports(reports_dir: Path, mode: str, track: str) -> dict[str, dict[str, Any]]:
    source_reports = {
        "product_smoke": {
            "path": reports_dir / f"product_smoke_{mode}.json",
            "expected_report_id": "product_smoke_v1",
            "expected_mode": mode,
        },
        "product_readiness": {
            "path": reports_dir / f"product_readiness_{mode}.json",
            "expected_report_id": None,
            "expected_mode": mode,
        },
    }
    if track == "intelligence":
        source_reports.update({
            "human_review_status": {
                "path": reports_dir / "intelligence_brief_review_status_v1.json",
                "expected_report_id": "intelligence_brief_review_status_v1",
                "expected_mode": None,
            },
            "acceptance_review": {
                "path": reports_dir / "intelligence_brief_feedback_v1.json",
                "expected_report_id": "intelligence_brief_feedback_v1",
                "expected_mode": None,
            },
        })
        return source_reports
    source_reports.update({
        "human_review_status": {
            "path": reports_dir / "human_review_status_v1.json",
            "expected_report_id": "human_review_status_v1",
            "expected_mode": mode,
        },
        "acceptance_review": {
            "path": reports_dir / "industry_pilot_acceptance_v1.json",
            "expected_report_id": "industry_pilot_acceptance_v1",
            "expected_mode": mode,
        },
    })
    return source_reports


def default_acceptance_track(mode: str) -> str:
    return "intelligence" if mode == "ai_infra_geo_risk_demo" else "industry"


def acceptance_component(track: str, report: dict[str, Any]) -> dict[str, Any]:
    if track == "intelligence":
        return component(
            "intelligence_feedback",
            report.get("status") == "ready",
            report.get("status"),
            {
                "record_count": report.get("record_count", 0),
                "reviewed_record_count": report.get("reviewed_record_count", 0),
                "regression_candidate_count": len(report.get("regression_candidates") or []),
            },
        )
    return component(
        "industry_pilot",
        report.get("status") == "pass",
        report.get("status"),
        {
            "release_blockers": report.get("release_blockers", []),
            "reviewed_question_count": report.get("reviewed_question_count", 0),
            "minimum_question_count": report.get("minimum_question_count", 30),
        },
    )


def repository_root(settings: Any) -> Path:
    root = getattr(settings, "root", None)
    if root:
        return Path(root)
    reports_dir = Path(settings.reports_dir)
    return reports_dir.parent if reports_dir.name == "reports" else reports_dir


def component(
    identifier: str,
    passed: bool,
    observed_status: Any,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "observed_status": observed_status or "missing",
        "details": details or {},
    }


def source_report_integrity(
    source_reports: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    for identifier, spec in source_reports.items():
        path = spec["path"]
        payload = payloads.get(identifier) or {}
        if not path.exists():
            errors.append(
                {
                    "source": identifier,
                    "field": "path",
                    "message": "source report is missing",
                    "value": str(path),
                }
            )
            continue
        if payload.get("_load_error"):
            errors.append(
                {
                    "source": identifier,
                    "field": "json",
                    "message": "source report could not be parsed",
                    "value": payload.get("_load_error"),
                }
            )
            continue
        expected_report_id = spec.get("expected_report_id")
        if expected_report_id and payload.get("report_id") != expected_report_id:
            errors.append(
                {
                    "source": identifier,
                    "field": "report_id",
                    "message": f"expected {expected_report_id}",
                    "value": payload.get("report_id"),
                }
            )
        expected_mode = spec.get("expected_mode")
        if expected_mode and payload.get("mode") != expected_mode:
            errors.append(
                {
                    "source": identifier,
                    "field": "mode",
                    "message": f"expected {expected_mode}",
                    "value": payload.get("mode"),
                }
            )
    return errors


def public_repository_hygiene_component(root: Path) -> dict[str, Any]:
    checks = public_repository_hygiene_checks(root)
    return component(
        "public_repository_hygiene",
        all(item["passed"] for item in checks),
        "pass" if all(item["passed"] for item in checks) else "blocked",
        {"checks": checks},
    )


def public_repository_hygiene_checks(root: Path) -> list[dict[str, Any]]:
    pyproject = load_pyproject(root / "pyproject.toml")
    return [
        hygiene_check("readme_present", (root / "README.md").exists(), "README.md is present."),
        hygiene_check(
            "security_policy_present",
            (root / "SECURITY.md").exists(),
            "SECURITY.md is present.",
        ),
        hygiene_check("env_example_present", (root / ".env.example").exists(), ".env.example is present."),
        hygiene_check(
            "gitignore_blocks_env_files",
            gitignore_blocks_env_files(root / ".gitignore"),
            ".gitignore blocks local env files.",
        ),
        hygiene_check("license_file_present", license_file_present(root), "A LICENSE file is present."),
        hygiene_check(
            "package_license_declared",
            package_license_declared(pyproject),
            "Package metadata declares license terms.",
        ),
    ]


def hygiene_check(identifier: str, passed: bool, description: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "description": description,
    }


def gitignore_blocks_env_files(path: Path) -> bool:
    if not path.exists():
        return False
    lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    return ".env" in lines and ".env.*" in lines and "!.env.example" in lines


def license_file_present(root: Path) -> bool:
    return any((root / name).exists() for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"])


def load_pyproject(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return {}


def package_license_declared(pyproject: dict[str, Any]) -> bool:
    project = pyproject.get("project") or {}
    return bool(project.get("license") or project.get("license-files"))


def human_review_status_passes(report: dict[str, Any]) -> bool:
    reviewed = int(report.get("reviewed_question_count") or 0)
    minimum = int(report.get("minimum_question_count") or 30)
    missing = int(report.get("missing_field_count") or 0)
    validation_errors = int(report.get("validation_error_count") or 0)
    return (
        report.get("status") == "complete"
        and reviewed >= minimum
        and missing == 0
        and validation_errors == 0
    )


def release_claim(components: list[dict[str, Any]]) -> str:
    if not next(item for item in components if item["id"] == "source_report_integrity")["passed"]:
        return "Product release is not ready: source reports are missing or inconsistent."
    if all(item["passed"] for item in components):
        return "Industry pilot complete: automated gates and human-reviewed acceptance criteria passed."
    if not next(item for item in components if item["id"] == "public_repository_hygiene")["passed"]:
        return "Product release is not ready: public repository hygiene is incomplete."
    automated = {
        item["id"]: item["passed"]
        for item in components
        if item["id"] in {"product_smoke", "product_readiness"}
    }
    if all(automated.values()):
        if next((item for item in components if item["id"] == "intelligence_feedback"), None):
            return "Automated product gates pass; intelligence review remains blocked on human labels."
        return "Automated product gates pass; industry pilot remains blocked on human-reviewed acceptance evidence."
    return "Product release is not ready: automated gates are incomplete or failing."


def next_required_action(components: list[dict[str, Any]]) -> str:
    by_id = {item["id"]: item for item in components}
    if not by_id.get("source_report_integrity", {}).get("passed"):
        return "Regenerate missing or inconsistent source reports before auditing release readiness."
    if not by_id.get("public_repository_hygiene", {}).get("passed"):
        failed = [
            item["id"]
            for item in by_id["public_repository_hygiene"]["details"].get("checks", [])
            if not item.get("passed")
        ]
        if set(failed) & {"license_file_present", "package_license_declared"}:
            return "Choose a license, add a LICENSE file, and declare the package license before public release."
        return f"Fix public repository hygiene checks: {', '.join(failed)}."
    if not by_id.get("product_smoke", {}).get("passed"):
        return "Run and fix python -m canon.product.smoke."
    if not by_id.get("product_readiness", {}).get("passed"):
        return "Run and fix python -m canon.product.readiness."
    if not by_id.get("human_review_status", {}).get("passed"):
        details = by_id["human_review_status"]["details"]
        return (
            "Complete human review labels for "
            f"{details.get('minimum_question_count', 30) - details.get('reviewed_question_count', 0)} "
            "remaining acceptance questions."
        )
    if not by_id.get("intelligence_feedback", {}).get("passed", True):
        details = by_id["intelligence_feedback"]["details"]
        return (
            "Complete intelligence review labels for "
            f"{details.get('record_count', 0) - details.get('reviewed_record_count', 0)} "
            "remaining brief task(s)."
        )
    if not by_id.get("industry_pilot", {}).get("passed", True):
        blockers = by_id["industry_pilot"]["details"].get("release_blockers", [])
        return f"Resolve industry pilot release blockers: {', '.join(blockers)}."
    return "No required action remains."


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": str(exc)}


def render_markdown(report: dict[str, Any]) -> str:
    components = "\n".join(
        f"- {'PASS' if item['passed'] else 'BLOCKED'}: `{item['id']}` observed={item['observed_status']}"
        for item in report["components"]
    )
    sources = "\n".join(f"- `{key}`: `{value}`" for key, value in report["source_reports"].items())
    return f"""# CANON Product Release Audit

Status: **{report['status']}**

Mode: `{report['mode']}`

Claim: {report['claim']}

Next required action: {report['next_required_action']}

## Components

{components}

## Source Reports

{sources}
"""


def write_json(path: Path, payload: object) -> None:
    report_io.write_json(path, payload)


def write_markdown(path: Path, text: str) -> None:
    report_io.write_markdown(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON product release audit.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_release_audit(args.mode)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

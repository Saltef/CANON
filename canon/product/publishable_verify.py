from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product.publishable_package import DEFAULT_EXPERIMENT_ID


DEFAULT_PACKAGE_PATH = Path("reports") / f"publishable_package_{DEFAULT_EXPERIMENT_ID}.json"
DEFAULT_OUTPUT_PATH = Path("reports") / "publishable_artifact_verification_v1.json"


def verify_publishable_artifacts(
    package_path: Path | None = None,
    output_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_package_path = resolve_inside_root(settings.root, package_path or DEFAULT_PACKAGE_PATH)
    package = load_json(resolved_package_path)
    artifact_rows = package.get("artifact_manifest", {}).get("artifacts") or []
    rows = [verify_artifact(settings.root, row) for row in artifact_rows]
    missing = [row for row in rows if row["status"] == "missing"]
    mismatched = [row for row in rows if row["status"] == "mismatch"]
    unsafe = [row for row in rows if row["status"] == "unsafe_path"]
    report = {
        "report_id": "publishable_artifact_verification_v1",
        "package_path": relative(settings.root, resolved_package_path),
        "package_report_id": package.get("report_id"),
        "package_status": package.get("status"),
        "suite_id": (package.get("suite") or {}).get("suite_id"),
        "status": verification_status(rows),
        "artifact_count": len(rows),
        "verified_count": sum(1 for row in rows if row["status"] == "verified"),
        "missing_count": len(missing),
        "mismatch_count": len(mismatched),
        "unsafe_path_count": len(unsafe),
        "rows": rows,
        "claim_boundary": {
            "meaning": "This verifies the frozen package artifact hashes and byte counts only.",
            "not_proven": [
                "human-review completeness",
                "factual correctness",
                "model superiority",
                "whether generated reports are publishable claims",
            ],
        },
    }
    if write_report:
        output = resolve_inside_root(settings.root, output_path or DEFAULT_OUTPUT_PATH)
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def verify_artifact(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    manifest_path = str(row.get("path") or "")
    try:
        path = resolve_inside_root(root, Path(manifest_path))
    except ValueError as exc:
        return {
            "path": manifest_path,
            "status": "unsafe_path",
            "message": str(exc),
            "expected_sha256": row.get("sha256"),
            "actual_sha256": None,
            "expected_bytes": row.get("bytes"),
            "actual_bytes": None,
        }
    if not path.exists():
        return {
            "path": manifest_path,
            "status": "missing",
            "message": "Artifact is missing.",
            "expected_sha256": row.get("sha256"),
            "actual_sha256": None,
            "expected_bytes": row.get("bytes"),
            "actual_bytes": None,
        }
    data = path.read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    actual_bytes = len(data)
    expected_sha = str(row.get("sha256") or "")
    try:
        expected_bytes = int(row.get("bytes"))
    except (TypeError, ValueError):
        expected_bytes = -1
    if actual_sha != expected_sha or actual_bytes != expected_bytes:
        return {
            "path": manifest_path,
            "status": "mismatch",
            "message": "Artifact hash or byte count differs from the package manifest.",
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
        }
    return {
        "path": manifest_path,
        "status": "verified",
        "message": "Artifact matches the package manifest.",
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
    }


def verification_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "failed_empty_manifest"
    if all(row["status"] == "verified" for row in rows):
        return "verified"
    return "failed"


def resolve_inside_root(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- `{row['path']}`: `{row['status']}`"
        for row in report["rows"]
    )
    return f"""# CANON Publishable Artifact Verification

Status: `{report['status']}`

Package: `{report['package_path']}`

Suite: `{report.get('suite_id')}`

Verified artifacts: `{report['verified_count']}` / `{report['artifact_count']}`

Missing artifacts: `{report['missing_count']}`

Mismatched artifacts: `{report['mismatch_count']}`

Unsafe paths: `{report['unsafe_path_count']}`

## Artifacts

{rows}

## Boundary

This verifies hashes and byte counts. It does not prove factual correctness,
human-review completeness, or model superiority.
"""


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "package_path": report["package_path"],
        "artifact_count": report["artifact_count"],
        "verified_count": report["verified_count"],
        "missing_count": report["missing_count"],
        "mismatch_count": report["mismatch_count"],
        "unsafe_path_count": report["unsafe_path_count"],
        "artifacts": {
            "json": "reports/publishable_artifact_verification_v1.json",
            "markdown": "reports/publishable_artifact_verification_v1.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a frozen CANON publishable package manifest.")
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    report = verify_publishable_artifacts(
        package_path=Path(args.package),
        output_path=Path(args.output),
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product.publishable_verify import DEFAULT_PACKAGE_PATH, verify_publishable_artifacts


DEFAULT_OUTPUT_DIR = Path("reports") / "publishable_bundle_canon_publishable_evidence_workflow_v1"
DEFAULT_ARCHIVE_PATH = Path("reports") / "publishable_bundle_canon_publishable_evidence_workflow_v1.zip"
DEFAULT_EXPORT_REPORT = Path("reports") / "publishable_bundle_export_v1.json"


def export_publishable_bundle(
    package_path: Path | None = None,
    output_dir: Path | None = None,
    archive_path: Path | None = None,
    write_archive: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_package_path = resolve_inside(settings.root, package_path or DEFAULT_PACKAGE_PATH)
    resolved_output_dir = resolve_inside(settings.root, output_dir or DEFAULT_OUTPUT_DIR)
    resolved_archive_path = resolve_inside(settings.root, archive_path or DEFAULT_ARCHIVE_PATH)
    verification = verify_publishable_artifacts(package_path=resolved_package_path, write_report=True)
    package = load_json(resolved_package_path)
    report = {
        "report_id": "publishable_bundle_export_v1",
        "status": "export_blocked_verification_failed",
        "package_path": relative(settings.root, resolved_package_path),
        "package_status": package.get("status"),
        "verification_status": verification.get("status"),
        "output_dir": relative(settings.root, resolved_output_dir),
        "archive_path": relative(settings.root, resolved_archive_path) if write_archive else None,
        "artifact_count": len((package.get("artifact_manifest") or {}).get("artifacts") or []),
        "copied_file_count": 0,
        "copied_files": [],
        "claim_boundary": claim_boundary(),
    }
    if verification.get("status") != "verified":
        if write_report:
            write_export_report(settings.root, report)
        return report
    copied = copy_bundle_files(settings.root, resolved_output_dir, resolved_package_path, package)
    report["status"] = "exported"
    report["copied_file_count"] = len(copied)
    report["copied_files"] = [relative(resolved_output_dir, path) for path in copied]
    if write_archive:
        write_zip(resolved_output_dir, resolved_archive_path, copied)
    if write_report:
        write_export_report(settings.root, report)
    return report


def copy_bundle_files(root: Path, output_dir: Path, package_path: Path, package: dict[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    copied.append(copy_source(root, output_dir, package_path))
    package_markdown = package_path.with_suffix(".md")
    if package_markdown.exists():
        copied.append(copy_source(root, output_dir, package_markdown))
    for row in (package.get("artifact_manifest") or {}).get("artifacts") or []:
        copied.append(copy_source(root, output_dir, resolve_inside(root, Path(str(row.get("path") or "")))))
    verification_json = root / "reports" / "publishable_artifact_verification_v1.json"
    verification_markdown = verification_json.with_suffix(".md")
    if verification_json.exists():
        copied.append(copy_source(root, output_dir, verification_json))
    if verification_markdown.exists():
        copied.append(copy_source(root, output_dir, verification_markdown))
    copied.append(write_bundle_readme(output_dir, package))
    return unique_paths(copied)


def copy_source(root: Path, output_dir: Path, source: Path) -> Path:
    resolved_source = resolve_inside(root, source)
    relative_source = resolved_source.relative_to(root.resolve())
    destination = resolve_inside(output_dir, output_dir / relative_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, destination)
    return destination


def write_bundle_readme(output_dir: Path, package: dict[str, Any]) -> Path:
    path = output_dir / "BUNDLE_README.md"
    text = f"""# CANON Publishable Evidence Bundle

Package status: `{package.get('status')}`

Suite: `{(package.get('suite') or {}).get('suite_id')}`

This bundle contains the package manifest, manifest-verified benchmark inputs,
generated benchmark/synthesis/review artifacts, and the verification report.

The bundle does not complete human review and does not prove factual correctness
or model superiority. Treat `publication_blocked_evidence_incomplete` as an
honest review handoff state until completed reviewer labels are imported.
"""
    report_io.write_markdown(path, text)
    return path


def write_zip(output_dir: Path, archive_path: Path, files: list[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique_paths(files):
            archive.write(path, path.relative_to(output_dir.resolve()).as_posix())


def write_export_report(root: Path, report: dict[str, Any]) -> None:
    output = root / DEFAULT_EXPORT_REPORT
    report_io.write_json(output, report)
    report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))


def render_markdown(report: dict[str, Any]) -> str:
    copied = "\n".join(f"- `{path}`" for path in report["copied_files"])
    return f"""# CANON Publishable Bundle Export

Status: `{report['status']}`

Package: `{report['package_path']}`

Verification: `{report['verification_status']}`

Bundle directory: `{report['output_dir']}`

Archive: `{report['archive_path']}`

Copied files: `{report['copied_file_count']}`

## Files

{copied}

## Boundary

This export packages already-frozen artifacts. It does not complete human
review, validate factual correctness, or establish model superiority.
"""


def claim_boundary() -> dict[str, Any]:
    return {
        "meaning": "This export packages a verified frozen artifact manifest for handoff.",
        "not_proven": [
            "human-review completeness",
            "factual correctness",
            "model superiority",
            "publication readiness without reviewer labels",
        ],
    }


def unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def resolve_inside(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes root: {path}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "package_status": report.get("package_status"),
        "verification_status": report.get("verification_status"),
        "artifact_count": report["artifact_count"],
        "copied_file_count": report["copied_file_count"],
        "output_dir": report["output_dir"],
        "archive_path": report["archive_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a verified CANON publishable evidence bundle.")
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE_PATH))
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    report = export_publishable_bundle(
        package_path=Path(args.package),
        output_dir=Path(args.output_dir),
        archive_path=Path(args.archive),
        write_archive=not args.no_archive,
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.corpus.build import run_phase16
from canon.ingest.flexible import SUPPORTED_FILE_EXTENSIONS, ingest_flexible_source, is_supported_file, profile_source
from canon.product import report_io
from canon.product.prehuman_check import run_prehuman_check


def run_mounted_corpus(
    input_path: Path,
    mode: str,
    corpus_id: str | None = None,
    input_format: str | None = None,
    domain: str | None = None,
    provenance: str = "mounted_drive",
    source_name: str | None = None,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
    profile_only: bool = False,
    corpus_only: bool = True,
    run_prehuman: bool = False,
    top_k: int = 10,
    candidate_k: int = 25,
) -> dict[str, Any]:
    settings = load_settings()
    resolved = resolve_mounted_path(input_path)
    corpus_id = corpus_id or f"{mode}_corpus"
    profile = profile_source(resolved, input_format=input_format)
    supported = supported_file_summary(resolved)
    report: dict[str, Any] = {
        "report_id": "mounted_corpus_v1",
        "status": "profiled" if profile_only else "complete",
        "input_path": str(resolved),
        "mode": mode,
        "corpus_id": corpus_id,
        "profile": profile,
        "supported_files": supported,
        "security_boundary": {
            "drive_access": "filesystem_mount_only",
            "credentials": "not_read_or_stored",
            "raw_files_committed": False,
            "notes": [
                "Mount Google Drive locally, then pass the mounted folder path to this command.",
                "CANON reads supported local files from the mounted path; it does not use Google APIs here.",
                "Raw and processed corpus outputs are under data/ and are gitignored by default.",
            ],
        },
    }
    if profile_only:
        write_report(settings, mode, report)
        return report
    ingest = ingest_flexible_source(
        input_path=resolved,
        mode=mode,
        input_format=input_format,
        domain=domain,
        provenance=provenance,
        source_name=source_name,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    corpus = run_phase16(
        corpus_id=corpus_id,
        from_modes=[mode],
        harvest=False,
        corpus_only=corpus_only,
        top_k=top_k,
    )
    report["ingest"] = compact_ingest(ingest)
    report["corpus"] = corpus.get("corpus", corpus)
    report["validation"] = corpus.get("validation", {})
    if run_prehuman:
        report["prehuman_check"] = run_prehuman_check(
            mode=corpus_id,
            benchmark_id=f"llm_judged_{corpus_id}",
            top_k=top_k,
            candidate_k=candidate_k,
        )
    write_report(settings, mode, report)
    return report


def resolve_mounted_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Mounted corpus path does not exist: {path}")
    if not resolved.is_dir() and not is_supported_file(resolved):
        raise ValueError(
            "Mounted corpus input must be a folder or supported file type: "
            f"{', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
        )
    return resolved


def supported_file_summary(path: Path) -> dict[str, Any]:
    files = [path] if path.is_file() else [child for child in path.rglob("*") if child.is_file()]
    supported = [child for child in files if is_supported_file(child)]
    unsupported = [child for child in files if not is_supported_file(child)]
    return {
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_extensions": sorted({child.suffix.lower() or "<none>" for child in supported}),
        "unsupported_extensions": sorted({child.suffix.lower() or "<none>" for child in unsupported})[:25],
        "sample_supported_files": [str(child) for child in supported[:10]],
        "limitations": [
            "Text-extractable mounted files include JSON, JSONL, CSV, TXT, Markdown, PDF, DOCX, XLSX, PPTX, HTML, notebooks, source code, and folders containing those files.",
            "Google native pointer files, images, and legacy presentations are detected but not treated as evidence text unless exported or OCR-extracted first.",
        ],
    }


def compact_ingest(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report.get("mode"),
        "raw_record_count": report.get("raw_record_count", 0),
        "normalized_record_count": report.get("normalized_record_count", 0),
        "skipped_record_count": report.get("skipped_record_count", 0),
        "skipped_reasons": report.get("skipped_reasons", {}),
        "work_count": report.get("work_count", 0),
        "chunk_count": report.get("chunk_count", 0),
        "limitations": report.get("limitations", []),
        "works_path": report.get("works_path"),
        "chunks_path": report.get("chunks_path"),
    }


def write_report(settings: Any, mode: str, report: dict[str, Any]) -> None:
    report_io.write_json(settings.reports_dir / f"mounted_corpus_{mode}.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CANON corpus from a locally mounted folder, including Google Drive mounts.")
    parser.add_argument("--input", required=True, help="Mounted folder or supported file path.")
    parser.add_argument("--mode", required=True, help="Ingest mode, e.g. ai_infra_geo_risk_v1.")
    parser.add_argument("--corpus-id", default=None, help="Corpus id; defaults to <mode>_corpus.")
    parser.add_argument("--format", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--provenance", default="mounted_drive")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--full-validation", action="store_true", help="Run full corpus validation instead of corpus-only build.")
    parser.add_argument("--run-prehuman", action="store_true", help="Run provisional pre-human evaluation after corpus build.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=25)
    args = parser.parse_args()
    report = run_mounted_corpus(
        input_path=Path(args.input),
        mode=args.mode,
        corpus_id=args.corpus_id,
        input_format=args.format,
        domain=args.domain,
        provenance=args.provenance,
        source_name=args.source_name,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        profile_only=args.profile_only,
        corpus_only=not args.full_validation,
        run_prehuman=args.run_prehuman,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

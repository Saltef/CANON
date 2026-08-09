from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.flexible import IGNORED_CORPUS_DIR_NAMES, is_supported_file, iter_corpus_files, resolve_input_path


MANIFEST_VERSION = "corpus_source_manifest_v1"
DEFAULT_HASH_BYTES = None


@dataclass(frozen=True)
class SourceManifestOptions:
    mode: str
    input_path: Path
    hash_file_bytes: int | None = DEFAULT_HASH_BYTES
    include_unsupported: bool = True


def build_source_manifest(options: SourceManifestOptions) -> dict[str, Any]:
    resolved = resolve_input_path(options.input_path)
    files = list_source_files(resolved, include_unsupported=options.include_unsupported)
    entries = [fingerprint_file(path, root=resolved, hash_file_bytes=options.hash_file_bytes) for path in files]
    supported = [entry for entry in entries if entry["supported"]]
    unsupported = [entry for entry in entries if not entry["supported"]]
    content_digest = stable_manifest_digest(supported)
    return {
        "report_id": MANIFEST_VERSION,
        "manifest_version": 1,
        "mode": options.mode,
        "input_path": str(resolved),
        "input_kind": "folder" if resolved.is_dir() else "file",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash_file_bytes": options.hash_file_bytes or "full_file",
        "file_count": len(entries),
        "supported_file_count": len(supported),
        "unsupported_file_count": len(unsupported),
        "content_digest": content_digest,
        "ignored_dir_names": sorted(IGNORED_CORPUS_DIR_NAMES),
        "entries": entries,
        "boundary": (
            "This manifest fingerprints local source files for refresh decisions. "
            "Processed corpus JSON remains the canonical retrievable corpus."
        ),
    }


def list_source_files(path: Path, include_unsupported: bool = True) -> list[Path]:
    if path.is_file():
        return [path] if include_unsupported or is_supported_file(path) else []
    files = iter_corpus_files(path)
    if include_unsupported:
        return files
    return [file_path for file_path in files if is_supported_file(file_path)]


def fingerprint_file(path: Path, *, root: Path, hash_file_bytes: int | None = DEFAULT_HASH_BYTES) -> dict[str, Any]:
    stat = path.stat()
    supported = is_supported_file(path)
    return {
        "relative_path": relative_path(path, root),
        "path": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "supported": supported,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": partial_sha256(path, hash_file_bytes) if supported else None,
    }


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def partial_sha256(path: Path, max_bytes: int | None = DEFAULT_HASH_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining is None or remaining > 0:
            read_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = handle.read(read_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def stable_manifest_digest(entries: list[dict[str, Any]]) -> str:
    rows = [
        {
            "relative_path": entry["relative_path"],
            "size_bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        for entry in sorted(entries, key=lambda row: row["relative_path"])
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_path(mode: str, reports_dir: Path | None = None) -> Path:
    base_dir = reports_dir or load_settings().reports_dir
    return base_dir / f"corpus_source_manifest_{mode}.json"


def load_previous_manifest(mode: str, reports_dir: Path | None = None) -> dict[str, Any] | None:
    path = manifest_path(mode, reports_dir=reports_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def write_source_manifest(manifest: dict[str, Any], reports_dir: Path | None = None) -> Path:
    path = manifest_path(str(manifest["mode"]), reports_dir=reports_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def diff_manifests(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    previous_entries = manifest_entries_by_path(previous)
    current_entries = manifest_entries_by_path(current)
    added = sorted(path for path in current_entries if path not in previous_entries)
    removed = sorted(path for path in previous_entries if path not in current_entries)
    changed = sorted(
        path
        for path in current_entries.keys() & previous_entries.keys()
        if comparable_entry(current_entries[path]) != comparable_entry(previous_entries[path])
    )
    unchanged = sorted(
        path
        for path in current_entries.keys() & previous_entries.keys()
        if comparable_entry(current_entries[path]) == comparable_entry(previous_entries[path])
    )
    return {
        "status": "changed" if previous is None or added or removed or changed else "unchanged",
        "previous_manifest_found": previous is not None,
        "previous_digest": previous.get("content_digest") if previous else None,
        "current_digest": current.get("content_digest"),
        "added_count": len(added),
        "changed_count": len(changed),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
        "unsupported_file_count": current.get("unsupported_file_count", 0),
        "added": added[:50],
        "changed": changed[:50],
        "removed": removed[:50],
        "boundary": "Refresh decisions use supported local files only; unsupported files are counted for user visibility.",
    }


def manifest_entries_by_path(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    entries = manifest.get("entries") or []
    return {
        str(entry.get("relative_path")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("relative_path") and entry.get("supported")
    }


def comparable_entry(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("size_bytes"),
        entry.get("sha256"),
    )

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.chunker import chunk_text
from canon.ingest.unstructured import sections_from_record, serialize_work, work_from_record
from canon.reports.data_card import document_type_profile_summary


SUPPORTED_FILE_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt", ".md", ".markdown"}
TEXT_FIELDS = ["text", "content", "body", "abstract", "description", "notes", "message", "comment"]
TITLE_FIELDS = ["title", "name", "subject", "headline", "file_name"]
DATE_FIELDS = ["year", "date", "created_at", "updated_at", "timestamp", "published_at"]
SOURCE_FIELDS = ["source_name", "source", "platform", "publisher", "organization", "system"]
AUTHOR_FIELDS = ["author", "authors", "owner", "created_by", "sender", "speaker"]
URL_FIELDS = ["url", "link", "permalink", "landing_page_url"]


def profile_source(input_path: Path, input_format: str | None = None, sample_size: int = 25) -> dict[str, Any]:
    resolved = resolve_input_path(input_path)
    records = load_source_records(resolved, input_format=input_format, profile_only=True)
    sample = records[:sample_size]
    field_counts = count_fields(sample)
    shape = infer_source_shape(resolved, sample, field_counts)
    mapping = proposed_mapping(shape, field_counts)
    warnings = profile_warnings(sample, mapping)
    return {
        "input_path": str(resolved),
        "input_kind": "folder" if resolved.is_dir() else "file",
        "detected_format": detect_format(resolved, input_format),
        "record_count": len(records),
        "sample_count": len(sample),
        "source_shape": shape,
        "field_counts": field_counts,
        "proposed_mapping": mapping,
        "chunking_strategy": chunking_strategy_for_shape(shape),
        "warnings": warnings,
    }


def ingest_flexible_source(
    input_path: Path,
    mode: str = "flexible_demo",
    input_format: str | None = None,
    domain: str | None = None,
    provenance: str | None = None,
    source_name: str | None = None,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    profile = profile_source(input_path, input_format=input_format)
    raw_records = load_source_records(resolve_input_path(input_path), input_format=input_format)
    normalized, skipped = normalize_records(
        raw_records,
        source_shape=profile["source_shape"],
        domain=domain,
        provenance=provenance,
        source_name=source_name,
    )
    chunk_size = chunk_tokens or int(settings.raw["ingest"]["chunk_tokens"])
    overlap = overlap_tokens if overlap_tokens is not None else int(settings.raw["ingest"]["chunk_overlap_tokens"])
    works = [work_from_record(record, index) for index, record in enumerate(normalized, start=1)]
    chunks = [
        chunk
        for record, work in zip(normalized, works, strict=True)
        for section, text in sections_from_record(record)
        for chunk in chunk_text(work, section, text, chunk_size, overlap)
    ]
    serialized_works = [serialize_work(work) for work in works]
    raw_path = settings.data_dir / "raw" / f"flexible_{mode}.json"
    unstructured_raw_path = settings.data_dir / "raw" / f"unstructured_{mode}.json"
    works_path = settings.data_dir / "processed" / f"works_{mode}.json"
    chunks_path = settings.data_dir / "processed" / f"chunks_{mode}.json"
    write_json(raw_path, normalized)
    write_json(unstructured_raw_path, normalized)
    write_json(works_path, serialized_works)
    write_json(chunks_path, [chunk.__dict__ for chunk in chunks])
    report = {
        "mode": mode,
        "input_path": str(resolve_input_path(input_path)),
        "profile": profile,
        "raw_record_count": len(raw_records),
        "normalized_record_count": len(normalized),
        "skipped_record_count": skipped["total"],
        "skipped_reasons": dict(sorted(skipped["reasons"].items())),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "raw_path": str(raw_path),
        "unstructured_raw_path": str(unstructured_raw_path),
        "works_path": str(works_path),
        "chunks_path": str(chunks_path),
        "document_type_profile": document_type_profile_summary(serialized_works),
        "limitations": flexible_limitations(profile, normalized, skipped, chunks),
    }
    write_json(settings.reports_dir / f"flexible_ingest_{mode}.json", report)
    return report


def resolve_input_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return resolved


def load_source_records(
    path: Path,
    input_format: str | None = None,
    profile_only: bool = False,
) -> list[dict[str, Any]]:
    if path.is_dir():
        records = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                records.extend(load_source_records(child, profile_only=profile_only))
        return records
    fmt = detect_format(path, input_format)
    if fmt == "jsonl":
        return load_jsonl(path)
    if fmt == "json":
        return load_json(path)
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [add_file_metadata(dict(row), path, index) for index, row in enumerate(csv.DictReader(handle), start=1)]
    if fmt in {"txt", "md", "markdown"}:
        return [record_from_text_file(path)]
    raise ValueError(f"Unsupported flexible ingest format: {fmt}")


def detect_format(path: Path, input_format: str | None = None) -> str:
    if input_format:
        return input_format.lower().lstrip(".")
    if path.is_dir():
        return "folder"
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "markdown":
        return "md"
    return suffix or "txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        rows.append(add_file_metadata(payload, path, line_number))
    return rows


def load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [add_file_metadata(expect_object(row, path, index), path, index) for index, row in enumerate(payload, start=1)]
    if isinstance(payload, dict):
        rows = payload.get("records") or payload.get("documents") or payload.get("items") or payload.get("data")
        if isinstance(rows, list):
            return [add_file_metadata(expect_object(row, path, index), path, index) for index, row in enumerate(rows, start=1)]
        return [add_file_metadata(payload, path, 1)]
    raise ValueError(f"{path}: JSON input must be an object or list")


def expect_object(value: Any, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{index}: expected object record")
    return value


def record_from_text_file(path: Path) -> dict[str, Any]:
    return {
        "id": f"file:{stable_digest(str(path))}",
        "title": path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
        "source_name": path.parent.name or "local_file",
        "document_type": "note" if path.suffix.lower() in {".md", ".markdown"} else "unknown_unstructured_text",
        "provenance": "user_file",
        "domain": "user_supplied",
        "text": path.read_text(encoding="utf-8"),
        "metadata": {
            "source_file": str(path),
            "file_extension": path.suffix.lower(),
        },
    }


def add_file_metadata(record: dict[str, Any], path: Path, index: int) -> dict[str, Any]:
    enriched = dict(record)
    metadata = dict(enriched.get("metadata") or {})
    metadata.setdefault("source_file", str(path))
    metadata.setdefault("source_row", index)
    enriched["metadata"] = metadata
    return enriched


def count_fields(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for key, value in record.items():
            if value is not None and str(value).strip():
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def infer_source_shape(path: Path, records: list[dict[str, Any]], field_counts: dict[str, int]) -> str:
    if path.is_dir():
        return "mixed_source"
    if not records:
        return "empty_source"
    keys = {key.lower() for key in field_counts}
    if {"sender", "timestamp", "message"} <= keys or {"speaker", "message"} <= keys:
        return "message_thread"
    if {"account_id", "opportunity_stage"} & keys or {"deal_id", "company", "notes"} & keys:
        return "crm_record"
    if {"doi", "abstract"} <= keys or {"authors", "abstract", "title"} <= keys:
        return "research_paper"
    if any(key in keys for key in ["sections", "section"]):
        return "sectioned_document"
    if path.suffix.lower() == ".csv":
        return "table_record"
    if path.suffix.lower() in {".txt", ".md", ".markdown"}:
        return "document_file"
    if len(field_counts) >= 8 and any(field in keys for field in TEXT_FIELDS):
        return "structured_records"
    return "unstructured_records"


def proposed_mapping(shape: str, field_counts: dict[str, int]) -> dict[str, Any]:
    fields = set(field_counts)
    text_fields = [field for field in TEXT_FIELDS if field in fields]
    title_field = first_present(TITLE_FIELDS, fields)
    date_field = first_present(DATE_FIELDS, fields)
    source_field = first_present(SOURCE_FIELDS, fields)
    author_field = first_present(AUTHOR_FIELDS, fields)
    url_field = first_present(URL_FIELDS, fields)
    return {
        "id_field": "id" if "id" in fields else None,
        "title_field": title_field,
        "text_fields": text_fields,
        "date_field": date_field,
        "source_field": source_field,
        "author_field": author_field,
        "url_field": url_field,
        "metadata_fields": sorted(fields - set(text_fields) - {field for field in [title_field, date_field, source_field, author_field, url_field] if field}),
        "document_type": default_document_type(shape),
        "confidence": mapping_confidence(shape, text_fields),
    }


def first_present(candidates: list[str], fields: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def mapping_confidence(shape: str, text_fields: list[str]) -> str:
    if shape == "empty_source" or not text_fields:
        return "low"
    if shape in {"crm_record", "message_thread", "research_paper", "sectioned_document", "document_file"}:
        return "high"
    return "medium"


def chunking_strategy_for_shape(shape: str) -> str:
    return {
        "message_thread": "speaker_turn_window",
        "crm_record": "field_aware_record",
        "research_paper": "section_sentence_boundary",
        "sectioned_document": "section_sentence_boundary",
        "document_file": "conservative_boundary",
        "table_record": "field_aware_record",
        "mixed_source": "per_record_profile",
    }.get(shape, "conservative_boundary")


def profile_warnings(records: list[dict[str, Any]], mapping: dict[str, Any]) -> list[str]:
    warnings = []
    if not records:
        warnings.append("No records were detected.")
    if not mapping.get("text_fields"):
        warnings.append("No obvious text field was detected; rows may be skipped.")
    if not mapping.get("source_field"):
        warnings.append("No source field was detected; source_name will be inferred.")
    if not any("provenance" in record for record in records):
        warnings.append("No explicit provenance field was detected.")
    return warnings


def normalize_records(
    records: list[dict[str, Any]],
    source_shape: str,
    domain: str | None,
    provenance: str | None,
    source_name: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = []
    skipped = {"total": 0, "reasons": {}}
    for index, record in enumerate(records, start=1):
        text_parts = text_parts_from_record(record, source_shape)
        if not text_parts:
            increment_skip(skipped, "missing_text")
            continue
        normalized.append(normalize_record(record, index, source_shape, text_parts, domain, provenance, source_name))
    return normalized, skipped


def text_parts_from_record(record: dict[str, Any], source_shape: str) -> list[tuple[str, str]]:
    sections = record.get("sections")
    if isinstance(sections, list):
        rows = []
        for index, section in enumerate(sections, start=1):
            if isinstance(section, dict):
                name = str(section.get("section") or section.get("title") or f"section_{index}")
                text = str(section.get("text") or section.get("content") or "")
            else:
                name = f"section_{index}"
                text = str(section)
            if text.strip():
                rows.append((name, text.strip()))
        if rows:
            return rows
    if source_shape in {"crm_record", "table_record", "structured_records"}:
        rows = [
            (key, str(value).strip())
            for key, value in record.items()
            if key not in {"metadata", "sections"}
            and value is not None
            and str(value).strip()
            and should_use_field_as_text(key, value)
        ]
        if rows:
            return rows
    for field in TEXT_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return [("body" if field != "abstract" else "abstract", str(value).strip())]
    return []


def should_use_field_as_text(key: str, value: Any) -> bool:
    if key in TEXT_FIELDS:
        return True
    text = str(value).strip()
    if len(text.split()) >= 4 and key not in {"id", "year", "date", "created_at", "url", "link"}:
        return True
    return False


def normalize_record(
    record: dict[str, Any],
    index: int,
    source_shape: str,
    text_parts: list[tuple[str, str]],
    domain: str | None,
    provenance: str | None,
    source_name: str | None,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    for key, value in record.items():
        if key not in {"text", "content", "body", "sections", "metadata"} and value is not None:
            metadata.setdefault(key, value)
    title = first_record_value(record, TITLE_FIELDS) or f"{source_shape.replace('_', ' ').title()} {index}"
    source = source_name or first_record_value(record, SOURCE_FIELDS) or metadata.get("source_file") or "user_supplied"
    document_type = str(record.get("document_type") or record.get("source_type") or default_document_type(source_shape))
    normalized = {
        "id": str(record.get("id") or stable_record_id(record, index)),
        "title": str(title),
        "year": first_record_value(record, DATE_FIELDS),
        "source_name": str(source),
        "document_type": document_type,
        "source_type": document_type,
        "provenance": str(provenance or record.get("provenance") or "user_supplied"),
        "domain": str(domain or record.get("domain") or "user_supplied"),
        "url": first_record_value(record, URL_FIELDS),
        "language": record.get("language") or record.get("lang") or "en",
        "sections": [{"section": section, "text": text} for section, text in text_parts],
        "metadata": {
            **metadata,
            "source_shape": source_shape,
            "chunking_strategy": chunking_strategy_for_shape(source_shape),
        },
    }
    authors = first_record_value(record, AUTHOR_FIELDS)
    if authors:
        normalized["authors"] = authors if isinstance(authors, list) else [str(authors)]
    return normalized


def default_document_type(source_shape: str) -> str:
    return {
        "message_thread": "transcript",
        "crm_record": "crm_record",
        "research_paper": "academic_article",
        "sectioned_document": "unknown_unstructured_text",
        "document_file": "note",
        "table_record": "record",
        "structured_records": "record",
        "mixed_source": "unknown_unstructured_text",
    }.get(source_shape, "unknown_unstructured_text")


def first_record_value(record: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return value
    return None


def stable_record_id(record: dict[str, Any], index: int) -> str:
    payload = json.dumps(
        {
            "index": index,
            "title": first_record_value(record, TITLE_FIELDS),
            "source": first_record_value(record, SOURCE_FIELDS),
            "text": text_parts_from_record(record, "unstructured_records"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"flex:{stable_digest(payload)}"


def stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def increment_skip(skipped: dict[str, Any], reason: str) -> None:
    skipped["total"] += 1
    skipped["reasons"][reason] = skipped["reasons"].get(reason, 0) + 1


def flexible_limitations(
    profile: dict[str, Any],
    normalized: list[dict[str, Any]],
    skipped: dict[str, Any],
    chunks: list[Any],
) -> list[str]:
    limitations = []
    if not normalized:
        limitations.append("No usable records were normalized from the source.")
    if skipped["total"]:
        limitations.append("Some records were skipped because usable text could not be identified.")
    if len(chunks) < len(normalized):
        limitations.append("Some normalized records produced no chunks; inspect mapping and chunking settings.")
    if profile["proposed_mapping"].get("confidence") != "high":
        limitations.append("Source-shape mapping was not high confidence and should be reviewed by a human.")
    if profile.get("source_shape") == "mixed_source":
        limitations.append("Folder inputs may combine different source shapes; inspect per-record metadata before production use.")
    return limitations


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and ingest user-owned files with flexible source-shape detection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", default="flexible_demo")
    parser.add_argument("--format", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if args.profile_only:
        payload = profile_source(Path(args.input), input_format=args.format)
    else:
        payload = ingest_flexible_source(
            input_path=Path(args.input),
            mode=args.mode,
            input_format=args.format,
            domain=args.domain,
            provenance=args.provenance,
            source_name=args.source_name,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

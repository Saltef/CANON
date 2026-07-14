from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.chunker import chunk_text
from canon.models import Work
from canon.reports.data_card import document_type_profile_summary


def ingest_unstructured_jsonl(
    input_path: Path,
    mode: str,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    raw = load_jsonl(input_path)
    chunk_size = chunk_tokens or int(settings.raw["ingest"]["chunk_tokens"])
    overlap = overlap_tokens if overlap_tokens is not None else int(settings.raw["ingest"]["chunk_overlap_tokens"])
    works = [work_from_record(record, index) for index, record in enumerate(raw, start=1)]
    chunks = [
        chunk
        for record, work in zip(raw, works, strict=True)
        for section, text in sections_from_record(record)
        for chunk in chunk_text(work, section, text, chunk_size, overlap)
    ]
    raw_path = settings.data_dir / "raw" / f"unstructured_{mode}.json"
    works_path = settings.data_dir / "processed" / f"works_{mode}.json"
    chunks_path = settings.data_dir / "processed" / f"chunks_{mode}.json"
    write_json(raw_path, raw)
    write_json(works_path, [serialize_work(work) for work in works])
    write_json(chunks_path, [chunk.__dict__ for chunk in chunks])
    report = {
        "mode": mode,
        "input_path": str(input_path),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "document_type_profile": document_type_profile_summary([serialize_work(work) for work in works]),
        "schema": {
            "required_any_text_field": ["text", "content", "body", "sections"],
            "recommended_metadata": ["id", "title", "year", "source_name", "document_type", "provenance", "domain", "url"],
        },
        "limitations": limitations(raw, works, chunks),
    }
    write_json(settings.reports_dir / f"unstructured_ingest_{mode}.json", report)
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
        if not any(row.get(field) for field in ["text", "content", "body", "sections"]):
            raise ValueError(f"{path}:{line_number}: row must include text, content, body, or sections")
        rows.append(row)
    return rows


def work_from_record(record: dict[str, Any], index: int) -> Work:
    raw = dict(record.get("metadata") or {})
    for key in ["document_type", "source_type", "provenance", "domain", "jurisdiction", "publisher", "organization"]:
        if record.get(key) is not None:
            raw[key] = record.get(key)
    if record.get("document_type") and not raw.get("source_type"):
        raw["source_type"] = record["document_type"]
    return Work(
        id=str(record.get("id") or stable_record_id(record, index)),
        doi=optional_str(record.get("doi")),
        title=str(record.get("title") or f"Untitled unstructured document {index}"),
        year=parse_year(record.get("year") or record.get("date")),
        language=optional_str(record.get("language") or "en"),
        abstract=abstract_from_record(record),
        source_name=optional_str(record.get("source_name") or record.get("source") or record.get("publisher")),
        is_open_access=bool(record.get("is_open_access", True)),
        is_retracted=bool(record.get("is_retracted", False)),
        cited_by_count=int(record.get("cited_by_count") or 0),
        referenced_work_count=int(record.get("referenced_work_count") or len(record.get("references") or [])),
        pdf_url=optional_str(record.get("pdf_url")),
        landing_page_url=optional_str(record.get("landing_page_url") or record.get("url")),
        raw=raw,
        author_display_names=list(record.get("author_display_names") or record.get("authors") or []),
        author_count=int(record.get("author_count") or len(record.get("authors") or [])),
        max_author_cited_by_count=int(record.get("max_author_cited_by_count") or 0),
        max_author_works_count=int(record.get("max_author_works_count") or 0),
    )


def sections_from_record(record: dict[str, Any]) -> list[tuple[str, str]]:
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
                rows.append((name, text))
        if rows:
            return rows
    text = str(record.get("text") or record.get("content") or record.get("body") or "")
    section = str(record.get("section") or "body")
    return [(section, text)] if text.strip() else []


def abstract_from_record(record: dict[str, Any], max_chars: int = 2000) -> str | None:
    if record.get("abstract"):
        return str(record["abstract"])
    text = " ".join(text for _, text in sections_from_record(record))
    return text[:max_chars] if text else None


def serialize_work(work: Work) -> dict[str, Any]:
    return {
        "id": work.id,
        "doi": work.doi,
        "title": work.title,
        "year": work.year,
        "language": work.language,
        "abstract": work.abstract,
        "source_name": work.source_name,
        "is_open_access": work.is_open_access,
        "is_retracted": work.is_retracted,
        "cited_by_count": work.cited_by_count,
        "referenced_work_count": work.referenced_work_count,
        "pdf_url": work.pdf_url,
        "landing_page_url": work.landing_page_url,
        "raw": work.raw,
        "author_display_names": work.author_display_names,
        "author_count": work.author_count,
        "max_author_cited_by_count": work.max_author_cited_by_count,
        "max_author_works_count": work.max_author_works_count,
        "domain": work.raw.get("domain") if work.raw else None,
        "provenance": work.raw.get("provenance") if work.raw else None,
    }


def limitations(raw: list[dict[str, Any]], works: list[Work], chunks: list[Any]) -> list[str]:
    limits = []
    if not raw:
        limits.append("No records were ingested.")
    if len(chunks) < len(works):
        limits.append("Some works produced no chunks; inspect text fields and chunking settings.")
    missing_type = sum(1 for record in raw if not record.get("document_type") and not (record.get("metadata") or {}).get("document_type"))
    if raw and missing_type / len(raw) > 0.2:
        limits.append("More than 20% of records lack explicit document_type metadata.")
    missing_provenance = sum(1 for record in raw if not record.get("provenance") and not (record.get("metadata") or {}).get("provenance"))
    if raw and missing_provenance / len(raw) > 0.2:
        limits.append("More than 20% of records lack provenance metadata.")
    return limits


def stable_record_id(record: dict[str, Any], index: int) -> str:
    payload = json.dumps(
        {
            "index": index,
            "title": record.get("title"),
            "source": record.get("source_name") or record.get("source"),
            "text": record.get("text") or record.get("content") or record.get("body"),
            "sections": record.get("sections"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"unstructured:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    digits = "".join(character for character in text[:10] if character.isdigit())
    if len(digits) >= 4:
        year = int(digits[:4])
        if 1000 <= year <= 3000:
            return year
    return None


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest generic unstructured JSONL into CANON processed corpus files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", default="unstructured_demo")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            ingest_unstructured_jsonl(
                input_path=Path(args.input),
                mode=args.mode,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

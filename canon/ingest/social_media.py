from __future__ import annotations

import argparse
import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.chunker import chunk_text
from canon.ingest.unstructured import (
    sections_from_record,
    serialize_work,
    work_from_record,
)
from canon.reports.data_card import document_type_profile_summary


PUBLIC_VISIBILITIES = {"public", "public_web", "open"}
PRIVATE_VISIBILITIES = {"private", "direct", "dm", "friends", "followers_only", "deleted"}
TEXT_FIELDS = ["text", "content", "body", "message", "comment", "post_text"]
SUPPORTED_FORMATS = {"jsonl", "json", "csv", "html"}
APPROVED_TERMS_STATUS = {"allowed", "reviewed_allowed", "not_applicable", "fixture"}
APPROVED_ROBOTS_STATUS = {"allowed", "not_applicable", "fixture"}
KNOWN_LICENSE_STATUS = {"public_domain", "cc_by", "cc0", "platform_terms_allowed", "fair_use_reviewed", "fixture"}
KNOWN_CONSENT_BASIS = {"public_post", "platform_export", "research_fixture", "user_provided_public_archive"}


def ingest_social_media(
    input_path: Path,
    mode: str = "social_media_demo",
    input_format: str | None = None,
    platform: str | None = None,
    collection_method: str = "export_or_allowed_public_capture",
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> dict[str, Any]:
    resolved_format = (input_format or input_path.suffix.lstrip(".") or "jsonl").lower()
    if resolved_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported social-media input format: {resolved_format}")
    raw_records = load_records(input_path, resolved_format)
    normalized, skipped = normalize_records(
        raw_records,
        platform=platform,
        collection_method=collection_method,
    )
    ingest_report = write_processed_social_media(
        normalized,
        mode=mode,
        input_path=input_path,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    acquisition = collection_audit(
        raw_records,
        normalized,
        skipped,
        collection_method=collection_method,
    )
    report = {
        "mode": mode,
        "input_path": str(input_path),
        "input_format": resolved_format,
        "collection_method": collection_method,
        "record_count": len(raw_records),
        "normalized_public_record_count": len(normalized),
        "skipped_record_count": skipped["total"],
        "skipped_reasons": dict(sorted(skipped["reasons"].items())),
        "platform_counts": count_values(normalized, "platform"),
        "thread_count": len({record.get("thread_id") for record in normalized if record.get("thread_id")}),
        "collection_audit": acquisition,
        "ingest_report": ingest_report,
        "schema": {
            "required_text_fields_any": TEXT_FIELDS,
            "required_public_visibility": sorted(PUBLIC_VISIBILITIES),
            "recommended_metadata": [
                "id",
                "platform",
                "url",
                "created_at",
                "author_id",
                "thread_id",
                "parent_id",
                "like_count",
                "share_count",
                "reply_count",
                "language",
                "terms_status",
                "robots_txt_status",
                "license",
                "consent_basis",
                "sampling_frame",
            ],
        },
        "limitations": social_media_limitations(raw_records, normalized, skipped),
    }
    settings = load_settings()
    output_path = settings.reports_dir / f"social_media_ingest_{mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def collection_audit(
    raw_records: list[dict[str, Any]],
    normalized: list[dict[str, Any]],
    skipped: dict[str, Any],
    collection_method: str,
) -> dict[str, Any]:
    record_count = len(raw_records)
    normalized_count = len(normalized)
    platform_count = len(count_values(normalized, "platform"))
    terms_count = count_metadata(raw_records, "terms_status", APPROVED_TERMS_STATUS)
    robots_count = count_metadata(raw_records, "robots_txt_status", APPROVED_ROBOTS_STATUS)
    license_count = count_metadata(raw_records, "license", KNOWN_LICENSE_STATUS)
    consent_count = count_metadata(raw_records, "consent_basis", KNOWN_CONSENT_BASIS)
    sampling_known_count = count_present(raw_records, "sampling_frame")
    privacy = privacy_risk_summary(raw_records, normalized)
    coverage = {
        "terms_review_share": ratio(terms_count, record_count),
        "robots_allowed_share": ratio(robots_count, record_count),
        "license_known_share": ratio(license_count, record_count),
        "consent_basis_share": ratio(consent_count, record_count),
        "sampling_frame_share": ratio(sampling_known_count, record_count),
    }
    status = collection_status(
        record_count=record_count,
        normalized_count=normalized_count,
        platform_count=platform_count,
        skipped=skipped,
        coverage=coverage,
        privacy=privacy,
    )
    return {
        "status": status,
        "collection_method": collection_method,
        "record_count": record_count,
        "normalized_public_record_count": normalized_count,
        "platform_count": platform_count,
        "skipped_record_count": skipped.get("total", 0),
        "skipped_reasons": dict(sorted((skipped.get("reasons") or {}).items())),
        "metadata_coverage": coverage,
        "privacy_risk": privacy,
        "claim_boundaries": collection_claim_boundaries(status),
        "recommended_actions": collection_recommended_actions(status, coverage, privacy, skipped),
    }


def collection_status(
    record_count: int,
    normalized_count: int,
    platform_count: int,
    skipped: dict[str, Any],
    coverage: dict[str, float],
    privacy: dict[str, Any],
) -> str:
    if normalized_count <= 0:
        return "blocked_no_public_records"
    if privacy.get("direct_identifier_record_count", 0) > 0 or privacy.get("contact_pattern_record_count", 0) > 0:
        return "needs_privacy_review"
    if skipped.get("reasons", {}).get("non_public_visibility", 0) > 0:
        return "fixture_only_private_records_skipped"
    enough_records = record_count >= 50 and normalized_count >= 50
    compliance_ready = (
        coverage.get("terms_review_share", 0.0) >= 0.8
        and coverage.get("robots_allowed_share", 0.0) >= 0.8
        and coverage.get("consent_basis_share", 0.0) >= 0.8
    )
    if enough_records and platform_count >= 2 and compliance_ready:
        return "ready_for_exploratory_public_ingest"
    return "fixture_only_or_needs_collection_metadata"


def privacy_risk_summary(raw_records: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> dict[str, Any]:
    direct_identifier_fields = {"email", "phone", "real_name", "ip_address", "location_exact"}
    direct_identifier_count = 0
    contact_pattern_count = 0
    for record in raw_records:
        if any(record.get(field) for field in direct_identifier_fields):
            direct_identifier_count += 1
        text = first_text(record, TEXT_FIELDS)
        if contains_contact_pattern(text):
            contact_pattern_count += 1
    author_hash_count = sum(1 for record in normalized if (record.get("metadata") or {}).get("author_hash"))
    return {
        "direct_identifier_record_count": direct_identifier_count,
        "contact_pattern_record_count": contact_pattern_count,
        "author_hash_count": author_hash_count,
        "author_hash_share": ratio(author_hash_count, len(normalized)),
        "raw_author_ids_retained": False,
    }


def contains_contact_pattern(text: str) -> bool:
    if not text:
        return False
    email_like = "@" in text and "." in text
    phone_like = any(character.isdigit() for character in text) and any(marker in text for marker in ["-", "(", ")"])
    return email_like or phone_like


def collection_claim_boundaries(status: str) -> list[str]:
    return [
        "Allowed: use this audit to decide whether public-expression records can enter a fixture or exploratory public-opinion corpus.",
        "Allowed: retain hashed author identifiers for concentration diagnostics while excluding raw author IDs from processed records.",
        "Not allowed: treat public visibility alone as permission for live scraping, broad public-opinion inference, or legal compliance.",
        f"Current acquisition status: {status}.",
    ]


def collection_recommended_actions(
    status: str,
    coverage: dict[str, float],
    privacy: dict[str, Any],
    skipped: dict[str, Any],
) -> list[str]:
    actions = []
    if coverage.get("terms_review_share", 0.0) < 0.8:
        actions.append("Record platform terms or data-license review status before using live scraped data.")
    if coverage.get("robots_allowed_share", 0.0) < 0.8:
        actions.append("Record robots.txt or API-permission status for web-captured records.")
    if coverage.get("consent_basis_share", 0.0) < 0.8:
        actions.append("Record consent/public-basis metadata such as public_post, platform_export, or research_fixture.")
    if coverage.get("sampling_frame_share", 0.0) < 0.5:
        actions.append("Document sampling frame, query terms, time window, and inclusion/exclusion criteria.")
    if privacy.get("direct_identifier_record_count", 0) or privacy.get("contact_pattern_record_count", 0):
        actions.append("Review and redact direct identifiers or contact details before processing.")
    if (skipped.get("reasons") or {}).get("non_public_visibility", 0):
        actions.append("Keep skipped private/deleted records out of processed corpora and audit collection source.")
    if not actions:
        actions.append("Proceed only with aggregate public-opinion analysis and keep claim boundaries visible.")
    if status.startswith("blocked"):
        actions.append("Do not build retrieval or synthesis artifacts until public records are available.")
    return actions


def write_processed_social_media(
    records: list[dict[str, Any]],
    mode: str,
    input_path: Path,
    chunk_tokens: int | None,
    overlap_tokens: int | None,
) -> dict[str, Any]:
    settings = load_settings()
    chunk_size = chunk_tokens or int(settings.raw["ingest"]["chunk_tokens"])
    overlap = overlap_tokens if overlap_tokens is not None else int(settings.raw["ingest"]["chunk_overlap_tokens"])
    works = [work_from_record(record, index) for index, record in enumerate(records, start=1)]
    chunks = [
        chunk
        for record, work in zip(records, works, strict=True)
        for section, text in sections_from_record(record)
        for chunk in chunk_text(work, section, text, chunk_size, overlap)
    ]
    raw_path = settings.data_dir / "raw" / f"unstructured_{mode}.json"
    social_raw_path = settings.data_dir / "raw" / f"social_media_{mode}.json"
    works_path = settings.data_dir / "processed" / f"works_{mode}.json"
    chunks_path = settings.data_dir / "processed" / f"chunks_{mode}.json"
    serialized_works = [serialize_work(work) for work in works]
    write_json(raw_path, records)
    write_json(social_raw_path, records)
    write_json(works_path, serialized_works)
    write_json(chunks_path, [chunk.__dict__ for chunk in chunks])
    return {
        "mode": mode,
        "input_path": str(input_path),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "raw_path": str(raw_path),
        "social_raw_path": str(social_raw_path),
        "works_path": str(works_path),
        "chunks_path": str(chunks_path),
        "document_type_profile": document_type_profile_summary(serialized_works),
        "limitations": social_media_processed_limitations(records, chunks),
    }


def load_records(path: Path, input_format: str) -> list[dict[str, Any]]:
    if input_format == "jsonl":
        return load_jsonl(path)
    if input_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [expect_object(row, path, index) for index, row in enumerate(payload, start=1)]
        if isinstance(payload, dict):
            rows = payload.get("records") or payload.get("posts") or payload.get("comments")
            if isinstance(rows, list):
                return [expect_object(row, path, index) for index, row in enumerate(rows, start=1)]
        raise ValueError(f"{path}: JSON input must be a list or an object with records/posts/comments")
    if input_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if input_format == "html":
        return PublicPostHTMLParser.parse(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported format: {input_format}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        rows.append(expect_object(row, path, line_number))
    return rows


def expect_object(row: Any, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{index}: expected object record")
    return row


def normalize_records(
    records: list[dict[str, Any]],
    platform: str | None,
    collection_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = []
    skipped = {"total": 0, "reasons": {}}
    for index, record in enumerate(records, start=1):
        reason = skip_reason(record)
        if reason:
            skipped["total"] += 1
            skipped["reasons"][reason] = skipped["reasons"].get(reason, 0) + 1
            continue
        normalized.append(normalize_record(record, index, platform, collection_method))
    return normalized, skipped


def skip_reason(record: dict[str, Any]) -> str | None:
    text = first_text(record, TEXT_FIELDS)
    if not text:
        return "missing_text"
    visibility = visibility_value(record)
    if visibility in PRIVATE_VISIBILITIES:
        return "non_public_visibility"
    if visibility not in PUBLIC_VISIBILITIES and record.get("is_public") is not True:
        return "missing_public_visibility"
    return None


def normalize_record(
    record: dict[str, Any],
    index: int,
    platform: str | None,
    collection_method: str,
) -> dict[str, Any]:
    platform_name = str(platform or record.get("platform") or record.get("source") or "unknown_platform")
    record_id = str(record.get("id") or record.get("post_id") or record.get("comment_id") or stable_social_id(record, index))
    text = first_text(record, TEXT_FIELDS)
    author_hash = hash_author(record.get("author_id") or record.get("author") or record.get("username"))
    visibility = visibility_value(record) if visibility_value(record) in PUBLIC_VISIBILITIES else "public"
    thread_id = optional_str(record.get("thread_id") or record.get("conversation_id"))
    parent_id = optional_str(record.get("parent_id") or record.get("in_reply_to_id"))
    created_at = optional_str(record.get("created_at") or record.get("timestamp") or record.get("date"))
    document_type = social_document_type(record)
    metadata = nested_metadata(record)
    return {
        "id": f"social:{platform_name}:{record_id}",
        "title": str(record.get("title") or f"{platform_name} public {document_type} {record_id}"),
        "year": record.get("year") or record.get("created_at") or record.get("timestamp") or record.get("date"),
        "source_name": platform_name,
        "document_type": document_type,
        "source_type": "social_media" if document_type == "social_media_post" else "forum_comment",
        "provenance": "social_media" if document_type == "social_media_post" else "comment",
        "domain": str(record.get("domain") or "public_opinion"),
        "platform": platform_name,
        "visibility": visibility,
        "collection_method": collection_method,
        "thread_id": thread_id,
        "parent_id": parent_id,
        "created_at": created_at,
        "url": record.get("url") or record.get("permalink"),
        "language": record.get("language") or record.get("lang") or "en",
        "text": text,
        "metadata": {
            "platform": platform_name,
            "visibility": visibility,
            "collection_method": collection_method,
            "terms_status": optional_str(record.get("terms_status") or metadata.get("terms_status")),
            "robots_txt_status": optional_str(record.get("robots_txt_status") or metadata.get("robots_txt_status")),
            "license": optional_str(record.get("license") or metadata.get("license")),
            "consent_basis": optional_str(record.get("consent_basis") or metadata.get("consent_basis")),
            "sampling_frame": optional_str(record.get("sampling_frame") or metadata.get("sampling_frame")),
            "collection_query": optional_str(record.get("collection_query") or metadata.get("collection_query")),
            "collection_window": optional_str(record.get("collection_window") or metadata.get("collection_window")),
            "author_hash": author_hash,
            "thread_id": thread_id,
            "parent_id": parent_id,
            "created_at": created_at,
            "like_count": numeric(record.get("like_count") or record.get("likes")),
            "share_count": numeric(record.get("share_count") or record.get("shares") or record.get("reposts")),
            "reply_count": numeric(record.get("reply_count") or record.get("replies") or record.get("comments")),
            "quote_count": numeric(record.get("quote_count") or record.get("quotes")),
        },
    }


def social_document_type(record: dict[str, Any]) -> str:
    value = str(record.get("document_type") or record.get("source_type") or record.get("type") or "").lower()
    platform = str(record.get("platform") or record.get("source") or "").lower()
    if value in {"forum_comment", "comment", "forum"} or platform in {"reddit", "hackernews", "forum"}:
        return "forum_comment"
    return "social_media_post"


def first_text(record: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def visibility_value(record: dict[str, Any]) -> str:
    if record.get("is_public") is True:
        return "public"
    return str(record.get("visibility") or record.get("privacy") or "").strip().lower()


def stable_social_id(record: dict[str, Any], index: int) -> str:
    payload = json.dumps(
        {
            "index": index,
            "platform": record.get("platform") or record.get("source"),
            "url": record.get("url") or record.get("permalink"),
            "text": first_text(record, TEXT_FIELDS),
            "created_at": record.get("created_at") or record.get("timestamp") or record.get("date"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def hash_author(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()[:16]


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def numeric(value: Any) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_values(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def count_metadata(records: list[dict[str, Any]], key: str, allowed: set[str]) -> int:
    return sum(1 for record in records if metadata_value(record, key) in allowed)


def count_present(records: list[dict[str, Any]], key: str) -> int:
    return sum(1 for record in records if bool(metadata_value(record, key)))


def metadata_value(record: dict[str, Any], key: str) -> str:
    return str(record.get(key) or nested_metadata(record).get(key) or "").strip().lower()


def nested_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def social_media_limitations(
    raw_records: list[dict[str, Any]],
    normalized: list[dict[str, Any]],
    skipped: dict[str, Any],
) -> list[str]:
    limits = [
        "Social-media text is public-expression evidence, not factual authority or representative polling by default.",
        "Platform terms, robots rules, consent, privacy, and legal requirements must be checked before live scraping.",
    ]
    if skipped["total"]:
        limits.append("Some records were skipped because they lacked text or explicit public visibility.")
    if len(normalized) < 50:
        limits.append("Corpus is too small for public-opinion inference; use it only as a pipeline fixture.")
    if raw_records and len(normalized) / len(raw_records) < 0.8:
        limits.append("More than 20% of source records were skipped; inspect collection metadata.")
    if len(count_values(normalized, "platform")) < 2:
        limits.append("Single-platform data cannot support broad public-opinion claims.")
    return limits


def social_media_processed_limitations(records: list[dict[str, Any]], chunks: list[Any]) -> list[str]:
    limits = []
    if len(chunks) < len(records):
        limits.append("Some public social records produced no chunks; inspect text fields and chunking settings.")
    if len(records) < 50:
        limits.append("Processed social corpus is too small for public-opinion inference.")
    if len(count_values(records, "platform")) < 2:
        limits.append("Processed social corpus is single-platform; do not generalize to broad public opinion.")
    return limits


class PublicPostHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.active_field: str | None = None
        self.buffer: list[str] = []

    @classmethod
    def parse(cls, html: str) -> list[dict[str, Any]]:
        parser = cls()
        parser.feed(html)
        parser.close()
        return parser.records

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        role = attrs_dict.get("data-canon-social")
        if role == "post":
            self.current = {
                "id": attrs_dict.get("data-id"),
                "platform": attrs_dict.get("data-platform"),
                "url": attrs_dict.get("data-url"),
                "created_at": attrs_dict.get("data-created-at"),
                "visibility": attrs_dict.get("data-visibility") or "public",
                "is_public": True,
            }
        elif self.current is not None and role in {"text", "author", "thread_id"}:
            self.active_field = role
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.active_field:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is not None and self.active_field:
            self.current[self.active_field if self.active_field != "text" else "text"] = " ".join(
                piece.strip() for piece in self.buffer if piece.strip()
            )
            self.active_field = None
            self.buffer = []
            return
        if self.current is not None and tag in {"article", "div"} and self.current.get("text"):
            self.records.append(self.current)
            self.current = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest public social-media or forum records into CANON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", default="social_media_demo")
    parser.add_argument("--format", default=None, choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--platform", default=None)
    parser.add_argument("--collection-method", default="export_or_allowed_public_capture")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            ingest_social_media(
                input_path=Path(args.input),
                mode=args.mode,
                input_format=args.format,
                platform=args.platform,
                collection_method=args.collection_method,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

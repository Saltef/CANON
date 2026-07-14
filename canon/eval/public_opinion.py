from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.ingest.social_media import (
    APPROVED_ROBOTS_STATUS,
    APPROVED_TERMS_STATUS,
    KNOWN_CONSENT_BASIS,
    KNOWN_LICENSE_STATUS,
    PUBLIC_VISIBILITIES,
    contains_contact_pattern,
)
from canon.safety.scanner import scan_text


THEME_LEXICON = {
    "safety_risk": ("safety", "fire", "risk", "emergency", "harm", "hazard"),
    "trust_accountability": ("trust", "distrust", "clear", "transparent", "answers", "commitment"),
    "economic_material": ("jobs", "rent", "affordability", "cost", "price", "wage", "market"),
    "reliability_benefit": ("reliability", "grid", "benefit", "resilience", "backup"),
    "process_governance": ("planning board", "public meeting", "permit", "hearing", "regulator"),
    "support_condition": ("support", "if", "provided that", "commitment", "publish"),
    "opposition_concern": ("oppose", "against", "worry", "concern", "distrust", "split"),
}

SUPPORT_TERMS = ("support", "favor", "benefit", "approve", "help")
OPPOSE_TERMS = ("oppose", "against", "reject", "distrust", "worry", "concern")
CONDITION_TERMS = ("if", "provided that", "unless", "but", "only if", "commitment")


def analyze_public_opinion(mode: str) -> dict[str, Any]:
    settings = load_settings()
    works = load_json(settings.data_dir / "processed" / f"works_{mode}.json", [])
    chunks = load_json(settings.data_dir / "processed" / f"chunks_{mode}.json", [])
    work_by_id = {work["id"]: work for work in works}
    chunk_rows = [chunk_analysis(chunk, work_by_id.get(chunk.get("work_id"), {})) for chunk in chunks]
    safe_rows = [row for row in chunk_rows if row["safety_decision"] == "allow"]
    report = {
        "report_id": "public_opinion_analysis_v1",
        "mode": mode,
        "work_count": len(works),
        "chunk_count": len(chunks),
        "platform_counts": count_raw(works, "platform", fallback="source_name"),
        "thread_counts": count_raw(works, "thread_id"),
        "visibility_counts": count_raw(works, "visibility"),
        "collection_method_counts": count_raw(works, "collection_method"),
        "safety_decision_counts": dict(sorted(Counter(row["safety_decision"] for row in chunk_rows).items())),
        "safety_category_counts": dict(sorted(Counter(category for row in chunk_rows for category in row["safety_categories"]).items())),
        "theme_counts_all_public": theme_counts(chunk_rows),
        "theme_counts_safety_allowed": theme_counts(safe_rows),
        "stance_counts_all_public": dict(sorted(Counter(row["stance"] for row in chunk_rows).items())),
        "stance_counts_safety_allowed": dict(sorted(Counter(row["stance"] for row in safe_rows).items())),
        "duplicate_text": duplicate_text_summary(works),
        "author_activity": author_activity_summary(works),
        "temporal_coverage": temporal_coverage(works),
        "collection_audit": collection_audit_from_works(works),
        "aggregation_readiness": aggregation_readiness(works, chunk_rows),
        "chunk_sample": chunk_rows[:20],
        "limitations": public_opinion_limitations(works, chunk_rows),
    }
    write_json(settings.reports_dir / f"public_opinion_analysis_{mode}.json", report)
    write_markdown(settings.reports_dir / f"public_opinion_analysis_{mode}.md", report)
    return report


def chunk_analysis(chunk: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
    raw = work.get("raw") or {}
    text = str(chunk.get("text") or "")
    scan = scan_text(text, provenance=str(raw.get("provenance") or "social_media"))
    return {
        "chunk_id": chunk.get("id"),
        "work_id": chunk.get("work_id"),
        "platform": raw.get("platform") or work.get("source_name") or "unknown",
        "thread_id": raw.get("thread_id"),
        "document_type": raw.get("document_type"),
        "safety_decision": scan.decision,
        "safety_score": scan.risk_score,
        "safety_categories": scan.categories,
        "themes": matched_themes(text),
        "stance": stance_label(text),
        "preview": text[:240],
    }


def matched_themes(text: str) -> list[str]:
    lower = text.lower()
    matches = []
    for theme, terms in THEME_LEXICON.items():
        if any(term in lower for term in terms):
            matches.append(theme)
    return matches or ["uncategorized"]


def stance_label(text: str) -> str:
    lower = text.lower()
    support = any(term in lower for term in SUPPORT_TERMS)
    oppose = any(term in lower for term in OPPOSE_TERMS)
    condition = any(term in lower for term in CONDITION_TERMS)
    if support and (oppose or condition):
        return "mixed_or_conditional"
    if support:
        return "support"
    if oppose:
        return "concern_or_opposition"
    if condition:
        return "conditional_unclear"
    return "unclear"


def theme_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(theme for row in rows for theme in row["themes"])
    return dict(sorted(counts.items()))


def count_raw(works: list[dict[str, Any]], key: str, fallback: str | None = None) -> dict[str, int]:
    counts = Counter()
    for work in works:
        raw = work.get("raw") or {}
        value = raw.get(key)
        if value is None and fallback:
            value = work.get(fallback) or raw.get(fallback)
        counts[str(value or "unknown")] += 1
    return dict(sorted(counts.items()))


def duplicate_text_summary(works: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for work in works:
        text = normalize_text(str(work.get("abstract") or ""))
        if text:
            groups[hash_text(text)].append(work["id"])
    duplicate_groups = {key: ids for key, ids in groups.items() if len(ids) > 1}
    duplicate_work_count = sum(len(ids) for ids in duplicate_groups.values())
    return {
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_work_count": duplicate_work_count,
        "duplicate_work_share": ratio(duplicate_work_count, len(works)),
        "groups": duplicate_groups,
    }


def author_activity_summary(works: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str((work.get("raw") or {}).get("author_hash") or "unknown") for work in works)
    unknown = counts.pop("unknown", 0)
    max_count = max(counts.values(), default=0)
    return {
        "known_author_hash_count": len(counts),
        "unknown_author_count": unknown,
        "max_records_by_author_hash": max_count,
        "max_author_record_share": ratio(max_count, len(works)),
    }


def temporal_coverage(works: list[dict[str, Any]]) -> dict[str, Any]:
    days = Counter()
    timestamps = []
    for work in works:
        value = (work.get("raw") or {}).get("created_at")
        parsed = parse_datetime(value)
        if parsed:
            timestamps.append(parsed)
            days[parsed.date().isoformat()] += 1
    if not timestamps:
        return {"known_timestamp_count": 0, "day_counts": {}, "first_seen": None, "last_seen": None}
    return {
        "known_timestamp_count": len(timestamps),
        "day_counts": dict(sorted(days.items())),
        "first_seen": min(timestamps).isoformat(),
        "last_seen": max(timestamps).isoformat(),
        "covered_day_count": len(days),
    }


def aggregation_readiness(works: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    platform_count = len({str((work.get("raw") or {}).get("platform") or work.get("source_name") or "") for work in works})
    thread_count = len({str((work.get("raw") or {}).get("thread_id") or "") for work in works if (work.get("raw") or {}).get("thread_id")})
    safety_block_share = ratio(sum(1 for row in rows if row["safety_decision"] == "block"), len(rows))
    duplicate_share = duplicate_text_summary(works)["duplicate_work_share"]
    checks = {
        "minimum_records_50": len(works) >= 50,
        "minimum_platforms_2": platform_count >= 2,
        "minimum_threads_10": thread_count >= 10,
        "safety_block_share_below_0_2": safety_block_share < 0.2,
        "duplicate_share_below_0_3": duplicate_share < 0.3,
    }
    status = "ready_for_exploratory_aggregation" if all(checks.values()) else "fixture_or_insufficient_for_inference"
    return {
        "status": status,
        "checks": checks,
        "platform_count": platform_count,
        "thread_count": thread_count,
        "safety_block_share": safety_block_share,
        "duplicate_work_share": duplicate_share,
    }


def collection_audit_from_works(works: list[dict[str, Any]]) -> dict[str, Any]:
    raw_rows = [work.get("raw") or {} for work in works]
    work_count = len(works)
    platform_count = len({str(raw.get("platform") or "unknown") for raw in raw_rows if raw.get("platform")})
    visibility_counts = Counter(str(raw.get("visibility") or "unknown") for raw in raw_rows)
    public_visibility_count = sum(count for visibility, count in visibility_counts.items() if visibility in PUBLIC_VISIBILITIES)
    coverage = {
        "public_visibility_share": ratio(public_visibility_count, work_count),
        "terms_review_share": metadata_share(raw_rows, "terms_status", APPROVED_TERMS_STATUS),
        "robots_allowed_share": metadata_share(raw_rows, "robots_txt_status", APPROVED_ROBOTS_STATUS),
        "license_known_share": metadata_share(raw_rows, "license", KNOWN_LICENSE_STATUS),
        "consent_basis_share": metadata_share(raw_rows, "consent_basis", KNOWN_CONSENT_BASIS),
        "sampling_frame_share": present_share(raw_rows, "sampling_frame"),
    }
    privacy = public_opinion_privacy_summary(works)
    status = public_collection_status(work_count, platform_count, visibility_counts, coverage, privacy)
    return {
        "status": status,
        "work_count": work_count,
        "platform_count": platform_count,
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "metadata_coverage": coverage,
        "privacy_risk": privacy,
        "claim_boundaries": [
            "Allowed: use processed public-opinion corpora for aggregate expression analysis when acquisition metadata and sampling boundaries are visible.",
            "Not allowed: treat public visibility alone as permission, representativeness, or factual reliability.",
            f"Current processed-corpus acquisition status: {status}.",
        ],
    }


def public_collection_status(
    work_count: int,
    platform_count: int,
    visibility_counts: Counter,
    coverage: dict[str, float],
    privacy: dict[str, Any],
) -> str:
    if work_count <= 0:
        return "blocked_no_public_records"
    non_public = sum(count for visibility, count in visibility_counts.items() if visibility not in PUBLIC_VISIBILITIES)
    if non_public:
        return "needs_visibility_review"
    if privacy.get("direct_identifier_record_count", 0) or privacy.get("contact_pattern_record_count", 0):
        return "needs_privacy_review"
    if (
        work_count >= 50
        and platform_count >= 2
        and coverage.get("terms_review_share", 0.0) >= 0.8
        and coverage.get("robots_allowed_share", 0.0) >= 0.8
        and coverage.get("consent_basis_share", 0.0) >= 0.8
    ):
        return "ready_for_exploratory_public_ingest"
    return "fixture_only_or_needs_collection_metadata"


def public_opinion_privacy_summary(works: list[dict[str, Any]]) -> dict[str, Any]:
    direct_fields = {"author_id", "username", "email", "phone", "ip_address", "location_exact"}
    direct_identifier_count = 0
    contact_pattern_count = 0
    raw_author_id_count = 0
    for work in works:
        raw = work.get("raw") or {}
        if raw.get("author_id") or raw.get("username"):
            raw_author_id_count += 1
        if any(raw.get(field) for field in direct_fields):
            direct_identifier_count += 1
        if contains_contact_pattern(str(work.get("abstract") or "")):
            contact_pattern_count += 1
    known_author_hash_count = sum(1 for work in works if (work.get("raw") or {}).get("author_hash"))
    return {
        "direct_identifier_record_count": direct_identifier_count,
        "contact_pattern_record_count": contact_pattern_count,
        "known_author_hash_count": known_author_hash_count,
        "known_author_hash_share": ratio(known_author_hash_count, len(works)),
        "raw_author_id_count": raw_author_id_count,
    }


def metadata_share(rows: list[dict[str, Any]], key: str, allowed: set[str]) -> float:
    return ratio(sum(1 for row in rows if str(row.get(key) or "").strip().lower() in allowed), len(rows))


def present_share(rows: list[dict[str, Any]], key: str) -> float:
    return ratio(sum(1 for row in rows if row.get(key)), len(rows))


def public_opinion_limitations(works: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    limits = [
        "Deterministic theme and stance labels are lexical probes, not validated sentiment or opinion models.",
        "Public-opinion evidence requires aggregation and sampling context before use in investor, legal, or policy conclusions.",
    ]
    if len(works) < 50:
        limits.append("Corpus is too small for public-opinion inference.")
    if len({row["platform"] for row in rows}) < 2:
        limits.append("Single-platform public-opinion evidence cannot support broad public claims.")
    if any(row["safety_decision"] != "allow" for row in rows):
        limits.append("Some retrieved public text is unsafe for generation and must remain quarantined.")
    collection = collection_audit_from_works(works)
    if collection["status"] != "ready_for_exploratory_public_ingest":
        limits.append("Collection metadata is insufficient for broad public-opinion claims.")
    return limits


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Public Opinion Analysis: {report['mode']}",
        "",
        f"- Works: {report['work_count']}",
        f"- Chunks: {report['chunk_count']}",
        f"- Aggregation status: {report['aggregation_readiness']['status']}",
        "",
        "## Platforms",
        "",
    ]
    for key, value in report["platform_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", ""])
    for key, value in report["safety_decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Themes (Safety Allowed)", ""])
    for key, value in report["theme_counts_safety_allowed"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Stance (Safety Allowed)", ""])
    for key, value in report["stance_counts_safety_allowed"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Collection Audit", ""])
    lines.append(f"- Status: {report['collection_audit']['status']}")
    for key, value in report["collection_audit"]["metadata_coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze public-opinion social/forum corpus readiness.")
    parser.add_argument("--mode", default="social_public_opinion_demo_corpus")
    args = parser.parse_args()
    print(json.dumps(analyze_public_opinion(args.mode), indent=2))


if __name__ == "__main__":
    main()

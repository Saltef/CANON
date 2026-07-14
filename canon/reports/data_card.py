from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from canon.config import load_settings
from canon.evidence.chunk_context import parent_context_profile
from canon.quality.domain_profiles import domain_fit_vector, domain_quality_profile
from canon.quality.document_types import document_type_profile
from canon.quality.signals import source_trust_vector
from canon.retrieval.corpus import work_model_from_dict


def build_data_card(mode: str) -> dict:
    settings = load_settings()
    works = load_json(settings.data_dir / "processed" / f"works_{mode}.json", [])
    chunks = load_json(settings.data_dir / "processed" / f"chunks_{mode}.json", [])
    corpus = load_json(settings.reports_dir / f"corpus_{mode}.json", {})
    quality = load_json(settings.reports_dir / f"quality_diagnostics_{mode}.json", {})
    card = {
        "mode": mode,
        "corpus_id": corpus.get("corpus_id", mode),
        "source_modes": corpus.get("source_modes", []),
        "work_count": len(works),
        "chunk_count": len(chunks),
        "topic_mode_counts": topic_mode_counts(works),
        "coverage": coverage(works, chunks),
        "chunking_profile": chunking_profile(chunks),
        "document_type_profile": document_type_profile_summary(works),
        "domain_profile": domain_profile_summary(works),
        "year_distribution": dict(sorted(Counter(str(work.get("year")) for work in works if work.get("year")).items())),
        "top_sources": dict(Counter(work.get("source_name") or "UNKNOWN" for work in works).most_common(20)),
        "openalex_filters": inferred_openalex_filters(settings),
        "quality_signal_summary": quality.get("signals", {}),
        "limitations": limitations(works, chunks, quality),
    }
    write_json(settings.reports_dir / f"data_card_{mode}.json", card)
    write_markdown(settings.reports_dir / f"data_card_{mode}.md", card)
    return card


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def topic_mode_counts(works: list[dict]) -> dict:
    counts = Counter()
    for work in works:
        modes = work.get("corpus_source_modes") or []
        for mode in modes:
            counts[mode] += 1
    return dict(sorted(counts.items()))


def coverage(works: list[dict], chunks: list[dict]) -> dict:
    total = len(works)
    return {
        "abstract_coverage": ratio(sum(1 for work in works if work.get("abstract")), total),
        "open_access_coverage": ratio(sum(1 for work in works if work.get("is_open_access")), total),
        "pdf_url_coverage": ratio(sum(1 for work in works if work.get("pdf_url")), total),
        "reference_list_coverage": ratio(sum(1 for work in works if int(work.get("referenced_work_count") or 0) > 0), total),
        "authorship_coverage": ratio(sum(1 for work in works if work.get("author_display_names")), total),
        "author_impact_coverage": ratio(sum(1 for work in works if int(work.get("max_author_cited_by_count") or 0) > 0), total),
        "chunks_per_work": round(len(chunks) / total, 6) if total else 0.0,
    }


def chunking_profile(chunks: list[dict]) -> dict:
    token_counts = [chunk_token_count(chunk) for chunk in chunks]
    boundary_values = [
        float((chunk.get("importance") or {}).get("boundary_aligned", 0.0) or 0.0)
        for chunk in chunks
    ]
    strategy_counts = Counter(
        str((chunk.get("importance") or {}).get("chunk_strategy") or "unknown")
        for chunk in chunks
    )
    policy_document_type_counts = Counter(
        str((chunk.get("importance") or {}).get("chunk_policy_document_type") or "unknown")
        for chunk in chunks
    )
    resolution_counts = Counter(
        str((chunk.get("importance") or {}).get("chunk_resolution") or "unknown")
        for chunk in chunks
    )
    parent_context_counts = Counter(
        str((chunk.get("importance") or {}).get("parent_context_mode") or "unknown")
        for chunk in chunks
    )
    generation_role_counts = Counter(
        str((chunk.get("importance") or {}).get("generation_context_role") or "unknown")
        for chunk in chunks
    )
    quality_tier_counts = Counter(
        str((chunk.get("importance") or {}).get("chunk_quality_tier") or "unknown")
        for chunk in chunks
    )
    expansion_policy_counts = Counter(
        str((chunk.get("importance") or {}).get("context_expansion_policy") or "unknown")
        for chunk in chunks
    )
    expansion_values = [
        float((chunk.get("importance") or {}).get("parent_context_recommended", 0.0) or 0.0)
        for chunk in chunks
    ]
    contamination_values = [
        float((chunk.get("importance") or {}).get("chunk_safety_contamination_risk", 0.0) or 0.0)
        for chunk in chunks
    ]
    containment_values = [
        float((chunk.get("importance") or {}).get("evidence_containment_score", 0.0) or 0.0)
        for chunk in chunks
    ]
    return {
        "chunk_count": len(chunks),
        "average_chunk_tokens": average(token_counts),
        "min_chunk_tokens": min(token_counts, default=0),
        "max_chunk_tokens": max(token_counts, default=0),
        "boundary_aligned_share": average(boundary_values),
        "token_window_fallback_share": round(1.0 - average(boundary_values), 6) if chunks else 0.0,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "policy_document_type_counts": dict(sorted(policy_document_type_counts.items())),
        "chunk_resolution_counts": dict(sorted(resolution_counts.items())),
        "parent_context_mode_counts": dict(sorted(parent_context_counts.items())),
        "generation_context_role_counts": dict(sorted(generation_role_counts.items())),
        "chunk_quality_tier_counts": dict(sorted(quality_tier_counts.items())),
        "context_expansion_policy_counts": dict(sorted(expansion_policy_counts.items())),
        "parent_context_recommended_share": average(expansion_values),
        "average_safety_contamination_risk": average(contamination_values),
        "average_evidence_containment_score": average(containment_values),
        **parent_context_profile(chunks),
    }


def chunk_token_count(chunk: dict) -> int:
    importance = chunk.get("importance") or {}
    if importance.get("chunk_token_count") is not None:
        return int(float(importance["chunk_token_count"]))
    return max(0, int(chunk.get("token_end") or 0) - int(chunk.get("token_start") or 0))


def document_type_profile_summary(works: list[dict]) -> dict:
    profiles = [document_type_profile(work_model_from_dict(work)) for work in works]
    type_counts = Counter(profile.document_type for profile in profiles)
    family_counts = Counter(profile.evidence_family for profile in profiles)
    role_counts = Counter(profile.recommended_decision_role for profile in profiles)
    grey_count = sum(1 for profile in profiles if profile.grey_literature)
    unknown_count = type_counts.get("unknown_unstructured_text", 0)
    return {
        "document_type_counts": dict(sorted(type_counts.items())),
        "evidence_family_counts": dict(sorted(family_counts.items())),
        "recommended_decision_role_counts": dict(sorted(role_counts.items())),
        "grey_literature_count": grey_count,
        "grey_literature_share": ratio(grey_count, len(works)),
        "unknown_type_count": unknown_count,
        "unknown_type_share": ratio(unknown_count, len(works)),
        "average_document_type_confidence": average(profile.confidence for profile in profiles),
        "average_authority_signal": average(profile.authority_signal for profile in profiles),
        "average_timeliness_need": average(profile.timeliness_need for profile in profiles),
        "average_source_intent_risk": average(profile.source_intent_risk for profile in profiles),
        "average_adversarial_surface": average(profile.adversarial_surface for profile in profiles),
        "average_extraction_risk": average(profile.extraction_risk for profile in profiles),
    }


def domain_profile_summary(works: list[dict]) -> dict:
    rows = []
    for work in works:
        model = work_model_from_dict(work)
        domain = str(work.get("domain") or (work.get("raw") or {}).get("domain") or "general_academic")
        doc_profile = document_type_profile(model).to_dict()
        trust = source_trust_vector(model, domain=domain)
        rows.append(domain_fit_vector(model, domain, doc_profile, trust))
    domains = Counter(row["domain_profile"] for row in rows)
    tiers = Counter(row["domain_fit_tier"] for row in rows)
    gate_flags = Counter(flag for row in rows for flag in row.get("domain_gate_flags", []))
    low_fit_count = sum(1 for row in rows if row.get("domain_fit_tier") in {"low_fit", "weak_fit", "quarantine_review", "blocked"})
    general_count = domains.get("general_academic", 0)
    return {
        "domain_counts": dict(sorted(domains.items())),
        "domain_fit_tier_counts": dict(sorted(tiers.items())),
        "domain_gate_flag_counts": dict(sorted(gate_flags.items())),
        "low_or_review_fit_count": low_fit_count,
        "low_or_review_fit_share": ratio(low_fit_count, len(rows)),
        "general_academic_share": ratio(general_count, len(rows)),
        "average_domain_fit_score": average(row["domain_fit_score"] for row in rows),
        "average_document_type_fit": average(row["domain_document_type_fit"] for row in rows),
        "average_evidence_family_fit": average(row["domain_evidence_family_fit"] for row in rows),
        "average_transparency_fit": average(row["domain_transparency_fit"] for row in rows),
        "average_provenance_fit": average(row["domain_provenance_fit"] for row in rows),
        "available_domain_profiles": sorted(domain_quality_profile(domain).domain for domain in set(domains) or {"general_academic"}),
    }


def inferred_openalex_filters(settings) -> dict:
    raw = settings.raw.get("openalex", {})
    return {
        "from_publication_year": raw.get("from_publication_year"),
        "to_publication_year": raw.get("to_publication_year"),
        "default_per_page": raw.get("per_page"),
    }


def limitations(works: list[dict], chunks: list[dict], quality: dict) -> list[str]:
    limits = []
    cov = coverage(works, chunks)
    public_opinion_share = domain_share(works, "public_opinion")
    mostly_public_opinion = public_opinion_share >= 0.8
    if cov["pdf_url_coverage"] < 0.5 and not mostly_public_opinion:
        limits.append("PDF URL coverage is low; full-text coverage is not representative yet.")
    if cov["reference_list_coverage"] < 0.8 and not mostly_public_opinion:
        limits.append("Reference-list coverage is incomplete; citation graph signals may understate some works.")
    if cov["author_impact_coverage"] < 0.8 and not mostly_public_opinion:
        limits.append("Author-impact coverage is incomplete; author-score comparisons should be treated as partial.")
    if len(works) < 100:
        limits.append("Corpus is small; results are suitable for technical testing, not field-level claims.")
    chunk_profile = chunking_profile(chunks)
    if chunks and chunk_profile["boundary_aligned_share"] < 0.25:
        limits.append("Most chunks used token-window fallback rather than sentence boundaries; inspect OCR, markup, or sentence segmentation quality.")
    doc_profile = document_type_profile_summary(works)
    if doc_profile["unknown_type_share"] > 0.2:
        limits.append("More than 20% of works have unknown document type; heterogeneous-corpus claims need better metadata.")
    if doc_profile["document_type_counts"] and len(doc_profile["document_type_counts"]) < 2:
        limits.append("Document-type diversity is low; mixed-unstructured conclusions are not supported by this corpus.")
    if doc_profile["average_adversarial_surface"] >= 0.5:
        limits.append("Average adversarial surface is high; prompt-injection and extraction-risk checks should be prioritized.")
    domain_profile = domain_profile_summary(works)
    if domain_profile["general_academic_share"] > 0.8 and len(domain_profile["domain_counts"]) == 1:
        limits.append("Corpus is mostly general_academic; claims about cross-domain unstructured performance are not supported.")
    if domain_profile["low_or_review_fit_share"] > 0.3:
        limits.append("More than 30% of works have low or review-needed domain fit; inspect domain metadata and source mix.")
    if mostly_public_opinion:
        limits.extend(public_opinion_limitations(works))
    if quality and int(quality.get("chunk_count") or 0) != len(chunks):
        limits.append("Quality diagnostics chunk count differs from current processed chunks.")
    return limits


def domain_share(works: list[dict], domain: str) -> float:
    if not works:
        return 0.0
    count = 0
    for work in works:
        work_domain = str(work.get("domain") or (work.get("raw") or {}).get("domain") or "general_academic")
        if work_domain == domain:
            count += 1
    return ratio(count, len(works))


def public_opinion_limitations(works: list[dict]) -> list[str]:
    limits = [
        "Public-opinion records are expression evidence, not factual authority or representative polling by default.",
        "Public-opinion use requires aggregation; individual posts/comments should not be used as standalone truth evidence.",
    ]
    platforms = {str((work.get("raw") or {}).get("platform") or work.get("source_name") or "") for work in works}
    platforms.discard("")
    if len(platforms) < 2:
        limits.append("Public-opinion corpus is single-platform; do not generalize across broader publics.")
    if len(works) < 50:
        limits.append("Public-opinion corpus is too small for stance, trend, or market-sentiment inference.")
    if any((work.get("raw") or {}).get("visibility") not in {"public", "public_web", "open"} for work in works):
        limits.append("Some public-opinion records lack explicit public visibility metadata.")
    return limits


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, card: dict) -> None:
    lines = [
        f"# CANON Data Card: {card['mode']}",
        "",
        f"- Works: {card['work_count']}",
        f"- Chunks: {card['chunk_count']}",
        f"- Source modes: {', '.join(card['source_modes']) if card['source_modes'] else 'unknown'}",
        "",
        "## Coverage",
        "",
    ]
    for key, value in card["coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Chunking", ""])
    for key, value in card.get("chunking_profile", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Document Types", ""])
    profile = card.get("document_type_profile", {})
    for key, value in (profile.get("document_type_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- grey_literature_share: {profile.get('grey_literature_share', 0.0)}",
            f"- unknown_type_share: {profile.get('unknown_type_share', 0.0)}",
            f"- average_adversarial_surface: {profile.get('average_adversarial_surface', 0.0)}",
        ]
    )
    lines.extend(["", "## Domain Profiles", ""])
    domain_profile = card.get("domain_profile", {})
    for key, value in (domain_profile.get("domain_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- average_domain_fit_score: {domain_profile.get('average_domain_fit_score', 0.0)}",
            f"- low_or_review_fit_share: {domain_profile.get('low_or_review_fit_share', 0.0)}",
            f"- general_academic_share: {domain_profile.get('general_academic_share', 0.0)}",
        ]
    )
    lines.extend(["", "## Limitations", ""])
    for item in card["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus data card for CANON.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    args = parser.parse_args()
    print(json.dumps(build_data_card(args.mode), indent=2))


if __name__ == "__main__":
    main()

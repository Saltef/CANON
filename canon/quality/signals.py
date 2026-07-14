from __future__ import annotations

from dataclasses import dataclass

from canon.models import Work


TRUST_TIERS = {"blocked", "low", "limited", "moderate", "strong"}


@dataclass(frozen=True)
class SourceTrustProfile:
    """Domain-aware defaults for interpreting available source metadata."""

    domain: str
    values_formal_peer_review: bool
    values_preregistration: bool
    values_archival_provenance: bool
    values_method_transparency: bool
    citation_half_life_years: int


DOMAIN_TRUST_PROFILES: dict[str, SourceTrustProfile] = {
    "general_academic": SourceTrustProfile(
        domain="general_academic",
        values_formal_peer_review=True,
        values_preregistration=True,
        values_archival_provenance=False,
        values_method_transparency=True,
        citation_half_life_years=8,
    ),
    "economics": SourceTrustProfile(
        domain="economics",
        values_formal_peer_review=True,
        values_preregistration=True,
        values_archival_provenance=False,
        values_method_transparency=True,
        citation_half_life_years=7,
    ),
    "psychology": SourceTrustProfile(
        domain="psychology",
        values_formal_peer_review=True,
        values_preregistration=True,
        values_archival_provenance=False,
        values_method_transparency=True,
        citation_half_life_years=6,
    ),
    "anthropology": SourceTrustProfile(
        domain="anthropology",
        values_formal_peer_review=True,
        values_preregistration=False,
        values_archival_provenance=True,
        values_method_transparency=True,
        citation_half_life_years=12,
    ),
    "history": SourceTrustProfile(
        domain="history",
        values_formal_peer_review=True,
        values_preregistration=False,
        values_archival_provenance=True,
        values_method_transparency=False,
        citation_half_life_years=16,
    ),
    "cultural_studies": SourceTrustProfile(
        domain="cultural_studies",
        values_formal_peer_review=True,
        values_preregistration=False,
        values_archival_provenance=True,
        values_method_transparency=False,
        citation_half_life_years=14,
    ),
    "legal_market": SourceTrustProfile(
        domain="legal_market",
        values_formal_peer_review=False,
        values_preregistration=False,
        values_archival_provenance=True,
        values_method_transparency=True,
        citation_half_life_years=5,
    ),
    "public_opinion": SourceTrustProfile(
        domain="public_opinion",
        values_formal_peer_review=False,
        values_preregistration=False,
        values_archival_provenance=True,
        values_method_transparency=False,
        citation_half_life_years=2,
    ),
}


def citation_impact_score(work: Work, high_watermark: int = 500) -> float:
    if work.cited_by_count <= 0:
        return 0.0
    return min(1.0, work.cited_by_count / high_watermark)


def retraction_score(work: Work) -> float:
    return 0.0 if work.is_retracted else 1.0


def open_access_score(work: Work) -> float:
    if work.pdf_url:
        return 1.0
    if work.is_open_access:
        return 0.7
    return 0.2


def reference_coverage_score(work: Work, target_reference_count: int = 30) -> float:
    return min(1.0, work.referenced_work_count / target_reference_count)


def author_score(work: Work, cited_by_high_watermark: int = 5000) -> float:
    if work.max_author_cited_by_count <= 0:
        return 0.0
    return min(1.0, work.max_author_cited_by_count / cited_by_high_watermark)


def source_quality_vector(work: Work) -> dict[str, float]:
    """Legacy source metadata vector kept for existing diagnostics/tests.

    Prefer `source_trust_vector` for new retrieval or audit work. This function
    intentionally exposes open access and PDF availability as access/provenance
    signals, not proof of reliability.
    """

    return {
        "retraction": retraction_score(work),
        "citation_impact": citation_impact_score(work),
        "open_access": open_access_score(work),
        "reference_coverage": reference_coverage_score(work),
        "author_score": author_score(work),
    }


def source_trust_vector(work: Work, domain: str = "general_academic") -> dict[str, float | str | bool]:
    """Estimate source trust dimensions without collapsing them into prestige.

    The vector follows evidence-appraisal practice: separate hard integrity
    problems, study/reporting transparency, provenance, and influence. Access
    fields such as open access and PDF URL are retained as provenance/access
    metadata only.
    """

    profile = DOMAIN_TRUST_PROFILES.get(domain, DOMAIN_TRUST_PROFILES["general_academic"])
    integrity_gate = not work.is_retracted
    vector: dict[str, float | str | bool] = {
        "domain_profile": profile.domain,
        "integrity_gate": integrity_gate,
        "retraction_risk": 1.0 if work.is_retracted else 0.0,
        "bibliographic_provenance": bibliographic_provenance_score(work),
        "publication_channel": publication_channel_score(work, profile),
        "reporting_transparency": reporting_transparency_score(work, profile),
        "open_science": open_science_score(work, profile),
        "field_normalized_influence": field_normalized_influence_score(work, profile),
        "author_influence": author_score(work),
        "accessibility": access_metadata_score(work),
    }
    vector["trust_tier"] = trust_tier(vector)
    vector["source_trust"] = source_trust_score(vector)
    return vector


def bibliographic_provenance_score(work: Work) -> float:
    score = 0.0
    if work.doi:
        score += 0.45
    if work.source_name:
        score += 0.25
    if work.landing_page_url:
        score += 0.15
    if work.year:
        score += 0.15
    return min(1.0, score)


def publication_channel_score(work: Work, profile: SourceTrustProfile) -> float:
    raw = work.raw or {}
    source_type = raw_source_type(raw)
    if work.is_retracted:
        return 0.0
    if source_type in {"journal", "book", "conference"}:
        return 0.8 if profile.values_formal_peer_review else 0.65
    if source_type in {"repository", "preprint"}:
        return 0.45
    if source_type in {"report", "dataset"}:
        return 0.65 if profile.values_archival_provenance else 0.5
    if source_type in {"social_media", "social_media_post", "forum", "forum_comment", "comment"}:
        return 0.25
    if work.source_name:
        return 0.55
    return 0.25


def raw_source_type(raw: dict) -> str | None:
    candidates = [
        raw.get("source_type"),
        raw.get("type"),
        (raw.get("primary_location") or {}).get("source", {}).get("type"),
        (raw.get("primary_location") or {}).get("source", {}).get("host_organization_lineage_names"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate.lower()
    return None


def reporting_transparency_score(work: Work, profile: SourceTrustProfile) -> float:
    score = 0.0
    abstract = (work.abstract or "").lower()
    if work.referenced_work_count >= 20:
        score += 0.25
    elif work.referenced_work_count > 0:
        score += 0.15
    if any(term in abstract for term in ["data", "sample", "method", "model", "archive"]):
        score += 0.2
    if any(term in abstract for term in ["robustness", "sensitivity", "replication"]):
        score += 0.2
    if any(term in abstract for term in ["limitation", "uncertainty", "bias"]):
        score += 0.15
    if profile.values_method_transparency:
        return min(1.0, score + 0.1)
    return min(1.0, score)


def open_science_score(work: Work, profile: SourceTrustProfile) -> float:
    raw_text = " ".join(
        str(value).lower()
        for value in [
            work.abstract,
            work.raw.get("data_availability") if work.raw else None,
            work.raw.get("repository") if work.raw else None,
        ]
        if value
    )
    score = 0.0
    if any(term in raw_text for term in ["preregister", "pre-register", "registered report"]):
        score += 0.35 if profile.values_preregistration else 0.15
    if any(term in raw_text for term in ["data available", "replication data", "open data"]):
        score += 0.25
    if any(term in raw_text for term in ["code available", "github", "osf", "dataverse"]):
        score += 0.2
    return min(1.0, score)


def field_normalized_influence_score(work: Work, profile: SourceTrustProfile, newest_year: int = 2026) -> float:
    if work.cited_by_count <= 0:
        return 0.0
    age = max(1, newest_year - (work.year or newest_year) + 1)
    expected = 12.0 * min(age, profile.citation_half_life_years)
    return min(1.0, work.cited_by_count / expected)


def access_metadata_score(work: Work) -> float:
    if work.pdf_url:
        return 1.0
    if work.is_open_access:
        return 0.7
    return 0.2


def source_trust_score(vector: dict[str, float | str | bool]) -> float:
    if not vector.get("integrity_gate", False):
        return 0.0
    weights = {
        "bibliographic_provenance": 0.2,
        "publication_channel": 0.25,
        "reporting_transparency": 0.25,
        "open_science": 0.15,
        "field_normalized_influence": 0.15,
    }
    return round(
        sum(float(vector.get(name, 0.0)) * weight for name, weight in weights.items())
        / sum(weights.values()),
        6,
    )


def trust_tier(vector: dict[str, float | str | bool]) -> str:
    if not vector.get("integrity_gate", False):
        return "blocked"
    score = source_trust_score(vector)
    if score >= 0.75:
        return "strong"
    if score >= 0.55:
        return "moderate"
    if score >= 0.35:
        return "limited"
    return "low"

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from canon.evidence.domain import domain_profile as decision_domain_profile
from canon.models import Work


@dataclass(frozen=True)
class DomainQualityProfile:
    domain: str
    preferred_document_types: tuple[str, ...]
    preferred_evidence_families: tuple[str, ...]
    critical_quality_signals: tuple[str, ...]
    gate_signals: tuple[str, ...]
    annotation_only_signals: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DOMAIN_QUALITY_PROFILES: dict[str, DomainQualityProfile] = {
    "general_academic": DomainQualityProfile(
        domain="general_academic",
        preferred_document_types=("academic_article", "preprint", "policy_report"),
        preferred_evidence_families=("scholarly", "scholarly_grey_literature", "institutional_report"),
        critical_quality_signals=("integrity_gate", "method_signal", "limitation_signal", "reporting_transparency"),
        gate_signals=("integrity_gate", "safety_gate"),
        annotation_only_signals=("citation_centrality", "author_influence", "accessibility"),
        notes=("General academic evidence should emphasize method fit and limitations over prestige shortcuts.",),
    ),
    "economics": DomainQualityProfile(
        domain="economics",
        preferred_document_types=("academic_article", "preprint", "policy_report", "regulatory_filing", "market_report"),
        preferred_evidence_families=("scholarly", "scholarly_grey_literature", "institutional_report", "official_record"),
        critical_quality_signals=("identification_detail", "method_signal", "data_detail", "robustness", "reporting_transparency"),
        gate_signals=("integrity_gate", "method_transparency_for_causal_claims"),
        annotation_only_signals=("citation_centrality", "author_influence", "journal_prestige"),
        notes=("Economics quality depends heavily on identification, data, robustness, and fit to the causal or market question.",),
    ),
    "psychology": DomainQualityProfile(
        domain="psychology",
        preferred_document_types=("academic_article", "preprint", "policy_report"),
        preferred_evidence_families=("scholarly", "scholarly_grey_literature", "institutional_report"),
        critical_quality_signals=("measurement_detail", "sample_detail", "replication", "preregistration", "open_science"),
        gate_signals=("integrity_gate", "measurement_detail_for_empirical_claims"),
        annotation_only_signals=("citation_centrality", "journal_prestige", "author_influence"),
        notes=("Psychology profiles should surface measurement, sample, preregistration, and replication status.",),
    ),
    "anthropology": DomainQualityProfile(
        domain="anthropology",
        preferred_document_types=("academic_article", "book", "transcript", "archival_primary_source", "policy_report"),
        preferred_evidence_families=("scholarly", "primary_or_near_primary_text", "institutional_report"),
        critical_quality_signals=("fieldwork_context", "positionality", "provenance", "interpretive_detail"),
        gate_signals=("integrity_gate", "provenance_for_primary_claims"),
        annotation_only_signals=("citation_centrality", "sample_size_language", "author_influence"),
        notes=("Anthropology profiles need context, positionality, and provenance rather than simple recency or sample-size priors.",),
    ),
    "history": DomainQualityProfile(
        domain="history",
        preferred_document_types=("academic_article", "book", "archival_primary_source", "transcript", "legal_authority"),
        preferred_evidence_families=("scholarly", "primary_or_near_primary_text", "official_record"),
        critical_quality_signals=("primary_source_context", "archival_provenance", "source_criticism", "historiography"),
        gate_signals=("integrity_gate", "provenance_for_primary_claims"),
        annotation_only_signals=("recency", "citation_centrality", "author_influence"),
        notes=("History profiles should separate primary-source provenance from historiographic influence and date.",),
    ),
    "cultural_studies": DomainQualityProfile(
        domain="cultural_studies",
        preferred_document_types=("academic_article", "book", "news_article", "archival_primary_source", "web_page"),
        preferred_evidence_families=("scholarly", "journalistic", "primary_or_near_primary_text", "open_web"),
        critical_quality_signals=("theoretical_frame", "object_of_analysis", "contextualization", "counter_reading"),
        gate_signals=("integrity_gate", "clear_object_of_analysis", "safety_gate"),
        annotation_only_signals=("citation_centrality", "replication_language", "author_influence"),
        notes=("Cultural-studies profiles should expose interpretive frame and object of analysis, not force consensus metrics.",),
    ),
    "legal_market": DomainQualityProfile(
        domain="legal_market",
        preferred_document_types=("legal_authority", "regulatory_filing", "market_report", "news_article", "transcript"),
        preferred_evidence_families=("official_record", "commercial_or_market_analysis", "journalistic", "primary_or_near_primary_text"),
        critical_quality_signals=("authority_signal", "jurisdiction", "temporal_validity", "filing_detail", "market_timing"),
        gate_signals=("integrity_gate", "authority_or_filing_provenance", "temporal_validity", "safety_gate"),
        annotation_only_signals=("academic_citation_count", "open_access", "author_influence"),
        notes=("Legal and market use cases need authority, jurisdiction, filing provenance, and timeliness gates.",),
    ),
    "public_opinion": DomainQualityProfile(
        domain="public_opinion",
        preferred_document_types=("social_media_post", "forum_comment", "news_article", "transcript"),
        preferred_evidence_families=("public_opinion", "journalistic", "primary_or_near_primary_text"),
        critical_quality_signals=("public_visibility", "platform_context", "time_window", "aggregation_count", "stance_or_theme"),
        gate_signals=("integrity_gate", "public_visibility", "safety_gate", "aggregation_required"),
        annotation_only_signals=("likes", "shares", "reposts", "follower_count", "raw_virality"),
        notes=("Public-opinion text should be used as aggregated expression evidence, not factual authority or representative polling by default.",),
    ),
}


def domain_quality_profile(domain: str | None) -> DomainQualityProfile:
    return DOMAIN_QUALITY_PROFILES.get(domain or "", DOMAIN_QUALITY_PROFILES["general_academic"])


def domain_fit_vector(
    work: Work,
    domain: str,
    document_profile: dict[str, Any],
    source_trust: dict[str, Any],
) -> dict[str, Any]:
    quality_profile = domain_quality_profile(domain)
    decision_profile = decision_domain_profile(domain)
    document_type = str(document_profile.get("document_type") or "unknown_unstructured_text")
    evidence_family = str(document_profile.get("evidence_family") or "unknown")
    preferred_type_hit = document_type in quality_profile.preferred_document_types
    preferred_family_hit = evidence_family in quality_profile.preferred_evidence_families
    authority_signal = float(document_profile.get("authority_signal", 0.0) or 0.0)
    adversarial_surface = float(document_profile.get("adversarial_surface", 0.0) or 0.0)
    source_intent_risk = float(document_profile.get("source_intent_risk", 0.0) or 0.0)
    transparency = float(source_trust.get("reporting_transparency", 0.0) or 0.0)
    provenance = float(source_trust.get("bibliographic_provenance", 0.0) or 0.0)
    publication_channel = float(source_trust.get("publication_channel", 0.0) or 0.0)
    open_science = float(source_trust.get("open_science", 0.0) or 0.0)
    safety_fit = max(0.0, 1.0 - max(adversarial_surface, source_intent_risk * 0.8))
    gate_flags = domain_gate_flags(
        work=work,
        domain=quality_profile.domain,
        document_profile=document_profile,
        source_trust=source_trust,
    )
    component_scores = {
        "document_type_fit": 1.0 if preferred_type_hit else 0.25 if document_type == "unknown_unstructured_text" else 0.45,
        "evidence_family_fit": 1.0 if preferred_family_hit else 0.35 if evidence_family == "unknown" else 0.55,
        "authority_fit": authority_signal,
        "transparency_fit": transparency,
        "provenance_fit": provenance,
        "publication_channel_fit": publication_channel,
        "open_science_fit": open_science,
        "safety_fit": safety_fit,
    }
    score = domain_fit_score(quality_profile.domain, component_scores, gate_flags)
    return {
        "domain_profile": quality_profile.domain,
        "domain_fit_score": score,
        "domain_fit_tier": domain_fit_tier(score, gate_flags),
        "domain_document_type_fit": round(component_scores["document_type_fit"], 6),
        "domain_evidence_family_fit": round(component_scores["evidence_family_fit"], 6),
        "domain_authority_fit": round(component_scores["authority_fit"], 6),
        "domain_transparency_fit": round(component_scores["transparency_fit"], 6),
        "domain_provenance_fit": round(component_scores["provenance_fit"], 6),
        "domain_publication_channel_fit": round(component_scores["publication_channel_fit"], 6),
        "domain_open_science_fit": round(component_scores["open_science_fit"], 6),
        "domain_safety_fit": round(component_scores["safety_fit"], 6),
        "domain_gate_flags": gate_flags,
        "domain_preferred_document_type_hit": preferred_type_hit,
        "domain_preferred_evidence_family_hit": preferred_family_hit,
        "domain_preferred_document_types": list(quality_profile.preferred_document_types),
        "domain_preferred_evidence_families": list(quality_profile.preferred_evidence_families),
        "domain_critical_quality_signals": list(quality_profile.critical_quality_signals),
        "domain_required_gates": list(quality_profile.gate_signals),
        "domain_annotation_only_signals": list(quality_profile.annotation_only_signals),
        "domain_recency_policy": decision_profile.recency_policy,
        "domain_corroboration_policy": decision_profile.corroboration_policy,
        "domain_safety_policy": decision_profile.safety_policy,
        "domain_profile_notes": list(quality_profile.notes),
    }


def domain_gate_flags(
    work: Work,
    domain: str,
    document_profile: dict[str, Any],
    source_trust: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    document_type = str(document_profile.get("document_type") or "unknown_unstructured_text")
    provenance = str((work.raw or {}).get("provenance") or "").lower()
    adversarial_surface = float(document_profile.get("adversarial_surface", 0.0) or 0.0)
    authority_signal = float(document_profile.get("authority_signal", 0.0) or 0.0)
    if not bool(source_trust.get("integrity_gate", True)):
        flags.append("integrity_gate_failed")
    if document_type == "unknown_unstructured_text":
        flags.append("missing_document_type")
    if adversarial_surface >= 0.7 and domain != "public_opinion":
        flags.append("high_adversarial_surface")
    if domain == "legal_market" and document_type in {"legal_authority", "regulatory_filing"} and authority_signal < 0.6:
        flags.append("weak_authority_for_authority_domain")
    if domain in {"history", "anthropology"} and document_type in {"archival_primary_source", "transcript"} and provenance not in {"archive", "official", "academic"}:
        flags.append("weak_primary_source_provenance")
    if domain == "public_opinion":
        visibility = str((work.raw or {}).get("visibility") or "").lower()
        platform = str((work.raw or {}).get("platform") or "").strip()
        if visibility not in {"public", "public_web", "open"}:
            flags.append("missing_public_visibility")
        if not platform:
            flags.append("missing_platform_context")
        if document_type in {"social_media_post", "forum_comment"}:
            flags.append("aggregation_required")
    return flags


def domain_fit_score(domain: str, components: dict[str, float], gate_flags: list[str]) -> float:
    if "integrity_gate_failed" in gate_flags:
        return 0.0
    weights = domain_fit_weights(domain)
    denominator = sum(weights.values())
    base = sum(components.get(name, 0.0) * weight for name, weight in weights.items()) / denominator
    penalty = 0.08 * len([flag for flag in gate_flags if flag != "high_adversarial_surface"])
    if "high_adversarial_surface" in gate_flags:
        penalty += 0.18
    return round(max(0.0, min(1.0, base - penalty)), 6)


def domain_fit_weights(domain: str) -> dict[str, float]:
    if domain == "legal_market":
        return {
            "document_type_fit": 0.2,
            "evidence_family_fit": 0.15,
            "authority_fit": 0.25,
            "provenance_fit": 0.15,
            "transparency_fit": 0.1,
            "safety_fit": 0.15,
        }
    if domain in {"history", "anthropology", "cultural_studies"}:
        return {
            "document_type_fit": 0.2,
            "evidence_family_fit": 0.2,
            "provenance_fit": 0.2,
            "transparency_fit": 0.15,
            "publication_channel_fit": 0.1,
            "safety_fit": 0.15,
        }
    if domain == "psychology":
        return {
            "document_type_fit": 0.15,
            "evidence_family_fit": 0.1,
            "transparency_fit": 0.25,
            "open_science_fit": 0.2,
            "publication_channel_fit": 0.15,
            "safety_fit": 0.15,
        }
    if domain == "public_opinion":
        return {
            "document_type_fit": 0.25,
            "evidence_family_fit": 0.2,
            "provenance_fit": 0.15,
            "transparency_fit": 0.05,
            "safety_fit": 0.25,
            "publication_channel_fit": 0.1,
        }
    return {
        "document_type_fit": 0.18,
        "evidence_family_fit": 0.12,
        "transparency_fit": 0.22,
        "provenance_fit": 0.15,
        "publication_channel_fit": 0.13,
        "safety_fit": 0.2,
    }


def domain_fit_tier(score: float, gate_flags: list[str]) -> str:
    if "integrity_gate_failed" in gate_flags:
        return "blocked"
    if "high_adversarial_surface" in gate_flags and score < 0.5:
        return "quarantine_review"
    if score >= 0.75:
        return "strong_fit"
    if score >= 0.55:
        return "moderate_fit"
    if score >= 0.35:
        return "weak_fit"
    return "low_fit"

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from canon.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class DomainDecisionProfile:
    domain: str
    high_value_roles: tuple[str, ...]
    trust_gates: tuple[str, ...]
    annotation_only: tuple[str, ...]
    recency_policy: str
    corroboration_policy: str
    safety_policy: str = "gate_before_rerank"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DOMAIN_DECISION_PROFILES: dict[str, DomainDecisionProfile] = {
    "general_academic": DomainDecisionProfile(
        domain="general_academic",
        high_value_roles=("direct_evidence", "method_detail", "limitation_or_caveat"),
        trust_gates=("integrity_gate",),
        annotation_only=("author_influence", "accessibility"),
        recency_policy="annotate_unless_query_demands_current_evidence",
        corroboration_policy="prefer_independent_support_and_surface_conflict",
    ),
    "economics": DomainDecisionProfile(
        domain="economics",
        high_value_roles=("identification", "method_detail", "data_detail", "robustness", "limitation_or_caveat"),
        trust_gates=("integrity_gate", "method_transparency_for_causal_claims"),
        annotation_only=("raw_citation_count", "author_influence"),
        recency_policy="boost_for_current_market_or_policy_questions",
        corroboration_policy="prefer_independent_designs_datasets_and_robustness_checks",
    ),
    "psychology": DomainDecisionProfile(
        domain="psychology",
        high_value_roles=("measurement", "sample_detail", "preregistration", "replication", "limitation_or_caveat"),
        trust_gates=("integrity_gate", "measurement_detail_for_empirical_claims"),
        annotation_only=("raw_citation_count", "journal_prestige"),
        recency_policy="boost_when_methods_or_replication_status_changed",
        corroboration_policy="prefer_replication_independent_samples_and_bias_assessment",
    ),
    "anthropology": DomainDecisionProfile(
        domain="anthropology",
        high_value_roles=("fieldwork_context", "positionality", "primary_account", "interpretive_detail"),
        trust_gates=("integrity_gate", "provenance_for_primary_claims"),
        annotation_only=("citation_count", "large_sample_language"),
        recency_policy="annotate_temporal_context_not_simple_freshness",
        corroboration_policy="prefer_convergent_archival_ethnographic_or_primary_accounts",
    ),
    "history": DomainDecisionProfile(
        domain="history",
        high_value_roles=("primary_source", "archival_context", "historiography", "source_criticism"),
        trust_gates=("integrity_gate", "provenance_for_primary_claims"),
        annotation_only=("recency", "citation_count"),
        recency_policy="annotate_historiographic_date",
        corroboration_policy="prefer_independent_primary_sources_and_historiographic_conflict",
    ),
    "cultural_studies": DomainDecisionProfile(
        domain="cultural_studies",
        high_value_roles=("theoretical_frame", "close_reading", "contextualization", "counter_reading"),
        trust_gates=("integrity_gate", "clear_object_of_analysis"),
        annotation_only=("replication_language", "citation_count"),
        recency_policy="annotate_cultural_period_and_theoretical_moment",
        corroboration_policy="surface_interpretive_disagreement_not_only_consensus",
    ),
    "legal_market": DomainDecisionProfile(
        domain="legal_market",
        high_value_roles=("authority", "jurisdiction", "filing_detail", "market_timing", "risk_disclosure"),
        trust_gates=("integrity_gate", "authority_or_filing_provenance", "temporal_validity"),
        annotation_only=("academic_citation_count", "open_access"),
        recency_policy="gate_when_law_or_market_state_can_change",
        corroboration_policy="prefer_authoritative_primary_documents_and_independent_market_checks",
    ),
    "public_opinion": DomainDecisionProfile(
        domain="public_opinion",
        high_value_roles=("stance_expression", "concern_theme", "community_context", "time_window", "platform_context"),
        trust_gates=("integrity_gate", "public_visibility", "safety_gate", "aggregation_required"),
        annotation_only=("likes", "shares", "follower_count", "raw_virality"),
        recency_policy="annotate_time_window_and_platform_event_context",
        corroboration_policy="aggregate_across_posts_threads_platforms_and_compare_to_surveys_or_representative_data",
    ),
}


def domain_profile(domain: str | None) -> DomainDecisionProfile:
    return DOMAIN_DECISION_PROFILES.get(domain or "", DOMAIN_DECISION_PROFILES["general_academic"])


def query_intent(query: str) -> str:
    terms = set(tokenize(query))
    if terms & {"method", "methods", "identification", "design", "sample", "data"}:
        return "method"
    if terms & {"limitation", "bias", "caveat", "weakness", "uncertainty"}:
        return "limitation"
    if terms & {"conflict", "contradict", "debate", "challenge", "mixed"}:
        return "conflict"
    if terms & {"current", "today", "recent", "latest", "market", "price"}:
        return "current_state"
    if terms & {"law", "legal", "court", "statute", "filing", "contract"}:
        return "authority"
    return "substantive"


def domain_decision_notes(
    query: str,
    signals: dict[str, Any],
    domain: str = "general_academic",
) -> dict[str, Any]:
    profile = domain_profile(domain)
    intent = query_intent(query)
    evidence_role = signals.get("evidence_role") or {}
    source_trust = signals.get("source_trust") or {}
    safety = signals.get("safety") or {}
    notes: list[str] = []
    gates: list[str] = []
    annotations: list[str] = list(profile.annotation_only)

    if safety.get("decision") in {"block", "quarantine"}:
        gates.append("safety_gate")
    if source_trust.get("integrity_gate") is False:
        gates.append("integrity_gate")
    if intent == "method" and float(evidence_role.get("method_signal", 0.0) or 0.0) >= 0.35:
        notes.append("query_intent_matches_method_detail")
    if intent == "limitation" and float(evidence_role.get("limitation_signal", 0.0) or 0.0) >= 0.35:
        notes.append("query_intent_matches_limitation")
    if intent == "current_state" and profile.recency_policy.startswith("gate"):
        gates.append("temporal_validity")
    if domain in {"history", "anthropology", "cultural_studies"}:
        notes.append("interpretive_domain_requires_contextual_quality_review")
    if domain == "public_opinion":
        notes.append("public_opinion_requires_aggregation_and_representativeness_caution")

    return {
        "domain": profile.domain,
        "query_intent": intent,
        "high_value_roles": list(profile.high_value_roles),
        "trust_gates": list(profile.trust_gates),
        "active_gates": gates,
        "annotations": annotations,
        "decision_notes": notes,
        "recency_policy": profile.recency_policy,
        "corroboration_policy": profile.corroboration_policy,
        "safety_policy": profile.safety_policy,
    }

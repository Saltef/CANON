from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from canon.models import Work


DOCUMENT_TYPES = {
    "academic_article",
    "preprint",
    "policy_report",
    "regulatory_filing",
    "legal_authority",
    "market_report",
    "news_article",
    "social_media_post",
    "forum_comment",
    "web_page",
    "note",
    "transcript",
    "archival_primary_source",
    "unknown_unstructured_text",
}


@dataclass(frozen=True)
class DocumentTypeProfile:
    document_type: str
    evidence_family: str
    confidence: float
    authority_signal: float
    timeliness_need: float
    source_intent_risk: float
    adversarial_surface: float
    extraction_risk: float
    grey_literature: bool
    recommended_decision_role: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_signals(self) -> dict[str, float | str | bool]:
        return {
            "document_type": self.document_type,
            "evidence_family": self.evidence_family,
            "document_type_confidence": self.confidence,
            "authority_signal": self.authority_signal,
            "timeliness_need": self.timeliness_need,
            "source_intent_risk": self.source_intent_risk,
            "adversarial_surface": self.adversarial_surface,
            "extraction_risk": self.extraction_risk,
            "grey_literature": self.grey_literature,
            "recommended_decision_role": self.recommended_decision_role,
        }


def document_type_profile(work: Work) -> DocumentTypeProfile:
    raw = work.raw or {}
    source_type = raw_source_type(raw)
    haystack = normalized_haystack(work)
    provenance = str(raw.get("provenance") or "").lower()

    if source_type in {"note", "memo", "user_note", "research_note"}:
        return profile(
            "note",
            "user_supplied_text",
            0.75,
            authority_signal=0.35,
            timeliness_need=0.6,
            source_intent_risk=0.45,
            adversarial_surface=0.45,
            extraction_risk=0.25,
            grey_literature=True,
            role="annotate_and_corroborate",
            notes=["User-supplied notes are useful working evidence but need source, author, and corroboration checks."],
        )
    if matches(haystack, ["10-k", "10-q", "8-k", "annual report", "sec filing", "edgar", "regulatory filing"]) or source_type in {"filing", "regulatory_filing"}:
        return profile(
            "regulatory_filing",
            "official_record",
            0.9,
            authority_signal=0.85,
            timeliness_need=0.9,
            source_intent_risk=0.45,
            adversarial_surface=0.25,
            extraction_risk=0.35,
            grey_literature=True,
            role="boost_or_gate_by_authority",
            notes=["Official filings can be authoritative for disclosed facts but still reflect strategic incentives."],
        )
    if matches(haystack, ["court", "supreme court", "district court", "appellate", "statute", "regulation", "v. "]) or source_type in {"legal", "case_law", "court_opinion", "statute"}:
        return profile(
            "legal_authority",
            "official_record",
            0.85,
            authority_signal=0.9,
            timeliness_need=0.85,
            source_intent_risk=0.2,
            adversarial_surface=0.2,
            extraction_risk=0.4,
            grey_literature=True,
            role="gate_by_jurisdiction_and_authority",
            notes=["Legal authority requires jurisdiction, date, procedural posture, and citator-style validity checks."],
        )
    if matches(haystack, ["transcript", "earnings call", "hearing", "interview transcript", "deposition"]):
        return profile(
            "transcript",
            "primary_or_near_primary_text",
            0.8,
            authority_signal=0.55,
            timeliness_need=0.75,
            source_intent_risk=0.55,
            adversarial_surface=0.35,
            extraction_risk=0.45,
            grey_literature=True,
            role="annotate_and_corroborate",
            notes=["Transcripts are useful primary evidence but often need speaker, context, and incentive checks."],
        )
    if source_type in {"preprint", "repository"} or matches(haystack, ["preprint", "arxiv", "ssrn", "osf preprints"]):
        return profile(
            "preprint",
            "scholarly_grey_literature",
            0.8,
            authority_signal=0.45,
            timeliness_need=0.65,
            source_intent_risk=0.25,
            adversarial_surface=0.2,
            extraction_risk=0.3,
            grey_literature=True,
            role="annotate_review_status",
            notes=["Preprints can be valuable and current, but review status should be annotated rather than hidden."],
        )
    if source_type in {"policy_report", "institutional_report", "government_report"} or (
        source_type in {"report", "dataset"} and matches(haystack, ["world bank", "imf", "oecd", "united nations", "government", "regulator", "department", "ministry", "agency"])
    ):
        return profile(
            "policy_report",
            "institutional_report",
            0.8,
            authority_signal=0.7,
            timeliness_need=0.7,
            source_intent_risk=0.35,
            adversarial_surface=0.2,
            extraction_risk=0.35,
            grey_literature=True,
            role="rank_with_institution_and_method_context",
            notes=["Institutional reports need mandate, method, and independence checks."],
        )
    if source_type in {"market_report", "analyst_report"} or source_type == "report" or matches(haystack, ["industry report", "market report", "investor presentation", "analyst report", "white paper"]):
        return profile(
            "market_report",
            "commercial_or_market_analysis",
            0.75,
            authority_signal=0.45,
            timeliness_need=0.9,
            source_intent_risk=0.65,
            adversarial_surface=0.35,
            extraction_risk=0.35,
            grey_literature=True,
            role="annotate_incentives_and_corroborate",
            notes=["Market reports can be decision-relevant but incentive and recency risks are high."],
        )
    if source_type in {"journal", "book", "conference", "academic_article"} or work.doi or work.referenced_work_count >= 10:
        return profile(
            "academic_article",
            "scholarly",
            0.85,
            authority_signal=0.7,
            timeliness_need=0.45,
            source_intent_risk=0.2,
            adversarial_surface=0.15,
            extraction_risk=0.3,
            grey_literature=False,
            role="rank_with_method_and_domain_quality",
            notes=["Academic status is not quality by itself; methods, limitations, and fit still need evaluation."],
        )
    if source_type in {"newspaper", "news", "magazine"} or matches(haystack, ["news", "newspaper", "press", "wire service"]):
        return profile(
            "news_article",
            "journalistic",
            0.7,
            authority_signal=0.45,
            timeliness_need=0.85,
            source_intent_risk=0.4,
            adversarial_surface=0.4,
            extraction_risk=0.25,
            grey_literature=True,
            role="annotate_and_corroborate",
            notes=["News is useful for events and timelines but should be corroborated for claims requiring analysis."],
        )
    if source_type in {"social_media", "social_media_post", "tweet", "post", "status"} or matches(haystack, ["social media", "public post", "tweet", "retweet", "likes", "shares", "reposts"]):
        return profile(
            "social_media_post",
            "public_opinion",
            0.75,
            authority_signal=0.15,
            timeliness_need=0.9,
            source_intent_risk=0.8,
            adversarial_surface=0.85,
            extraction_risk=0.55,
            grey_literature=True,
            role="aggregate_public_opinion_and_screen_safety",
            notes=["Social-media posts are signals of public expression, not reliable factual authority; aggregate, de-duplicate, and safety-screen before use."],
        )
    if source_type in {"forum", "forum_comment", "comment", "discussion_board"} or matches(haystack, ["forum comment", "discussion thread", "reddit", "hacker news", "public comment"]):
        return profile(
            "forum_comment",
            "public_opinion",
            0.75,
            authority_signal=0.15,
            timeliness_need=0.8,
            source_intent_risk=0.75,
            adversarial_surface=0.8,
            extraction_risk=0.55,
            grey_literature=True,
            role="aggregate_public_opinion_and_screen_safety",
            notes=["Forum comments can reveal public concerns or narratives but require aggregation, moderation-context, and adversarial screening."],
        )
    if matches(haystack, ["archive", "archival", "manuscript", "collection", "primary source"]):
        return profile(
            "archival_primary_source",
            "primary_or_near_primary_text",
            0.75,
            authority_signal=0.65,
            timeliness_need=0.25,
            source_intent_risk=0.25,
            adversarial_surface=0.2,
            extraction_risk=0.55,
            grey_literature=True,
            role="annotate_context_and_provenance",
            notes=["Archival material needs provenance, context, representativeness, and extraction-quality checks."],
        )
    if provenance == "web" or source_type in {"web", "blog", " webpage", "webpage"} or work.landing_page_url:
        return profile(
            "web_page",
            "open_web",
            0.65,
            authority_signal=0.2,
            timeliness_need=0.75,
            source_intent_risk=0.7,
            adversarial_surface=0.75,
            extraction_risk=0.45,
            grey_literature=True,
            role="screen_safety_and_corroborate",
            notes=["Open-web text has high injection, incentive, and provenance risks; treat as evidence only after screening."],
        )
    return profile(
        "unknown_unstructured_text",
        "unknown",
        0.25,
        authority_signal=0.2,
        timeliness_need=0.5,
        source_intent_risk=0.5,
        adversarial_surface=0.5,
        extraction_risk=0.5,
        grey_literature=True,
        role="annotate_missing_type",
        notes=["Unknown document type should lower confidence and trigger metadata completion."],
    )


def profile(
    document_type: str,
    evidence_family: str,
    confidence: float,
    authority_signal: float,
    timeliness_need: float,
    source_intent_risk: float,
    adversarial_surface: float,
    extraction_risk: float,
    grey_literature: bool,
    role: str,
    notes: list[str],
) -> DocumentTypeProfile:
    if document_type not in DOCUMENT_TYPES:
        raise ValueError(f"Unknown document type: {document_type}")
    return DocumentTypeProfile(
        document_type=document_type,
        evidence_family=evidence_family,
        confidence=bounded(confidence),
        authority_signal=bounded(authority_signal),
        timeliness_need=bounded(timeliness_need),
        source_intent_risk=bounded(source_intent_risk),
        adversarial_surface=bounded(adversarial_surface),
        extraction_risk=bounded(extraction_risk),
        grey_literature=grey_literature,
        recommended_decision_role=role,
        notes=notes,
    )


def normalized_haystack(work: Work) -> str:
    values = [
        work.title,
        work.source_name,
        work.abstract,
        work.landing_page_url,
        json.dumps(work.raw or {}, sort_keys=True, default=str),
    ]
    return " ".join(str(value).lower() for value in values if value)


def raw_source_type(raw: dict[str, Any]) -> str | None:
    candidates = [
        raw.get("document_type"),
        raw.get("source_type"),
        raw.get("type"),
        (raw.get("primary_location") or {}).get("source", {}).get("type"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate.strip().lower()
    return None


def matches(haystack: str, markers: list[str]) -> bool:
    return any(marker in haystack for marker in markers)


def bounded(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 6)))

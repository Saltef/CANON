from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from canon.evidence.domain import domain_decision_notes


@dataclass(frozen=True)
class EvidenceSignals:
    query_text_relevance: dict[str, float | int] = field(default_factory=dict)
    evidence_role: dict[str, float | str] = field(default_factory=dict)
    source_trust: dict[str, float | bool | str | None] = field(default_factory=dict)
    corroboration: dict[str, float | int | str | None] = field(default_factory=dict)
    temporal_validity: dict[str, float | int | str | None] = field(default_factory=dict)
    provenance: dict[str, str | int | None] = field(default_factory=dict)
    safety: dict[str, float | str | bool | list[str]] = field(default_factory=dict)
    domain_profile: dict[str, Any] = field(default_factory=dict)
    domain_fit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceDecision:
    chunk_id: str
    work_id: str
    decision: str
    score: float
    allowed_uses: list[str]
    reasons: list[str]
    signals: EvidenceSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "work_id": self.work_id,
            "decision": self.decision,
            "score": self.score,
            "allowed_uses": self.allowed_uses,
            "reasons": self.reasons,
            "signals": self.signals.to_dict(),
        }


def evidence_signals_from_result(result: dict[str, Any]) -> EvidenceSignals:
    decision = result.get("decision") or {}
    stage = decision.get("stage") or {}
    stage_signals = stage.get("signals") or {}
    components = result.get("components") or {}
    signals = EvidenceSignals(
        query_text_relevance=dict(stage_signals.get("query_text_relevance") or {}),
        evidence_role=dict(stage_signals.get("evidence_role") or evidence_role_from_components(components)),
        source_trust=dict(stage_signals.get("source_trust") or source_trust_from_components(components)),
        corroboration=dict(result.get("corroboration") or {}),
        temporal_validity={"recency": float(components.get("recency", 0.0)), "year": result.get("year")},
        provenance=dict(stage_signals.get("provenance") or provenance_from_result(result)),
        safety=dict((stage.get("safety_gate") or {}).get("safety") or safety_from_decision(decision)),
        domain_profile=dict(stage_signals.get("domain_profile") or {}),
    )
    query = str(result.get("query") or "")
    domain = str(result.get("domain") or signals.provenance.get("domain") or "general_academic")
    return EvidenceSignals(
        query_text_relevance=signals.query_text_relevance,
        evidence_role=signals.evidence_role,
        source_trust=signals.source_trust,
        corroboration=signals.corroboration,
        temporal_validity=signals.temporal_validity,
        provenance=signals.provenance,
        safety=signals.safety,
        domain_profile=signals.domain_profile,
        domain_fit=domain_decision_notes(query, signals.to_dict(), domain=domain),
    )


def evidence_decision_from_result(result: dict[str, Any]) -> EvidenceDecision:
    decision = result.get("decision") or {}
    safety_decision = decision.get("safety_decision", "allow")
    allowed_uses = ["inspect"]
    if decision.get("decision") != "reject" and safety_decision not in {"block", "quarantine"}:
        allowed_uses.append("generator_context")
    if safety_decision == "sanitize":
        allowed_uses.append("sanitized_context")
    return EvidenceDecision(
        chunk_id=result.get("chunk_id", ""),
        work_id=result.get("work_id", ""),
        decision=decision.get("decision", "allow"),
        score=float(decision.get("score", result.get("final_score", 0.0)) or 0.0),
        allowed_uses=allowed_uses,
        reasons=list(decision.get("reasons") or []),
        signals=evidence_signals_from_result(result),
    )


def evidence_role_from_components(components: dict[str, Any]) -> dict[str, float]:
    return {
        "section_role": float(components.get("section_role", 0.0)),
        "claim_density": float(components.get("claim_density", 0.0)),
        "method_signal": float(components.get("method_signal", 0.0)),
        "limitation_signal": float(components.get("limitation_signal", 0.0)),
        "evidence_specificity": float(components.get("evidence_specificity", 0.0)),
        "uncertainty_signal": float(components.get("uncertainty_signal", 0.0)),
    }


def source_trust_from_components(components: dict[str, Any]) -> dict[str, float]:
    return {
        "source_trust": float(components.get("source_quality", 0.0)),
        "source_quality_legacy": float(components.get("source_quality", 0.0)),
        "citation_centrality": float(components.get("citation_centrality", 0.0)),
        "author_score": float(components.get("author_score", 0.0)),
    }


def provenance_from_result(result: dict[str, Any]) -> dict[str, str | int | None]:
    return {
        "work_id": result.get("work_id"),
        "source_name": result.get("source_name"),
        "year": result.get("year"),
        "cluster_id": result.get("cluster_id"),
    }


def safety_from_decision(decision: dict[str, Any]) -> dict[str, float | str | list[str]]:
    return {
        "decision": decision.get("safety_decision", "allow"),
        "risk_score": float(decision.get("safety_risk", 0.0) or 0.0),
        "categories": [],
        "matched_patterns": [],
    }

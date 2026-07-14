from __future__ import annotations

from dataclasses import dataclass

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.policies import ScoreComponents
from canon.safety.scanner import SafetyScan


RETRACTION_QUERY_TERMS = {"retracted", "retraction", "withdrawn", "correction", "fraud"}


@dataclass(frozen=True)
class RetrievalDecision:
    score: float
    base_score: float
    decision: str
    reasons: list[str]
    safety_decision: str
    safety_risk: float
    utility: dict | None = None
    stage: dict | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "base_score": self.base_score,
            "decision": self.decision,
            "reasons": self.reasons,
            "safety_decision": self.safety_decision,
            "safety_risk": self.safety_risk,
            "utility": self.utility or {},
            "stage": self.stage or {},
        }


def apply_evidence_decision(
    query: str,
    document: RetrievalDocument,
    components: ScoreComponents,
    base_score: float,
    weights: dict[str, float],
    safety: SafetyScan,
    stage: dict | None = None,
) -> RetrievalDecision:
    reasons: list[str] = []
    relevance_floor = float(weights.get("relevance_floor", 0.0))
    if relevance_floor > 0 and components.relevance < relevance_floor:
        reasons.append("below_relevance_floor")

    integrity_gate = float(document.work_signals.get("integrity_gate", 1.0))
    if integrity_gate <= 0.0 and not is_retraction_query(query):
        return RetrievalDecision(
            score=0.0,
            base_score=round(base_score, 6),
            decision="reject",
            reasons=["integrity_gate_retracted"],
            safety_decision="allow",
            safety_risk=0.0,
            utility={},
            stage=stage,
        )

    utility = evidence_utility(base_score, components, weights)
    adjusted = utility["utility_score"]
    if safety.decision == "sanitize":
        adjusted *= 0.65
        reasons.append("safety_sanitize_penalty")

    if float(document.work_signals.get("source_trust", document.work_signals.get("source_quality", 0.0))) < 0.25:
        adjusted *= 0.85
        reasons.append("low_source_trust_penalty")

    if reasons and reasons[0] == "below_relevance_floor":
        adjusted *= 0.5

    return RetrievalDecision(
        score=round(max(0.0, adjusted), 6),
        base_score=round(base_score, 6),
        decision="allow",
        reasons=reasons,
        safety_decision=safety.decision,
        safety_risk=safety.risk_score,
        utility=utility | {"post_gate_score": round(max(0.0, adjusted), 6)},
        stage=stage,
    )


def non_linear_evidence_score(
    base_score: float,
    components: ScoreComponents,
    weights: dict[str, float],
) -> float:
    return evidence_utility(base_score, components, weights)["utility_score"]


def evidence_utility(
    base_score: float,
    components: ScoreComponents,
    weights: dict[str, float],
) -> dict:
    role_bonus = 0.0
    method_bonus = float(weights.get("method_signal_bonus", 0.0)) * components.method_signal
    limitation_bonus = float(weights.get("limitation_signal_bonus", 0.0)) * components.limitation_signal
    specificity_bonus = float(weights.get("evidence_specificity_bonus", 0.0)) * components.evidence_specificity
    role_bonus += method_bonus + limitation_bonus + specificity_bonus

    trust = max(components.source_quality, 0.05)
    relevance = max(components.relevance, components.semantic_similarity)
    trust_relevance_gate = 0.5 + 0.5 * relevance
    trust_relevance_multiplier = min(1.0, trust_relevance_gate + (0.15 * trust))
    utility_score = (base_score + role_bonus) * trust_relevance_multiplier
    return {
        "base_score": round(base_score, 6),
        "role_bonus": round(role_bonus, 6),
        "method_bonus": round(method_bonus, 6),
        "limitation_bonus": round(limitation_bonus, 6),
        "specificity_bonus": round(specificity_bonus, 6),
        "trust_relevance_multiplier": round(trust_relevance_multiplier, 6),
        "utility_score": round(utility_score, 6),
    }


def is_retraction_query(query: str) -> bool:
    terms = {term.strip(".,;:!?").lower() for term in query.split()}
    return bool(terms & RETRACTION_QUERY_TERMS)

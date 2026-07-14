from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from canon.config import load_settings
from canon.modeling.calibration_model import FEATURE_NAMES, CalibrationExample, average


@dataclass(frozen=True)
class AnchorPreferenceCase:
    case_id: str
    domain: str
    threat_model: str
    better_policy: str
    worse_policy: str
    better_features: dict[str, float]
    worse_features: dict[str, float]
    rationale: str
    target_source: str = "hard_negative_anchor"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["better_target"] = 1.0
        payload["worse_target"] = 0.0
        payload["feature_deltas"] = feature_delta(self.better_features, self.worse_features)
        return payload


def anchor_preference_cases() -> list[AnchorPreferenceCase]:
    return [
        AnchorPreferenceCase(
            case_id="anchor_prompt_injection_gate_001",
            domain="rag_security",
            threat_model="retrieved prompt injection should be gated before reranking or generation",
            better_policy="safety_gated_rag",
            worse_policy="semantic_attack_exposed",
            better_features=features(
                citation_support_score=0.7,
                answer_faithfulness_score=0.8,
                importance_fit_score=0.7,
                evidence_role_fit_score=0.7,
                source_trust_fit_score=0.6,
                safety_handling_score=1.0,
                relevance=0.7,
                semantic_similarity=0.7,
                source_quality=0.6,
            ),
            worse_features=features(
                citation_support_score=0.0,
                answer_faithfulness_score=0.0,
                importance_fit_score=0.8,
                evidence_role_fit_score=0.2,
                source_trust_fit_score=0.1,
                safety_handling_score=0.0,
                relevance=0.95,
                semantic_similarity=0.95,
                source_quality=0.1,
            ),
            rationale="A chunk that is highly semantically relevant but tries to instruct the model must lose to a safer evidence path.",
        ),
        AnchorPreferenceCase(
            case_id="anchor_independent_corroboration_001",
            domain="corroboration",
            threat_model="many dependent echoes should not outrank fewer independent sources",
            better_policy="independent_conflict_aware",
            worse_policy="echo_count_only",
            better_features=features(
                citation_support_score=0.8,
                answer_faithfulness_score=0.7,
                importance_fit_score=0.8,
                paradigm_coverage_score=0.4,
                evidence_role_fit_score=0.8,
                source_trust_fit_score=0.8,
                safety_handling_score=1.0,
                corroboration_handling_score=1.0,
                citation_count=0.5,
                claim_citation_count=0.5,
                conflict_note_count=0.4,
                distinct_sources=0.8,
                distinct_clusters=0.8,
                relevance=0.75,
                semantic_similarity=0.75,
                source_quality=0.8,
            ),
            worse_features=features(
                citation_support_score=0.7,
                answer_faithfulness_score=0.5,
                importance_fit_score=0.7,
                evidence_role_fit_score=0.5,
                source_trust_fit_score=0.25,
                safety_handling_score=1.0,
                corroboration_handling_score=0.0,
                citation_count=1.0,
                claim_citation_count=1.0,
                distinct_sources=0.8,
                distinct_clusters=0.1,
                relevance=0.8,
                semantic_similarity=0.8,
                source_quality=0.25,
            ),
            rationale="Support count without independence is weak evidence and should not dominate corroboration-aware ranking.",
        ),
        AnchorPreferenceCase(
            case_id="anchor_source_quality_not_access_001",
            domain="academic_quality",
            threat_model="open access, PDF availability, and source naming are not sufficient source quality",
            better_policy="methodologically_grounded_source",
            worse_policy="accessible_named_source_only",
            better_features=features(
                citation_support_score=0.8,
                answer_faithfulness_score=0.8,
                importance_fit_score=0.75,
                evidence_role_fit_score=0.9,
                source_trust_fit_score=0.85,
                safety_handling_score=1.0,
                citation_count=0.4,
                claim_citation_count=0.4,
                distinct_sources=0.4,
                distinct_clusters=0.4,
                relevance=0.75,
                semantic_similarity=0.7,
                source_quality=0.85,
                method_signal=0.8,
                limitation_signal=0.5,
                evidence_specificity=0.8,
                uncertainty_signal=0.3,
            ),
            worse_features=features(
                citation_support_score=0.3,
                answer_faithfulness_score=0.4,
                importance_fit_score=0.65,
                evidence_role_fit_score=0.3,
                source_trust_fit_score=0.2,
                safety_handling_score=1.0,
                citation_count=0.2,
                claim_citation_count=0.2,
                distinct_sources=0.2,
                distinct_clusters=0.2,
                relevance=0.8,
                semantic_similarity=0.75,
                source_quality=0.2,
            ),
            rationale="Accessibility and source naming should not substitute for methodological quality, provenance, and evidence specificity.",
        ),
        AnchorPreferenceCase(
            case_id="anchor_conflict_coverage_001",
            domain="social_science",
            threat_model="one-sided evidence should not outrank explicit conflict-aware synthesis for contested questions",
            better_policy="balanced_conflict_aware",
            worse_policy="one_sided_high_similarity",
            better_features=features(
                citation_support_score=0.8,
                answer_faithfulness_score=0.8,
                importance_fit_score=0.8,
                paradigm_coverage_score=0.6,
                evidence_role_fit_score=0.8,
                source_trust_fit_score=0.7,
                safety_handling_score=1.0,
                corroboration_handling_score=0.8,
                citation_count=0.6,
                claim_citation_count=0.6,
                conflict_note_count=0.7,
                distinct_sources=0.7,
                distinct_clusters=0.7,
                relevance=0.75,
                semantic_similarity=0.75,
                source_quality=0.7,
                limitation_signal=0.6,
                uncertainty_signal=0.4,
            ),
            worse_features=features(
                citation_support_score=0.6,
                answer_faithfulness_score=0.5,
                importance_fit_score=0.75,
                evidence_role_fit_score=0.5,
                source_trust_fit_score=0.5,
                safety_handling_score=1.0,
                corroboration_handling_score=0.1,
                citation_count=0.5,
                claim_citation_count=0.5,
                conflict_note_count=0.0,
                distinct_sources=0.3,
                distinct_clusters=0.3,
                relevance=0.95,
                semantic_similarity=0.95,
                source_quality=0.5,
            ),
            rationale="For contested social-science questions, conflict coverage and uncertainty handling should beat one-sided similarity.",
        ),
        AnchorPreferenceCase(
            case_id="anchor_recency_domain_policy_001",
            domain="market_analysis",
            threat_model="current market or legal questions require domain-specific recency/authority handling",
            better_policy="current_authoritative_update",
            worse_policy="older_semantic_match",
            better_features=features(
                citation_support_score=0.8,
                answer_faithfulness_score=0.8,
                importance_fit_score=0.75,
                evidence_role_fit_score=0.8,
                source_trust_fit_score=0.8,
                safety_handling_score=1.0,
                citation_count=0.4,
                claim_citation_count=0.4,
                distinct_sources=0.5,
                distinct_clusters=0.5,
                relevance=0.75,
                semantic_similarity=0.75,
                source_quality=0.8,
                evidence_specificity=0.8,
                uncertainty_signal=0.4,
            ),
            worse_features=features(
                citation_support_score=0.5,
                answer_faithfulness_score=0.5,
                importance_fit_score=0.7,
                evidence_role_fit_score=0.5,
                source_trust_fit_score=0.4,
                safety_handling_score=1.0,
                citation_count=0.4,
                claim_citation_count=0.4,
                distinct_sources=0.4,
                distinct_clusters=0.4,
                relevance=0.9,
                semantic_similarity=0.9,
                source_quality=0.4,
                evidence_specificity=0.5,
            ),
            rationale="For market/legal-style uses, current authoritative evidence should beat older semantic matches when the question is time-sensitive.",
        ),
    ]


def anchor_preference_examples() -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    for case in anchor_preference_cases():
        examples.append(
            CalibrationExample(
                task_id=f"{case.case_id}::{case.better_policy}",
                query_id=case.case_id,
                policy=case.better_policy,
                features=case.better_features,
                target=1.0,
                target_source=case.target_source,
                proxy_baseline=0.5,
            )
        )
        examples.append(
            CalibrationExample(
                task_id=f"{case.case_id}::{case.worse_policy}",
                query_id=case.case_id,
                policy=case.worse_policy,
                features=case.worse_features,
                target=0.0,
                target_source=case.target_source,
                proxy_baseline=0.5,
            )
        )
    return examples


def build_hard_negative_report() -> dict[str, Any]:
    cases = anchor_preference_cases()
    examples = anchor_preference_examples()
    report = {
        "benchmark_id": "hard_negative_anchor_preferences_v1",
        "case_count": len(cases),
        "example_count": len(examples),
        "domains": sorted({case.domain for case in cases}),
        "target_source": "hard_negative_anchor",
        "cases": [case.to_dict() for case in cases],
        "feature_summary": feature_summary(examples),
        "limitations": [
            "These are anchor stress tests, not broad expert labels.",
            "They encode minimum scientific and safety expectations for known failure modes.",
            "Passing these cases does not validate source trust, methodological rigor, or prompt-injection robustness in the wild.",
        ],
    }
    return report


def features(**values: float) -> dict[str, float]:
    payload = {name: 0.0 for name in FEATURE_NAMES}
    for name, value in values.items():
        if name not in payload:
            raise KeyError(f"Unknown feature: {name}")
        payload[name] = bounded(value)
    return payload


def feature_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: round(float(left.get(name, 0.0)) - float(right.get(name, 0.0)), 6) for name in FEATURE_NAMES}


def feature_summary(examples: list[CalibrationExample]) -> dict[str, Any]:
    if not examples:
        return {"feature_names": FEATURE_NAMES, "average_by_target": {}}
    grouped: dict[str, list[CalibrationExample]] = {"preferred": [], "dispreferred": []}
    for example in examples:
        key = "preferred" if example.target >= 0.5 else "dispreferred"
        grouped[key].append(example)
    return {
        "feature_names": FEATURE_NAMES,
        "average_by_target": {
            key: {name: average(item.features.get(name, 0.0) for item in items) for name in FEATURE_NAMES}
            for key, items in grouped.items()
        },
    }


def bounded(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 6)))


def write_report() -> dict[str, Any]:
    settings = load_settings()
    report = build_hard_negative_report()
    path = settings.reports_dir / "hard_negative_anchor_preferences_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-negative anchor preference fixtures.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = write_report() if args.write_report else build_hard_negative_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

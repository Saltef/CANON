from __future__ import annotations

from dataclasses import asdict, dataclass

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.policies import ScoreComponents


CONTROL_WEIGHT_KEYS = {
    "diversity",
    "diversity_focus_floor",
    "diversity_relevance_floor",
    "exclusion_penalty",
    "low_relevance_penalty",
    "relevance_floor",
}
COMPONENT_LABELS = {
    "relevance": "Lexical query match",
    "semantic_similarity": "Semantic query match",
    "source_quality": "Source quality",
    "citation_centrality": "Citation centrality",
    "author_score": "Author signal",
    "section_role": "Section importance",
    "claim_density": "Claim density",
    "recency": "Recency",
}


@dataclass(frozen=True)
class RetrievalTraceItem:
    rank: int
    chunk_id: str
    work_id: str
    title: str
    source_name: str | None
    year: int | None
    section: str
    cluster_id: int | None
    preview: str
    final_score: float
    base_score: float
    diversity_bonus: float
    components: dict[str, float]
    weights: dict[str, float]
    explanation: dict

    def to_dict(self) -> dict:
        return asdict(self)


def make_trace_item(
    rank: int,
    document: RetrievalDocument,
    final_score: float,
    components: ScoreComponents,
    weights: dict[str, float],
    preview_chars: int,
    base_score: float | None = None,
) -> RetrievalTraceItem:
    preview = document.text[:preview_chars].strip()
    if len(document.text) > preview_chars:
        preview += "..."
    rounded_final_score = round(final_score, 6)
    rounded_base_score = round(base_score if base_score is not None else final_score, 6)
    rounded_diversity_bonus = round(final_score - (base_score if base_score is not None else final_score), 6)
    rounded_components = {key: round(value, 6) for key, value in components.as_dict().items()}
    return RetrievalTraceItem(
        rank=rank,
        chunk_id=document.chunk_id,
        work_id=document.work_id,
        title=document.title,
        source_name=document.source_name,
        year=document.year,
        section=document.section,
        cluster_id=document.cluster_id,
        preview=preview,
        final_score=rounded_final_score,
        base_score=rounded_base_score,
        diversity_bonus=rounded_diversity_bonus,
        components=rounded_components,
        weights=weights,
        explanation=explain_trace_item(
            rank=rank,
            final_score=rounded_final_score,
            base_score=rounded_base_score,
            diversity_bonus=rounded_diversity_bonus,
            components=rounded_components,
            weights=weights,
        ),
    )


def explain_trace_item(
    rank: int,
    final_score: float,
    base_score: float,
    diversity_bonus: float,
    components: dict[str, float],
    weights: dict[str, float],
) -> dict:
    contributions = component_contributions(components, weights)
    return {
        "summary": explanation_summary(rank, contributions, diversity_bonus),
        "top_contributors": contributions[:4],
        "score_adjustments": {
            "base_score": round(base_score, 6),
            "diversity_bonus": round(diversity_bonus, 6),
            "final_score": round(final_score, 6),
        },
        "reasons": explanation_reasons(components, diversity_bonus),
    }


def component_contributions(components: dict[str, float], weights: dict[str, float]) -> list[dict]:
    active_weights = {
        key: float(value)
        for key, value in weights.items()
        if key not in CONTROL_WEIGHT_KEYS and float(value) > 0.0
    }
    denominator = sum(active_weights.values())
    if denominator <= 0:
        return []
    contributions = []
    for component, weight in active_weights.items():
        value = float(components.get(component, 0.0))
        weighted_value = value * weight
        contributions.append(
            {
                "component": component,
                "label": COMPONENT_LABELS.get(component, component.replace("_", " ").title()),
                "value": round(value, 6),
                "weight": round(weight, 6),
                "weighted_value": round(weighted_value, 6),
                "score_share": round(weighted_value / denominator, 6),
            }
        )
    return sorted(
        contributions,
        key=lambda item: (item["score_share"], item["value"], item["component"]),
        reverse=True,
    )


def explanation_summary(rank: int, contributions: list[dict], diversity_bonus: float) -> str:
    if not contributions:
        return f"Ranked {rank} from non-positive retrieval weights."
    labels = [item["label"] for item in contributions[:2] if item["score_share"] > 0.0]
    if not labels:
        return f"Ranked {rank} with no positive measured component contribution."
    summary = f"Ranked {rank} mainly because of {', '.join(labels).lower()}."
    if diversity_bonus > 0:
        summary += " The final score includes a diversity bonus for cluster coverage."
    return summary


def explanation_reasons(components: dict[str, float], diversity_bonus: float) -> list[str]:
    reasons = []
    thresholds = [
        ("relevance", 0.75, "high_lexical_relevance"),
        ("semantic_similarity", 0.75, "high_semantic_alignment"),
        ("source_quality", 0.75, "high_source_quality"),
        ("citation_centrality", 0.25, "citation_centrality_signal"),
        ("author_score", 0.25, "author_score_signal"),
        ("section_role", 0.75, "important_section"),
        ("claim_density", 0.5, "claim_dense"),
        ("recency", 0.75, "recent_work"),
    ]
    for component, threshold, reason in thresholds:
        if float(components.get(component, 0.0)) >= threshold:
            reasons.append(reason)
    if diversity_bonus > 0:
        reasons.append("cluster_diversity_bonus")
    return reasons

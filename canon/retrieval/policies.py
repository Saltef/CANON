from __future__ import annotations

from dataclasses import dataclass

from canon.retrieval.corpus import RetrievalDocument


@dataclass(frozen=True)
class ScoreComponents:
    relevance: float
    semantic_similarity: float
    source_quality: float
    citation_centrality: float
    author_score: float
    section_role: float
    claim_density: float
    recency: float

    def as_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "semantic_similarity": self.semantic_similarity,
            "source_quality": self.source_quality,
            "citation_centrality": self.citation_centrality,
            "author_score": self.author_score,
            "section_role": self.section_role,
            "claim_density": self.claim_density,
            "recency": self.recency,
        }


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    high = max(values)
    low = min(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def components_for_document(
    document: RetrievalDocument,
    relevance: float,
    semantic_similarity: float = 0.0,
) -> ScoreComponents:
    return ScoreComponents(
        relevance=relevance,
        semantic_similarity=semantic_similarity,
        source_quality=document.work_signals.get("source_quality", 0.0),
        citation_centrality=document.work_signals.get("citation_centrality", 0.0),
        author_score=document.work_signals.get("author_score", 0.0),
        section_role=document.chunk_importance.get("section_role", 0.0),
        claim_density=document.chunk_importance.get("claim_density", 0.0),
        recency=document.work_signals.get("recency", 0.0),
    )


def weighted_score(components: ScoreComponents, weights: dict[str, float]) -> float:
    component_values = components.as_dict()
    denominator = sum(value for value in weights.values() if value > 0)
    if denominator <= 0:
        return 0.0
    numerator = sum(component_values.get(name, 0.0) * weight for name, weight in weights.items())
    return numerator / denominator

from __future__ import annotations

from dataclasses import asdict, dataclass

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.policies import ScoreComponents


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
        final_score=round(final_score, 6),
        base_score=round(base_score if base_score is not None else final_score, 6),
        diversity_bonus=round(final_score - (base_score if base_score is not None else final_score), 6),
        components={key: round(value, 6) for key, value in components.as_dict().items()},
        weights=weights,
    )

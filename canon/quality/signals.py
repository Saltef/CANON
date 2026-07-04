from __future__ import annotations

from canon.models import Work


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
    return {
        "retraction": retraction_score(work),
        "citation_impact": citation_impact_score(work),
        "open_access": open_access_score(work),
        "reference_coverage": reference_coverage_score(work),
        "author_score": author_score(work),
    }

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from canon.models import Work


def corpus_diagnostics(works: list[Work]) -> dict[str, Any]:
    total = len(works)
    years = Counter(str(work.year) for work in works if work.year)
    sources = Counter(work.source_name or "UNKNOWN" for work in works)
    abstract_count = sum(1 for work in works if work.abstract)
    oa_count = sum(1 for work in works if work.is_open_access)
    pdf_count = sum(1 for work in works if work.pdf_url)
    references_count = sum(1 for work in works if work.referenced_work_count > 0)
    author_count = sum(1 for work in works if work.author_count > 0)
    author_impact_count = sum(1 for work in works if work.max_author_cited_by_count > 0)
    retracted_count = sum(1 for work in works if work.is_retracted)
    return {
        "total_works": total,
        "abstract_coverage": ratio(abstract_count, total),
        "open_access_coverage": ratio(oa_count, total),
        "pdf_url_coverage": ratio(pdf_count, total),
        "reference_list_coverage": ratio(references_count, total),
        "authorship_coverage": ratio(author_count, total),
        "author_impact_coverage": ratio(author_impact_count, total),
        "retracted_works": retracted_count,
        "year_distribution": dict(sorted(years.items())),
        "top_sources": dict(sources.most_common(20)),
    }


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def serialize_works(works: list[Work]) -> list[dict[str, Any]]:
    rows = []
    for work in works:
        row = asdict(work)
        row.pop("raw", None)
        rows.append(row)
    return rows

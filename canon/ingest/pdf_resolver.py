from __future__ import annotations

from dataclasses import dataclass

from canon.models import Work


@dataclass(frozen=True)
class PdfCandidate:
    work_id: str
    url: str
    source: str
    priority: int


def pdf_candidates(work: Work) -> list[PdfCandidate]:
    candidates: list[PdfCandidate] = []
    if work.pdf_url:
        candidates.append(PdfCandidate(work_id=work.id, url=work.pdf_url, source="openalex", priority=1))
    if work.landing_page_url and work.landing_page_url != work.pdf_url:
        candidates.append(
            PdfCandidate(work_id=work.id, url=work.landing_page_url, source="landing_page", priority=5)
        )
    return candidates

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canon.quality.domain_profiles import domain_fit_vector
from canon.quality.document_types import document_type_profile
from canon.quality.signals import source_trust_vector


@dataclass(frozen=True)
class RetrievalDocument:
    chunk_id: str
    work_id: str
    title: str
    year: int | None
    source_name: str | None
    section: str
    text: str
    chunk_importance: dict[str, float]
    work_signals: dict[str, Any]
    cluster_id: int | None = None


def load_processed_corpus(
    data_dir: Path,
    mode: str = "dry_run",
    cluster_assignments: dict[str, int] | None = None,
) -> list[RetrievalDocument]:
    works_path = data_dir / "processed" / f"works_{mode}.json"
    chunks_path = data_dir / "processed" / f"chunks_{mode}.json"
    works = {work["id"]: work for work in json.loads(works_path.read_text(encoding="utf-8"))}
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [
        document_from_chunk(chunk, works[chunk["work_id"]], cluster_assignments or {})
        for chunk in chunks
        if chunk["work_id"] in works
    ]


def document_from_chunk(
    chunk: dict[str, Any],
    work: dict[str, Any],
    cluster_assignments: dict[str, int] | None = None,
) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk["id"],
        work_id=chunk["work_id"],
        title=work["title"],
        year=work["year"],
        source_name=work["source_name"],
        section=chunk["section"],
        text=chunk["text"],
        chunk_importance=chunk["importance"],
        work_signals=work_signal_values(work),
        cluster_id=(cluster_assignments or {}).get(chunk["work_id"]),
    )


def work_signal_values(work: dict[str, Any]) -> dict[str, Any]:
    cited_by_count = int(work.get("cited_by_count") or 0)
    referenced_work_count = int(work.get("referenced_work_count") or 0)
    year = work.get("year")
    work_model = work_model_from_dict(work)
    domain = work.get("domain") or (work.get("raw") or {}).get("domain") or "general_academic"
    trust = source_trust_vector(work_model, domain=str(domain))
    document_profile = document_type_profile(work_model)
    domain_fit = domain_fit_vector(
        work_model,
        domain=str(domain),
        document_profile=document_profile.to_dict(),
        source_trust=trust,
    )
    signals = {
        "source_quality": float(trust["source_trust"]),
        "source_trust": float(trust["source_trust"]),
        "integrity_gate": 1.0 if trust["integrity_gate"] else 0.0,
        "domain": domain,
        "provenance": work.get("provenance") or (work.get("raw") or {}).get("provenance") or "academic",
        "citation_centrality": min(1.0, cited_by_count / 500.0),
        "author_score": author_score(work),
        "reference_coverage": min(1.0, referenced_work_count / 30.0),
        "recency": recency_score(year),
    }
    signals.update(document_profile.to_signals())
    signals.update(domain_fit)
    return signals


def work_model_from_dict(work: dict[str, Any]):
    from canon.models import Work

    return Work(
        id=work.get("id", ""),
        doi=work.get("doi"),
        title=work.get("title", ""),
        year=work.get("year"),
        language=work.get("language"),
        abstract=work.get("abstract"),
        source_name=work.get("source_name"),
        is_open_access=bool(work.get("is_open_access")),
        is_retracted=bool(work.get("is_retracted")),
        cited_by_count=int(work.get("cited_by_count") or 0),
        referenced_work_count=int(work.get("referenced_work_count") or 0),
        pdf_url=work.get("pdf_url"),
        landing_page_url=work.get("landing_page_url"),
        raw=work.get("raw") or {},
        author_display_names=list(work.get("author_display_names") or []),
        author_count=int(work.get("author_count") or 0),
        max_author_cited_by_count=int(work.get("max_author_cited_by_count") or 0),
        max_author_works_count=int(work.get("max_author_works_count") or 0),
    )


def source_quality(work: dict[str, Any]) -> float:
    if work.get("is_retracted"):
        return 0.0
    score = 0.45
    if work.get("is_open_access"):
        score += 0.2
    if work.get("pdf_url"):
        score += 0.15
    if work.get("source_name"):
        score += 0.1
    if int(work.get("referenced_work_count") or 0) > 0:
        score += 0.1
    return min(1.0, score)


def recency_score(year: int | None, newest_year: int = 2026, half_life_years: int = 12) -> float:
    if not year:
        return 0.0
    age = max(0, newest_year - year)
    return round(0.5 ** (age / half_life_years), 4)


def author_score(work: dict[str, Any], cited_by_high_watermark: int = 5000) -> float:
    count = int(work.get("max_author_cited_by_count") or 0)
    if count <= 0:
        return 0.0
    return min(1.0, count / cited_by_high_watermark)

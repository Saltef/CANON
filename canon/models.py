from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Work:
    id: str
    doi: str | None
    title: str
    year: int | None
    language: str | None
    abstract: str | None
    source_name: str | None
    is_open_access: bool
    is_retracted: bool
    cited_by_count: int
    referenced_work_count: int
    pdf_url: str | None
    landing_page_url: str | None
    raw: dict[str, Any] = field(repr=False)
    author_display_names: list[str] = field(default_factory=list)
    author_count: int = 0
    max_author_cited_by_count: int = 0
    max_author_works_count: int = 0


@dataclass(frozen=True)
class Chunk:
    id: str
    work_id: str
    section: str
    text: str
    token_start: int
    token_end: int
    importance: dict[str, Any]

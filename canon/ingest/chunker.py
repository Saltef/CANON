from __future__ import annotations

import hashlib
import re

from canon.models import Chunk, Work


TOKEN_RE = re.compile(r"\S+")


def rough_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def section_importance(section: str) -> float:
    normalized = section.lower()
    if normalized in {"abstract", "conclusion"}:
        return 0.9
    if normalized in {"results", "findings", "discussion"}:
        return 0.85
    if normalized in {"methods", "methodology", "research design"}:
        return 0.75
    if "literature" in normalized or "theory" in normalized:
        return 0.8
    return 0.55


def claim_density(text: str) -> float:
    markers = [
        "find",
        "finds",
        "found",
        "show",
        "shows",
        "suggest",
        "suggests",
        "argue",
        "argues",
        "evidence",
        "effect",
        "significant",
    ]
    lower = text.lower()
    hits = sum(lower.count(marker) for marker in markers)
    return min(1.0, hits / 8.0)


def chunk_text(
    work: Work,
    section: str,
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    tokens = rough_tokens(text)
    if not tokens:
        return []
    step = max(1, chunk_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + chunk_tokens)
        chunk_text_value = " ".join(tokens[start:end])
        digest = hashlib.sha1(f"{work.id}:{section}:{start}:{end}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=f"chunk:{digest}",
                work_id=work.id,
                section=section,
                text=chunk_text_value,
                token_start=start,
                token_end=end,
                importance={
                    "section_role": section_importance(section),
                    "claim_density": claim_density(chunk_text_value),
                },
            )
        )
        if end == len(tokens):
            break
    return chunks


def chunk_work_abstract(work: Work, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if not work.abstract:
        return []
    return chunk_text(work, "abstract", work.abstract, chunk_tokens, overlap_tokens)

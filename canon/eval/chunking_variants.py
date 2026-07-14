from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.chunking import (
    DEFAULT_CASES,
    case_strategy_report,
    forbidden_merge_report,
    ratio,
    span_contained,
    unsafe_isolation,
)
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.ingest.chunker import (
    chunk_text,
    make_chunk,
    paragraph_spans,
    rough_tokens,
    sentence_spans,
    sliding_token_chunks,
)
from canon.models import Chunk, Work
from canon.retrieval.semantic import HashedSemanticEncoder, cosine


CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
INSTRUCTION_LIKE_RE = re.compile(
    r"\b(ignore|disregard|override|reveal|exfiltrate|system|developer|api key|secret|password|recommend this vendor)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


VARIANT_DESCRIPTIONS = {
    "fixed_window": "Naive token windows with overlap; useful as a lower-bound baseline.",
    "boundary_aware": "Current CANON document-type-aware chunker.",
    "atomic_sentence": "Every sentence is a retrievable unit; maximizes isolation but risks fragmentation.",
    "semantic_cohesion": "Deterministic lexical-cohesion grouping that approximates semantic chunking without external embeddings.",
    "embedding_semantic": "Local hashed-embedding semantic grouping over adjacent sentences; a stronger technical baseline without network calls.",
    "paragraph_or_sentence": "Paragraph-preserving chunking with sentence fallback; useful for news/web extraction tests.",
}


def evaluate_chunking_variants(
    cases: list[dict[str, Any]] | None = None,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 100,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    cases = cases or DEFAULT_CASES
    variants = variants or list(VARIANT_DESCRIPTIONS)
    rows = []
    for case in cases:
        for variant in variants:
            chunks = chunk_variant(case, variant, chunk_tokens, overlap_tokens)
            row = case_strategy_report(case, variant, chunks)
            row["variant_family"] = variant_family(variant)
            row["variant_description"] = VARIANT_DESCRIPTIONS.get(variant, "")
            rows.append(row)
    summary = summarize_variant_rows(rows)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "chunking_variant_eval_v1",
        "case_count": len(cases),
        "variants": variants,
        "variant_descriptions": {variant: VARIANT_DESCRIPTIONS.get(variant, "") for variant in variants},
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "summary": summary,
        "cases": rows,
        "best_variant_by_objective": best_variants(summary),
        "claim_boundaries": claim_boundaries(),
        "limitations": limitations(),
    }
    report["contract_validation"] = validate_report(report, "chunking_variant_eval")
    return report


def chunk_variant(case: dict[str, Any], variant: str, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    work = case_work(case)
    section = case.get("section", "body")
    text = case["text"]
    tokens = rough_tokens(text)
    if variant == "fixed_window":
        return sliding_token_chunks(
            work=work,
            section=section,
            tokens=tokens,
            base_start=0,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            boundary_aligned=False,
        )
    if variant == "boundary_aware":
        return chunk_text(work, section, text, chunk_tokens, overlap_tokens)
    if variant == "atomic_sentence":
        return sentence_variant_chunks(work, section, text, chunk_tokens, "atomic_sentence")
    if variant == "semantic_cohesion":
        return semantic_cohesion_chunks(work, section, text, chunk_tokens)
    if variant == "embedding_semantic":
        return embedding_semantic_chunks(work, section, text, chunk_tokens)
    if variant == "paragraph_or_sentence":
        return paragraph_or_sentence_chunks(work, section, text, chunk_tokens)
    raise ValueError(f"Unknown chunking variant: {variant}")


def case_work(case: dict[str, Any]) -> Work:
    return Work(
        id=f"chunking_variant:{case['case_id']}",
        doi=None,
        title=case["case_id"],
        year=2026,
        language="en",
        abstract=case["text"],
        source_name="chunking_variant_fixture",
        is_open_access=True,
        is_retracted=False,
        cited_by_count=0,
        referenced_work_count=0,
        pdf_url=None,
        landing_page_url=None,
        raw={"document_type": case.get("document_type"), "domain": case.get("domain"), "provenance": "fixture"},
    )


def sentence_variant_chunks(work: Work, section: str, text: str, chunk_tokens: int, strategy: str) -> list[Chunk]:
    spans = sentence_spans(text)
    if not spans:
        return sliding_token_chunks(
            work,
            section,
            rough_tokens(text),
            0,
            max(1, chunk_tokens),
            0,
            boundary_aligned=False,
            policy=variant_policy(strategy, work),
        )
    total_tokens = len(rough_tokens(text))
    chunks: list[Chunk] = []
    for sentence_text, start, end in spans:
        if end - start > chunk_tokens:
            chunks.extend(
                sliding_token_chunks(
                    work,
                    section,
                    rough_tokens(sentence_text),
                    start,
                    max(1, chunk_tokens),
                    0,
                    boundary_aligned=False,
                    total_tokens=total_tokens,
                    policy=variant_policy(strategy, work),
                )
            )
        else:
            chunks.append(
                make_chunk(
                    work,
                    section,
                    sentence_text,
                    start,
                    end,
                    total_tokens,
                    boundary_aligned=True,
                    policy=variant_policy(strategy, work, parent_start=start, parent_end=end),
                )
            )
    return chunks


def semantic_cohesion_chunks(work: Work, section: str, text: str, chunk_tokens: int) -> list[Chunk]:
    spans = sentence_spans(text)
    if len(spans) <= 1:
        return sentence_variant_chunks(work, section, text, chunk_tokens, "semantic_cohesion")
    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0
    previous_terms: set[str] = set()
    for span in spans:
        sentence_text, start, end = span
        terms = content_terms(sentence_text)
        sentence_len = end - start
        should_break = False
        if current:
            cohesion = jaccard(previous_terms, terms)
            should_break = (
                current_len + sentence_len > chunk_tokens
                or cohesion < 0.08
                or instruction_like(sentence_text)
                or any(instruction_like(item[0]) for item in current)
            )
        if should_break:
            groups.append(current)
            current = []
            current_len = 0
        current.append(span)
        current_len += sentence_len
        previous_terms = terms
    if current:
        groups.append(current)
    return chunk_groups(work, section, text, groups, "semantic_cohesion")


def embedding_semantic_chunks(work: Work, section: str, text: str, chunk_tokens: int) -> list[Chunk]:
    spans = sentence_spans(text)
    if len(spans) <= 1:
        return sentence_variant_chunks(work, section, text, chunk_tokens, "embedding_semantic")
    encoder = HashedSemanticEncoder(dimensions=512, model_id="hashed-semantic-chunking-v1")
    embeddings = [encoder.encode(span[0]) for span in spans]
    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0
    for index, span in enumerate(spans):
        sentence_text, start, end = span
        sentence_len = end - start
        should_break = False
        if current:
            adjacent_similarity = cosine(embeddings[index - 1], embeddings[index])
            current_text = " ".join(item[0] for item in current)
            should_break = (
                current_len + sentence_len > chunk_tokens
                or adjacent_similarity < embedding_break_threshold(work)
                or instruction_like(sentence_text)
                or instruction_like(current_text)
            )
        if should_break:
            groups.append(current)
            current = []
            current_len = 0
        current.append(span)
        current_len += sentence_len
    if current:
        groups.append(current)
    return chunk_groups(work, section, text, groups, "embedding_semantic")


def embedding_break_threshold(work: Work) -> float:
    document_type = str((work.raw or {}).get("document_type") or "")
    domain = str((work.raw or {}).get("domain") or "")
    if document_type in {"social_media_post", "forum_comment", "web_page"}:
        return 0.18
    if domain in {"legal_market", "public_opinion"}:
        return 0.14
    return 0.1


def paragraph_or_sentence_chunks(work: Work, section: str, text: str, chunk_tokens: int) -> list[Chunk]:
    paragraphs = paragraph_spans(text)
    if len(paragraphs) <= 1:
        return semantic_cohesion_chunks(work, section, text, chunk_tokens)
    groups: list[list[tuple[str, int, int]]] = []
    for paragraph_text, paragraph_start, paragraph_end in paragraphs:
        if paragraph_end - paragraph_start <= chunk_tokens and not instruction_like(paragraph_text):
            groups.append([(paragraph_text, paragraph_start, paragraph_end)])
            continue
        paragraph_sentences = [
            (sentence_text, start + paragraph_start, end + paragraph_start)
            for sentence_text, start, end in sentence_spans(paragraph_text)
        ]
        groups.extend([[sentence] for sentence in paragraph_sentences])
    return chunk_groups(work, section, text, groups, "paragraph_or_sentence")


def chunk_groups(
    work: Work,
    section: str,
    original_text: str,
    groups: list[list[tuple[str, int, int]]],
    strategy: str,
) -> list[Chunk]:
    total_tokens = len(rough_tokens(original_text))
    chunks = []
    for group in groups:
        if not group:
            continue
        start = group[0][1]
        end = group[-1][2]
        chunk_text_value = " ".join(item[0] for item in group)
        chunks.append(
            make_chunk(
                work,
                section,
                chunk_text_value,
                start,
                end,
                total_tokens,
                boundary_aligned=True,
                policy=variant_policy(strategy, work, parent_start=start, parent_end=end),
            )
        )
    return chunks


def variant_policy(
    strategy: str,
    work: Work,
    parent_start: int | None = None,
    parent_end: int | None = None,
) -> dict[str, Any]:
    policy = {
        "strategy": strategy,
        "rationale": VARIANT_DESCRIPTIONS.get(strategy, ""),
        "document_type": str((work.raw or {}).get("document_type") or "unknown_unstructured_text"),
        "domain": str((work.raw or {}).get("domain") or "unknown"),
        "chunk_tokens": 0,
        "overlap_tokens": 0,
    }
    if parent_start is not None and parent_end is not None:
        policy["parent_context_token_start"] = int(parent_start)
        policy["parent_context_token_end"] = int(parent_end)
    return policy


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in CONTENT_TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def instruction_like(text: str) -> bool:
    return bool(INSTRUCTION_LIKE_RE.search(text))


def summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["strategy"], []).append(row)
    summary = {}
    for variant, variant_rows in sorted(by_variant.items()):
        forbidden_rows = [row["forbidden_merge"] for row in variant_rows if row["forbidden_merge"].get("forbidden_pair_count")]
        unsafe_rows = [row["unsafe_isolation"] for row in variant_rows if row["unsafe_isolation"]]
        chunk_counts = [row["chunking_profile"]["chunk_count"] for row in variant_rows]
        span_rates = [row["span_containment_rate"] for row in variant_rows if row["must_keep_span_count"]]
        profile_rows = [row["chunking_profile"] for row in variant_rows]
        summary[variant] = {
            "variant_family": variant_family(variant),
            "case_count": len(variant_rows),
            "average_chunk_count": average(chunk_counts),
            "average_span_containment_rate": average(span_rates),
            "forbidden_merge_count": sum(row.get("forbidden_merge_count", 0) for row in forbidden_rows),
            "forbidden_merge_case_count": len(forbidden_rows),
            "unsafe_isolated_count": sum(1 for row in unsafe_rows if row.get("unsafe_isolated")),
            "unsafe_case_count": len(unsafe_rows),
            "average_boundary_aligned_share": average(
                profile.get("boundary_aligned_share", 0.0) for profile in profile_rows
            ),
            "average_safety_contamination_risk": average(
                profile.get("average_safety_contamination_risk", 0.0) for profile in profile_rows
            ),
            "average_evidence_containment_score": average(
                profile.get("average_evidence_containment_score", 0.0) for profile in profile_rows
            ),
            "document_type_counts": dict(sorted(Counter(row.get("document_type") or "unknown" for row in variant_rows).items())),
            "objective_scores": objective_scores(variant_rows),
        }
    return summary


def objective_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    span_score = average(row["span_containment_rate"] for row in rows if row["must_keep_span_count"])
    forbidden_pairs = sum((row.get("forbidden_merge") or {}).get("forbidden_pair_count", 0) for row in rows)
    forbidden_merges = sum((row.get("forbidden_merge") or {}).get("forbidden_merge_count", 0) for row in rows)
    unsafe_rows = [row["unsafe_isolation"] for row in rows if row["unsafe_isolation"]]
    unsafe_isolated = sum(1 for row in unsafe_rows if row.get("unsafe_isolated"))
    avg_chunks = average(row["chunking_profile"]["chunk_count"] for row in rows)
    forbidden_score = 1.0 - ratio(forbidden_merges, forbidden_pairs) if forbidden_pairs else 1.0
    unsafe_score = ratio(unsafe_isolated, len(unsafe_rows)) if unsafe_rows else 1.0
    fragmentation_penalty = min(0.25, max(0.0, (avg_chunks - 2.5) / 20.0))
    utility_score = max(0.0, min(1.0, (0.5 * span_score) + (0.3 * forbidden_score) + (0.2 * unsafe_score) - fragmentation_penalty))
    safety_score = max(0.0, min(1.0, (0.55 * forbidden_score) + (0.45 * unsafe_score)))
    return {
        "span_preservation": round(span_score, 6),
        "forbidden_merge_avoidance": round(forbidden_score, 6),
        "unsafe_isolation": round(unsafe_score, 6),
        "fragmentation_penalty": round(fragmentation_penalty, 6),
        "balanced_utility": round(utility_score, 6),
        "safety_first": round(safety_score, 6),
    }


def best_variants(summary: dict[str, Any]) -> dict[str, str]:
    objectives = ["span_preservation", "forbidden_merge_avoidance", "unsafe_isolation", "balanced_utility", "safety_first"]
    winners = {}
    for objective in objectives:
        ranked = sorted(
            summary.items(),
            key=lambda item: (
                float((item[1].get("objective_scores") or {}).get(objective, 0.0)),
                -float(item[1].get("average_chunk_count", 0.0) or 0.0),
            ),
            reverse=True,
        )
        winners[objective] = ranked[0][0] if ranked else ""
    return winners


def variant_family(variant: str) -> str:
    if variant == "fixed_window":
        return "token_window"
    if variant == "boundary_aware":
        return "production_policy"
    if variant == "atomic_sentence":
        return "atomic_boundary"
    if variant == "semantic_cohesion":
        return "semantic_surrogate"
    if variant == "embedding_semantic":
        return "local_embedding_semantic"
    if variant == "paragraph_or_sentence":
        return "layout_surrogate"
    return "unknown"


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def claim_boundaries() -> list[str]:
    return [
        "Allowed: compare deterministic technical chunking variants on fixture-level evidence containment, harmful merge avoidance, unsafe-span isolation, and fragmentation.",
        "Allowed: use this report to identify candidate strategies for deeper human and external validation.",
        "Not allowed: claim that local hashed embedding chunking is equivalent to production neural semantic chunking or late chunking.",
        "Not allowed: claim production answer quality or safety robustness from this fixture-only benchmark.",
    ]


def limitations() -> list[str]:
    return [
        "The semantic-cohesion variant is lexical and deterministic; it is a proxy baseline, not a learned semantic chunker.",
        "The embedding-semantic variant uses local hashed embeddings, so it exercises semantic-vector plumbing without claiming neural embedding quality.",
        "The report evaluates known fixture spans and forbidden pairs, not all possible good or bad boundaries.",
        "Fragmentation is measured only by chunk count; human usefulness labels are still needed.",
        "Late chunking and hierarchical retrieval are represented as future comparison targets, not implemented neural methods.",
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Chunking Variant Evaluation",
        "",
        f"- Cases: {report['case_count']}",
        f"- Chunk tokens: {report['chunk_tokens']}",
        f"- Overlap tokens: {report['overlap_tokens']}",
        "",
        "## Best Variants By Objective",
        "",
    ]
    for objective, variant in report.get("best_variant_by_objective", {}).items():
        lines.append(f"- {objective}: {variant}")
    lines.extend(["", "## Summary", ""])
    for variant, values in report["summary"].items():
        lines.append(f"### {variant}")
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Claim Boundaries", ""])
    for item in report["claim_boundaries"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic technical chunking variants.")
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = evaluate_chunking_variants(chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
    if args.write_report:
        settings = load_settings()
        suffix = f"ct{args.chunk_tokens}_ov{args.overlap_tokens}"
        write_json(settings.reports_dir / f"chunking_variant_eval_{suffix}.json", report)
        write_markdown(settings.reports_dir / f"chunking_variant_eval_{suffix}.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

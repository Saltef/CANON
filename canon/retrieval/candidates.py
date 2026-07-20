from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from canon.config import load_settings
from canon.embeddings.providers import EmbeddingResult, embed_queries, get_embedding_provider
from canon.embeddings.store import build_embedding_store, embedding_store_path, load_embedding_records
from canon.product import report_io
from canon.retrieval.bm25 import BM25Index
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.topic_profile import ensure_topic_profile, topic_query_variants


@dataclass(frozen=True)
class CandidateHit:
    document: RetrievalDocument
    retrieval_sources: tuple[str, ...]
    best_rank: int
    best_score: float
    scores: dict[str, float]
    ranks: dict[str, int]

    def to_dict(self, preview_chars: int = 240) -> dict[str, Any]:
        return {
            "chunk_id": self.document.chunk_id,
            "work_id": self.document.work_id,
            "title": self.document.title,
            "source_name": self.document.source_name,
            "year": self.document.year,
            "section": self.document.section,
            "retrieval_sources": list(self.retrieval_sources),
            "best_rank": self.best_rank,
            "best_score": round(self.best_score, 6),
            "scores": {key: round(value, 6) for key, value in sorted(self.scores.items())},
            "ranks": dict(sorted(self.ranks.items())),
            "preview": self.document.text[:preview_chars],
        }


def build_candidate_pool(
    query: str,
    mode: str,
    lexical_k: int = 50,
    vector_k: int = 50,
    query_variants: list[str] | None = None,
    auto_query_expansion: bool = False,
    benchmark_oracle_expansion: bool = False,
    topic_profile: dict[str, Any] | None = None,
    parent_expansion_limit: int = 0,
    provider: str = "local",
    model: str | None = None,
    fusion: str = "union",
    ensure_embeddings: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    resolved_topic_profile = topic_profile
    if resolved_topic_profile is None and auto_query_expansion:
        resolved_topic_profile = ensure_topic_profile(settings.data_dir, mode, documents)
    hits = candidate_pool_from_documents(
        query=query,
        documents=documents,
        embeddings_path=embedding_store_path(settings.data_dir, mode, provider, model),
        lexical_k=lexical_k,
        vector_k=vector_k,
        query_variants=query_variants,
        auto_query_expansion=auto_query_expansion,
        benchmark_oracle_expansion=benchmark_oracle_expansion,
        topic_profile=resolved_topic_profile,
        parent_expansion_limit=parent_expansion_limit,
        provider=provider,
        model=model,
        fusion=fusion,
        ensure_embeddings=lambda: build_embedding_store(mode, provider, model),
    ) if ensure_embeddings else candidate_pool_from_documents(
        query=query,
        documents=documents,
        embeddings_path=embedding_store_path(settings.data_dir, mode, provider, model),
        lexical_k=lexical_k,
        vector_k=vector_k,
        query_variants=query_variants,
        auto_query_expansion=auto_query_expansion,
        benchmark_oracle_expansion=benchmark_oracle_expansion,
        topic_profile=resolved_topic_profile,
        parent_expansion_limit=parent_expansion_limit,
        provider=provider,
        model=model,
        fusion=fusion,
    )
    report = {
        "report_id": "candidate_pool_v1",
        "query": query,
        "mode": mode,
        "provider": provider,
        "model": model,
        "document_count": len(documents),
        "candidate_count": len(hits),
        "lexical_k": lexical_k,
        "vector_k": vector_k,
        "query_variants": query_variants or [],
        "auto_query_expansion": auto_query_expansion,
        "benchmark_oracle_expansion": benchmark_oracle_expansion,
        "parent_expansion_limit": parent_expansion_limit,
        "fusion": fusion,
        "source_counts": source_counts(hits),
        "candidates": [hit.to_dict() for hit in hits],
    }
    report_io.write_json(
        settings.reports_dir / f"candidate_pool_{mode}_{safe_slug(query)}.json",
        report,
    )
    return report


def candidate_pool_from_documents(
    query: str,
    documents: Sequence[RetrievalDocument],
    embeddings_path: Path,
    lexical_k: int = 50,
    vector_k: int = 50,
    query_variants: list[str] | None = None,
    auto_query_expansion: bool = False,
    benchmark_oracle_expansion: bool = False,
    topic_profile: dict[str, Any] | None = None,
    parent_expansion_limit: int = 0,
    provider: str = "local",
    model: str | None = None,
    fusion: str = "union",
    ensure_embeddings: Any | None = None,
    lexical_index: BM25Index | None = None,
    embedding_records: list[dict[str, Any]] | None = None,
) -> list[CandidateHit]:
    rows: dict[str, dict[str, Any]] = {}
    doc_by_id = {document.chunk_id: document for document in documents}
    variants = list(query_variants or [])
    if auto_query_expansion:
        if benchmark_oracle_expansion:
            variants.extend(expand_query_variants(query))
        variants.extend(topic_query_variants(query, topic_profile or {}))
    queries = dedupe_queries([query, *variants])
    for query_index, query_text in enumerate(queries):
        source_suffix = "" if query_index == 0 else f":variant:{query_index}"
        add_ranked_rows(
            rows,
            ranked_lexical(query_text, documents, lexical_k, index=lexical_index),
            source=f"lexical{source_suffix}",
        )
    embedding_records = embedding_records if embedding_records is not None else ensure_embedding_records(embeddings_path, ensure_embeddings)
    if embedding_records:
        for query_index, query_text in enumerate(queries):
            source_suffix = "" if query_index == 0 else f":variant:{query_index}"
            add_ranked_rows(
                rows,
                ranked_vectors(query_text, embedding_records, provider, model, vector_k),
                source=f"vector:{provider}{source_suffix}",
            )
    if parent_expansion_limit > 0:
        add_parent_neighborhood_rows(rows, documents, parent_expansion_limit)
    hits = [
        CandidateHit(
            document=doc_by_id[chunk_id],
            retrieval_sources=tuple(sorted(row["sources"])),
            best_rank=min(row["ranks"].values()),
            best_score=max(row["scores"].values()),
            scores=row["scores"],
            ranks=row["ranks"],
        )
        for chunk_id, row in sorted(
            rows.items(),
            key=lambda item: (min(item[1]["ranks"].values()), -max(item[1]["scores"].values()), item[0]),
        )
        if chunk_id in doc_by_id
    ]
    return sort_hits(hits, fusion)


def add_parent_neighborhood_rows(
    rows: dict[str, dict[str, Any]],
    documents: Sequence[RetrievalDocument],
    per_parent_limit: int,
) -> None:
    if not rows or per_parent_limit <= 0:
        return
    documents_by_parent: dict[str, list[RetrievalDocument]] = {}
    parent_best: dict[str, dict[str, float]] = {}
    document_by_id = {document.chunk_id: document for document in documents}
    for document in documents:
        documents_by_parent.setdefault(document.work_id, []).append(document)
    for chunk_id, row in rows.items():
        document = document_by_id.get(chunk_id)
        if document is None:
            continue
        best_rank = min(int(rank) for rank in row["ranks"].values())
        best_score = max(float(score) for score in row["scores"].values())
        current = parent_best.get(document.work_id)
        if current is None or best_rank < current["rank"]:
            parent_best[document.work_id] = {"rank": float(best_rank), "score": best_score}
    for work_id, parent_row in parent_best.items():
        siblings = sorted(
            documents_by_parent.get(work_id, []),
            key=lambda document: (section_priority(document.section), -float_signal(document.chunk_importance), document.chunk_id),
        )
        added = 0
        for sibling in siblings:
            if sibling.chunk_id in rows:
                continue
            if added >= per_parent_limit:
                break
            rank = int(parent_row["rank"]) + added + 1
            score = float(parent_row["score"]) * 0.97
            row = rows.setdefault(sibling.chunk_id, {"sources": set(), "scores": {}, "ranks": {}})
            row["sources"].add("parent_neighborhood")
            row["scores"]["parent_neighborhood"] = score
            row["ranks"]["parent_neighborhood"] = rank
            added += 1


def section_priority(section: str | None) -> int:
    normalized = str(section or "").lower()
    order = {
        "title": 0,
        "abstract": 1,
        "summary": 2,
        "introduction": 3,
        "methods": 4,
        "results": 5,
        "discussion": 6,
        "conclusion": 7,
    }
    return order.get(normalized, 20)


def float_signal(value: Any) -> float:
    if isinstance(value, dict):
        return max((float(item) for item in value.values() if isinstance(item, int | float)), default=0.0)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def sort_hits(hits: list[CandidateHit], fusion: str) -> list[CandidateHit]:
    normalized = fusion.lower()
    if normalized == "rrf":
        return sorted(hits, key=lambda hit: (rrf_score(hit), -hit.best_rank, hit.document.chunk_id), reverse=True)
    if normalized == "weighted_bm25_dense":
        scores = normalized_weighted_scores(hits)
        return sorted(
            hits,
            key=lambda hit: (scores.get(hit.document.chunk_id, 0.0), -hit.best_rank, hit.document.chunk_id),
            reverse=True,
        )
    if normalized == "dense_only":
        return sorted(hits, key=lambda hit: (best_source_score(hit, "vector:"), -hit.best_rank, hit.document.chunk_id), reverse=True)
    if normalized == "bm25_only":
        return sorted(hits, key=lambda hit: (best_source_score(hit, "lexical"), -hit.best_rank, hit.document.chunk_id), reverse=True)
    return hits


def rrf_score(hit: CandidateHit, k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in hit.ranks.values())


def weighted_score(hit: CandidateHit) -> float:
    return 0.45 * best_source_score(hit, "lexical") + 0.55 * best_source_score(hit, "vector:")


def normalized_weighted_scores(hits: list[CandidateHit]) -> dict[str, float]:
    lexical = best_source_scores(hits, "lexical")
    dense = best_source_scores(hits, "vector:")
    normalized_lexical = minmax_scores(lexical)
    normalized_dense = minmax_scores(dense)
    return {
        hit.document.chunk_id: (
            0.45 * normalized_lexical.get(hit.document.chunk_id, 0.0)
            + 0.55 * normalized_dense.get(hit.document.chunk_id, 0.0)
        )
        for hit in hits
    }


def best_source_scores(hits: list[CandidateHit], prefix: str) -> dict[str, float]:
    return {
        hit.document.chunk_id: best_source_score(hit, prefix)
        for hit in hits
        if best_source_score(hit, prefix) > 0.0
    }


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def best_source_score(hit: CandidateHit, prefix: str) -> float:
    scores = [score for source, score in hit.scores.items() if source.startswith(prefix)]
    return max(scores) if scores else 0.0


def ranked_lexical(
    query: str,
    documents: Sequence[RetrievalDocument],
    k: int,
    index: BM25Index | None = None,
) -> list[tuple[str, float]]:
    index = index or BM25Index([document.text for document in documents])
    scored = [
        (document.chunk_id, score)
        for document, score in zip(documents, index.scores(query), strict=True)
        if score > 0
    ]
    return sorted(scored, key=lambda row: (row[1], row[0]), reverse=True)[:k]


def ranked_vectors(
    query: str,
    records: list[dict[str, Any]],
    provider_name: str,
    model: str | None,
    k: int,
) -> list[tuple[str, float]]:
    provider = get_embedding_provider(provider_name, model)
    query_embedding = embed_queries(provider, [query])[0]
    scored = [
        (record["chunk_id"], dense_cosine(query_embedding, record["vector"]))
        for record in records
    ]
    return sorted(scored, key=lambda row: (row[1], row[0]), reverse=True)[:k]


def ensure_embedding_records(path: Path, ensure_embeddings: Any | None = None) -> list[dict[str, Any]]:
    records = load_embedding_records(path)
    if records or ensure_embeddings is None:
        return records
    ensure_embeddings()
    return load_embedding_records(path)


def add_ranked_rows(
    rows: dict[str, dict[str, Any]],
    ranked: list[tuple[str, float]],
    source: str,
) -> None:
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        row = rows.setdefault(chunk_id, {"sources": set(), "scores": {}, "ranks": {}})
        row["sources"].add(source)
        row["scores"][source] = float(score)
        row["ranks"][source] = rank


def dense_cosine(query_embedding: EmbeddingResult, vector: list[float]) -> float:
    left = query_embedding.vector
    if not left or not vector or len(left) != len(vector):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, vector, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in vector))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def source_counts(hits: list[CandidateHit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        for source in hit.retrieval_sources:
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def parse_variants(value: str | None) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in value.split("||") if piece.strip()]


def expand_query_variants(query: str) -> list[str]:
    normalized = query.lower()
    variants = []
    alias_groups = {
        "deafness": ["hearing loss auditory impairment"],
        "energy drinks": ["caffeine taurine stimulant beverages"],
        "fava beans": ["vicia faba favism vicine convicine"],
        "genetic manipulation": ["genetic engineering genetically modified gmo"],
        "dha": ["docosahexaenoic acid omega 3"],
        "ecmo": ["extracorporeal membrane oxygenation"],
        "adhd": ["attention deficit hyperactivity disorder"],
    }
    for trigger, aliases in alias_groups.items():
        if trigger in normalized:
            variants.extend(aliases)
    tokens = [token for token in normalized.replace("-", " ").split() if len(token) >= 4]
    if len(tokens) == 1:
        variants.append(tokens[0])
    return dedupe_queries(variants)


def dedupe_queries(queries: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for query in queries:
        normalized = " ".join(str(query or "").lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(query).strip())
    return deduped


def safe_slug(text: str, limit: int = 60) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in text.lower())[:limit].strip("-")
    return slug or "query"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pooled CANON first-stage candidate set.")
    parser.add_argument("query")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--lexical-k", type=int, default=50)
    parser.add_argument("--vector-k", type=int, default=50)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--fusion", default="union")
    parser.add_argument("--query-variants", default=None, help="Separate variants with ||")
    parser.add_argument("--auto-query-expansion", action="store_true")
    parser.add_argument("--benchmark-oracle-expansion", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_candidate_pool(
                query=args.query,
                mode=args.mode,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                query_variants=parse_variants(args.query_variants),
                auto_query_expansion=args.auto_query_expansion,
                benchmark_oracle_expansion=args.benchmark_oracle_expansion,
                provider=args.provider,
                model=args.model,
                fusion=args.fusion,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from canon.secrets import load_local_env
from canon.config import load_settings
from canon.embeddings.store import build_embedding_store, embedding_store_path
from canon.eval.external_ir import load_qrels
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.retrieval.bm25 import BM25Index
from canon.retrieval.candidates import build_candidate_pool, candidate_pool_from_documents, ensure_embedding_records
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.experiment import run as run_retrieval
from canon.retrieval.topic_profile import ensure_topic_profile
from canon.retrieval.tokenize import tokenize


DEFAULT_RERANKERS = ["heuristic", "cohere"]


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


@dataclass(frozen=True)
class PooledCandidateContext:
    documents: list[RetrievalDocument]
    lexical_index: BM25Index
    embeddings_path: Path
    embedding_records: list[dict]
    chunk_to_work: dict[str, str]
    topic_profile: dict


class RerankProvider(Protocol):
    provider: str
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        ...


class HeuristicRerankProvider:
    provider = "heuristic"
    model = "lexical-overlap-rerank-v1"

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        query_terms = content_terms(query)
        rows = []
        for index, document in enumerate(documents):
            terms = content_terms(document)
            score = safe_ratio(len(query_terms & terms), len(query_terms))
            rows.append(RerankResult(index=index, score=score))
        return sorted(rows, key=lambda row: (row.score, -row.index), reverse=True)[:top_n]


class CohereRerankProvider:
    provider = "cohere"

    def __init__(self, model: str = "rerank-v4.0-pro", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere rerank.")

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        request = urllib.request.Request(
            "https://api.cohere.com/v2/rerank",
            data=json.dumps(
                {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [
            RerankResult(index=int(row["index"]), score=float(row["relevance_score"]))
            for row in payload.get("results", [])
        ]


class OpenRouterRerankProvider:
    provider = "openrouter"

    def __init__(self, model: str = "cohere/rerank-v3.5", api_key: str | None = None) -> None:
        load_local_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter rerank.")

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/rerank",
            data=json.dumps(
                {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [
            RerankResult(index=int(row["index"]), score=float(row["relevance_score"]))
            for row in payload.get("results", [])
        ]


def get_rerank_provider(name: str, model: str | None = None) -> RerankProvider:
    normalized = name.lower()
    if normalized in {"heuristic", "local"}:
        return HeuristicRerankProvider()
    if normalized == "cohere":
        return CohereRerankProvider(model=model or "rerank-v4.0-pro")
    if normalized == "openrouter":
        return OpenRouterRerankProvider(model=model or "cohere/rerank-v3.5")
    raise ValueError(f"Unknown rerank provider: {name}")


def build_pooled_candidate_context(
    mode: str,
    vector_provider: str,
    vector_model: str | None,
) -> PooledCandidateContext:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    embeddings_path = embedding_store_path(settings.data_dir, mode, vector_provider, vector_model)
    records = ensure_embedding_records(
        embeddings_path,
        ensure_embeddings=lambda: build_embedding_store(mode, vector_provider, vector_model),
    )
    return PooledCandidateContext(
        documents=documents,
        lexical_index=BM25Index([document.text for document in documents]),
        embeddings_path=embeddings_path,
        embedding_records=records,
        chunk_to_work={document.chunk_id: document.work_id for document in documents},
        topic_profile=ensure_topic_profile(settings.data_dir, mode, documents),
    )


def evaluate_rerankers(
    mode: str,
    qrels_path: Path,
    rerankers: list[str] | None = None,
    base_policy: str = "rag",
    candidate_k: int = 25,
    k: int = 10,
    model: str | None = None,
    pooled_candidates: bool = False,
    lexical_k: int | None = None,
    vector_k: int | None = None,
    vector_provider: str = "local",
    vector_model: str | None = None,
    query_variants: list[str] | None = None,
    fusion: str = "union",
    document_format: str = "plain",
    max_chunks_per_parent: int = 0,
    auto_query_expansion: bool = False,
    parent_qrels: bool = False,
    parent_expansion_limit: int = 0,
) -> dict:
    settings = load_settings()
    qrels = load_qrels(qrels_path)
    provider_reports = [
        evaluate_reranker(
            provider_name=provider_name,
            mode=mode,
            queries=qrels["queries"],
            base_policy=base_policy,
            candidate_k=candidate_k,
            k=k,
            model=model if provider_name == "cohere" else None,
            pooled_candidates=pooled_candidates,
            lexical_k=lexical_k,
            vector_k=vector_k,
            vector_provider=vector_provider,
            vector_model=vector_model,
            query_variants=query_variants,
            fusion=fusion,
            document_format=document_format,
            max_chunks_per_parent=max_chunks_per_parent,
            auto_query_expansion=auto_query_expansion,
            parent_qrels=parent_qrels,
            parent_expansion_limit=parent_expansion_limit,
        )
        for provider_name in (rerankers or DEFAULT_RERANKERS)
    ]
    report = {
        "report_id": "rerank_evaluation_v1",
        "mode": mode,
        "benchmark_id": qrels.get("benchmark_id"),
        "base_policy": base_policy,
        "candidate_k": candidate_k,
        "candidate_generation": {
            "mode": "pooled_lexical_vector" if pooled_candidates else "single_policy",
            "lexical_k": lexical_k or candidate_k,
            "vector_k": vector_k or candidate_k,
            "vector_provider": vector_provider,
            "vector_model": vector_model,
            "query_variant_count": len(query_variants or []),
            "fusion": fusion,
            "document_format": document_format,
            "max_chunks_per_parent": max_chunks_per_parent,
            "auto_query_expansion": auto_query_expansion,
            "parent_qrels": parent_qrels,
            "parent_expansion_limit": parent_expansion_limit,
        },
        "metric_k": k,
        "rerankers": provider_reports,
        "leaderboard": leaderboard(provider_reports, k),
        "limitations": [
            "Reranking quality depends on the first-stage candidate pool.",
            "A reranker can improve ordering but cannot recover evidence absent from the candidate pool.",
            "Cohere rerank requires COHERE_API_KEY and should be compared on human-reviewed qrels before adoption.",
        ],
    }
    write_json(settings.reports_dir / f"rerank_evaluation_{mode}_{qrels.get('benchmark_id')}.json", report)
    return report


def evaluate_reranker(
    provider_name: str,
    mode: str,
    queries: list[dict],
    base_policy: str,
    candidate_k: int,
    k: int,
    model: str | None = None,
    pooled_candidates: bool = False,
    lexical_k: int | None = None,
    vector_k: int | None = None,
    vector_provider: str = "local",
    vector_model: str | None = None,
    query_variants: list[str] | None = None,
    fusion: str = "union",
    document_format: str = "plain",
    max_chunks_per_parent: int = 0,
    auto_query_expansion: bool = False,
    parent_qrels: bool = False,
    parent_expansion_limit: int = 0,
    pooled_context: PooledCandidateContext | None = None,
) -> dict:
    started = time.perf_counter()
    provider_key, provider_model = parse_provider_spec(provider_name)
    try:
        provider = get_rerank_provider(provider_key, model or provider_model)
        pooled_context = (
            build_pooled_candidate_context(mode, vector_provider, vector_model)
            if pooled_candidates
            else None
        )
        query_reports = [
            evaluate_query(
                provider,
                mode,
                query,
                base_policy,
                candidate_k,
                k,
                pooled_candidates=pooled_candidates,
                lexical_k=lexical_k,
                vector_k=vector_k,
                vector_provider=vector_provider,
                vector_model=vector_model,
                query_variants=query_variants,
                fusion=fusion,
                document_format=document_format,
                max_chunks_per_parent=max_chunks_per_parent,
                auto_query_expansion=auto_query_expansion,
                parent_qrels=parent_qrels,
                parent_expansion_limit=parent_expansion_limit,
                pooled_context=pooled_context,
            )
            for query in queries
        ]
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": provider_key,
            "model": provider_model,
            "provider_id": provider_identifier(provider_key, provider_model),
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "provider": provider.provider,
        "model": provider.model,
        "provider_id": provider_identifier(provider.provider, provider.model),
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "query_count": len(query_reports),
        "summary": provider_summary(query_reports, k),
        "queries": query_reports,
    }


def evaluate_query(
    provider: RerankProvider,
    mode: str,
    query: dict,
    base_policy: str,
    candidate_k: int,
    k: int,
    pooled_candidates: bool = False,
    lexical_k: int | None = None,
    vector_k: int | None = None,
    vector_provider: str = "local",
    vector_model: str | None = None,
    query_variants: list[str] | None = None,
    fusion: str = "union",
    document_format: str = "plain",
    max_chunks_per_parent: int = 0,
    auto_query_expansion: bool = False,
    parent_qrels: bool = False,
    parent_expansion_limit: int = 0,
    pooled_context: PooledCandidateContext | None = None,
) -> dict:
    candidate_report = candidate_report_for_query(
        query=query["query"],
        mode=mode,
        base_policy=base_policy,
        candidate_k=candidate_k,
        pooled_candidates=pooled_candidates,
        lexical_k=lexical_k,
        vector_k=vector_k,
        query_variants=query_variants,
        vector_provider=vector_provider,
        vector_model=vector_model,
        fusion=fusion,
        auto_query_expansion=auto_query_expansion,
        parent_expansion_limit=parent_expansion_limit,
        pooled_context=pooled_context,
    )
    candidates = candidate_report.get("results") or []
    documents = [document_text(result, document_format) for result in candidates]
    reranked = provider.rerank(query["query"], documents, top_n=len(documents))
    ranked_pairs = [
        (candidates[item.index], item)
        for item in reranked
        if 0 <= item.index < len(candidates)
    ]
    ranked_pairs = diversify_by_parent(ranked_pairs, max_chunks_per_parent)
    ranked_results = [result for result, _score in ranked_pairs]
    ranked_ids = [result["chunk_id"] for result in ranked_results]
    qrels = effective_qrels(
        candidates=candidates,
        qrels=query["relevant"],
        chunk_to_work=pooled_context.chunk_to_work if pooled_context else {},
        parent_qrels=parent_qrels,
    )
    metrics = evaluate_ranking(ranked_ids, qrels, k)
    candidate_scores = scored_candidates(candidates, reranked, qrels)
    return {
        "id": query["id"],
        "query": query["query"],
        "candidate_count": len(candidates),
        "candidate_generation": candidate_report.get("candidate_generation", {}),
        "candidate_recall": candidate_recall(candidates, qrels),
        "qrels_semantics": {
            "parent_qrels_enabled": parent_qrels,
            "effective_relevant_count": len(qrels),
            "original_relevant_count": len(query["relevant"]),
        },
        "metrics": metrics,
        "score_observability": score_observability(candidate_scores, k),
        "diversification": diversification_summary(ranked_results[:k], max_chunks_per_parent),
        "scored_candidates": candidate_scores,
        "ranked_chunk_ids": ranked_ids[:k],
        "relevant_hit_ids": [chunk_id for chunk_id in ranked_ids[:k] if chunk_id in query["relevant"]],
        "top_results": [
            {
                "rank": index,
                "chunk_id": result["chunk_id"],
                "title": result["title"],
                "source_name": result["source_name"],
                "work_id": result.get("work_id"),
                "rerank_score": round(score.score, 6),
            }
            for index, (result, score) in enumerate(ranked_pairs[:k], start=1)
        ],
    }


def diversify_by_parent(
    ranked_pairs: list[tuple[dict, RerankResult]],
    max_chunks_per_parent: int,
) -> list[tuple[dict, RerankResult]]:
    if max_chunks_per_parent <= 0:
        return ranked_pairs
    kept: list[tuple[dict, RerankResult]] = []
    overflow: list[tuple[dict, RerankResult]] = []
    counts: dict[str, int] = {}
    for result, score in ranked_pairs:
        parent_id = str(result.get("work_id") or result.get("chunk_id"))
        current = counts.get(parent_id, 0)
        if current < max_chunks_per_parent:
            kept.append((result, score))
            counts[parent_id] = current + 1
        else:
            overflow.append((result, score))
    return kept + overflow


def diversification_summary(results: list[dict], max_chunks_per_parent: int) -> dict:
    parent_ids = [str(result.get("work_id") or result.get("chunk_id")) for result in results]
    unique_count = len(set(parent_ids))
    return {
        "max_chunks_per_parent": max_chunks_per_parent,
        "enabled": max_chunks_per_parent > 0,
        "unique_parent_count": unique_count,
        "duplicate_chunk_count": max(0, len(parent_ids) - unique_count),
        "max_chunks_from_one_parent": max((parent_ids.count(parent_id) for parent_id in set(parent_ids)), default=0),
    }


def candidate_report_for_query(
    query: str,
    mode: str,
    base_policy: str,
    candidate_k: int,
    pooled_candidates: bool,
    lexical_k: int | None,
    vector_k: int | None,
    vector_provider: str,
    vector_model: str | None,
    fusion: str = "union",
    query_variants: list[str] | None = None,
    auto_query_expansion: bool = False,
    parent_expansion_limit: int = 0,
    pooled_context: PooledCandidateContext | None = None,
) -> dict:
    if not pooled_candidates:
        retrieval = run_retrieval(query, base_policy, mode, candidate_k)
        return {
            "candidate_generation": {"mode": "single_policy", "base_policy": base_policy},
            "results": retrieval.get("results") or [],
        }
    if pooled_context is None:
        pool = build_candidate_pool(
            query=query,
            mode=mode,
            lexical_k=lexical_k or candidate_k,
            vector_k=vector_k or candidate_k,
            query_variants=query_variants,
            auto_query_expansion=auto_query_expansion,
            topic_profile=pooled_context.topic_profile if pooled_context else None,
            parent_expansion_limit=parent_expansion_limit,
            provider=vector_provider,
            model=vector_model,
            fusion=fusion,
        )
        candidate_rows = pool.get("candidates", [])
        source_counts = pool.get("source_counts", {})
    else:
        hits = candidate_pool_from_documents(
            query=query,
            documents=pooled_context.documents,
            embeddings_path=pooled_context.embeddings_path,
            lexical_k=lexical_k or candidate_k,
            vector_k=vector_k or candidate_k,
            query_variants=query_variants,
            auto_query_expansion=auto_query_expansion,
            topic_profile=pooled_context.topic_profile,
            parent_expansion_limit=parent_expansion_limit,
            provider=vector_provider,
            model=vector_model,
            fusion=fusion,
            lexical_index=pooled_context.lexical_index,
            embedding_records=pooled_context.embedding_records,
        )
        candidate_rows = [hit.to_dict() for hit in hits]
        source_counts = source_counts_for_rows(candidate_rows)
    return {
        "candidate_generation": {
            "mode": "pooled_lexical_vector",
            "source_counts": source_counts,
            "vector_provider": vector_provider,
            "vector_model": vector_model,
            "query_variant_count": len(query_variants or []),
            "auto_query_expansion": auto_query_expansion,
            "parent_expansion_limit": parent_expansion_limit,
            "fusion": fusion,
        },
        "results": [
            {
                "chunk_id": row["chunk_id"],
                "work_id": row["work_id"],
                "title": row["title"],
                "source_name": row["source_name"],
                "year": row["year"],
                "preview": row["preview"],
                "candidate_scores": row["scores"],
                "candidate_ranks": row["ranks"],
                "retrieval_sources": row["retrieval_sources"],
                "section": row.get("section"),
                "best_rank": row.get("best_rank"),
                "best_score": row.get("best_score"),
            }
            for row in candidate_rows
        ],
    }


def scored_candidates(
    candidates: list[dict],
    reranked: list[RerankResult],
    qrels: dict[str, float],
) -> list[dict]:
    by_index = {item.index: item.score for item in reranked}
    return [
        {
            "candidate_rank": index,
            "chunk_id": result.get("chunk_id"),
            "rerank_score": float(by_index.get(index - 1, 0.0)),
            "relevance": float(qrels.get(result.get("chunk_id"), 0.0)),
        }
        for index, result in enumerate(candidates, start=1)
    ]


def effective_qrels(
    candidates: list[dict],
    qrels: dict[str, float],
    chunk_to_work: dict[str, str],
    parent_qrels: bool,
) -> dict[str, float]:
    if not parent_qrels:
        return qrels
    relevant_works = {
        chunk_to_work.get(str(chunk_id), str(chunk_id))
        for chunk_id, relevance in qrels.items()
        if float(relevance) > 0
    }
    effective = dict(qrels)
    for candidate in candidates:
        chunk_id = str(candidate.get("chunk_id"))
        work_id = str(candidate.get("work_id") or chunk_to_work.get(chunk_id, chunk_id))
        if work_id in relevant_works and chunk_id not in effective:
            effective[chunk_id] = 1.0
    return effective


def candidate_recall(candidates: list[dict], qrels: dict[str, float]) -> dict:
    relevant_ids = {chunk_id for chunk_id, relevance in qrels.items() if float(relevance) > 0}
    candidate_ids = {str(result.get("chunk_id")) for result in candidates}
    hit_ids = sorted(relevant_ids & candidate_ids)
    return {
        "relevant_count": len(relevant_ids),
        "candidate_hit_count": len(hit_ids),
        "candidate_recall": round(safe_ratio(len(hit_ids), len(relevant_ids)), 6),
        "hit_ids": hit_ids,
    }


def score_observability(scored: list[dict], k: int) -> dict:
    scores = [row["rerank_score"] for row in scored]
    relevant_scores = [row["rerank_score"] for row in scored if row["relevance"] > 0]
    nonrelevant_scores = [row["rerank_score"] for row in scored if row["relevance"] <= 0]
    top_scores = [row["rerank_score"] for row in sorted(scored, key=lambda row: row["rerank_score"], reverse=True)[:k]]
    return {
        "score_distribution": score_distribution(scores),
        "relevant_score_mean": round(average(relevant_scores), 6),
        "nonrelevant_score_mean": round(average(nonrelevant_scores), 6),
        "relevant_nonrelevant_score_gap": round(average(relevant_scores) - average(nonrelevant_scores), 6),
        "top_k_min_score": round(min(top_scores), 6) if top_scores else 0.0,
        "low_confidence_top_k_count": sum(1 for score in top_scores if score < 0.25),
        "scored_candidate_count": len(scored),
    }


def score_distribution(scores: list[float]) -> dict:
    if not scores:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(scores)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "min": round(ordered[0], 6),
        "median": round(median, 6),
        "mean": round(average(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def source_counts_for_rows(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for source in row.get("retrieval_sources") or []:
            counts[str(source)] = counts.get(str(source), 0) + 1
    return dict(sorted(counts.items()))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def document_text(result: dict, document_format: str = "plain") -> str:
    if document_format == "structured":
        return "\n".join(
            str(piece)
            for piece in [
                f"title: {result.get('title')}",
                f"source: {result.get('source_name')}",
                f"year: {result.get('year')}",
                f"section: {result.get('section')}",
                f"retrieval_sources: {', '.join(result.get('retrieval_sources') or [])}",
                f"best_rank: {result.get('best_rank')}",
                f"bm25_rank: {best_rank_for_prefix(result, 'lexical')}",
                f"dense_rank: {best_rank_for_prefix(result, 'vector:')}",
                f"bm25_score: {best_score_for_prefix(result, 'lexical')}",
                f"dense_score: {best_score_for_prefix(result, 'vector:')}",
                f"text: {result.get('preview')}",
            ]
            if piece is not None
        )
    return "\n".join(
        str(piece)
        for piece in [
            f"title: {result.get('title')}",
            f"source: {result.get('source_name')}",
            f"year: {result.get('year')}",
            f"text: {result.get('preview')}",
        ]
        if piece is not None
    )


def best_rank_for_prefix(result: dict, prefix: str) -> int | None:
    ranks = [
        int(rank)
        for source, rank in (result.get("candidate_ranks") or {}).items()
        if str(source).startswith(prefix)
    ]
    return min(ranks) if ranks else None


def best_score_for_prefix(result: dict, prefix: str) -> float | None:
    scores = [
        float(score)
        for source, score in (result.get("candidate_scores") or {}).items()
        if str(source).startswith(prefix)
    ]
    return round(max(scores), 6) if scores else None


def provider_summary(query_reports: list[dict], k: int) -> dict:
    metric_names = [f"nDCG@{k}", f"Recall@{k}", f"MAP@{k}", f"MRR@{k}"]
    rows = [query["metrics"] for query in query_reports if query.get("metrics")]
    return {metric: mean_metric(rows, metric) for metric in metric_names}


def leaderboard(provider_reports: list[dict], k: int) -> list[dict]:
    primary = f"nDCG@{k}"
    available = [report for report in provider_reports if report.get("status") == "ok"]
    return [
        {
            "rank": index,
            "provider": report["provider"],
            "model": report["model"],
            "provider_id": report.get("provider_id") or provider_identifier(report["provider"], report["model"]),
            "primary_metric": primary,
            "primary_score": report["summary"].get(primary, 0.0),
            "Recall@k": report["summary"].get(f"Recall@{k}", 0.0),
            "MRR@k": report["summary"].get(f"MRR@{k}", 0.0),
            "elapsed_ms": report["elapsed_ms"],
        }
        for index, report in enumerate(
            sorted(available, key=lambda item: item["summary"].get(primary, 0.0), reverse=True),
            start=1,
        )
    ]


def content_terms(text: str) -> set[str]:
    stop = {"about", "and", "are", "does", "for", "from", "how", "the", "this", "what", "which", "with"}
    return {token for token in tokenize(text) if len(token) >= 3 and token not in stop}


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def parse_list(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_RERANKERS
    parsed = [piece.strip() for piece in value.split(",") if piece.strip()]
    return parsed or DEFAULT_RERANKERS


def parse_provider_spec(value: str) -> tuple[str, str | None]:
    provider, separator, model = value.partition(":")
    return provider.strip(), model.strip() if separator and model.strip() else None


def provider_identifier(provider: str, model: str | None) -> str:
    return f"{provider}:{model}" if model else provider


def parse_variants(value: str | None) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in value.split("||") if piece.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rerankers on a pooled first-stage retrieval set.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--rerankers", default=None)
    parser.add_argument("--base-policy", default="rag")
    parser.add_argument("--candidate-k", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pooled-candidates", action="store_true")
    parser.add_argument("--lexical-k", type=int, default=None)
    parser.add_argument("--vector-k", type=int, default=None)
    parser.add_argument("--vector-provider", default="local")
    parser.add_argument("--vector-model", default=None)
    parser.add_argument("--fusion", default="union")
    parser.add_argument("--document-format", choices=["plain", "structured"], default="plain")
    parser.add_argument("--max-chunks-per-parent", type=int, default=0)
    parser.add_argument("--auto-query-expansion", action="store_true")
    parser.add_argument("--parent-qrels", action="store_true")
    parser.add_argument("--parent-expansion-limit", type=int, default=0)
    parser.add_argument("--query-variants", default=None, help="Global variants separated with ||")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_rerankers(
                mode=args.mode,
                qrels_path=Path(args.qrels),
                rerankers=parse_list(args.rerankers),
                base_policy=args.base_policy,
                candidate_k=args.candidate_k,
                k=args.k,
                model=args.model,
                pooled_candidates=args.pooled_candidates,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                vector_provider=args.vector_provider,
                vector_model=args.vector_model,
                fusion=args.fusion,
                document_format=args.document_format,
                max_chunks_per_parent=args.max_chunks_per_parent,
                auto_query_expansion=args.auto_query_expansion,
                parent_qrels=args.parent_qrels,
                parent_expansion_limit=args.parent_expansion_limit,
                query_variants=parse_variants(args.query_variants),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

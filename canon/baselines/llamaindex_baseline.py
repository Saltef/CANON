from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.embeddings.providers import embed_queries, get_embedding_provider
from canon.embeddings.store import build_embedding_store, embedding_store_path, load_embedding_records
from canon.eval.external_ir import load_qrels
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.eval.rerank_evaluation import (
    PARENT_QRELS_PROTOCOL,
    build_pooled_candidate_context,
    effective_qrels,
    evaluate_rerankers,
)
from canon.product.stage1_repeat_spread import (
    comparison_decision,
    paired_delta,
    query_work_level_summary,
    round_metric,
    summarize_runs,
)
from canon.retrieval.candidates import candidate_pool_from_documents
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus


DEFAULT_MODE = "beir_nfcorpus_stage1_title_preserve"
DEFAULT_QRELS = "gold/beir_nfcorpus_stage1_title_preserve_qrels.json"
DEFAULT_CANON_REPORT = "reports/stage1_fixed_qrels_v2_repeat_spread.json"
DEFAULT_REPORT_JSON = "reports/llamaindex_stage1_head_to_head.json"
DEFAULT_REPORT_MD = "reports/llamaindex_stage1_head_to_head.md"
DEFAULT_QUERY_CACHE_ROOT = "reports/llamaindex_factor_query_cache"
DEFAULT_QWEN_PROVIDER = "openrouter"
DEFAULT_QWEN_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_HASH_PROVIDER = "local"
DEFAULT_HASH_MODEL = "hashed-semantic-v1"
DEFAULT_RERANKER = "cohere:rerank-v4.0-fast"


@dataclass(frozen=True)
class LlamaIndexRepeatConfig:
    id: str
    label: str
    parent_qrels: bool
    qrels_scope: str
    embedding_provider: str
    embedding_model: str | None


@dataclass(frozen=True)
class CanonRepeatConfig:
    id: str
    label: str
    parent_qrels: bool
    qrels_scope: str
    embedding_provider: str
    embedding_model: str | None
    pipeline: str
    reranker: str | None = None


DEFAULT_CONFIGS = [
    LlamaIndexRepeatConfig(
        id="llamaindex_hash_original_qrels",
        label="LlamaIndex VectorStoreIndex + hashed-semantic-v1, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_HASH_PROVIDER,
        embedding_model=DEFAULT_HASH_MODEL,
    ),
    LlamaIndexRepeatConfig(
        id="llamaindex_qwen_original_qrels",
        label="LlamaIndex VectorStoreIndex + Qwen3 Embedding 8B, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_QWEN_PROVIDER,
        embedding_model=DEFAULT_QWEN_MODEL,
    ),
]


DEFAULT_CANON_CONFIGS = [
    CanonRepeatConfig(
        id="canon_hash_cohere_original_qrels",
        label="CANON BM25+hash RRF + Cohere rerank, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_HASH_PROVIDER,
        embedding_model=DEFAULT_HASH_MODEL,
        pipeline="bm25_dense_rrf_cohere_rerank",
        reranker=DEFAULT_RERANKER,
    ),
    CanonRepeatConfig(
        id="canon_qwen_cohere_original_qrels",
        label="CANON BM25+Qwen RRF + Cohere rerank, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_QWEN_PROVIDER,
        embedding_model=DEFAULT_QWEN_MODEL,
        pipeline="bm25_dense_rrf_cohere_rerank",
        reranker=DEFAULT_RERANKER,
    ),
]


DEFAULT_RRF_ONLY_CONFIGS = [
    CanonRepeatConfig(
        id="canon_hash_rrf_only_original_qrels",
        label="CANON BM25+hash RRF without rerank, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_HASH_PROVIDER,
        embedding_model=DEFAULT_HASH_MODEL,
        pipeline="bm25_dense_rrf_no_rerank",
    ),
    CanonRepeatConfig(
        id="canon_qwen_rrf_only_original_qrels",
        label="CANON BM25+Qwen RRF without rerank, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
        embedding_provider=DEFAULT_QWEN_PROVIDER,
        embedding_model=DEFAULT_QWEN_MODEL,
        pipeline="bm25_dense_rrf_no_rerank",
    ),
]


class LlamaIndexUnavailable(RuntimeError):
    pass


def require_llamaindex() -> tuple[Any, Any]:
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import TextNode
    except ImportError as exc:
        raise LlamaIndexUnavailable(
            "llama-index-core is required for this baseline. "
            'Install with: python -m pip install -e ".[baselines]"'
        ) from exc
    return VectorStoreIndex, TextNode


def require_base_embedding():
    try:
        from llama_index.core.embeddings import BaseEmbedding
        from pydantic import PrivateAttr
    except ImportError as exc:
        raise LlamaIndexUnavailable(
            "llama-index-core and pydantic are required for the LlamaIndex embedding adapter."
        ) from exc

    class CanonLlamaIndexEmbedding(BaseEmbedding):
        _provider: Any = PrivateAttr()

        def __init__(self, provider_name: str = DEFAULT_HASH_PROVIDER, model: str | None = DEFAULT_HASH_MODEL) -> None:
            provider = get_embedding_provider(provider_name, model)
            super().__init__(model_name=f"canon-{provider.provider}-{provider.model}")
            self._provider = provider

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._provider.embed([text])[0].vector

        def _get_query_embedding(self, query: str) -> list[float]:
            return embed_queries(self._provider, [query])[0].vector

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return CanonLlamaIndexEmbedding


def chunk_to_work_map(documents: list[RetrievalDocument]) -> dict[str, str]:
    return {document.chunk_id: document.work_id for document in documents}


def build_nodes(
    documents: list[RetrievalDocument],
    embedding_by_chunk_id: dict[str, list[float]] | None = None,
) -> list[Any]:
    _index, text_node_cls = require_llamaindex()
    nodes = []
    for document in documents:
        node = text_node_cls(
            id_=document.chunk_id,
            text=document_text(document),
            metadata={
                "chunk_id": document.chunk_id,
                "work_id": document.work_id,
                "title": document.title,
                "source_name": document.source_name,
                "year": document.year,
                "section": document.section,
            },
        )
        if embedding_by_chunk_id and document.chunk_id in embedding_by_chunk_id:
            node.embedding = embedding_by_chunk_id[document.chunk_id]
        nodes.append(node)
    return nodes


def document_text(document: RetrievalDocument) -> str:
    return "\n".join(
        str(piece)
        for piece in [
            f"title: {document.title}",
            f"source: {document.source_name}",
            f"year: {document.year}",
            f"section: {document.section}",
            f"text: {document.text}",
        ]
        if piece is not None
    )


def build_retriever(
    documents: list[RetrievalDocument],
    *,
    similarity_top_k: int,
    mode: str,
    embedding_provider: str,
    embedding_model: str | None,
) -> tuple[Any, dict[str, Any]]:
    index_cls, _text_node_cls = require_llamaindex()
    embedding_cls = require_base_embedding()
    embed_model = embedding_cls(embedding_provider, embedding_model)
    embedding_records, embedding_metadata = load_or_build_embedding_records(
        mode=mode,
        provider_name=embedding_provider,
        model=embedding_model,
    )
    embedding_by_chunk_id = {
        str(record["chunk_id"]): [float(value) for value in record["vector"]]
        for record in embedding_records
        if record.get("chunk_id") and record.get("vector")
    }
    index = index_cls(
        nodes=build_nodes(documents, embedding_by_chunk_id),
        embed_model=embed_model,
        show_progress=False,
    )
    return index.as_retriever(similarity_top_k=similarity_top_k), embedding_metadata


def load_or_build_embedding_records(
    *,
    mode: str,
    provider_name: str,
    model: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = load_settings()
    provider = get_embedding_provider(provider_name, model)
    path = embedding_store_path(settings.data_dir, mode, provider.provider, provider.model)
    records = load_embedding_records(path)
    built = False
    if not records:
        build_embedding_store(mode, provider.provider, provider.model)
        records = load_embedding_records(path)
        built = True
    dimensions = int(records[0].get("dimensions", 0)) if records else 0
    return records, {
        "provider": provider.provider,
        "model": provider.model,
        "dimensions": dimensions,
        "path": display_path(path),
        "precomputed_document_embedding_count": len(records),
        "built_embedding_store": built,
    }


def evaluate_llamaindex_query(
    retriever: Any,
    query: dict[str, Any],
    *,
    k: int,
    parent_qrels: bool,
    chunk_to_work: dict[str, str],
) -> dict[str, Any]:
    results = retriever.retrieve(query["query"])
    ranked = [node_result_to_row(result) for result in results]
    ranked_ids = [row["chunk_id"] for row in ranked if row.get("chunk_id")]
    qrels = effective_qrels(query.get("relevant") or {}, chunk_to_work, parent_qrels)
    metrics = evaluate_ranking(ranked_ids, qrels, k)
    return {
        "id": query["id"],
        "query": query["query"],
        "candidate_count": len(ranked),
        "candidate_generation": {"mode": "llamaindex_vectorstoreindex_default_retriever"},
        "candidate_recall": candidate_recall(ranked_ids, qrels),
        "qrels_semantics": {
            "parent_qrels_enabled": parent_qrels,
            "parent_qrels_protocol": PARENT_QRELS_PROTOCOL if parent_qrels else None,
            "qrels_expansion_scope": "full_corpus_parent_map" if parent_qrels else "original_chunk_qrels",
            "effective_relevant_count": positive_relevance_count(qrels),
            "original_relevant_count": positive_relevance_count(query.get("relevant") or {}),
        },
        "metrics": metrics,
        "scored_candidates": [
            {
                "candidate_rank": index,
                "chunk_id": row.get("chunk_id"),
                "score": row.get("score"),
                "relevance": float(qrels.get(row.get("chunk_id"), 0.0)),
            }
            for index, row in enumerate(ranked, start=1)
        ],
        "ranked_chunk_ids": ranked_ids[:k],
        "relevant_hit_ids": [chunk_id for chunk_id in ranked_ids[:k] if float(qrels.get(chunk_id, 0.0)) > 0],
        "top_results": [
            {
                "rank": index,
                "chunk_id": row.get("chunk_id"),
                "work_id": row.get("work_id"),
                "title": row.get("title"),
                "source_name": row.get("source_name"),
                "score": row.get("score"),
            }
            for index, row in enumerate(ranked[:k], start=1)
        ],
    }


def node_result_to_row(result: Any) -> dict[str, Any]:
    node = getattr(result, "node", result)
    metadata = dict(getattr(node, "metadata", {}) or {})
    chunk_id = metadata.get("chunk_id") or getattr(node, "node_id", None) or getattr(node, "id_", None)
    return {
        "chunk_id": str(chunk_id) if chunk_id is not None else "",
        "work_id": metadata.get("work_id"),
        "title": metadata.get("title"),
        "source_name": metadata.get("source_name"),
        "year": metadata.get("year"),
        "section": metadata.get("section"),
        "score": round_metric(float(getattr(result, "score", 0.0) or 0.0)),
    }


def candidate_recall(ranked_ids: list[str], qrels: dict[str, float]) -> dict[str, Any]:
    relevant_ids = {str(chunk_id) for chunk_id, relevance in qrels.items() if float(relevance) > 0}
    hit_ids = sorted(relevant_ids & set(ranked_ids))
    return {
        "relevant_count": len(relevant_ids),
        "candidate_hit_count": len(hit_ids),
        "candidate_recall": round_metric(len(hit_ids) / len(relevant_ids) if relevant_ids else 0.0),
        "hit_ids": hit_ids,
    }


def positive_relevance_count(qrels: dict[str, float]) -> int:
    return sum(1 for value in qrels.values() if float(value) > 0)


def provider_summary(query_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metric_names = [f"nDCG@{k}", f"Recall@{k}", f"MAP@{k}", f"MRR@{k}"]
    rows = [query["metrics"] for query in query_reports if query.get("metrics")]
    return {metric: mean_metric(rows, metric) for metric in metric_names}


def extract_run_metrics(
    query_reports: list[dict[str, Any]],
    qrels: dict[str, Any],
    chunk_map: dict[str, str],
    k: int,
) -> dict[str, Any]:
    qrels_by_id = {str(query["id"]): query.get("relevant", {}) for query in qrels.get("queries", [])}
    work_rows = [
        query_work_level_summary(query, qrels_by_id.get(str(query.get("id")), {}), chunk_map)
        for query in query_reports
    ]
    candidate_recall_values = [
        float((query.get("candidate_recall") or {}).get("candidate_recall", 0.0))
        for query in query_reports
    ]
    summary = provider_summary(query_reports, k)
    metrics = {
        f"nDCG@{k}": round_metric(summary.get(f"nDCG@{k}", 0.0)),
        f"Recall@{k}": round_metric(summary.get(f"Recall@{k}", 0.0)),
        f"MAP@{k}": round_metric(summary.get(f"MAP@{k}", 0.0)),
        f"MRR@{k}": round_metric(summary.get(f"MRR@{k}", 0.0)),
        "candidate_recall_mean": round_metric(average(candidate_recall_values)),
        "work_level_hit_rate_mean": round_metric(
            average([row["relevant_work_hit_rate"] for row in work_rows])
        ),
    }
    metrics["queries_with_any_relevant_work_in_pool_count"] = sum(
        1 for row in work_rows if row["any_relevant_work_in_pool"]
    )
    metrics["zero_candidate_recall_query_count"] = sum(
        1 for value in candidate_recall_values if value <= 0
    )
    metrics["queries_with_any_relevant_work_in_pool"] = (
        f"{metrics['queries_with_any_relevant_work_in_pool_count']}/{len(work_rows)}"
    )
    metrics["zero_candidate_recall_queries"] = (
        f"{metrics['zero_candidate_recall_query_count']}/{len(query_reports)}"
    )
    return metrics


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_once(
    config: LlamaIndexRepeatConfig,
    *,
    repeat: int,
    mode: str,
    qrels: dict[str, Any],
    documents: list[RetrievalDocument],
    chunk_map: dict[str, str],
    candidate_k: int,
    k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        retriever, embedding_metadata = build_retriever(
            documents,
            similarity_top_k=candidate_k,
            mode=mode,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
        )
        query_reports = [
            evaluate_llamaindex_query(
                retriever,
                query,
                k=k,
                parent_qrels=config.parent_qrels,
                chunk_to_work=chunk_map,
            )
            for query in qrels.get("queries", [])
        ]
    except Exception as exc:  # noqa: BLE001 - optional baseline availability belongs in the report.
        return {
            **asdict(config),
            "arm": "llamaindex",
            "configuration": config.id,
            "repeat": repeat,
            "status": "unavailable",
            "pipeline": "vectorstoreindex_single_dense_retriever",
            "reason": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        **asdict(config),
        "arm": "llamaindex",
        "configuration": config.id,
        "repeat": repeat,
        "status": "ok",
        "provider_id": f"llamaindex:vectorstoreindex:{embedding_metadata['provider']}:{embedding_metadata['model']}",
        "embedding": embedding_metadata,
        "pipeline": "vectorstoreindex_single_dense_retriever",
        "elapsed_ms": elapsed_ms,
        "metrics": extract_run_metrics(query_reports, qrels, chunk_map, k),
        "summary": provider_summary(query_reports, k),
        "query_count": len(query_reports),
        "queries": query_reports,
    }


def run_canon_rerank_once(
    config: CanonRepeatConfig,
    *,
    repeat: int,
    mode: str,
    qrels_path: Path,
    qrels: dict[str, Any],
    chunk_map: dict[str, str],
    candidate_k: int,
    lexical_k: int,
    vector_k: int,
    k: int,
    fusion: str,
    document_format: str,
    query_cache_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    query_cache_dir = query_cache_root / f"repeat_{repeat}" / config.id
    try:
        report = evaluate_rerankers(
            mode=mode,
            qrels_path=qrels_path,
            rerankers=[config.reranker or DEFAULT_RERANKER],
            candidate_k=candidate_k,
            k=k,
            pooled_candidates=True,
            lexical_k=lexical_k,
            vector_k=vector_k,
            vector_provider=config.embedding_provider,
            vector_model=config.embedding_model,
            fusion=fusion,
            document_format=document_format,
            parent_qrels=config.parent_qrels,
            resume=True,
            query_cache_dir=query_cache_dir,
        )
        reranker_report = (report.get("rerankers") or [{}])[0]
        if reranker_report.get("status") != "ok":
            return unavailable_canon_run(config, repeat, started, reranker_report.get("reason"))
        query_reports = reranker_report.get("queries") or []
        return {
            **asdict(config),
            "arm": "canon",
            "configuration": config.id,
            "repeat": repeat,
            "status": "ok",
            "provider_id": reranker_report.get("provider_id"),
            "elapsed_ms": float(reranker_report.get("elapsed_ms", 0.0)),
            "wall_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "candidate_k": candidate_k,
            "lexical_k": lexical_k,
            "vector_k": vector_k,
            "fusion": fusion,
            "document_format": document_format,
            "query_cache_dir": display_path(query_cache_dir),
            "cache": reranker_report.get("cache", {}),
            "metrics": extract_run_metrics(query_reports, qrels, chunk_map, k),
            "summary": provider_summary(query_reports, k),
            "query_count": len(query_reports),
            "queries": query_reports,
        }
    except Exception as exc:  # noqa: BLE001 - optional hosted dependencies belong in the report.
        return unavailable_canon_run(config, repeat, started, str(exc))


def unavailable_canon_run(
    config: CanonRepeatConfig,
    repeat: int,
    started: float,
    reason: str | None,
) -> dict[str, Any]:
    return {
        **asdict(config),
        "arm": "canon",
        "configuration": config.id,
        "repeat": repeat,
        "status": "unavailable",
        "reason": reason or "CANON comparison cell unavailable.",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def run_canon_rrf_once(
    config: CanonRepeatConfig,
    *,
    repeat: int,
    mode: str,
    qrels: dict[str, Any],
    chunk_map: dict[str, str],
    candidate_k: int,
    lexical_k: int,
    vector_k: int,
    k: int,
    fusion: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        context = build_pooled_candidate_context(mode, config.embedding_provider, config.embedding_model)
        query_reports = [
            evaluate_rrf_query(
                query=query,
                context=context,
                chunk_map=chunk_map,
                k=k,
                lexical_k=lexical_k,
                vector_k=vector_k,
                provider=config.embedding_provider,
                model=config.embedding_model,
                fusion=fusion,
                parent_qrels=config.parent_qrels,
            )
            for query in qrels.get("queries", [])
        ]
    except Exception as exc:  # noqa: BLE001
        return unavailable_canon_run(config, repeat, started, str(exc))
    return {
        **asdict(config),
        "arm": "canon",
        "configuration": config.id,
        "repeat": repeat,
        "status": "ok",
        "provider_id": f"canon:{config.embedding_provider}:{config.embedding_model}:rrf_only",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "candidate_k": candidate_k,
        "lexical_k": lexical_k,
        "vector_k": vector_k,
        "fusion": fusion,
        "metrics": extract_run_metrics(query_reports, qrels, chunk_map, k),
        "summary": provider_summary(query_reports, k),
        "query_count": len(query_reports),
        "queries": query_reports,
    }


def evaluate_rrf_query(
    *,
    query: dict[str, Any],
    context: Any,
    chunk_map: dict[str, str],
    k: int,
    lexical_k: int,
    vector_k: int,
    provider: str,
    model: str | None,
    fusion: str,
    parent_qrels: bool,
) -> dict[str, Any]:
    hits = candidate_pool_from_documents(
        query=query["query"],
        documents=context.documents,
        embeddings_path=context.embeddings_path,
        lexical_k=lexical_k,
        vector_k=vector_k,
        provider=provider,
        model=model,
        fusion=fusion,
        lexical_index=context.lexical_index,
        embedding_records=context.embedding_records,
    )
    ranked_ids = [hit.document.chunk_id for hit in hits]
    qrels = effective_qrels(query.get("relevant") or {}, chunk_map, parent_qrels)
    metrics = evaluate_ranking(ranked_ids, qrels, k)
    return {
        "id": query["id"],
        "query": query["query"],
        "candidate_count": len(hits),
        "candidate_generation": {
            "mode": "pooled_lexical_vector_rrf_no_rerank",
            "vector_provider": provider,
            "vector_model": model,
            "fusion": fusion,
        },
        "candidate_recall": candidate_recall([hit.document.chunk_id for hit in hits], qrels),
        "qrels_semantics": {
            "parent_qrels_enabled": parent_qrels,
            "parent_qrels_protocol": PARENT_QRELS_PROTOCOL if parent_qrels else None,
            "qrels_expansion_scope": "full_corpus_parent_map" if parent_qrels else "original_chunk_qrels",
            "effective_relevant_count": positive_relevance_count(qrels),
            "original_relevant_count": positive_relevance_count(query.get("relevant") or {}),
        },
        "metrics": metrics,
        "scored_candidates": [
            {
                "candidate_rank": index,
                "chunk_id": hit.document.chunk_id,
                "relevance": float(qrels.get(hit.document.chunk_id, 0.0)),
            }
            for index, hit in enumerate(hits, start=1)
        ],
        "ranked_chunk_ids": ranked_ids[:k],
        "relevant_hit_ids": [
            chunk_id
            for chunk_id in ranked_ids[:k]
            if float(qrels.get(chunk_id, 0.0)) > 0
        ],
        "top_results": [
            {
                "rank": index,
                "chunk_id": hit.document.chunk_id,
                "work_id": hit.document.work_id,
                "title": hit.document.title,
                "source_name": hit.document.source_name,
                "score": round_metric(hit.best_score),
            }
            for index, hit in enumerate(hits[:k], start=1)
        ],
    }


def completed_ok_keys(runs: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(run.get("configuration")), int(run.get("repeat", 0)))
        for run in runs
        if run.get("status") == "ok"
    }


def replace_run(runs: list[dict[str, Any]], replacement: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if not (
            run.get("configuration") == replacement.get("configuration")
            and int(run.get("repeat", 0)) == int(replacement.get("repeat", 0))
        )
    ] + [replacement]


def load_existing_runs(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("llamaindex_runs") or payload.get("runs") or [])


def load_existing_run_groups(report_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not report_path.exists():
        return [], [], []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], [], []
    return (
        list(payload.get("llamaindex_runs") or payload.get("runs") or []),
        list(payload.get("canon_runs") or []),
        list(payload.get("canon_rrf_only_runs") or []),
    )


def load_canon_runs(canon_report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not canon_report_path.exists():
        return [], {"status": "missing", "path": str(canon_report_path).replace("\\", "/")}
    payload = json.loads(canon_report_path.read_text(encoding="utf-8"))
    runs = list(payload.get("runs") or [])
    return runs, {
        "status": payload.get("status"),
        "path": str(canon_report_path).replace("\\", "/"),
        "report_id": payload.get("report_id"),
        "retrieval_stack": payload.get("retrieval_stack"),
    }


def build_report(
    *,
    llama_runs: list[dict[str, Any]],
    canon_runs: list[dict[str, Any]],
    rrf_runs: list[dict[str, Any]],
    legacy_canon_report: dict[str, Any],
    configs: list[LlamaIndexRepeatConfig],
    canon_configs: list[CanonRepeatConfig],
    rrf_configs: list[CanonRepeatConfig],
    repeats: int,
    mode: str,
    qrels_path: Path,
    qrels: dict[str, Any],
    k: int,
    candidate_k: int,
    lexical_k: int,
    vector_k: int,
    fusion: str,
    document_format: str,
    include_query_details: bool = False,
) -> dict[str, Any]:
    all_runs = list(llama_runs) + list(canon_runs) + list(rrf_runs)
    llama_summaries = summarize_runs(llama_runs, repeat_configs_for_summary(configs), k)
    canon_summaries = summarize_runs(canon_runs, repeat_configs_for_summary(canon_configs), k)
    rrf_summaries = summarize_runs(rrf_runs, repeat_configs_for_summary(rrf_configs), k)
    all_summaries = summarize_runs(all_runs, repeat_configs_for_summary([*configs, *canon_configs, *rrf_configs]), k)
    comparisons = [
        paired_delta(
            all_runs,
            treatment_config="llamaindex_qwen_original_qrels",
            baseline_config="llamaindex_hash_original_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            all_runs,
            treatment_config="canon_qwen_cohere_original_qrels",
            baseline_config="canon_hash_cohere_original_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            all_runs,
            treatment_config="canon_hash_cohere_original_qrels",
            baseline_config="llamaindex_hash_original_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            all_runs,
            treatment_config="canon_qwen_cohere_original_qrels",
            baseline_config="llamaindex_qwen_original_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            all_runs,
            treatment_config="canon_hash_cohere_original_qrels",
            baseline_config="canon_hash_rrf_only_original_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            all_runs,
            treatment_config="canon_qwen_cohere_original_qrels",
            baseline_config="canon_qwen_rrf_only_original_qrels",
            metric=f"nDCG@{k}",
        ),
    ]
    claim_checks = [comparison_decision(comparison, all_summaries) for comparison in comparisons]
    completed = sum(1 for run in all_runs if run.get("status") == "ok")
    expected = repeats * (len(configs) + len(canon_configs) + len(rrf_configs))
    verdict = standing_verdict(claim_checks)
    return {
        "report_id": "llamaindex_canon_stage1_factorial_correction_v2",
        "created_at": time.strftime("%Y-%m-%d"),
        "status": "factorial_correction_complete" if completed == expected else "factorial_correction_partial",
        "mode": mode,
        "qrels_path": display_path(qrels_path),
        "query_count": len(qrels.get("queries", [])),
        "label_count": sum(len(query.get("relevant", {})) for query in qrels.get("queries", [])),
        "metric_k": k,
        "repeat_count_requested": repeats,
        "completed_run_count": completed,
        "expected_run_count": expected,
        "protocol": {
            "benchmark": "BEIR NFCorpus 30-query public slice expanded to CANON chunks",
            "parent_qrels_protocol": PARENT_QRELS_PROTOCOL,
            "paired_repeat_protocol": "Repeat numbers are paired across all cells 1..N.",
            "primary_question": (
                "How much of the old LlamaIndex-vs-CANON gap is encoder choice, "
                "how much is retrieval pipeline, and how much is reranking?"
            ),
            "supersedes": {
                "report": "llamaindex_stage1_head_to_head_v1",
                "reason": (
                    "The old primary framing compared CANON with Qwen embeddings, BM25, RRF, "
                    "and Cohere rerank against LlamaIndex with a hashed embedding adapter and no "
                    "fusion/rerank. That was not a pipeline-only comparison."
                ),
            },
            "llamaindex_default_verification": {
                "verified_on": "2026-08-08",
                "finding": (
                    "Current official LlamaIndex embedding defaults are version-sensitive. "
                    "This benchmark therefore passes explicit CANON/OpenRouter embedding adapters and "
                    "does not claim to measure an implicit hosted provider default."
                ),
                "sources": [
                    "https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/",
                    "https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/settings/",
                    "https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local/",
                ],
            },
        },
        "llamaindex_stack": {
            "framework": "llama-index-core",
            "index": "VectorStoreIndex",
            "retriever": "index.as_retriever(similarity_top_k=candidate_k)",
            "candidate_k": candidate_k,
            "embedding": "explicit CANON embedding adapter; hash and Qwen cells are run separately",
            "llm": "disabled",
            "boundary": (
                "This measures LlamaIndex VectorStoreIndex retrieval with explicit embeddings, "
                "not LlamaIndex generation and not an implicit hosted provider default."
            ),
        },
        "canon_stack": {
            "candidate_k": candidate_k,
            "lexical_k": lexical_k,
            "vector_k": vector_k,
            "sparse": "bm25",
            "fusion": fusion,
            "reranker": DEFAULT_RERANKER,
            "document_format": document_format,
            "boundary": (
                "The CANON full-pipeline cells compare BM25+dense RRF plus Cohere rerank "
                "against LlamaIndex single dense retrieval. RRF-only cells isolate reranker contribution."
            ),
        },
        "include_query_details": include_query_details,
        "legacy_canon_report": legacy_canon_report,
        "factor_matrix": build_factor_matrix(all_summaries, k),
        "llamaindex_configuration_summaries": llama_summaries,
        "canon_configuration_summaries": canon_summaries,
        "canon_rrf_only_summaries": rrf_summaries,
        "all_configuration_summaries": all_summaries,
        "paired_deltas": comparisons,
        "claim_checks": claim_checks,
        "standing_verdict": verdict,
        "retrieval_factor_diagnostics": build_retrieval_factor_diagnostics(all_runs, k),
        "llamaindex_runs": [
            compact_run(run, include_query_details=include_query_details)
            for run in sorted(llama_runs, key=lambda row: (row.get("repeat", 0), str(row.get("configuration"))))
        ],
        "canon_runs": [
            compact_run(run, include_query_details=include_query_details)
            for run in sorted(canon_runs, key=lambda row: (row.get("repeat", 0), str(row.get("configuration"))))
        ],
        "canon_rrf_only_runs": [
            compact_run(run, include_query_details=include_query_details)
            for run in sorted(rrf_runs, key=lambda row: (row.get("repeat", 0), str(row.get("configuration"))))
        ],
    }


def build_retrieval_factor_diagnostics(all_runs: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metric = f"nDCG@{k}"
    config_queries = {
        config_id: summarize_config_queries(all_runs, config_id, metric)
        for config_id in [
            "llamaindex_hash_original_qrels",
            "llamaindex_qwen_original_qrels",
            "canon_hash_cohere_original_qrels",
            "canon_qwen_cohere_original_qrels",
            "canon_hash_rrf_only_original_qrels",
            "canon_qwen_rrf_only_original_qrels",
        ]
    }
    missing = [
        config_id
        for config_id in [
            "llamaindex_hash_original_qrels",
            "llamaindex_qwen_original_qrels",
            "canon_hash_cohere_original_qrels",
            "canon_qwen_cohere_original_qrels",
        ]
        if not config_queries.get(config_id)
    ]
    if missing:
        return {
            "status": "query_details_unavailable",
            "reason": (
                "Per-query factor diagnostics require raw query details. "
                "Regenerate the report from fresh runs rather than resuming from compact runs."
            ),
            "missing_configurations": missing,
            "required_command": (
                "python -m canon.baselines.llamaindex_baseline --repeats 3 --no-resume"
            ),
        }
    query_ids = sorted(
        set(config_queries["llamaindex_hash_original_qrels"])
        & set(config_queries["llamaindex_qwen_original_qrels"])
        & set(config_queries["canon_hash_cohere_original_qrels"])
        & set(config_queries["canon_qwen_cohere_original_qrels"])
    )
    rows = [
        diagnose_factor_query(query_id, config_queries, metric)
        for query_id in query_ids
    ]
    class_counts = Counter(
        failure_class
        for row in rows
        for failure_class in row["failure_classes"]
    )
    winners = Counter(row["qwen_same_encoder_winner"] for row in rows)
    return {
        "status": "ok",
        "metric": metric,
        "query_count": len(rows),
        "diagnostic_scope": (
            "Per-query summaries are averaged across repeats, then compared across the "
            "same encoder/framework cells. They explain retrieval behavior; they are not "
            "human relevance judgments."
        ),
        "winner_counts_qwen_same_encoder": dict(sorted(winners.items())),
        "failure_class_counts": dict(sorted(class_counts.items())),
        "mean_deltas_pp": {
            "llamaindex_qwen_minus_hash": average_delta(rows, "llamaindex_qwen_minus_hash_pp"),
            "canon_qwen_minus_hash_with_cohere": average_delta(rows, "canon_qwen_minus_hash_pp"),
            "canon_minus_llamaindex_with_hash": average_delta(rows, "canon_hash_pipeline_minus_llamaindex_hash_pp"),
            "canon_minus_llamaindex_with_qwen": average_delta(rows, "canon_qwen_pipeline_minus_llamaindex_qwen_pp"),
            "cohere_minus_rrf_with_hash": average_delta(rows, "cohere_minus_rrf_hash_pp"),
            "cohere_minus_rrf_with_qwen": average_delta(rows, "cohere_minus_rrf_qwen_pp"),
        },
        "largest_canon_qwen_losses": compact_factor_rows(
            sorted(rows, key=lambda row: row["deltas_pp"]["canon_qwen_pipeline_minus_llamaindex_qwen_pp"])[:8]
        ),
        "largest_canon_qwen_wins": compact_factor_rows(
            sorted(
                rows,
                key=lambda row: row["deltas_pp"]["canon_qwen_pipeline_minus_llamaindex_qwen_pp"],
                reverse=True,
            )[:8]
        ),
        "reranker_hurts_qwen_queries": compact_factor_rows(
            [
                row
                for row in rows
                if "cohere_rerank_hurts_qwen" in row["failure_classes"]
            ][:8]
        ),
        "diagnostic_reading": diagnostic_reading(rows),
        "queries": rows,
    }


def summarize_config_queries(
    all_runs: list[dict[str, Any]],
    configuration: str,
    metric: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        if run.get("configuration") != configuration or run.get("status") != "ok":
            continue
        for query in run.get("queries") or []:
            query_id = str(query.get("id") or "")
            if query_id:
                grouped.setdefault(query_id, []).append(query)
    return {
        query_id: summarize_query_rows(rows, metric)
        for query_id, rows in grouped.items()
        if rows
    }


def summarize_query_rows(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    first = rows[0]
    metric_values = [float((row.get("metrics") or {}).get(metric, 0.0) or 0.0) for row in rows]
    recall_values = [
        float((row.get("candidate_recall") or {}).get("candidate_recall", 0.0) or 0.0)
        for row in rows
    ]
    top_hit_counts = [len(row.get("relevant_hit_ids") or []) for row in rows]
    relevant_counts = [
        int((row.get("qrels_semantics") or {}).get("effective_relevant_count", 0) or 0)
        for row in rows
    ]
    return {
        "id": str(first.get("id") or ""),
        "query": str(first.get("query") or ""),
        "repeat_count": len(rows),
        metric: round_metric(average(metric_values)),
        "candidate_recall": round_metric(average(recall_values)),
        "top10_relevant_hit_count_mean": round_metric(average([float(value) for value in top_hit_counts])),
        "top10_hit_repeat_count": sum(1 for value in top_hit_counts if value > 0),
        "relevant_count_mean": round_metric(average([float(value) for value in relevant_counts])),
    }


def diagnose_factor_query(
    query_id: str,
    config_queries: dict[str, dict[str, dict[str, Any]]],
    metric: str,
) -> dict[str, Any]:
    li_hash = config_queries["llamaindex_hash_original_qrels"][query_id]
    li_qwen = config_queries["llamaindex_qwen_original_qrels"][query_id]
    canon_hash = config_queries["canon_hash_cohere_original_qrels"][query_id]
    canon_qwen = config_queries["canon_qwen_cohere_original_qrels"][query_id]
    canon_hash_rrf = config_queries.get("canon_hash_rrf_only_original_qrels", {}).get(query_id, {})
    canon_qwen_rrf = config_queries.get("canon_qwen_rrf_only_original_qrels", {}).get(query_id, {})
    deltas = {
        "llamaindex_qwen_minus_hash_pp": delta_pp(li_qwen, li_hash, metric),
        "canon_qwen_minus_hash_pp": delta_pp(canon_qwen, canon_hash, metric),
        "canon_hash_pipeline_minus_llamaindex_hash_pp": delta_pp(canon_hash, li_hash, metric),
        "canon_qwen_pipeline_minus_llamaindex_qwen_pp": delta_pp(canon_qwen, li_qwen, metric),
        "cohere_minus_rrf_hash_pp": delta_pp(canon_hash, canon_hash_rrf, metric),
        "cohere_minus_rrf_qwen_pp": delta_pp(canon_qwen, canon_qwen_rrf, metric),
    }
    metrics = {
        "llamaindex_hash": metric_snapshot(li_hash, metric),
        "llamaindex_qwen": metric_snapshot(li_qwen, metric),
        "canon_hash_cohere": metric_snapshot(canon_hash, metric),
        "canon_qwen_cohere": metric_snapshot(canon_qwen, metric),
        "canon_hash_rrf_only": metric_snapshot(canon_hash_rrf, metric),
        "canon_qwen_rrf_only": metric_snapshot(canon_qwen_rrf, metric),
    }
    winner = qwen_winner(li_qwen, canon_qwen, metric)
    failure_classes = classify_factor_query(metrics, deltas)
    return {
        "id": query_id,
        "query": li_qwen.get("query") or li_hash.get("query") or canon_qwen.get("query"),
        "qwen_same_encoder_winner": winner,
        "failure_classes": failure_classes,
        "metrics": metrics,
        "deltas_pp": deltas,
    }


def metric_snapshot(row: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        metric: round_metric(float(row.get(metric, 0.0) or 0.0)),
        "candidate_recall": round_metric(float(row.get("candidate_recall", 0.0) or 0.0)),
        "top10_hit_repeat_count": int(row.get("top10_hit_repeat_count", 0) or 0),
        "repeat_count": int(row.get("repeat_count", 0) or 0),
    }


def qwen_winner(
    llamaindex_qwen: dict[str, Any],
    canon_qwen: dict[str, Any],
    metric: str,
    tolerance: float = 0.000001,
) -> str:
    delta = float(canon_qwen.get(metric, 0.0) or 0.0) - float(llamaindex_qwen.get(metric, 0.0) or 0.0)
    if delta > tolerance:
        return "canon_qwen_cohere"
    if delta < -tolerance:
        return "llamaindex_qwen"
    return "tie"


def classify_factor_query(metrics: dict[str, dict[str, Any]], deltas: dict[str, float]) -> list[str]:
    classes: list[str] = []
    li_qwen = metrics["llamaindex_qwen"]
    canon_qwen = metrics["canon_qwen_cohere"]
    if li_qwen["candidate_recall"] <= 0 and canon_qwen["candidate_recall"] <= 0:
        classes.append("shared_candidate_miss")
    elif li_qwen["candidate_recall"] > canon_qwen["candidate_recall"] + 0.05:
        classes.append("llamaindex_candidate_recall_advantage")
    elif canon_qwen["candidate_recall"] > li_qwen["candidate_recall"] + 0.05:
        classes.append("canon_candidate_recall_advantage")
    if deltas["canon_qwen_pipeline_minus_llamaindex_qwen_pp"] <= -5:
        classes.append("llamaindex_qwen_ranking_advantage")
    elif deltas["canon_qwen_pipeline_minus_llamaindex_qwen_pp"] >= 5:
        classes.append("canon_multistage_advantage")
    if deltas["cohere_minus_rrf_qwen_pp"] <= -1:
        classes.append("cohere_rerank_hurts_qwen")
    elif deltas["cohere_minus_rrf_qwen_pp"] >= 1:
        classes.append("cohere_rerank_helps_qwen")
    if deltas["llamaindex_qwen_minus_hash_pp"] >= 10:
        classes.append("encoder_rescues_llamaindex")
    if deltas["canon_qwen_minus_hash_pp"] >= 10:
        classes.append("encoder_rescues_canon")
    if not classes:
        classes.append("near_tie_or_low_signal")
    return classes


def delta_pp(
    treatment: dict[str, Any],
    baseline: dict[str, Any],
    metric: str,
) -> float:
    return round(
        (float(treatment.get(metric, 0.0) or 0.0) - float(baseline.get(metric, 0.0) or 0.0)) * 100,
        3,
    )


def average_delta(rows: list[dict[str, Any]], key: str) -> float:
    return round(average([float(row["deltas_pp"].get(key, 0.0)) for row in rows]), 3)


def compact_factor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "query": row["query"],
            "qwen_same_encoder_winner": row["qwen_same_encoder_winner"],
            "canon_qwen_minus_llamaindex_qwen_pp": row["deltas_pp"][
                "canon_qwen_pipeline_minus_llamaindex_qwen_pp"
            ],
            "cohere_minus_rrf_qwen_pp": row["deltas_pp"]["cohere_minus_rrf_qwen_pp"],
            "failure_classes": row["failure_classes"],
        }
        for row in rows
    ]


def diagnostic_reading(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No per-query details were available."
    qwen_pipeline_delta = average_delta(rows, "canon_qwen_pipeline_minus_llamaindex_qwen_pp")
    encoder_delta = average_delta(rows, "llamaindex_qwen_minus_hash_pp")
    rerank_delta = average_delta(rows, "cohere_minus_rrf_qwen_pp")
    if qwen_pipeline_delta < 0 and encoder_delta > 10:
        return (
            "The correction points to encoder strength as the dominant factor on this slice; "
            "CANON's multi-stage path is not yet earning its extra complexity once Qwen is shared."
        )
    if qwen_pipeline_delta > 0 and rerank_delta > 1:
        return (
            "CANON's multi-stage path is adding measurable value beyond shared Qwen retrieval, "
            "with reranking contributing to top-k placement."
        )
    return (
        "The query-level pattern is mixed; use the largest wins/losses lists before making "
        "architecture claims."
    )


def repeat_configs_for_summary(configs: list[LlamaIndexRepeatConfig | CanonRepeatConfig]):
    from canon.product.stage1_repeat_spread import RepeatConfig

    return [
        RepeatConfig(
            id=config.id,
            label=config.label,
            auto_query_expansion=False,
            parent_qrels=config.parent_qrels,
            parent_expansion_limit=0,
            qrels_scope=config.qrels_scope,
        )
        for config in configs
    ]


def canon_and_llama_summary_configs(configs: list[LlamaIndexRepeatConfig]):
    from canon.product.stage1_repeat_spread import DEFAULT_CONFIGS

    return list(DEFAULT_CONFIGS) + repeat_configs_for_summary(configs)


def build_factor_matrix(summaries: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metric = f"nDCG@{k}"
    by_id = {summary["id"]: summary for summary in summaries}
    cells = {
        "llamaindex_hashed": matrix_cell(by_id.get("llamaindex_hash_original_qrels", {}), metric),
        "llamaindex_qwen": matrix_cell(by_id.get("llamaindex_qwen_original_qrels", {}), metric),
        "canon_hashed_cohere": matrix_cell(by_id.get("canon_hash_cohere_original_qrels", {}), metric),
        "canon_qwen_cohere": matrix_cell(by_id.get("canon_qwen_cohere_original_qrels", {}), metric),
    }
    rrf_only = {
        "canon_hashed_rrf_only": matrix_cell(by_id.get("canon_hash_rrf_only_original_qrels", {}), metric),
        "canon_qwen_rrf_only": matrix_cell(by_id.get("canon_qwen_rrf_only_original_qrels", {}), metric),
    }
    return {
        "primary_metric": metric,
        "cells": cells,
        "rrf_only_ablation": rrf_only,
        "boundary": (
            "The four primary cells decompose encoder and pipeline effects. "
            "The RRF-only cells are deterministic ablations for the Cohere reranker contribution."
        ),
    }


def matrix_cell(summary: dict[str, Any], metric: str) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    primary = metrics.get(metric) or {}
    return {
        "configuration": summary.get("id"),
        "label": summary.get("label"),
        "completed_runs": summary.get("completed_runs", 0),
        "mean": primary.get("mean", 0.0),
        "spread_pp": primary.get("spread_pp", 0.0),
        "candidate_recall_mean": (metrics.get("candidate_recall_mean") or {}).get("mean", 0.0),
        "zero_candidate_recall_queries": summary.get("zero_candidate_recall_queries_values", []),
    }


def standing_verdict(claim_checks: list[dict[str, Any]]) -> dict[str, Any]:
    by_comparison = {check.get("comparison"): check for check in claim_checks}
    qwen_pipeline = by_comparison.get(
        "canon_qwen_cohere_original_qrels - llamaindex_qwen_original_qrels",
        {},
    )
    standing = str(qwen_pipeline.get("standing") or "unknown")
    if standing == "insufficient_repeat_count":
        conclusion = "incomplete"
    elif qwen_pipeline:
        conclusion = "corrected_factorial_result_available"
    else:
        conclusion = "missing_primary_qwen_pipeline_comparison"
    return {
        "primary_metric": qwen_pipeline.get("metric", "nDCG@10"),
        "primary_comparison": qwen_pipeline.get("comparison"),
        "conclusion": conclusion,
        "standing": standing,
        "mean_delta_pp": qwen_pipeline.get("mean_delta_pp", 0.0),
        "reason": qwen_pipeline.get("reason", "No primary Qwen-vs-Qwen pipeline comparison was available."),
        "publication_boundary": (
            "Publish the result as a corrected 2x2 retrieval-factor pilot, not as proof "
            "that one framework or architecture generally wins."
        ),
        "superseded_claim": (
            "The old -26.6pp LlamaIndex comparison should not be used as evidence that "
            "CANON's architecture earned its complexity, because encoder and pipeline changed together."
        ),
    }


def compact_run(run: dict[str, Any], *, include_query_details: bool) -> dict[str, Any]:
    if include_query_details:
        return run
    return {key: value for key, value in run.items() if key != "queries"}


def markdown_report(report: dict[str, Any]) -> str:
    primary_metric = report["factor_matrix"]["primary_metric"]
    lines = [
        "# LlamaIndex vs CANON Stage 1 Factor Correction",
        "",
        f"Status: `{report['status']}`",
        "",
        (
            f"Protocol: {report['query_count']} NFCorpus queries, "
            f"{report['label_count']} fixed labels, {report['repeat_count_requested']} repeats per cell."
        ),
        "",
        f"Corrected conclusion: `{report['standing_verdict']['conclusion']}`.",
        "",
        "## Correction",
        "",
        f"- {report['standing_verdict']['superseded_claim']}",
        f"- {report['protocol']['supersedes']['reason']}",
        f"- {report['standing_verdict']['publication_boundary']}",
        "",
        "## Primary 2x2",
        "",
        f"| Encoder | LlamaIndex single dense {primary_metric} | CANON BM25+dense RRF+Cohere {primary_metric} |",
        "|---|---:|---:|",
    ]
    cells = report["factor_matrix"]["cells"]
    lines.append(
        "| hashed-semantic-v1 | {llama:.6f} | {canon:.6f} |".format(
            llama=float(cells["llamaindex_hashed"]["mean"]),
            canon=float(cells["canon_hashed_cohere"]["mean"]),
        )
    )
    lines.append(
        "| qwen/qwen3-embedding-8b | {llama:.6f} | {canon:.6f} |".format(
            llama=float(cells["llamaindex_qwen"]["mean"]),
            canon=float(cells["canon_qwen_cohere"]["mean"]),
        )
    )
    diagnostics = report.get("retrieval_factor_diagnostics") or {}
    lines.extend(render_diagnostics_markdown(diagnostics))
    lines.extend(
        [
            "",
            "## Reranker Ablation",
            "",
            f"| Encoder | CANON RRF-only {primary_metric} | CANON RRF+Cohere {primary_metric} |",
            "|---|---:|---:|",
        ]
    )
    rrf = report["factor_matrix"]["rrf_only_ablation"]
    lines.append(
        "| hashed-semantic-v1 | {rrf_value:.6f} | {cohere:.6f} |".format(
            rrf_value=float(rrf["canon_hashed_rrf_only"]["mean"]),
            cohere=float(cells["canon_hashed_cohere"]["mean"]),
        )
    )
    lines.append(
        "| qwen/qwen3-embedding-8b | {rrf_value:.6f} | {cohere:.6f} |".format(
            rrf_value=float(rrf["canon_qwen_rrf_only"]["mean"]),
            cohere=float(cells["canon_qwen_cohere"]["mean"]),
        )
    )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| Comparison | Paired runs | Mean delta pp | Min delta pp | Max delta pp | Delta spread pp | Standing |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    checks_by_comparison = {check["comparison"]: check for check in report.get("claim_checks", [])}
    for comparison in report.get("paired_deltas", []):
        check = checks_by_comparison.get(comparison["comparison"], {})
        lines.append(
            "| {comparison} | {n} | {mean:.3f} | {min_delta:.3f} | {max_delta:.3f} | {spread:.3f} | `{standing}` |".format(
                comparison=comparison["comparison"],
                n=comparison["n"],
                mean=comparison["mean_delta_pp"],
                min_delta=comparison["min_delta"] * 100,
                max_delta=comparison["max_delta"] * 100,
                spread=comparison["spread_delta_pp"],
                standing=check.get("standing", "unknown"),
            )
        )
    lines.extend(
        [
            "",
            "## Stack Boundaries",
            "",
            f"- LlamaIndex: {report['llamaindex_stack']['boundary']}",
            f"- CANON: {report['canon_stack']['boundary']}",
            f"- LlamaIndex default note: {report['protocol']['llamaindex_default_verification']['finding']}",
            "",
            "The compact machine-readable report is in `reports/llamaindex_stage1_head_to_head.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_diagnostics_markdown(diagnostics: dict[str, Any]) -> list[str]:
    if diagnostics.get("status") != "ok":
        return [
            "",
            "## Retrieval Diagnostics",
            "",
            f"Status: `{diagnostics.get('status', 'unavailable')}`",
            "",
            str(diagnostics.get("reason") or "Per-query diagnostics were not available."),
        ]
    lines = [
        "",
        "## Retrieval Diagnostics",
        "",
        diagnostics["diagnostic_reading"],
        "",
        "### Winner Counts With Shared Qwen",
        "",
        json.dumps(diagnostics["winner_counts_qwen_same_encoder"], indent=2),
        "",
        "### Failure Classes",
        "",
        json.dumps(diagnostics["failure_class_counts"], indent=2),
        "",
        "### Largest CANON Losses With Shared Qwen",
        "",
        "| Query | Delta pp | Rerank pp | Classes |",
        "|---|---:|---:|---|",
    ]
    for row in diagnostics.get("largest_canon_qwen_losses", [])[:5]:
        lines.append(
            "| `{id}` {query} | {delta:.3f} | {rerank:.3f} | `{classes}` |".format(
                id=row["id"],
                query=short_text(str(row.get("query") or ""), 72),
                delta=float(row["canon_qwen_minus_llamaindex_qwen_pp"]),
                rerank=float(row["cohere_minus_rrf_qwen_pp"]),
                classes="`, `".join(row.get("failure_classes") or []),
            )
        )
    lines.extend(
        [
            "",
            "### Largest CANON Wins With Shared Qwen",
            "",
            "| Query | Delta pp | Rerank pp | Classes |",
            "|---|---:|---:|---|",
        ]
    )
    for row in diagnostics.get("largest_canon_qwen_wins", [])[:5]:
        lines.append(
            "| `{id}` {query} | {delta:.3f} | {rerank:.3f} | `{classes}` |".format(
                id=row["id"],
                query=short_text(str(row.get("query") or ""), 72),
                delta=float(row["canon_qwen_minus_llamaindex_qwen_pp"]),
                rerank=float(row["cohere_minus_rrf_qwen_pp"]),
                classes="`, `".join(row.get("failure_classes") or []),
            )
        )
    return lines


def short_text(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_head_to_head(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(markdown_report(report), encoding="utf-8")


def display_path(path: Path | str) -> str:
    path = Path(str(path))
    settings = load_settings()
    try:
        return str(path.resolve().relative_to(settings.root.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return str(path).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LlamaIndex baseline against CANON Stage 1 qrels.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--qrels", default=DEFAULT_QRELS)
    parser.add_argument("--canon-report", default=DEFAULT_CANON_REPORT)
    parser.add_argument("--candidate-k", type=int, default=250)
    parser.add_argument("--lexical-k", type=int, default=250)
    parser.add_argument("--vector-k", type=int, default=250)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fusion", default="rrf")
    parser.add_argument("--document-format", default="structured", choices=["plain", "structured"])
    parser.add_argument("--query-cache-root", default=DEFAULT_QUERY_CACHE_ROOT)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--include-query-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    settings = load_settings()
    qrels_path = resolve(settings.root, args.qrels)
    canon_report_path = resolve(settings.root, args.canon_report)
    report_json = resolve(settings.root, args.report_json)
    report_md = resolve(settings.root, args.report_md)
    query_cache_root = resolve(settings.root, args.query_cache_root)
    qrels = load_qrels(qrels_path)
    documents = load_processed_corpus(
        settings.data_dir,
        args.mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, args.mode),
    )
    chunk_map = chunk_to_work_map(documents)
    _legacy_canon_runs, legacy_canon_report = load_canon_runs(canon_report_path)
    if args.no_resume:
        llama_runs, canon_runs, rrf_runs = [], [], []
    else:
        llama_runs, canon_runs, rrf_runs = load_existing_run_groups(report_json)
    print(
        f"[llamaindex-baseline] mode={args.mode} repeats={args.repeats} "
        f"queries={len(qrels.get('queries', []))} documents={len(documents)} resume={not args.no_resume}",
        flush=True,
    )
    for repeat in range(1, args.repeats + 1):
        for config in DEFAULT_CONFIGS:
            if (config.id, repeat) in completed_ok_keys(llama_runs):
                print(f"[llamaindex-baseline] skip repeat={repeat} config={config.id}", flush=True)
                continue
            print(f"[llamaindex-baseline] start repeat={repeat} config={config.id}", flush=True)
            run = run_once(
                config,
                repeat=repeat,
                mode=args.mode,
                qrels=qrels,
                documents=documents,
                chunk_map=chunk_map,
                candidate_k=args.candidate_k,
                k=args.k,
            )
            llama_runs = replace_run(llama_runs, run)
            write_progress_report(args, report_json, report_md, qrels_path, qrels, legacy_canon_report, llama_runs, canon_runs, rrf_runs)
            log_run("llamaindex-baseline", run, args.k)

        for config in DEFAULT_CANON_CONFIGS:
            if (config.id, repeat) in completed_ok_keys(canon_runs):
                print(f"[llamaindex-baseline] skip repeat={repeat} config={config.id}", flush=True)
                continue
            print(f"[llamaindex-baseline] start repeat={repeat} config={config.id}", flush=True)
            run = run_canon_rerank_once(
                config,
                repeat=repeat,
                mode=args.mode,
                qrels_path=qrels_path,
                qrels=qrels,
                chunk_map=chunk_map,
                candidate_k=args.candidate_k,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                k=args.k,
                fusion=args.fusion,
                document_format=args.document_format,
                query_cache_root=query_cache_root,
            )
            canon_runs = replace_run(canon_runs, run)
            write_progress_report(args, report_json, report_md, qrels_path, qrels, legacy_canon_report, llama_runs, canon_runs, rrf_runs)
            log_run("llamaindex-baseline", run, args.k)

        for config in DEFAULT_RRF_ONLY_CONFIGS:
            if (config.id, repeat) in completed_ok_keys(rrf_runs):
                print(f"[llamaindex-baseline] skip repeat={repeat} config={config.id}", flush=True)
                continue
            print(f"[llamaindex-baseline] start repeat={repeat} config={config.id}", flush=True)
            run = run_canon_rrf_once(
                config,
                repeat=repeat,
                mode=args.mode,
                qrels=qrels,
                chunk_map=chunk_map,
                candidate_k=args.candidate_k,
                lexical_k=args.lexical_k,
                vector_k=args.vector_k,
                k=args.k,
                fusion=args.fusion,
            )
            rrf_runs = replace_run(rrf_runs, run)
            write_progress_report(args, report_json, report_md, qrels_path, qrels, legacy_canon_report, llama_runs, canon_runs, rrf_runs)
            log_run("llamaindex-baseline", run, args.k)
    final_report = build_report(
        llama_runs=llama_runs,
        canon_runs=canon_runs,
        rrf_runs=rrf_runs,
        legacy_canon_report=legacy_canon_report,
        configs=DEFAULT_CONFIGS,
        canon_configs=DEFAULT_CANON_CONFIGS,
        rrf_configs=DEFAULT_RRF_ONLY_CONFIGS,
        repeats=args.repeats,
        mode=args.mode,
        qrels_path=qrels_path,
        qrels=qrels,
        k=args.k,
        candidate_k=args.candidate_k,
        lexical_k=args.lexical_k,
        vector_k=args.vector_k,
        fusion=args.fusion,
        document_format=args.document_format,
        include_query_details=args.include_query_details,
    )
    write_head_to_head(final_report, report_json, report_md)
    print(json.dumps(final_report["standing_verdict"], indent=2), flush=True)


def write_progress_report(
    args: argparse.Namespace,
    report_json: Path,
    report_md: Path,
    qrels_path: Path,
    qrels: dict[str, Any],
    legacy_canon_report: dict[str, Any],
    llama_runs: list[dict[str, Any]],
    canon_runs: list[dict[str, Any]],
    rrf_runs: list[dict[str, Any]],
) -> None:
    report = build_report(
        llama_runs=llama_runs,
        canon_runs=canon_runs,
        rrf_runs=rrf_runs,
        legacy_canon_report=legacy_canon_report,
        configs=DEFAULT_CONFIGS,
        canon_configs=DEFAULT_CANON_CONFIGS,
        rrf_configs=DEFAULT_RRF_ONLY_CONFIGS,
        repeats=args.repeats,
        mode=args.mode,
        qrels_path=qrels_path,
        qrels=qrels,
        k=args.k,
        candidate_k=args.candidate_k,
        lexical_k=args.lexical_k,
        vector_k=args.vector_k,
        fusion=args.fusion,
        document_format=args.document_format,
        include_query_details=args.include_query_details,
    )
    write_head_to_head(report, report_json, report_md)


def log_run(prefix: str, run: dict[str, Any], k: int) -> None:
    if run.get("status") == "ok":
        print(
            f"[{prefix}] ok repeat={run.get('repeat')} config={run.get('configuration')} "
            f"nDCG@{k}={run['metrics'][f'nDCG@{k}']:.6f} "
            f"candidate_recall={run['metrics']['candidate_recall_mean']:.6f}",
            flush=True,
        )
    else:
        print(
            f"[{prefix}] unavailable repeat={run.get('repeat')} config={run.get('configuration')} "
            f"reason={run.get('reason')}",
            flush=True,
        )


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()

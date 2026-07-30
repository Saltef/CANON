from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.embeddings.store import build_embedding_store, embedding_store_path, load_embedding_records
from canon.embeddings.providers import (
    EmbeddingResult,
    embedding_preparation,
    embed_documents as provider_embed_documents,
    embed_queries,
    get_embedding_provider,
)
from canon.eval.external_ir import load_qrels
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.tokenize import tokenize


DEFAULT_PROVIDERS = ["local", "openai", "cohere"]
RELATIONAL_TERMS = {
    "affect",
    "affects",
    "because",
    "cause",
    "causes",
    "caused",
    "drives",
    "impact",
    "influence",
    "lead",
    "leads",
    "relationship",
    "risk",
    "tradeoff",
    "why",
}
GENERIC_QUERY_TERMS = {
    "about",
    "any",
    "can",
    "does",
    "evidence",
    "find",
    "how",
    "main",
    "show",
    "tell",
    "the",
    "what",
    "which",
}


def evaluate_semantic_models(
    mode: str,
    queries_path: Path | None = None,
    qrels_path: Path | None = None,
    providers: list[str] | None = None,
    k: int = 10,
    batch_size: int = 32,
) -> dict[str, Any]:
    settings = load_settings()
    documents = load_processed_corpus(
        settings.data_dir,
        mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, mode),
    )
    queries = load_eval_queries(queries_path, qrels_path)
    provider_names = providers or DEFAULT_PROVIDERS
    provider_reports = [
        evaluate_provider(provider_name, mode, documents, queries, k=k, batch_size=batch_size)
        for provider_name in provider_names
    ]
    report = {
        "report_id": "semantic_model_evaluation_v1",
        "mode": mode,
        "metric_k": k,
        "query_count": len(queries),
        "document_count": len(documents),
        "providers": provider_reports,
        "leaderboard": model_leaderboard(provider_reports, k),
        "recommendation": model_recommendation(provider_reports, k),
        "limitations": report_limitations(queries, provider_reports),
    }
    write_json(settings.reports_dir / f"semantic_model_evaluation_{mode}.json", report)
    return report


def evaluate_provider(
    provider_name: str,
    mode: str,
    documents: list[RetrievalDocument],
    queries: list[dict[str, Any]],
    k: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    provider_key, model = parse_provider_spec(provider_name)
    try:
        provider = get_embedding_provider(provider_key, model)
        document_vectors, document_cache = embed_documents(
            provider,
            documents,
            mode=mode,
            batch_size=batch_size,
        )
        query_reports = [
            evaluate_query(provider, document_vectors, query, k=k)
            for query in queries
        ]
    except Exception as exc:  # noqa: BLE001 - provider availability belongs in the report.
        return {
            "provider": provider_key,
            "model": model,
            "provider_id": provider_identifier(provider_key, model),
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    summary = provider_summary(query_reports, k)
    return {
        "provider": provider.provider,
        "model": provider.model,
        "provider_id": provider_identifier(provider.provider, provider.model),
        "preparation": embedding_preparation(provider),
        "status": "ok",
        "dimensions": document_vectors[0]["dimensions"] if document_vectors else 0,
        "document_embedding_cache": document_cache,
        "elapsed_ms": elapsed_ms,
        "query_count": len(query_reports),
        "summary": summary,
        "slice_summary": slice_summary(query_reports, k),
        "queries": query_reports,
    }


def embed_documents(
    provider: Any,
    documents: list[RetrievalDocument],
    mode: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = load_settings()
    cache_path = embedding_store_path(settings.data_dir, mode, provider.provider, provider.model)
    records = load_embedding_records(cache_path)
    if len(records) < len(documents):
        build_embedding_store(mode, provider.provider, provider.model, batch_size=batch_size)
        records = load_embedding_records(cache_path)
    if len(records) >= len(documents):
        by_chunk = {str(record["chunk_id"]): record for record in records}
        rows = [
            document_vector_from_record(document, by_chunk[str(document.chunk_id)])
            for document in documents
            if str(document.chunk_id) in by_chunk
        ]
        if len(rows) == len(documents):
            return rows, {
                "status": "hit",
                "path": str(cache_path).replace("\\", "/"),
                "record_count": len(records),
            }
    rows = embed_documents_uncached(provider, documents, batch_size=batch_size)
    return rows, {
        "status": "miss",
        "path": str(cache_path).replace("\\", "/"),
        "record_count": len(records),
    }


def document_vector_from_record(document: RetrievalDocument, record: dict[str, Any]) -> dict[str, Any]:
    embedding = EmbeddingResult(
        provider=str(record["provider"]),
        model=str(record["model"]),
        dimensions=int(record["dimensions"]),
        vector=[float(value) for value in record["vector"]],
    )
    return document_vector_row(document, embedding)


def embed_documents_uncached(
    provider: Any,
    documents: list[RetrievalDocument],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        embeddings = provider_embed_documents(provider, [document.text for document in batch])
        rows.extend(
            document_vector_row(document, embedding)
            for document, embedding in zip(batch, embeddings, strict=True)
        )
    return rows


def document_vector_row(document: RetrievalDocument, embedding: EmbeddingResult) -> dict[str, Any]:
    sparse = sparse_vector(embedding.vector)
    return {
        "chunk_id": document.chunk_id,
        "work_id": document.work_id,
        "title": document.title,
        "source_name": document.source_name,
        "document_type": document.work_signals.get("document_type"),
        "domain": document.work_signals.get("domain") or document.work_signals.get("domain_profile"),
        "text": document.text,
        "embedding": embedding,
        "embedding_sparse": sparse,
        "embedding_norm": sparse_norm(sparse),
        "dimensions": embedding.dimensions,
    }


def evaluate_query(provider: Any, document_vectors: list[dict[str, Any]], query: dict[str, Any], k: int) -> dict[str, Any]:
    query_embedding = embed_queries(provider, [query["query"]])[0]
    ranked = rank_vectors(query_embedding, document_vectors)
    ranked_ids = [row["chunk_id"] for row in ranked]
    qrels = query.get("relevant") or {}
    metrics = evaluate_ranking(ranked_ids, qrels, k) if qrels else {}
    top_rows = ranked[:k]
    return {
        "id": query["id"],
        "query": query["query"],
        "slices": query_slices(query["query"]),
        "qrel_count": len(qrels),
        "metrics": metrics,
        "top_score": round(top_rows[0]["score"], 6) if top_rows else 0.0,
        "score_margin": score_margin(top_rows),
        "ranked_chunk_ids": ranked_ids[:k],
        "relevant_hit_ids": [chunk_id for chunk_id in ranked_ids[:k] if chunk_id in qrels],
        "top_results": [
            {
                "rank": index,
                "chunk_id": row["chunk_id"],
                "work_id": row["work_id"],
                "title": row["title"],
                "source_name": row["source_name"],
                "document_type": row["document_type"],
                "score": round(row["score"], 6),
            }
            for index, row in enumerate(top_rows, start=1)
        ],
    }


def rank_vectors(query_embedding: EmbeddingResult, document_vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_sparse = sparse_vector(query_embedding.vector)
    query_norm = sparse_norm(query_sparse)
    rows = []
    for row in document_vectors:
        score = sparse_cosine(
            query_sparse,
            query_norm,
            row.get("embedding_sparse") or sparse_vector(row["embedding"].vector),
            float(row.get("embedding_norm") or 0.0),
        )
        rows.append({**row, "score": score})
    return sorted(rows, key=lambda item: (item["score"], item["chunk_id"]), reverse=True)


def dense_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def sparse_vector(vector: list[float]) -> dict[int, float]:
    return {index: float(value) for index, value in enumerate(vector) if float(value) != 0.0}


def sparse_norm(vector: dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def sparse_cosine(
    left: dict[int, float],
    left_norm: float,
    right: dict[int, float],
    right_norm: float,
) -> float:
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(index, 0.0) for index, value in left.items())
    return numerator / (left_norm * right_norm)


def score_margin(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    return round(float(rows[0]["score"]) - float(rows[1]["score"]), 6)


def provider_summary(query_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metric_names = [f"nDCG@{k}", f"Recall@{k}", f"MAP@{k}", f"MRR@{k}"]
    metric_rows = [query["metrics"] for query in query_reports if query.get("metrics")]
    return {
        **{metric: mean_metric(metric_rows, metric) for metric in metric_names},
        "coverage_at_1": coverage_at(query_reports, 1),
        "coverage_at_3": coverage_at(query_reports, min(3, k)),
        f"coverage_at_{k}": coverage_at(query_reports, k),
        "average_top_score": average([query["top_score"] for query in query_reports]),
        "average_score_margin": average([query["score_margin"] for query in query_reports]),
        "labeled_query_count": len(metric_rows),
    }


def coverage_at(query_reports: list[dict[str, Any]], k: int) -> float:
    labeled = [query for query in query_reports if query.get("qrel_count", 0) > 0]
    if not labeled:
        return 0.0
    hits = 0
    for query in labeled:
        if set(query.get("ranked_chunk_ids", [])[:k]) & set((query.get("relevant_hit_ids") or [])):
            hits += 1
    return round(hits / len(labeled), 6)


def slice_summary(query_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    slices = sorted({slice_name for query in query_reports for slice_name in query.get("slices", [])})
    summary = {}
    for slice_name in slices:
        rows = [query for query in query_reports if slice_name in query.get("slices", [])]
        summary[slice_name] = provider_summary(rows, k)
    return summary


def model_leaderboard(provider_reports: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
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


def model_recommendation(provider_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    leaderboard = model_leaderboard(provider_reports, k)
    if not leaderboard:
        return {
            "status": "no_available_models",
            "message": "No semantic providers were available for evaluation.",
        }
    best = leaderboard[0]
    if best["primary_score"] <= 0:
        confidence = "low"
        reason = "No labeled retrieval gains were observed; inspect qrels and corpus coverage."
    elif len(leaderboard) == 1:
        confidence = "limited"
        reason = "Only one semantic provider was available, so this is not a comparative claim."
    else:
        runner_up = leaderboard[1]["primary_score"]
        gap = round(best["primary_score"] - runner_up, 6)
        confidence = "moderate" if gap >= 0.05 else "low"
        reason = f"Best model leads the runner-up by {gap} on nDCG@{k}."
    return {
        "status": "recommendation_available",
        "provider": best["provider"],
        "model": best["model"],
        "provider_id": best.get("provider_id") or provider_identifier(best["provider"], best["model"]),
        "confidence": confidence,
        "reason": reason,
        "human_review_required": True,
    }


def query_slices(query: str) -> list[str]:
    tokens = set(tokenize(query))
    slices = []
    if len(tokens - GENERIC_QUERY_TERMS) <= 3:
        slices.append("beginner_or_vague")
    if tokens & RELATIONAL_TERMS:
        slices.append("relational_semantics")
    if any(len(token) >= 11 for token in tokens):
        slices.append("domain_terminology")
    if not slices:
        slices.append("general")
    return slices


def load_eval_queries(queries_path: Path | None, qrels_path: Path | None) -> list[dict[str, Any]]:
    if qrels_path:
        qrels = load_qrels(qrels_path)
        return [
            {
                "id": str(query["id"]),
                "query": str(query["query"]),
                "relevant": {str(key): float(value) for key, value in (query.get("relevant") or {}).items()},
            }
            for query in qrels.get("queries", [])
        ]
    if not queries_path:
        raise ValueError("Either --queries or --qrels is required for semantic model evaluation.")
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Queries input must be a non-empty list or an object with a queries list.")
    return [
        {
            "id": str(row.get("id") or index),
            "query": str(row["query"]),
            "relevant": {str(key): float(value) for key, value in (row.get("relevant") or {}).items()},
        }
        for index, row in enumerate(rows, start=1)
    ]


def report_limitations(queries: list[dict[str, Any]], provider_reports: list[dict[str, Any]]) -> list[str]:
    limitations = [
        "Semantic model evaluation is corpus- and task-specific; do not treat the winner as globally best.",
        "Human relevance labels or qrels are required for production model-selection claims.",
    ]
    if not any(query.get("relevant") for query in queries):
        limitations.append("No relevance labels were supplied, so IR metrics are unavailable.")
    unavailable = [report["provider"] for report in provider_reports if report.get("status") != "ok"]
    if unavailable:
        limitations.append(f"Some providers were unavailable and excluded from ranking: {', '.join(unavailable)}.")
    return limitations


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def parse_providers(value: str | None) -> list[str] | None:
    if not value:
        return None
    providers = [piece.strip() for piece in value.split(",") if piece.strip()]
    return providers or None


def parse_provider_spec(value: str) -> tuple[str, str | None]:
    provider, separator, model = value.partition(":")
    return provider.strip(), model.strip() if separator and model.strip() else None


def provider_identifier(provider: str, model: str | None) -> str:
    return f"{provider}:{model}" if model else provider


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic embedding models on a CANON corpus.")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--providers", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_semantic_models(
                mode=args.mode,
                queries_path=Path(args.queries) if args.queries else None,
                qrels_path=Path(args.qrels) if args.qrels else None,
                providers=parse_providers(args.providers),
                k=args.k,
                batch_size=args.batch_size,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

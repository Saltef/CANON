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
from canon.eval.external_ir import load_qrels
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.retrieval.experiment import run as run_retrieval
from canon.retrieval.tokenize import tokenize


DEFAULT_RERANKERS = ["heuristic", "cohere"]


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


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


def get_rerank_provider(name: str, model: str | None = None) -> RerankProvider:
    normalized = name.lower()
    if normalized in {"heuristic", "local"}:
        return HeuristicRerankProvider()
    if normalized == "cohere":
        return CohereRerankProvider(model=model or "rerank-v4.0-pro")
    raise ValueError(f"Unknown rerank provider: {name}")


def evaluate_rerankers(
    mode: str,
    qrels_path: Path,
    rerankers: list[str] | None = None,
    base_policy: str = "rag",
    candidate_k: int = 25,
    k: int = 10,
    model: str | None = None,
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
        )
        for provider_name in (rerankers or DEFAULT_RERANKERS)
    ]
    report = {
        "report_id": "rerank_evaluation_v1",
        "mode": mode,
        "benchmark_id": qrels.get("benchmark_id"),
        "base_policy": base_policy,
        "candidate_k": candidate_k,
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
) -> dict:
    started = time.perf_counter()
    try:
        provider = get_rerank_provider(provider_name, model)
        query_reports = [
            evaluate_query(provider, mode, query, base_policy, candidate_k, k)
            for query in queries
        ]
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": provider_name,
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "provider": provider.provider,
        "model": provider.model,
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "query_count": len(query_reports),
        "summary": provider_summary(query_reports, k),
        "queries": query_reports,
    }


def evaluate_query(provider: RerankProvider, mode: str, query: dict, base_policy: str, candidate_k: int, k: int) -> dict:
    retrieval = run_retrieval(query["query"], base_policy, mode, candidate_k)
    candidates = retrieval.get("results") or []
    documents = [document_text(result) for result in candidates]
    reranked = provider.rerank(query["query"], documents, top_n=min(k, len(documents)))
    ranked_results = [candidates[item.index] for item in reranked if 0 <= item.index < len(candidates)]
    ranked_ids = [result["chunk_id"] for result in ranked_results]
    metrics = evaluate_ranking(ranked_ids, query["relevant"], k)
    return {
        "id": query["id"],
        "query": query["query"],
        "candidate_count": len(candidates),
        "metrics": metrics,
        "ranked_chunk_ids": ranked_ids[:k],
        "relevant_hit_ids": [chunk_id for chunk_id in ranked_ids[:k] if chunk_id in query["relevant"]],
        "top_results": [
            {
                "rank": index,
                "chunk_id": result["chunk_id"],
                "title": result["title"],
                "source_name": result["source_name"],
                "rerank_score": round(reranked[index - 1].score, 6) if index - 1 < len(reranked) else 0.0,
            }
            for index, result in enumerate(ranked_results[:k], start=1)
        ],
    }


def document_text(result: dict) -> str:
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
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

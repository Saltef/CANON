from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canon.config import load_settings
from canon.eval.llm_judge import heuristic_answer_judgment, llm_answer_judgment
from canon.eval.external_ir import load_qrels
from canon.eval.rerank_evaluation import evaluate_rerankers
from canon.generation.providers import get_generation_provider
from canon.product import service


EMBEDDING_PROVIDERS = [
    ("local", "hashed-semantic-v1"),
    ("sentence-transformers", "BAAI/bge-small-en-v1.5"),
    ("openrouter", "openai/text-embedding-3-small"),
    ("openrouter", "baai/bge-m3"),
    ("openrouter", "qwen/qwen3-embedding-8b"),
    ("cohere", "embed-v4.0"),
]

RERANKERS = [
    "heuristic",
    "cohere:rerank-v4.0-pro",
    "openrouter:cohere/rerank-v3.5",
]

BASE_GENERATION_MODELS = [
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
]

KIMI_GENERATION_MODELS = [
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.7-code",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2-thinking",
]

GENERATION_MODELS = BASE_GENERATION_MODELS + KIMI_GENERATION_MODELS

PUBLIC_GENERATION_QUERIES = [
    "0-dimensional biomaterials show inductive properties.",
    "Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.",
]
PUBLIC_QRELS_PATH = "gold/beir_scifact_full_qrels.json"
PUBLIC_QRELS_PROBE_LIMIT = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON product model matrix benchmarks.")
    parser.add_argument("--qrels", default="gold/beir_scifact_mini_qrels.json")
    parser.add_argument("--mode", default="beir_scifact_mini")
    parser.add_argument("--candidate-k", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--app-mode", default=service.DEFAULT_PRODUCTION_MODE)
    parser.add_argument("--generation-query", action="append", default=None)
    parser.add_argument("--judge-provider", default="openrouter")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--candidate-scope", default=service.DEFAULT_PRODUCTION_CANDIDATE_SCOPE)
    parser.add_argument("--lexical-window-k", type=int, default=service.DEFAULT_PRODUCTION_LEXICAL_WINDOW_K)
    parser.add_argument("--public-qrels", default=PUBLIC_QRELS_PATH)
    parser.add_argument("--public-qrels-limit", type=int, default=PUBLIC_QRELS_PROBE_LIMIT)
    parser.add_argument("--generation-model", action="append", default=None)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-public-qrels", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = load_settings()
    started = time.perf_counter()
    rerank_rows = []
    if not args.skip_rerank:
        for vector_provider, vector_model in EMBEDDING_PROVIDERS:
            for fusion in ["union", "weighted_bm25_dense"]:
                report = evaluate_rerankers(
                    mode=args.mode,
                    qrels_path=Path(args.qrels),
                    rerankers=RERANKERS,
                    candidate_k=args.candidate_k,
                    k=args.k,
                    lexical_k=args.candidate_k,
                    vector_k=args.candidate_k,
                    vector_provider=vector_provider,
                    vector_model=vector_model,
                    fusion=fusion,
                    query_cache_dir=settings.reports_dir / "product_model_matrix_rerank_cache",
                )
                for row in report.get("leaderboard", []):
                    rerank_rows.append(
                        {
                            "benchmark": report.get("benchmark_id"),
                            "candidate_vector_provider": vector_provider,
                            "candidate_vector_model": vector_model,
                            "fusion": fusion,
                            "candidate_k": args.candidate_k,
                            "reranker_provider": row.get("provider"),
                            "reranker_model": row.get("model"),
                            "provider_id": row.get("provider_id"),
                            "nDCG@10": row.get("primary_score"),
                            "Recall@10": row.get("Recall@k"),
                            "MRR@10": row.get("MRR@k"),
                            "elapsed_ms": row.get("elapsed_ms"),
                        }
                    )
    candidate_k_rows = []
    if not args.skip_rerank:
        for candidate_k in [12, 25, 50]:
            report = evaluate_rerankers(
                mode=args.mode,
                qrels_path=Path(args.qrels),
                rerankers=RERANKERS,
                candidate_k=candidate_k,
                k=args.k,
                lexical_k=candidate_k,
                vector_k=candidate_k,
                vector_provider="openrouter",
                vector_model="qwen/qwen3-embedding-8b",
                fusion="weighted_bm25_dense",
                query_cache_dir=settings.reports_dir / "product_model_matrix_rerank_cache",
            )
            for row in report.get("leaderboard", []):
                candidate_k_rows.append(
                    {
                        "candidate_k": candidate_k,
                        "reranker_provider": row.get("provider"),
                        "reranker_model": row.get("model"),
                        "nDCG@10": row.get("primary_score"),
                        "Recall@10": row.get("Recall@k"),
                        "MRR@10": row.get("MRR@k"),
                    }
                )
    generation_queries = args.generation_query or PUBLIC_GENERATION_QUERIES
    generation_models = args.generation_model or GENERATION_MODELS
    generation_rows = [] if args.skip_generation else generation_matrix(
        app_mode=args.app_mode,
        generation_queries=generation_queries,
        generation_models=generation_models,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        skip_judge=args.skip_judge,
        candidate_scope=args.candidate_scope,
        lexical_window_k=args.lexical_window_k,
    )
    public_qrels_rows = [] if args.skip_public_qrels or args.skip_generation else public_qrels_probe(
        app_mode=args.app_mode,
        qrels_path=Path(args.public_qrels),
        limit=args.public_qrels_limit,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        skip_judge=args.skip_judge,
        candidate_scope=args.candidate_scope,
        lexical_window_k=args.lexical_window_k,
    )
    wrong_corpus_guard = None if args.skip_generation else wrong_corpus_check()
    report = {
        "report_id": "product_model_matrix_v1",
        "created_at": service.utc_now(),
        "benchmark": {
            "mode": args.mode,
            "qrels_path": args.qrels,
            "metric_k": args.k,
            "rerank_skipped": args.skip_rerank,
            "human_review_required": True,
        },
        "public_app_benchmark": {
            "mode": args.app_mode,
            "qrels_path": args.public_qrels or public_qrels_for_mode(args.app_mode),
            "generation_queries": generation_queries,
            "generation_models": generation_models,
            "candidate_scope": args.candidate_scope,
            "lexical_window_k": args.lexical_window_k,
        },
        "llm_judge": {
            "enabled": not args.skip_judge and not args.skip_generation,
            "provider": None if args.skip_judge or args.skip_generation else args.judge_provider,
            "model": None if args.skip_judge or args.skip_generation else args.judge_model,
            "scope": "all generation_matrix rows",
            "human_review_required": True,
        },
        "rerank_matrix": rerank_rows,
        "candidate_k_sensitivity": candidate_k_rows,
        "generation_matrix": generation_rows,
        "public_qrels_evaluation": {
            "mode": args.app_mode,
            "qrels_path": args.public_qrels,
            "query_limit": args.public_qrels_limit,
            "rows": public_qrels_rows,
            "summary": public_qrels_summary(public_qrels_rows),
            "human_review_required": True,
        },
        "wrong_corpus_guard": wrong_corpus_guard,
        "recommended_combination": recommend(rerank_rows, generation_rows),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "limitations": [
            "The default matrix qrels are a quick public benchmark slice unless --mode beir_scifact_full is selected.",
            "LLM judge labels are provisional triage, not human factual review.",
            "The product default is optimized for a usable inspection experience, not autonomous conclusions.",
        ],
    }
    output_path = Path(args.output) if args.output else settings.reports_dir / "product_model_matrix_v1.json"
    if not output_path.is_absolute():
        output_path = settings.root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(compact_report(report), indent=2))


def generation_matrix(
    app_mode: str,
    generation_queries: list[str],
    generation_models: list[str],
    judge_provider: str,
    judge_model: str | None,
    skip_judge: bool,
    candidate_scope: str,
    lexical_window_k: int,
) -> list[dict[str, Any]]:
    rows = []
    judge = None if skip_judge else get_generation_provider(judge_provider, judge_model)
    for query in generation_queries:
        for model in generation_models:
            report = service.production_evidence_workbench(
                {
                    "query": query,
                    "mode": app_mode,
                    "top_k": service.DEFAULT_PRODUCTION_TOP_K,
                    "candidate_k": service.DEFAULT_PRODUCTION_CANDIDATE_K,
                    "retrieval_engine": "model_candidate_pool",
                    "retrieval_provider": "openrouter",
                    "retrieval_model": "qwen/qwen3-embedding-8b",
                    "candidate_scope": candidate_scope,
                    "lexical_window_k": lexical_window_k,
                    "reranker_provider": "cohere",
                    "reranker_model": "rerank-v4.0-pro",
                    "generator_provider": "openrouter",
                    "generator_model": model,
                    "fusion": "weighted_bm25_dense",
                    "write_telemetry": False,
                }
            )
            draft = report.get("draft_brief") or {}
            text = draft.get("text") or ""
            row = {
                "query": query,
                "mode": app_mode,
                "model": model,
                "session_status": report.get("status"),
                "relevance_gate": (report.get("relevance_gate") or {}).get("status"),
                "draft_status": draft.get("status"),
                "citation_count": draft.get("citation_count"),
                "usable_evidence_count": (report.get("evidence_packet") or {}).get("usable_evidence_count"),
                "diagnostic_candidate_count": draft.get("diagnostic_candidate_count"),
                "used_citation_ids": sorted(set(re.findall(r"\bC\d+\b", text))),
                "text_chars": len(text),
                "text_preview": text[:240],
            }
            row["llm_judge"] = (
                {"status": "skipped"}
                if skip_judge
                else judge_product_session(report, judge, judge_provider, judge_model, category="product_model_matrix")
            )
            rows.append(row)
    return rows


def public_qrels_probe(
    app_mode: str,
    qrels_path: Path,
    limit: int,
    judge_provider: str,
    judge_model: str | None,
    skip_judge: bool,
    candidate_scope: str,
    lexical_window_k: int,
) -> list[dict[str, Any]]:
    qrels = load_qrels(qrels_path)
    rows = []
    judge = None if skip_judge else get_generation_provider(judge_provider, judge_model)
    for query in qrels.get("queries", [])[: max(0, limit)]:
        report = service.production_evidence_workbench(
            {
                "query": query["query"],
                "mode": app_mode,
                "top_k": service.DEFAULT_PRODUCTION_TOP_K,
                "candidate_k": service.DEFAULT_PRODUCTION_CANDIDATE_K,
                "retrieval_engine": "model_candidate_pool",
                "retrieval_provider": "openrouter",
                "retrieval_model": "qwen/qwen3-embedding-8b",
                "candidate_scope": candidate_scope,
                "lexical_window_k": lexical_window_k,
                "reranker_provider": "cohere",
                "reranker_model": "rerank-v4.0-pro",
                "generator_provider": "openrouter",
                "generator_model": "openai/gpt-4.1-mini",
                "fusion": "weighted_bm25_dense",
                "write_telemetry": False,
            }
        )
        evidence_ids = [row.get("chunk_id") for row in report.get("evidence_cards", [])]
        relevant = set(query.get("relevant", {}))
        hits = [chunk_id for chunk_id in evidence_ids if chunk_id in relevant]
        draft = report.get("draft_brief") or {}
        rows.append(
            {
                "query_id": query.get("id"),
                "query": query.get("query"),
                "session_status": report.get("status"),
                "relevance_gate": (report.get("relevance_gate") or {}).get("status"),
                "draft_status": draft.get("status"),
                "retrieved_count": len(evidence_ids),
                "relevant_count": len(relevant),
                "hit_count": len(hits),
                "hit_at_k": bool(hits),
                "recall_at_k": round(len(hits) / len(relevant), 6) if relevant else 0.0,
                "hit_ids": hits,
                "llm_judge": (
                    {"status": "skipped"}
                    if skip_judge
                    else judge_product_session(report, judge, judge_provider, judge_model, category="public_qrels_probe")
                ),
            }
        )
    return rows


def public_qrels_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0, "hit_rate": 0.0, "mean_recall_at_k": 0.0}
    return {
        "query_count": len(rows),
        "hit_rate": round(sum(1 for row in rows if row.get("hit_at_k")) / len(rows), 6),
        "mean_recall_at_k": round(
            sum(float(row.get("recall_at_k") or 0.0) for row in rows) / len(rows),
            6,
        ),
        "judged_count": sum(1 for row in rows if (row.get("llm_judge") or {}).get("status") == "judged"),
    }


def judge_product_session(
    session: dict[str, Any],
    provider: Any | None,
    provider_name: str,
    model: str | None,
    category: str,
) -> dict[str, Any]:
    draft = session.get("draft_brief") or {}
    record = {
        "id": session.get("session_id") or f"matrix_{abs(hash((session.get('query'), draft.get('generator', {}).get('model'))))}",
        "category": category,
        "query": session.get("query"),
        "answer_response": {
            "answer": draft.get("text") or draft.get("abstention") or "",
            "citations": draft.get("citations") or [],
            "evidence": usable_evidence_cards(session),
            "support_assessment": draft.get("support_assessment") or {},
            "query_diagnostics": session.get("query_lingo") or {},
        },
    }
    if provider is None or provider_name == "heuristic":
        judgment = heuristic_answer_judgment(record)
        judgment["status"] = "judged"
        return judgment
    judgment = llm_answer_judgment(provider, record)
    judgment["status"] = "judged"
    judgment["judge_provider"] = provider_name
    judgment["judge_model"] = model
    return judgment


def usable_evidence_cards(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": row.get("evidence_id"),
            "title": row.get("title"),
            "source_name": row.get("source_name"),
            "text": row.get("text"),
            "relevance": row.get("relevance"),
        }
        for row in session.get("evidence_cards", [])
        if (row.get("relevance") or {}).get("usage_status") == "usable_candidate_evidence"
    ]


def wrong_corpus_check() -> dict[str, Any]:
    report = service.production_evidence_workbench(
        {
            "query": "What are the AI risks?",
            "mode": service.DEFAULT_MODE,
            "top_k": service.DEFAULT_PRODUCTION_TOP_K,
            "retrieval_engine": "canon_synthesis",
            "generator_provider": "openrouter",
            "generator_model": "openai/gpt-4.1-mini",
            "write_telemetry": False,
        }
    )
    draft = report.get("draft_brief") or {}
    return {
        "session_status": report.get("status"),
        "relevance_gate": (report.get("relevance_gate") or {}).get("status"),
        "draft_status": draft.get("status"),
        "citation_count": draft.get("citation_count"),
        "abstention": draft.get("abstention"),
    }


def recommend(rerank_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rerank_rows,
        key=lambda row: (
            float(row.get("nDCG@10") or 0.0),
            float(row.get("Recall@10") or 0.0),
            float(row.get("MRR@10") or 0.0),
        ),
        reverse=True,
    )
    generation = next(
        (
            row
            for row in generation_rows
            if row.get("model") == "openai/gpt-4.1-mini" and row.get("draft_status") == "model_draft_ready"
        ),
        None,
    )
    generation = generation or next(
        (row for row in generation_rows if row.get("draft_status") == "model_draft_ready"),
        None,
    )
    return {
        "retrieval_engine": "model_candidate_pool",
        "candidate_k": service.DEFAULT_PRODUCTION_CANDIDATE_K,
        "candidate_scope": service.DEFAULT_PRODUCTION_CANDIDATE_SCOPE,
        "lexical_window_k": service.DEFAULT_PRODUCTION_LEXICAL_WINDOW_K,
        "retrieval_provider": "openrouter",
        "retrieval_model": "qwen/qwen3-embedding-8b",
        "fusion": "weighted_bm25_dense",
        "reranker_provider": "cohere",
        "reranker_model": "rerank-v4.0-pro",
        "generator_provider": "openrouter",
        "generator_model": "openai/gpt-4.1-mini",
        "qrels_matrix_best_row": ranked[0] if ranked else None,
        "generation_default_row": generation,
        "human_review_required": True,
        "reason": (
            "Qwen3 embeddings led the semantic benchmark; Cohere rerank led candidate-k sensitivity; "
            "GPT-4.1 mini passed citation-bearing draft checks and receives provisional LLM-judge review."
        ),
    }


def public_qrels_for_mode(mode: str) -> str | None:
    if mode == "beir_scifact_full":
        return "gold/beir_scifact_full_qrels.json"
    if mode == "beir_scifact_mini":
        return "gold/beir_scifact_mini_qrels.json"
    if mode == "beir_nfcorpus_full":
        return "gold/beir_nfcorpus_full_qrels.json"
    return None


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "public_app_benchmark": report["public_app_benchmark"],
        "llm_judge": report["llm_judge"],
        "public_qrels_evaluation": report["public_qrels_evaluation"],
        "rerank_top_10": sorted(
            report["rerank_matrix"],
            key=lambda row: (
                float(row.get("nDCG@10") or 0.0),
                float(row.get("Recall@10") or 0.0),
                float(row.get("MRR@10") or 0.0),
            ),
            reverse=True,
        )[:10],
        "candidate_k_sensitivity": report["candidate_k_sensitivity"],
        "generation_matrix": report["generation_matrix"],
        "wrong_corpus_guard": report["wrong_corpus_guard"],
        "recommended_combination": report["recommended_combination"],
        "elapsed_ms": report["elapsed_ms"],
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.embeddings.providers import HashedEmbeddingProvider
from canon.eval.external_ir import load_qrels
from canon.eval.ir_metrics import evaluate_ranking, mean_metric
from canon.eval.rerank_evaluation import PARENT_QRELS_PROTOCOL, effective_qrels
from canon.product.stage1_repeat_spread import (
    comparison_decision,
    paired_delta,
    query_work_level_summary,
    round_metric,
    summarize_runs,
)
from canon.retrieval.clusters import load_cluster_assignments
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus


DEFAULT_MODE = "beir_nfcorpus_stage1_title_preserve"
DEFAULT_QRELS = "gold/beir_nfcorpus_stage1_title_preserve_qrels.json"
DEFAULT_CANON_REPORT = "reports/stage1_fixed_qrels_v2_repeat_spread.json"
DEFAULT_REPORT_JSON = "reports/llamaindex_stage1_head_to_head.json"
DEFAULT_REPORT_MD = "reports/llamaindex_stage1_head_to_head.md"


@dataclass(frozen=True)
class LlamaIndexRepeatConfig:
    id: str
    label: str
    parent_qrels: bool
    qrels_scope: str


DEFAULT_CONFIGS = [
    LlamaIndexRepeatConfig(
        id="llamaindex_vector_original_qrels",
        label="LlamaIndex VectorStoreIndex + CANON hash embeddings, original qrels",
        parent_qrels=False,
        qrels_scope="original_chunk_qrels",
    ),
    LlamaIndexRepeatConfig(
        id="llamaindex_vector_fixed_parent_qrels",
        label="LlamaIndex VectorStoreIndex + CANON hash embeddings, fixed parent qrels",
        parent_qrels=True,
        qrels_scope="full_corpus_parent_map_v2",
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

    class CanonHashedLlamaIndexEmbedding(BaseEmbedding):
        _provider: Any = PrivateAttr()

        def __init__(self) -> None:
            super().__init__(model_name="canon-local-hashed-semantic-v1")
            self._provider = HashedEmbeddingProvider()

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._provider.embed([text])[0].vector

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._provider.embed([query])[0].vector

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return CanonHashedLlamaIndexEmbedding


def chunk_to_work_map(documents: list[RetrievalDocument]) -> dict[str, str]:
    return {document.chunk_id: document.work_id for document in documents}


def build_nodes(documents: list[RetrievalDocument]) -> list[Any]:
    _index, text_node_cls = require_llamaindex()
    nodes = []
    for document in documents:
        nodes.append(
            text_node_cls(
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
        )
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


def build_retriever(documents: list[RetrievalDocument], *, similarity_top_k: int):
    index_cls, _text_node_cls = require_llamaindex()
    embedding_cls = require_base_embedding()
    embed_model = embedding_cls()
    index = index_cls(nodes=build_nodes(documents), embed_model=embed_model, show_progress=False)
    return index.as_retriever(similarity_top_k=similarity_top_k)


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
        retriever = build_retriever(documents, similarity_top_k=candidate_k)
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
            "configuration": config.id,
            "repeat": repeat,
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        **asdict(config),
        "configuration": config.id,
        "repeat": repeat,
        "status": "ok",
        "provider_id": "llamaindex:vectorstoreindex:canon-local-hash",
        "elapsed_ms": elapsed_ms,
        "metrics": extract_run_metrics(query_reports, qrels, chunk_map, k),
        "summary": provider_summary(query_reports, k),
        "query_count": len(query_reports),
        "queries": query_reports,
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
    canon_report: dict[str, Any],
    configs: list[LlamaIndexRepeatConfig],
    repeats: int,
    mode: str,
    qrels_path: Path,
    qrels: dict[str, Any],
    k: int,
    candidate_k: int,
    include_query_details: bool = False,
) -> dict[str, Any]:
    llama_summaries = summarize_runs(llama_runs, repeat_configs_for_summary(configs), k)
    combined_runs = list(canon_runs) + list(llama_runs)
    comparisons = [
        paired_delta(
            combined_runs,
            treatment_config="llamaindex_vector_original_qrels",
            baseline_config="base_qwen_cohere",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            combined_runs,
            treatment_config="llamaindex_vector_fixed_parent_qrels",
            baseline_config="auto_query_expansion_fixed_parent_qrels",
            metric=f"nDCG@{k}",
        ),
        paired_delta(
            combined_runs,
            treatment_config="llamaindex_vector_fixed_parent_qrels",
            baseline_config="auto_query_expansion_fixed_parent_qrels_parent_neighborhood",
            metric=f"nDCG@{k}",
        ),
    ]
    all_summaries = summarize_runs(combined_runs, canon_and_llama_summary_configs(configs), k)
    claim_checks = [comparison_decision(comparison, all_summaries) for comparison in comparisons]
    completed = sum(1 for run in llama_runs if run.get("status") == "ok")
    expected = repeats * len(configs)
    verdict = standing_verdict(claim_checks)
    return {
        "report_id": "llamaindex_stage1_head_to_head_v1",
        "created_at": time.strftime("%Y-%m-%d"),
        "status": "head_to_head_complete" if completed == expected else "head_to_head_partial",
        "mode": mode,
        "qrels_path": display_path(qrels_path),
        "query_count": len(qrels.get("queries", [])),
        "label_count": sum(len(query.get("relevant", {})) for query in qrels.get("queries", [])),
        "metric_k": k,
        "repeat_count_requested": repeats,
        "completed_llamaindex_run_count": completed,
        "expected_llamaindex_run_count": expected,
        "protocol": {
            "benchmark": "BEIR NFCorpus 30-query public slice expanded to CANON chunks",
            "parent_qrels_protocol": PARENT_QRELS_PROTOCOL,
            "paired_repeat_protocol": "CANON repeat numbers are paired with LlamaIndex repeat numbers 1..N.",
            "primary_question": (
                "Does LlamaIndex's off-the-shelf VectorStoreIndex retrieval path beat "
                "CANON on the same fixed-qrels NFCorpus pilot?"
            ),
        },
        "llamaindex_stack": {
            "framework": "llama-index-core",
            "index": "VectorStoreIndex",
            "retriever": "index.as_retriever(similarity_top_k=candidate_k)",
            "candidate_k": candidate_k,
            "embedding": "CANON local hashed-semantic-v1 adapter",
            "llm": "disabled",
            "boundary": (
                "This measures LlamaIndex's default in-memory vector index and retriever plumbing. "
                "It does not use LlamaIndex's hosted OpenAI default embedding because no OpenAI key "
                "is configured and CANON keeps public benchmark reruns reproducible by default."
            ),
        },
        "include_query_details": include_query_details,
        "canon_report": canon_report,
        "llamaindex_configuration_summaries": llama_summaries,
        "all_configuration_summaries": all_summaries,
        "paired_deltas": comparisons,
        "claim_checks": claim_checks,
        "standing_verdict": verdict,
        "llamaindex_runs": [
            compact_run(run, include_query_details=include_query_details)
            for run in sorted(llama_runs, key=lambda row: (row.get("repeat", 0), str(row.get("configuration"))))
        ],
    }


def repeat_configs_for_summary(configs: list[LlamaIndexRepeatConfig]):
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


def standing_verdict(claim_checks: list[dict[str, Any]]) -> dict[str, Any]:
    primary = claim_checks[0] if claim_checks else {}
    mean_delta = float(primary.get("mean_delta", 0.0))
    standing = str(primary.get("standing") or "unknown")
    if standing in {"stands_under_repeat_spread_check", "stands_as_repeatable_but_tiny"}:
        winner = "llamaindex" if mean_delta > 0 else "canon" if mean_delta < 0 else "tie"
    elif standing == "does_not_stand_direction_unstable":
        winner = "no_stable_winner"
    elif standing == "insufficient_repeat_count":
        winner = "incomplete"
    else:
        winner = "no_material_winner"
    return {
        "primary_metric": primary.get("metric", "nDCG@10"),
        "primary_comparison": primary.get("comparison"),
        "winner": winner,
        "standing": standing,
        "mean_delta_pp": primary.get("mean_delta_pp", 0.0),
        "reason": primary.get("reason", "No primary comparison was available."),
        "publication_boundary": (
            "Publish the result either way, but describe it as a 30-query NFCorpus pilot "
            "rather than a full BEIR leaderboard claim."
        ),
    }


def compact_run(run: dict[str, Any], *, include_query_details: bool) -> dict[str, Any]:
    if include_query_details:
        return run
    return {key: value for key, value in run.items() if key != "queries"}


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# LlamaIndex vs CANON Stage 1 Head-to-Head",
        "",
        f"Status: `{report['status']}`",
        "",
        (
            f"Protocol: {report['query_count']} NFCorpus queries, "
            f"{report['label_count']} fixed labels, {report['repeat_count_requested']} repeats per LlamaIndex configuration."
        ),
        "",
        f"Standing verdict: `{report['standing_verdict']['winner']}` "
        f"({report['standing_verdict']['standing']}, "
        f"{report['standing_verdict']['mean_delta_pp']} pp on {report['standing_verdict']['primary_metric']}).",
        "",
        "## LlamaIndex Stack",
        "",
        f"- Index: `{report['llamaindex_stack']['index']}`",
        f"- Retriever: `{report['llamaindex_stack']['retriever']}`",
        f"- Embedding: `{report['llamaindex_stack']['embedding']}`",
        f"- Boundary: {report['llamaindex_stack']['boundary']}",
        "",
        "## LlamaIndex Repeat Spread",
        "",
        "| Configuration | Runs | nDCG@10 mean | nDCG spread pp | MAP@10 mean | Candidate recall mean | Work-hit mean | Zero-hit queries |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in report["llamaindex_configuration_summaries"]:
        metrics = summary["metrics"]
        lines.append(
            "| {label} | {runs} | {ndcg_mean:.6f} | {ndcg_spread:.3f} | {map_mean:.6f} | {candidate_mean:.6f} | {work_mean:.6f} | {zero} |".format(
                label=summary["label"],
                runs=summary["completed_runs"],
                ndcg_mean=metrics["nDCG@10"]["mean"],
                ndcg_spread=metrics["nDCG@10"]["spread_pp"],
                map_mean=metrics["MAP@10"]["mean"],
                candidate_mean=metrics["candidate_recall_mean"]["mean"],
                work_mean=metrics["work_level_hit_rate_mean"]["mean"],
                zero=", ".join(
                    str(value)
                    for value in summary.get("zero_candidate_recall_queries_values", [])
                    if value is not None
                )
                or "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Paired nDCG@10 Deltas",
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
            "## Interpretation Boundary",
            "",
            f"- {report['standing_verdict']['reason']}",
            f"- {report['standing_verdict']['publication_boundary']}",
            "- This is a framework retrieval comparison, not a generation or answer-faithfulness comparison.",
            "",
            "The compact machine-readable report is in `reports/llamaindex_stage1_head_to_head.json`.",
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--k", type=int, default=10)
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
    qrels = load_qrels(qrels_path)
    documents = load_processed_corpus(
        settings.data_dir,
        args.mode,
        cluster_assignments=load_cluster_assignments(settings.data_dir, args.mode),
    )
    chunk_map = chunk_to_work_map(documents)
    canon_runs, canon_report = load_canon_runs(canon_report_path)
    runs = [] if args.no_resume else load_existing_runs(report_json)
    print(
        f"[llamaindex-baseline] mode={args.mode} repeats={args.repeats} "
        f"queries={len(qrels.get('queries', []))} documents={len(documents)} resume={not args.no_resume}",
        flush=True,
    )
    for repeat in range(1, args.repeats + 1):
        for config in DEFAULT_CONFIGS:
            if (config.id, repeat) in completed_ok_keys(runs):
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
            runs = replace_run(runs, run)
            report = build_report(
                llama_runs=runs,
                canon_runs=canon_runs,
                canon_report=canon_report,
                configs=DEFAULT_CONFIGS,
                repeats=args.repeats,
                mode=args.mode,
                qrels_path=qrels_path,
                qrels=qrels,
                k=args.k,
                candidate_k=args.candidate_k,
                include_query_details=args.include_query_details,
            )
            write_head_to_head(report, report_json, report_md)
            if run.get("status") == "ok":
                print(
                    f"[llamaindex-baseline] ok repeat={repeat} config={config.id} "
                    f"nDCG@{args.k}={run['metrics'][f'nDCG@{args.k}']:.6f} "
                    f"candidate_recall={run['metrics']['candidate_recall_mean']:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[llamaindex-baseline] unavailable repeat={repeat} config={config.id} "
                    f"reason={run.get('reason')}",
                    flush=True,
                )
    final_report = build_report(
        llama_runs=runs,
        canon_runs=canon_runs,
        canon_report=canon_report,
        configs=DEFAULT_CONFIGS,
        repeats=args.repeats,
        mode=args.mode,
        qrels_path=qrels_path,
        qrels=qrels,
        k=args.k,
        candidate_k=args.candidate_k,
        include_query_details=args.include_query_details,
    )
    write_head_to_head(final_report, report_json, report_md)
    print(json.dumps(final_report["standing_verdict"], indent=2), flush=True)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.harness import load_gold
from canon.eval.qrels import validate_qrels
from canon.retrieval.experiment import run as run_retrieval


DEFAULT_POLICIES = ["lexical", "semantic", "rag", "focus_diverse", "source_quality_heavy", "recency_heavy"]
QRELS_REVIEW_FIELDS = [
    "query_id",
    "query",
    "query_type",
    "chunk_id",
    "title",
    "source_name",
    "year",
    "cluster_id",
    "best_rank",
    "best_score",
    "retrieved_by_policies",
    "policy_ranks",
    "preview",
    "relevance",
    "reviewer_notes",
]


def build_qrels_review_packet(
    mode: str,
    queries_path: Path | None = None,
    policies: list[str] | None = None,
    top_k: int = 10,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    queries = load_gold(queries_path or settings.root / "gold" / "seed_queries.json")
    policy_ids = policies or DEFAULT_POLICIES
    records = [
        query_record(query=query, mode=mode, policies=policy_ids, top_k=top_k)
        for query in queries
    ]
    report = {
        "report_id": "qrels_review_tasks_v1",
        "mode": mode,
        "status": "ready_for_human_relevance_review",
        "benchmark_purpose": "Create graded qrels for retrieval and embedding-model evaluation.",
        "policies": policy_ids,
        "top_k": top_k,
        "query_count": len(records),
        "candidate_count": sum(len(record["candidates"]) for record in records),
        "instructions": review_instructions(),
        "records": records,
    }
    if write_report:
        json_path, csv_path, md_path = artifact_paths(settings.reports_dir, mode)
        write_json(json_path, report)
        export_review_csv(report, csv_path)
        md_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def query_record(query: dict[str, Any], mode: str, policies: list[str], top_k: int) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for policy in policies:
        retrieval = run_retrieval(query["query"], policy, mode, top_k)
        for result in retrieval.get("results", [])[:top_k]:
            chunk_id = str(result["chunk_id"])
            row = candidates.setdefault(chunk_id, candidate_row(query, result))
            row["retrieved_by_policies"].append(policy)
            row["policy_ranks"][policy] = int(result.get("rank") or top_k)
            row["policy_scores"][policy] = float(result.get("final_score") or 0.0)
            row["best_rank"] = min(int(row["best_rank"]), int(result.get("rank") or top_k))
            row["best_score"] = max(float(row["best_score"]), float(result.get("final_score") or 0.0))
    ordered = sorted(
        candidates.values(),
        key=lambda row: (row["best_rank"], -row["best_score"], row["chunk_id"]),
    )
    return {
        "query_id": str(query.get("id")),
        "query": str(query.get("query")),
        "query_type": query.get("type"),
        "importance_expectation": query.get("importance_expectation"),
        "candidate_count": len(ordered),
        "candidates": ordered,
    }


def candidate_row(query: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(query.get("id")),
        "query": str(query.get("query")),
        "query_type": query.get("type"),
        "chunk_id": str(result.get("chunk_id")),
        "work_id": result.get("work_id"),
        "title": result.get("title"),
        "source_name": result.get("source_name"),
        "year": result.get("year"),
        "cluster_id": result.get("cluster_id"),
        "best_rank": int(result.get("rank") or 9999),
        "best_score": float(result.get("final_score") or 0.0),
        "retrieved_by_policies": [],
        "policy_ranks": {},
        "policy_scores": {},
        "preview": result.get("preview"),
        "relevance": "",
        "reviewer_notes": "",
    }


def export_review_csv(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QRELS_REVIEW_FIELDS)
        writer.writeheader()
        for record in report.get("records", []):
            for candidate in record.get("candidates", []):
                writer.writerow(csv_row(record, candidate))
    return output_path


def csv_row(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": record.get("query_id"),
        "query": record.get("query"),
        "query_type": record.get("query_type"),
        "chunk_id": candidate.get("chunk_id"),
        "title": candidate.get("title"),
        "source_name": candidate.get("source_name"),
        "year": candidate.get("year"),
        "cluster_id": candidate.get("cluster_id"),
        "best_rank": candidate.get("best_rank"),
        "best_score": candidate.get("best_score"),
        "retrieved_by_policies": ";".join(candidate.get("retrieved_by_policies") or []),
        "policy_ranks": json.dumps(candidate.get("policy_ranks") or {}, sort_keys=True),
        "preview": candidate.get("preview"),
        "relevance": candidate.get("relevance", ""),
        "reviewer_notes": candidate.get("reviewer_notes", ""),
    }


def qrels_from_review_csv(
    csv_path: Path,
    benchmark_id: str,
    output_path: Path | None = None,
    benchmark_kind: str = "human_reviewed_qrels",
    description: str = "Human-reviewed CANON qrels from pooled retrieval candidates.",
) -> dict[str, Any]:
    rows = read_review_csv(csv_path)
    queries: dict[str, dict[str, Any]] = {}
    validation_errors = []
    for index, row in enumerate(rows, start=2):
        query_id = str(row.get("query_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not query_id or not chunk_id:
            validation_errors.append(error(index, "id", "query_id and chunk_id are required."))
            continue
        relevance_text = str(row.get("relevance") or "").strip()
        if relevance_text == "":
            continue
        try:
            relevance = float(relevance_text)
        except ValueError:
            validation_errors.append(error(index, "relevance", "relevance must be 0, 1, 2, or 3."))
            continue
        if relevance not in {0.0, 1.0, 2.0, 3.0}:
            validation_errors.append(error(index, "relevance", "relevance must be 0, 1, 2, or 3."))
            continue
        query = queries.setdefault(
            query_id,
            {
                "id": query_id,
                "query": str(row.get("query") or query_id),
                "relevant": {},
            },
        )
        if relevance > 0:
            query["relevant"][chunk_id] = relevance
    qrels = {
        "benchmark_id": benchmark_id,
        "benchmark_kind": benchmark_kind,
        "description": description,
        "queries": sorted(queries.values(), key=lambda item: item["id"]),
    }
    validation_errors.extend(validate_qrels(qrels))
    if validation_errors:
        return {
            "status": "failed_validation",
            "benchmark_id": benchmark_id,
            "csv_path": str(csv_path),
            "validation_errors": validation_errors,
            "qrels": qrels,
        }
    if output_path:
        write_json(output_path, qrels)
    return {
        "status": "qrels_written" if output_path else "qrels_ready",
        "benchmark_id": benchmark_id,
        "csv_path": str(csv_path),
        "output_path": str(output_path) if output_path else None,
        "query_count": len(qrels["queries"]),
        "judgment_count": sum(len(query["relevant"]) for query in qrels["queries"]),
        "validation_errors": [],
        "qrels": qrels,
    }


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def error(row_number: int, field: str, message: str) -> dict[str, Any]:
    return {"row": row_number, "field": field, "message": message}


def artifact_paths(reports_dir: Path, mode: str) -> tuple[Path, Path, Path]:
    return (
        reports_dir / f"qrels_review_tasks_{mode}.json",
        reports_dir / f"qrels_review_tasks_{mode}.csv",
        reports_dir / f"qrels_review_tasks_{mode}.md",
    )


def review_instructions() -> dict[str, Any]:
    return {
        "task": "Label whether each retrieved chunk is relevant evidence for the query.",
        "scale": {
            "3": "directly answers or strongly supports the query",
            "2": "partially useful evidence or important qualification",
            "1": "background/context only",
            "0": "not relevant, misleading, or too tangential",
            "blank": "not yet reviewed",
        },
        "review_guidance": [
            "Judge the chunk against the query, not against whether the final answer sounds good.",
            "Prefer direct evidence over broad topical similarity.",
            "Use 0 for semantically related but wrong-subdomain evidence.",
            "Leave notes when a result is useful for representation but weak as evidence.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- `{record['query_id']}` {record['candidate_count']} candidates: {record['query']}"
        for record in report.get("records", [])
    )
    return f"""# CANON Qrels Review Tasks

Status: **{report['status']}**

Mode: `{report['mode']}`

Policies: {', '.join(report['policies'])}

Top-k per policy: {report['top_k']}

Queries: {report['query_count']}

Pooled candidates: {report['candidate_count']}

## Reviewer Scale

- `3`: directly answers or strongly supports the query
- `2`: partially useful evidence or important qualification
- `1`: background/context only
- `0`: not relevant, misleading, or too tangential
- blank: not yet reviewed

## Queries

{rows}
"""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_policies(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_POLICIES
    policies = [piece.strip() for piece in value.split(",") if piece.strip()]
    return policies or DEFAULT_POLICIES


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "mode": report["mode"],
        "query_count": report["query_count"],
        "candidate_count": report["candidate_count"],
        "artifacts": {
            "json": f"reports/qrels_review_tasks_{report['mode']}.json",
            "csv": f"reports/qrels_review_tasks_{report['mode']}.csv",
            "md": f"reports/qrels_review_tasks_{report['mode']}.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or import human qrels review tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--mode", default="social_science_ir_v1_harvest10")
    prepare.add_argument("--queries", default=None)
    prepare.add_argument("--policies", default=None)
    prepare.add_argument("--top-k", type=int, default=10)
    import_csv = subparsers.add_parser("import-csv")
    import_csv.add_argument("--csv", required=True)
    import_csv.add_argument("--benchmark-id", required=True)
    import_csv.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = build_qrels_review_packet(
            mode=args.mode,
            queries_path=Path(args.queries) if args.queries else None,
            policies=parse_policies(args.policies),
            top_k=args.top_k,
        )
        print(json.dumps(cli_summary(report), indent=2))
        return
    result = qrels_from_review_csv(
        csv_path=Path(args.csv),
        benchmark_id=args.benchmark_id,
        output_path=Path(args.output),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "qrels"}, indent=2))


if __name__ == "__main__":
    main()

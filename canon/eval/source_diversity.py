from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.harness import load_gold
from canon.synthesis.answer import synthesize


DEFAULT_MIN_DISTINCT_SOURCES = 3
DEFAULT_MAX_DOMINANT_SOURCE_SHARE = 0.5
DEFAULT_MIN_CLUSTER_COUNT = 2
DEFAULT_PASS_RATE = 0.8


def evaluate_source_diversity(
    mode: str,
    queries_path: Path | None = None,
    policy: str = "rag",
    top_k: int = 10,
    min_distinct_sources: int = DEFAULT_MIN_DISTINCT_SOURCES,
    max_dominant_source_share: float = DEFAULT_MAX_DOMINANT_SOURCE_SHARE,
    min_cluster_count: int = DEFAULT_MIN_CLUSTER_COUNT,
) -> dict[str, Any]:
    settings = load_settings()
    queries = load_gold(queries_path or settings.root / "gold" / "seed_queries.json")
    rows = []
    for query in queries:
        report = synthesize(query=query["query"], policy=policy, mode=mode, top_k=top_k)
        rows.append(
            source_diversity_row(
                query=query,
                evidence=report.get("evidence") or [],
                top_k=top_k,
                min_distinct_sources=min_distinct_sources,
                max_dominant_source_share=max_dominant_source_share,
                min_cluster_count=min_cluster_count,
            )
        )
    summary = aggregate_rows(rows)
    status = "pass" if summary["query_pass_rate"] >= DEFAULT_PASS_RATE else "review"
    report = {
        "report_id": "source_diversity_v1",
        "mode": mode,
        "policy": policy,
        "top_k": top_k,
        "status": status,
        "thresholds": {
            "min_distinct_sources": min_distinct_sources,
            "max_dominant_source_share": max_dominant_source_share,
            "min_cluster_count": min_cluster_count,
            "minimum_query_pass_rate": DEFAULT_PASS_RATE,
        },
        "summary": summary,
        "queries": rows,
        "limitations": [
            "Source diversity is a retrieval-quality signal, not proof that the evidence is correct.",
            "Some corpora may not contain enough independent sources for every query; low diversity should trigger review rather than automatic synthesis confidence.",
            "Graph clusters are used as evidence neighborhoods and should not be interpreted as scholarly paradigms without human review.",
        ],
    }
    write_json(settings.reports_dir / f"source_diversity_{mode}_{policy}_k{top_k}.json", report)
    write_markdown(settings.reports_dir / f"source_diversity_{mode}_{policy}_k{top_k}.md", render_markdown(report))
    return report


def source_diversity_row(
    query: dict[str, Any],
    evidence: list[dict[str, Any]],
    top_k: int,
    min_distinct_sources: int,
    max_dominant_source_share: float,
    min_cluster_count: int,
) -> dict[str, Any]:
    top_evidence = evidence[:top_k]
    source_counts = Counter(source_key(item) for item in top_evidence)
    work_counts = Counter(str(item.get("work_id") or "unknown") for item in top_evidence)
    cluster_counts = Counter(str(item.get("cluster_id")) for item in top_evidence if item.get("cluster_id") is not None)
    evidence_count = len(top_evidence)
    dominant_source, dominant_count = source_counts.most_common(1)[0] if source_counts else ("none", 0)
    dominant_share = safe_ratio(dominant_count, evidence_count)
    distinct_sources = len(source_counts)
    distinct_works = len(work_counts)
    distinct_clusters = len(cluster_counts)
    warnings = diversity_warnings(
        evidence_count=evidence_count,
        distinct_sources=distinct_sources,
        dominant_source=dominant_source,
        dominant_share=dominant_share,
        distinct_clusters=distinct_clusters,
        min_distinct_sources=min_distinct_sources,
        max_dominant_source_share=max_dominant_source_share,
        min_cluster_count=min_cluster_count,
    )
    passed = not warnings
    return {
        "id": query.get("id"),
        "type": query.get("type"),
        "query": query.get("query"),
        "passed": passed,
        "evidence_count": evidence_count,
        "distinct_sources": distinct_sources,
        "distinct_works": distinct_works,
        "distinct_clusters": distinct_clusters,
        "dominant_source": dominant_source,
        "dominant_source_share": dominant_share,
        "source_counts": dict(sorted(source_counts.items())),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "warnings": warnings,
        "top_evidence": compact_evidence(top_evidence),
    }


def source_key(item: dict[str, Any]) -> str:
    return str(item.get("source_name") or item.get("title") or item.get("work_id") or "unknown").strip() or "unknown"


def diversity_warnings(
    evidence_count: int,
    distinct_sources: int,
    dominant_source: str,
    dominant_share: float,
    distinct_clusters: int,
    min_distinct_sources: int,
    max_dominant_source_share: float,
    min_cluster_count: int,
) -> list[dict[str, Any]]:
    warnings = []
    if evidence_count == 0:
        warnings.append({"id": "no_evidence", "message": "No evidence was retrieved."})
        return warnings
    if distinct_sources < min(min_distinct_sources, evidence_count):
        warnings.append(
            {
                "id": "low_source_diversity",
                "message": "Top evidence uses too few distinct sources.",
                "observed": distinct_sources,
                "threshold": min(min_distinct_sources, evidence_count),
            }
        )
    if dominant_share > max_dominant_source_share and evidence_count >= 3:
        warnings.append(
            {
                "id": "dominant_source_concentration",
                "message": f"One source dominates top evidence: {dominant_source}.",
                "observed": dominant_share,
                "threshold": max_dominant_source_share,
            }
        )
    if distinct_clusters < min(min_cluster_count, evidence_count):
        warnings.append(
            {
                "id": "low_graph_neighborhood_diversity",
                "message": "Top evidence covers too few graph clusters/evidence neighborhoods.",
                "observed": distinct_clusters,
                "threshold": min(min_cluster_count, evidence_count),
            }
        )
    return warnings


def compact_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item.get("rank"),
            "citation_id": item.get("citation_id"),
            "chunk_id": item.get("chunk_id"),
            "work_id": item.get("work_id"),
            "title": item.get("title"),
            "source_name": item.get("source_name"),
            "year": item.get("year"),
            "cluster_id": item.get("cluster_id"),
            "final_score": item.get("final_score"),
        }
        for item in evidence
    ]


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "query_count": 0,
            "query_pass_rate": 0.0,
            "average_distinct_sources": 0.0,
            "average_distinct_works": 0.0,
            "average_distinct_clusters": 0.0,
            "average_dominant_source_share": 0.0,
            "warning_counts": {},
        }
    warning_counts: Counter[str] = Counter()
    for row in rows:
        for warning in row["warnings"]:
            warning_counts[warning["id"]] += 1
    return {
        "query_count": len(rows),
        "query_pass_rate": round(sum(1 for row in rows if row["passed"]) / len(rows), 6),
        "average_distinct_sources": average(row["distinct_sources"] for row in rows),
        "average_distinct_works": average(row["distinct_works"] for row in rows),
        "average_distinct_clusters": average(row["distinct_clusters"] for row in rows),
        "average_dominant_source_share": average(row["dominant_source_share"] for row in rows),
        "warning_counts": dict(sorted(warning_counts.items())),
        "queries_with_warnings": [row["id"] for row in rows if row["warnings"]],
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 6)


def render_markdown(report: dict[str, Any]) -> str:
    warning_counts = "\n".join(
        f"- `{key}`: {value}" for key, value in report["summary"].get("warning_counts", {}).items()
    ) or "- None"
    rows = "\n".join(
        f"- `{row['id']}` {'PASS' if row['passed'] else 'REVIEW'}: "
        f"{row['distinct_sources']} sources, {row['distinct_works']} works, "
        f"{row['distinct_clusters']} clusters, dominant source share {row['dominant_source_share']}"
        for row in report["queries"]
    )
    return f"""# CANON Source Diversity Report

Status: **{report['status']}**

Mode: `{report['mode']}`

Policy: `{report['policy']}`

Top-k: {report['top_k']}

Query pass rate: {report['summary']['query_pass_rate']}

Average distinct sources: {report['summary']['average_distinct_sources']}

Average distinct works: {report['summary']['average_distinct_works']}

Average graph clusters: {report['summary']['average_distinct_clusters']}

Average dominant source share: {report['summary']['average_dominant_source_share']}

## Warning Counts

{warning_counts}

## Query Rows

{rows}
"""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "mode": report["mode"],
        "policy": report["policy"],
        "top_k": report["top_k"],
        "status": report["status"],
        "summary": report["summary"],
        "artifacts": {
            "json": f"reports/source_diversity_{report['mode']}_{report['policy']}_k{report['top_k']}.json",
            "md": f"reports/source_diversity_{report['mode']}_{report['policy']}_k{report['top_k']}.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source and graph-neighborhood diversity in retrieved evidence.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-distinct-sources", type=int, default=DEFAULT_MIN_DISTINCT_SOURCES)
    parser.add_argument("--max-dominant-source-share", type=float, default=DEFAULT_MAX_DOMINANT_SOURCE_SHARE)
    parser.add_argument("--min-cluster-count", type=int, default=DEFAULT_MIN_CLUSTER_COUNT)
    args = parser.parse_args()
    report = evaluate_source_diversity(
        mode=args.mode,
        queries_path=Path(args.queries) if args.queries else None,
        policy=args.policy,
        top_k=args.top_k,
        min_distinct_sources=args.min_distinct_sources,
        max_dominant_source_share=args.max_dominant_source_share,
        min_cluster_count=args.min_cluster_count,
    )
    print(json.dumps(cli_summary(report), indent=2))


if __name__ == "__main__":
    main()

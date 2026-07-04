from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


def validate_qrels(qrels: dict) -> list[dict]:
    issues = []
    if not qrels.get("benchmark_id"):
        issues.append(issue("missing_benchmark_id", "Qrels file must include benchmark_id."))
    queries = qrels.get("queries")
    if not isinstance(queries, list) or not queries:
        issues.append(issue("missing_queries", "Qrels file must include at least one query."))
        return issues
    seen_ids = set()
    for index, query in enumerate(queries):
        query_id = query.get("id")
        if not query_id:
            issues.append(issue("missing_query_id", f"Query at index {index} has no id."))
        elif query_id in seen_ids:
            issues.append(issue("duplicate_query_id", f"Duplicate query id: {query_id}."))
        seen_ids.add(query_id)
        if not query.get("query"):
            issues.append(issue("missing_query_text", f"Query {query_id or index} has no query text."))
        relevant = query.get("relevant")
        if not isinstance(relevant, dict) or not relevant:
            issues.append(issue("missing_relevance", f"Query {query_id or index} has no relevance judgments."))
            continue
        for doc_id, relevance in relevant.items():
            if not str(doc_id).strip():
                issues.append(issue("blank_doc_id", f"Query {query_id or index} contains a blank doc id."))
            try:
                value = float(relevance)
            except (TypeError, ValueError):
                issues.append(issue("invalid_relevance", f"Query {query_id or index} has non-numeric relevance for {doc_id}."))
                continue
            if value < 0:
                issues.append(issue("negative_relevance", f"Query {query_id or index} has negative relevance for {doc_id}."))
    return issues


def issue(identifier: str, message: str) -> dict:
    return {"id": identifier, "message": message}


def parse_trec_qrels(lines: list[str]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid TREC qrels line {line_number}: expected at least 4 columns.")
        query_id, _, doc_id, relevance = parts[:4]
        qrels.setdefault(query_id, {})[doc_id] = float(relevance)
    return qrels


def canonical_from_trec(
    lines: list[str],
    benchmark_id: str,
    queries: dict[str, str] | None = None,
    benchmark_kind: str = "external_qrels",
    description: str = "",
) -> dict:
    by_query = parse_trec_qrels(lines)
    return {
        "benchmark_id": benchmark_id,
        "benchmark_kind": benchmark_kind,
        "description": description,
        "queries": [
            {
                "id": query_id,
                "query": (queries or {}).get(query_id, query_id),
                "relevant": relevant,
            }
            for query_id, relevant in sorted(by_query.items())
        ],
    }


def load_query_texts(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    return {str(row["id"]): str(row["query"]) for row in payload}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or convert qrels into CANON canonical JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=["canon", "trec"], default="canon")
    parser.add_argument("--benchmark-id", default="external_qrels")
    parser.add_argument("--benchmark-kind", default="external_qrels")
    parser.add_argument("--queries-json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    settings = load_settings()
    input_path = Path(args.input)
    if args.format == "trec":
        qrels = canonical_from_trec(
            input_path.read_text(encoding="utf-8").splitlines(),
            benchmark_id=args.benchmark_id,
            benchmark_kind=args.benchmark_kind,
            queries=load_query_texts(Path(args.queries_json) if args.queries_json else None),
            description="Imported from TREC qrels format.",
        )
    else:
        qrels = json.loads(input_path.read_text(encoding="utf-8"))
    issues = validate_qrels(qrels)
    report = {
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "query_count": len(qrels.get("queries") or []),
        "benchmark_id": qrels.get("benchmark_id"),
    }
    if args.output:
        write_json(Path(args.output), qrels)
    write_json(settings.reports_dir / f"qrels_validation_{qrels.get('benchmark_id', 'unknown')}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

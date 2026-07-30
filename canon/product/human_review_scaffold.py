from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.disagreement_preservation import DEFAULT_BENCHMARK_PATH, load_benchmark, resolve_path
from canon.eval.external_ir import load_qrels
from canon.product import report_io
from canon.product.automated_benchmark_suite import load_suite


DEFAULT_RETRIEVAL_TARGET = 100
DEFAULT_SYNTHESIS_TARGET = 30
DEFAULT_PUBLISHABLE_SUITE_PATH = Path("conf") / "benchmark_suites" / "stage1_public_full.json"

RETRIEVAL_REVIEW_FIELDS = [
    "review_set",
    "reviewer_id",
    "benchmark_id",
    "benchmark_kind",
    "query_id",
    "query",
    "candidate_chunk_id",
    "candidate_title",
    "candidate_preview",
    "relevance",
    "evidence_role",
    "reviewer_notes",
]

SYNTHESIS_REVIEW_FIELDS = [
    "review_set",
    "reviewer_id",
    "case_id",
    "question",
    "claim_id",
    "claim",
    "evidence_id",
    "expected_stance",
    "citation_valid",
    "stance_correct",
    "claim_supported",
    "missed_key_evidence",
    "reviewer_action",
    "reason",
]


def build_human_review_scaffold(
    suite_path: Path | None = None,
    disagreement_benchmark_path: Path | None = None,
    retrieval_target: int = DEFAULT_RETRIEVAL_TARGET,
    synthesis_target: int = DEFAULT_SYNTHESIS_TARGET,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_suite_path = resolve_path(settings.root, suite_path or DEFAULT_PUBLISHABLE_SUITE_PATH)
    suite = load_suite(resolved_suite_path)
    disagreement_path = resolve_path(settings.root, disagreement_benchmark_path or DEFAULT_BENCHMARK_PATH)
    disagreement = load_benchmark(disagreement_path)
    retrieval_rows = retrieval_review_rows(settings.root, suite)
    synthesis_rows = synthesis_review_rows(disagreement)
    report = {
        "report_id": "publishable_human_review_scaffold_v1",
        "status": "review_scaffold_ready_human_labels_required",
        "suite_id": suite["suite_id"],
        "suite_path": relative(settings.root, resolved_suite_path),
        "disagreement_benchmark_id": disagreement["benchmark_id"],
        "disagreement_benchmark_path": relative(settings.root, disagreement_path),
        "targets": {
            "minimum_public_retrieval_queries": retrieval_target,
            "minimum_synthesis_review_questions": synthesis_target,
        },
        "completion_status": completion_status(
            retrieval_rows=retrieval_rows,
            synthesis_rows=synthesis_rows,
            retrieval_target=retrieval_target,
            synthesis_target=synthesis_target,
        ),
        "label_schemas": {
            "reviewer_provenance": {
                "reviewer_id": "Required stable reviewer identifier for every completed label row.",
            },
            "retrieval_qrels": retrieval_label_schema(),
            "synthesis_disagreement": synthesis_label_schema(),
        },
        "retrieval_review": {
            "row_count": len(retrieval_rows),
            "rows": retrieval_rows,
        },
        "synthesis_review": {
            "row_count": len(synthesis_rows),
            "rows": synthesis_rows,
        },
        "required_external_work": [
            "Generate pooled retrieval candidates for full public and private acceptance queries.",
            "Have reviewers complete retrieval relevance and evidence-role labels.",
            "Have reviewers complete synthesis citation, stance, claim-support, and action labels.",
            "Run canon.product.publishable_review to validate completed labels and write the publishable human-review status artifact.",
        ],
        "claim_boundary": {
            "status": "scaffold_only",
            "meaning": "This artifact prepares review work; it does not supply human labels.",
            "human_review_complete": False,
        },
    }
    if write_report:
        json_path = settings.reports_dir / "publishable_human_review_scaffold_v1.json"
        report_io.write_json(json_path, report)
        write_csv(json_path.with_name("publishable_retrieval_review_scaffold.csv"), RETRIEVAL_REVIEW_FIELDS, retrieval_rows)
        write_csv(json_path.with_name("publishable_synthesis_review_scaffold.csv"), SYNTHESIS_REVIEW_FIELDS, synthesis_rows)
        report_io.write_markdown(json_path.with_suffix(".md"), render_markdown(report))
    return report


def retrieval_review_rows(root: Path, suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for benchmark in suite.get("benchmarks") or []:
        qrels_path = resolve_path(root, Path(benchmark["qrels"]))
        qrels = load_qrels(qrels_path) if qrels_path.exists() else {"queries": []}
        for query in qrels.get("queries") or []:
            relevant = query.get("relevant") or {}
            if relevant:
                for chunk_id in sorted(relevant):
                    rows.append(retrieval_row(qrels, query, str(chunk_id)))
            else:
                rows.append(retrieval_row(qrels, query, ""))
    return rows


def retrieval_row(qrels: dict[str, Any], query: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return {
        "review_set": "retrieval_qrels",
        "reviewer_id": "",
        "benchmark_id": qrels.get("benchmark_id"),
        "benchmark_kind": qrels.get("benchmark_kind"),
        "query_id": str(query.get("id") or ""),
        "query": str(query.get("query") or ""),
        "candidate_chunk_id": chunk_id,
        "candidate_title": "",
        "candidate_preview": "",
        "relevance": "",
        "evidence_role": "",
        "reviewer_notes": "",
    }


def synthesis_review_rows(disagreement: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in disagreement.get("cases") or []:
        for evidence in case.get("evidence") or []:
            rows.append(
                {
                    "review_set": "synthesis_disagreement",
                    "reviewer_id": "",
                    "case_id": case.get("id"),
                    "question": case.get("question"),
                    "claim_id": "",
                    "claim": "",
                    "evidence_id": evidence.get("evidence_id"),
                    "expected_stance": evidence.get("expected_stance", ""),
                    "citation_valid": "",
                    "stance_correct": "",
                    "claim_supported": "",
                    "missed_key_evidence": "",
                    "reviewer_action": "",
                    "reason": "",
                }
            )
    return rows


def completion_status(
    retrieval_rows: list[dict[str, Any]],
    synthesis_rows: list[dict[str, Any]],
    retrieval_target: int,
    synthesis_target: int,
) -> dict[str, Any]:
    retrieval_query_count = len({row["query_id"] for row in retrieval_rows if row.get("query_id")})
    synthesis_case_count = len({row["case_id"] for row in synthesis_rows if row.get("case_id")})
    return {
        "status": "labels_required",
        "retrieval_query_count": retrieval_query_count,
        "retrieval_target": retrieval_target,
        "retrieval_target_met": retrieval_query_count >= retrieval_target,
        "synthesis_case_count": synthesis_case_count,
        "synthesis_target": synthesis_target,
        "synthesis_target_met": synthesis_case_count >= synthesis_target,
        "reviewed_label_count": 0,
    }


def retrieval_label_schema() -> dict[str, Any]:
    return {
        "relevance": {
            "2": "directly relevant evidence",
            "1": "partially relevant context or qualification",
            "0": "not relevant or misleading",
        },
        "evidence_role": [
            "background",
            "supports",
            "contradicts",
            "qualifies",
            "methods",
            "context",
        ],
    }


def synthesis_label_schema() -> dict[str, Any]:
    return {
        "citation_valid": ["yes", "no"],
        "stance_correct": ["yes", "no", "partial"],
        "claim_supported": ["yes", "no", "partial"],
        "missed_key_evidence": ["yes", "no"],
        "reviewer_action": ["accept", "revise", "reject"],
        "reason": "Free-text explanation for any no, partial, revise, or reject label.",
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_markdown(report: dict[str, Any]) -> str:
    status = report["completion_status"]
    return f"""# CANON Publishable Human Review Scaffold

Status: `{report['status']}`

Suite: `{report['suite_id']}`

Retrieval review rows: `{report['retrieval_review']['row_count']}`

Synthesis review rows: `{report['synthesis_review']['row_count']}`

Retrieval queries: `{status['retrieval_query_count']}` / `{status['retrieval_target']}`

Synthesis cases: `{status['synthesis_case_count']}` / `{status['synthesis_target']}`

This is a scaffold only. Human labels remain required before publication-quality
claims about retrieval quality, synthesis usefulness, or unsupported-claim rates.

After reviewers complete the CSV labels, run:

`python -m canon.product.publishable_review --retrieval-csv reports/publishable_retrieval_review_scaffold.csv --synthesis-csv reports/publishable_synthesis_review_scaffold.csv`
"""


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare publishable CANON human-review scaffolding.")
    parser.add_argument("--suite", default=str(DEFAULT_PUBLISHABLE_SUITE_PATH))
    parser.add_argument("--disagreement-benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--retrieval-target", type=int, default=DEFAULT_RETRIEVAL_TARGET)
    parser.add_argument("--synthesis-target", type=int, default=DEFAULT_SYNTHESIS_TARGET)
    args = parser.parse_args()
    report = build_human_review_scaffold(
        suite_path=Path(args.suite),
        disagreement_benchmark_path=Path(args.disagreement_benchmark),
        retrieval_target=args.retrieval_target,
        synthesis_target=args.synthesis_target,
    )
    print(json.dumps(cli_summary(report), indent=2, ensure_ascii=True))


def cli_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "suite_id": report["suite_id"],
        "completion_status": report["completion_status"],
        "retrieval_review_rows": report["retrieval_review"]["row_count"],
        "synthesis_review_rows": report["synthesis_review"]["row_count"],
        "artifacts": {
            "json": "reports/publishable_human_review_scaffold_v1.json",
            "retrieval_csv": "reports/publishable_retrieval_review_scaffold.csv",
            "synthesis_csv": "reports/publishable_synthesis_review_scaffold.csv",
            "markdown": "reports/publishable_human_review_scaffold_v1.md",
        },
    }


if __name__ == "__main__":
    main()

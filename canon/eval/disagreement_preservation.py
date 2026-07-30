from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.eval.disagreement_fixture import DEFAULT_OUTPUT_PATH
from canon.product.stage2_synthesis import run_stage2_synthesis


DEFAULT_BENCHMARK_PATH = DEFAULT_OUTPUT_PATH
DEFAULT_MIN_SCORE = 0.75


def evaluate_disagreement_preservation(
    benchmark_path: Path | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    max_claims: int = 6,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_path = resolve_path(settings.root, benchmark_path or DEFAULT_BENCHMARK_PATH)
    benchmark = load_benchmark(resolved_path)
    cases = [
        evaluate_case(case, max_claims=max_claims)
        for case in benchmark.get("cases", [])
    ]
    summary = summarize_cases(cases)
    report = {
        "report_id": "disagreement_preservation_benchmark_v1",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_path": relative(settings.root, resolved_path),
        "status": benchmark_status(summary, min_score),
        "min_score": min_score,
        "case_count": len(cases),
        "summary": summary,
        "cases": cases,
        "human_review_required": True,
        "claim_boundary": {
            "status": "automated_pre_human_signal",
            "meaning": (
                "This benchmark checks whether Stage 2 preserves known support, contradiction, "
                "qualification, and citation structure in controlled evidence packets."
            ),
            "not_proven": [
                "general factual correctness",
                "clinical or policy safety",
                "performance on naturally occurring contested evidence",
                "human reviewer usefulness",
            ],
        },
    }
    if write_report:
        output = settings.reports_dir / f"disagreement_preservation_{benchmark['benchmark_id']}.json"
        report_io.write_json(output, report)
        report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    return report


def evaluate_case(case: dict[str, Any], max_claims: int) -> dict[str, Any]:
    expected = expected_stances(case)
    workflow = workflow_from_case(case)
    stage2 = run_stage2_synthesis(
        {
            "question": case["question"],
            "research_workflow": workflow,
            "max_claims": max_claims,
            "min_evidence_count": 2,
            "min_distinct_sources": 2,
            "write_report": False,
        }
    )
    metrics = score_stage2(stage2, expected)
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_stances": expected,
        "stage2_status": stage2.get("status"),
        "claim_count": len(stage2.get("cited_claims") or []),
        "metrics": metrics,
        "disagreement_map": compact_disagreement_map(stage2.get("disagreement_map") or {}),
        "quality_gate": {
            "status": (stage2.get("quality_gate") or {}).get("status"),
            "unsupported_claims": (stage2.get("quality_gate") or {}).get("unsupported_claims") or [],
        },
    }


def score_stage2(stage2: dict[str, Any], expected: dict[str, str]) -> dict[str, Any]:
    expected_relevant = {
        evidence_id
        for evidence_id, stance in expected.items()
        if stance in {"supports", "contradicts", "qualifies"}
    }
    predicted = predicted_stances(stage2.get("cited_claims") or [])
    predicted_ids = set(predicted)
    linked_expected = expected_relevant & predicted_ids
    link_recall = ratio(len(linked_expected), len(expected_relevant))
    link_precision = ratio(len(linked_expected), len(predicted_ids))
    stance_accuracy = ratio(
        sum(1 for evidence_id in linked_expected if predicted[evidence_id] == expected[evidence_id]),
        len(linked_expected),
    )
    contradiction_recall = contradiction_preserved(stage2.get("cited_claims") or [], expected)
    citation_integrity = citation_integrity_score(stage2.get("quality_gate") or {})
    unsupported_claim_rate = unsupported_claim_rate_score(stage2.get("quality_gate") or {}, stage2.get("cited_claims") or [])
    score = round(
        (0.25 * link_recall)
        + (0.20 * link_precision)
        + (0.20 * stance_accuracy)
        + (0.20 * contradiction_recall)
        + (0.10 * citation_integrity)
        + (0.05 * (1.0 - unsupported_claim_rate)),
        6,
    )
    return {
        "disagreement_preservation_score": score,
        "link_recall": link_recall,
        "link_precision": link_precision,
        "stance_accuracy": stance_accuracy,
        "contradiction_recall": contradiction_recall,
        "citation_integrity": citation_integrity,
        "unsupported_claim_rate": unsupported_claim_rate,
        "predicted_stances": predicted,
        "missed_expected_evidence": sorted(expected_relevant - predicted_ids),
        "unexpected_linked_evidence": sorted(predicted_ids - expected_relevant),
    }


def predicted_stances(claims: list[dict[str, Any]]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for claim in claims:
        for link in claim.get("links") or []:
            evidence_id = str(link.get("evidence_id") or "")
            stance = str(link.get("stance") or "undetermined")
            if evidence_id and evidence_id not in rows:
                rows[evidence_id] = stance
    return rows


def contradiction_preserved(claims: list[dict[str, Any]], expected: dict[str, str]) -> float:
    expected_support = {
        evidence_id for evidence_id, stance in expected.items() if stance in {"supports", "qualifies"}
    }
    expected_contradict = {
        evidence_id for evidence_id, stance in expected.items() if stance == "contradicts"
    }
    if not expected_support or not expected_contradict:
        return 1.0
    for claim in claims:
        links = claim.get("links") or []
        stances_by_id = {
            str(link.get("evidence_id")): str(link.get("stance") or "")
            for link in links
        }
        has_support = any(
            stances_by_id.get(evidence_id) in {"supports", "qualifies"}
            for evidence_id in expected_support
        )
        has_contradiction = any(
            stances_by_id.get(evidence_id) == "contradicts"
            for evidence_id in expected_contradict
        )
        if has_support and has_contradiction:
            return 1.0
    return 0.0


def citation_integrity_score(gate: dict[str, Any]) -> float:
    checks = gate.get("checks") or []
    for row in checks:
        if row.get("id") == "citation_integrity":
            return 1.0 if row.get("status") == "pass" else 0.0
    return 0.0


def unsupported_claim_rate_score(gate: dict[str, Any], claims: list[dict[str, Any]]) -> float:
    claim_count = len(claims)
    if claim_count == 0:
        return 1.0
    return ratio(len(gate.get("unsupported_claims") or []), claim_count)


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "disagreement_preservation_score",
        "link_recall",
        "link_precision",
        "stance_accuracy",
        "contradiction_recall",
        "citation_integrity",
        "unsupported_claim_rate",
    ]
    return {
        "case_count": len(cases),
        "average_metrics": {
            metric: average([case["metrics"][metric] for case in cases])
            for metric in metric_names
        },
        "failed_cases": [
            case["id"]
            for case in cases
            if case["metrics"]["contradiction_recall"] < 1.0
            or case["metrics"]["citation_integrity"] < 1.0
        ],
    }


def benchmark_status(summary: dict[str, Any], min_score: float) -> str:
    if summary["case_count"] == 0:
        return "automated_fail"
    metrics = summary["average_metrics"]
    if (
        metrics["disagreement_preservation_score"] >= min_score
        and metrics["contradiction_recall"] >= 1.0
        and metrics["citation_integrity"] >= 1.0
    ):
        return "automated_pass_human_review_required"
    return "needs_revision_human_review_required"


def workflow_from_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": "disagreement_preservation_fixture_workflow_v1",
        "status": "ready_for_human_review",
        "mode": case.get("mode", "disagreement_fixture"),
        "policy": "rag",
        "layers": {
            "research_guidance_layer": {
                "next_actions": [{"description": "Review whether disagreement was preserved."}],
                "synthesis_gate": {"status": "blocked_pending_human_review"},
            }
        },
        "evidence_packet_response": {
            "coverage_gaps": [],
            "evidence_packets": [
                {
                    "supporting_evidence": [
                        evidence_item(row)
                        for row in case.get("evidence", [])
                    ]
                }
            ],
        },
    }


def evidence_item(row: dict[str, Any]) -> dict[str, Any]:
    evidence_id = str(row["evidence_id"])
    return {
        "evidence_id": evidence_id,
        "chunk_id": row.get("chunk_id") or f"chunk:{evidence_id}",
        "document_id": row.get("document_id") or f"doc:{evidence_id}",
        "title": row.get("title") or evidence_id,
        "source_name": row.get("source_name") or "fixture",
        "source_type": row.get("source_type") or "fixture",
        "language": row.get("language") or "English",
        "domain": row.get("domain") or "fixture",
        "cluster_id": row.get("cluster_id") or "fixture",
        "text": row["text"],
    }


def expected_stances(case: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["evidence_id"]): str(row.get("expected_stance") or "neutral")
        for row in case.get("evidence", [])
    }


def compact_disagreement_map(disagreement_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": disagreement_map.get("status"),
        "claim_cluster_count": disagreement_map.get("claim_cluster_count", 0),
        "contested_cluster_count": disagreement_map.get("contested_cluster_count", 0),
        "axes": disagreement_map.get("axes") or {},
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["summary"]["average_metrics"]
    cases = "\n".join(
        f"- `{case['id']}` score={case['metrics']['disagreement_preservation_score']} "
        f"contradiction_recall={case['metrics']['contradiction_recall']} "
        f"unexpected_links={len(case['metrics']['unexpected_linked_evidence'])}"
        for case in report["cases"]
    )
    return f"""# CANON Disagreement Preservation Benchmark

Status: `{report['status']}`

Benchmark: `{report['benchmark_id']}`

Average score: `{metrics['disagreement_preservation_score']}`

Contradiction recall: `{metrics['contradiction_recall']}`

Citation integrity: `{metrics['citation_integrity']}`

## Cases

{cases}

This is an automated pre-human signal. Human review remains required before
publication-quality claims about synthesis usefulness or factual correctness.
"""


def load_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("benchmark_id"):
        raise ValueError("Disagreement benchmark must include benchmark_id.")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("Disagreement benchmark must include at least one case.")
    return payload


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def average(values: list[float]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 disagreement preservation.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-claims", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_disagreement_preservation(
                benchmark_path=Path(args.benchmark),
                min_score=args.min_score,
                max_claims=args.max_claims,
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

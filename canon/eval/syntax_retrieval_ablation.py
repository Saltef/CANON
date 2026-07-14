from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.document_type_slices import resolve_policy_weights
from canon.retrieval.corpus import RetrievalDocument, load_processed_corpus
from canon.retrieval.engine import retrieve_with_stages
from canon.retrieval.stages import candidate_pool_size, retrieve_candidates
from canon.retrieval.syntax import document_syntax_vector, primary_syntax_category


DEFAULT_MODES = ["mixed_domain_demo_corpus", "social_public_opinion_demo_corpus"]
DEFAULT_TOP_K = 3

DEFAULT_SYNTAX_ABLATION_CASES = [
    {
        "id": "economics-method",
        "query": "What identification method, panel data, and robustness evidence support battery subsidy effects?",
        "expected_work_ids": ["econ-1", "econ-preprint-1"],
        "expected_syntax_categories": ["method_detail", "limitation_caveat"],
        "intent": "method_evidence",
    },
    {
        "id": "psychology-limitations",
        "query": "Which psychology evidence describes measurement validity, sample recruitment, uncertainty, and bias limitations?",
        "expected_work_ids": ["psych-1"],
        "expected_syntax_categories": ["limitation_caveat", "method_detail"],
        "intent": "limitation_audit",
    },
    {
        "id": "legal-filing-risk",
        "query": "Which legal market filing discloses current risk, regulation, contract exposure, and interconnection backlog?",
        "expected_work_ids": ["legal-market-1"],
        "expected_syntax_categories": ["risk_disclosure", "legal_authority"],
        "intent": "risk_disclosure_lookup",
    },
    {
        "id": "market-quantitative-risk",
        "query": "Which market report estimates demand, pricing risk, financing sensitivity, and customer concentration?",
        "expected_work_ids": ["market-1"],
        "expected_syntax_categories": ["quantitative_result", "risk_disclosure"],
        "intent": "market_due_diligence",
    },
    {
        "id": "anthropology-transcript",
        "query": "Which interview transcript or fieldwork evidence gives community testimony and positionality context?",
        "expected_work_ids": ["anthro-1"],
        "expected_syntax_categories": ["dialogue_or_transcript", "narrative_context"],
        "intent": "fieldwork_context",
    },
    {
        "id": "history-archive",
        "query": "Which archival primary source gives provenance notes, collection context, source criticism, and historiographic dispute?",
        "expected_work_ids": ["history-1"],
        "expected_syntax_categories": ["narrative_context"],
        "intent": "archival_context",
    },
    {
        "id": "public-opinion-expression",
        "query": "Which public opinion comments mention battery storage safety concerns, trust, and support or opposition?",
        "expected_work_ids": [
            "social:MicroBlog:post-1",
            "social:MicroBlog:post-2",
            "social:Forum:comment-1",
        ],
        "expected_syntax_categories": ["public_expression"],
        "intent": "public_expression_aggregation",
    },
]


def evaluate_syntax_retrieval_ablation(
    modes: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_modes = modes or DEFAULT_MODES
    documents = load_documents(settings.data_dir, resolved_modes)
    retrieval_policies = settings.raw["retrieval"]["policies"]
    balanced = resolve_policy_weights(retrieval_policies, ["balanced"])["balanced"]
    variants = ablation_variants(balanced)
    rows = [
        run_ablation_case(case, variant, documents, top_k, int(settings.raw["retrieval"]["preview_chars"]))
        for case in DEFAULT_SYNTAX_ABLATION_CASES
        for variant in variants
    ]
    summary = summarize_rows(rows, variants)
    checks = ablation_checks(summary, rows)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "syntax_retrieval_ablation_v1",
        "modes": resolved_modes,
        "document_count": len(documents),
        "case_count": len(DEFAULT_SYNTAX_ABLATION_CASES),
        "top_k": top_k,
        "variants": variants,
        "rows": rows,
        "summary": summary,
        "checks": checks,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "limitations": [
            "This is a deterministic fixture ablation, not proof of general retrieval quality.",
            "Syntax relevance detects evidence form, not reliability, factual truth, legal authority status, or representativeness.",
            "Expected work IDs are hand-authored technical anchors and should be replaced with larger qrels and human labels.",
            "The corpus-learned syntax profile uses rarity weights, not supervised discourse labels.",
        ],
    }
    report["contract_validation"] = validate_report(report, "syntax_retrieval_ablation")
    if write_report:
        path = settings.reports_dir / "syntax_retrieval_ablation_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), render_markdown(report))
    return report


def load_documents(data_dir: Path, modes: list[str]) -> list[RetrievalDocument]:
    documents: list[RetrievalDocument] = []
    seen: set[str] = set()
    for mode in modes:
        for document in load_processed_corpus(data_dir, mode):
            key = f"{document.work_id}::{document.chunk_id}"
            if key in seen:
                continue
            seen.add(key)
            documents.append(document)
    return documents


def ablation_variants(base_weights: dict[str, float]) -> list[dict[str, Any]]:
    no_syntax = dict(base_weights)
    no_syntax["syntax_relevance"] = 0.0
    no_syntax["candidate_syntax_weight"] = 0.0
    candidate_only = dict(base_weights)
    candidate_only["syntax_relevance"] = 0.0
    candidate_only["candidate_syntax_weight"] = 0.1
    candidate_and_rerank = dict(base_weights)
    candidate_and_rerank["syntax_relevance"] = max(float(candidate_and_rerank.get("syntax_relevance", 0.0)), 0.25)
    candidate_and_rerank["candidate_syntax_weight"] = 0.1
    return [
        {"id": "lexical_semantic_no_syntax", "description": "Balanced retrieval with syntax disabled in candidate recall and final reranking.", "weights": no_syntax},
        {"id": "syntax_candidate_only", "description": "Syntax participates in candidate recall but has no final reranking weight.", "weights": candidate_only},
        {"id": "syntax_candidate_and_rerank", "description": "Syntax participates in candidate recall and final typed reranking.", "weights": candidate_and_rerank},
    ]


def run_ablation_case(
    case: dict[str, Any],
    variant: dict[str, Any],
    documents: list[RetrievalDocument],
    top_k: int,
    preview_chars: int,
) -> dict[str, Any]:
    weights = variant["weights"]
    candidate_k = candidate_pool_size(top_k, len(documents), int(weights["candidate_k"]) if "candidate_k" in weights else None)
    candidates = retrieve_candidates(
        case["query"],
        documents,
        candidate_k,
        candidate_syntax_weight=float(weights.get("candidate_syntax_weight", 0.1) or 0.0),
    )
    staged = retrieve_with_stages(case["query"], documents, weights, top_k, preview_chars)
    final_rows = [trace.to_dict() for trace in staged["trace"]]
    expected = set(case["expected_work_ids"])
    candidate_rank = first_rank([candidate.document.work_id for candidate in candidates], expected)
    final_rank = first_rank([row["work_id"] for row in final_rows], expected)
    top_result = final_rows[0] if final_rows else {}
    top_signals = ((top_result.get("decision") or {}).get("stage") or {}).get("signals") or {}
    top_syntax = top_signals.get("syntax_profile") or {}
    top_syntax_category = str(top_syntax.get("document_syntax_category", ""))
    syntax_category_match = top_syntax_category in set(case["expected_syntax_categories"])
    unsafe_final_count = sum(1 for row in final_rows if final_safety_decision(row) in {"block", "quarantine"})
    return {
        "case_id": case["id"],
        "query": case["query"],
        "intent": case["intent"],
        "variant": variant["id"],
        "expected_work_ids": case["expected_work_ids"],
        "expected_syntax_categories": case["expected_syntax_categories"],
        "candidate_k": candidate_k,
        "candidate_hit": candidate_rank is not None,
        "candidate_rank": candidate_rank,
        "final_hit": final_rank is not None,
        "final_rank": final_rank,
        "final_reciprocal_rank": reciprocal_rank(final_rank),
        "top_work_id": top_result.get("work_id"),
        "top_chunk_id": top_result.get("chunk_id"),
        "top_syntax_category": top_syntax_category,
        "top_syntax_relevance": float((top_result.get("components") or {}).get("syntax_relevance", 0.0) or 0.0),
        "syntax_category_match": syntax_category_match,
        "unsafe_final_count": unsafe_final_count,
        "final_work_ids": [row["work_id"] for row in final_rows],
        "candidate_sample": [
            {
                "rank": candidate.candidate_rank,
                "work_id": candidate.document.work_id,
                "chunk_id": candidate.document.chunk_id,
                "candidate_score": candidate.candidate_score,
                "lexical_relevance": round(candidate.lexical_relevance, 6),
                "semantic_similarity": round(candidate.semantic_similarity, 6),
                "syntax_relevance": round(candidate.syntax_relevance, 6),
                "syntax_category": primary_syntax_category(document_syntax_vector(candidate.document)),
            }
            for candidate in candidates[: min(len(candidates), 8)]
        ],
    }


def final_safety_decision(row: dict[str, Any]) -> str:
    safety = (((row.get("decision") or {}).get("stage") or {}).get("safety_gate") or {}).get("safety") or {}
    return str(safety.get("decision", "allow"))


def first_rank(ids: list[str], expected: set[str]) -> int | None:
    for index, identifier in enumerate(ids, start=1):
        if identifier in expected:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    return round(1.0 / rank, 6) if rank else 0.0


def summarize_rows(rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant["id"]]
        by_variant[variant["id"]] = {
            "case_count": len(variant_rows),
            "candidate_hit_rate": average_bool(row["candidate_hit"] for row in variant_rows),
            "final_hit_rate": average_bool(row["final_hit"] for row in variant_rows),
            "mean_reciprocal_final_rank": average(row["final_reciprocal_rank"] for row in variant_rows),
            "syntax_category_match_rate": average_bool(row["syntax_category_match"] for row in variant_rows),
            "unsafe_final_count": sum(int(row["unsafe_final_count"]) for row in variant_rows),
            "top_syntax_category_counts": dict(sorted(Counter(row["top_syntax_category"] for row in variant_rows).items())),
        }
    baseline = by_variant.get("lexical_semantic_no_syntax", {})
    syntax = by_variant.get("syntax_candidate_and_rerank", {})
    return {
        "by_variant": by_variant,
        "syntax_vs_no_syntax": {
            "final_hit_rate_delta": round(float(syntax.get("final_hit_rate", 0.0)) - float(baseline.get("final_hit_rate", 0.0)), 6),
            "mean_reciprocal_final_rank_delta": round(
                float(syntax.get("mean_reciprocal_final_rank", 0.0))
                - float(baseline.get("mean_reciprocal_final_rank", 0.0)),
                6,
            ),
            "syntax_category_match_rate_delta": round(
                float(syntax.get("syntax_category_match_rate", 0.0))
                - float(baseline.get("syntax_category_match_rate", 0.0)),
                6,
            ),
        },
        "case_deltas": case_deltas(rows),
    }


def case_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = sorted({row["case_id"] for row in rows})
    output = []
    for case_id in cases:
        baseline = next(row for row in rows if row["case_id"] == case_id and row["variant"] == "lexical_semantic_no_syntax")
        syntax = next(row for row in rows if row["case_id"] == case_id and row["variant"] == "syntax_candidate_and_rerank")
        output.append(
            {
                "case_id": case_id,
                "baseline_final_rank": baseline["final_rank"],
                "syntax_final_rank": syntax["final_rank"],
                "reciprocal_rank_delta": round(syntax["final_reciprocal_rank"] - baseline["final_reciprocal_rank"], 6),
                "baseline_top_work_id": baseline["top_work_id"],
                "syntax_top_work_id": syntax["top_work_id"],
                "syntax_category_match": syntax["syntax_category_match"],
            }
        )
    return output


def ablation_checks(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = summary["by_variant"]
    baseline = by_variant.get("lexical_semantic_no_syntax", {})
    syntax = by_variant.get("syntax_candidate_and_rerank", {})
    return [
        check("cases_present", len({row["case_id"] for row in rows}) >= 6, "Ablation covers multiple evidence-form cases."),
        check("variants_present", len(by_variant) == 3, "Ablation compares no-syntax, candidate-only syntax, and candidate-plus-rerank syntax."),
        check(
            "syntax_not_worse_than_no_syntax_mrr",
            float(syntax.get("mean_reciprocal_final_rank", 0.0)) >= float(baseline.get("mean_reciprocal_final_rank", 0.0)),
            "Syntax-enabled retrieval does not reduce mean reciprocal expected-evidence rank on the fixture.",
            summary.get("syntax_vs_no_syntax"),
        ),
        check(
            "syntax_category_trace_present",
            all(row["top_syntax_category"] for row in rows if row["variant"] == "syntax_candidate_and_rerank"),
            "Syntax-enabled rows expose top-result syntax categories.",
        ),
        check(
            "no_unsafe_final_exposure",
            sum(int(row["unsafe_final_count"]) for row in rows) == 0,
            "No blocked or quarantined evidence appears in final ablation results.",
        ),
    ]


def average(values: Any) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def average_bool(values: Any) -> float:
    rows = [1.0 if value else 0.0 for value in values]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def check(identifier: str, passed: bool, description: str, evidence: Any = None) -> dict[str, Any]:
    row = {"id": identifier, "passed": bool(passed), "description": description}
    if evidence is not None:
        row["evidence"] = evidence
    return row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] `{item['id']}`: {item['description']}"
        for item in report["checks"]
    )
    deltas = "\n".join(
        f"- `{row['case_id']}`: baseline rank {row['baseline_final_rank']}, syntax rank {row['syntax_final_rank']}, delta {row['reciprocal_rank_delta']}"
        for row in report["summary"]["case_deltas"]
    )
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Syntax Retrieval Ablation

- Status: {report["status"]}
- Modes: {', '.join(report["modes"])}
- Cases: {report["case_count"]}
- Top-k: {report["top_k"]}

## Summary

```json
{json.dumps(report["summary"], indent=2)}
```

## Checks

{checks}

## Case Deltas

{deltas}

## Limitations

{limitations}
"""


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run syntax/form relevance retrieval ablation.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = evaluate_syntax_retrieval_ablation(
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        top_k=args.top_k,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

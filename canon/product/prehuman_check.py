from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.llm_judge import judge_qrels_csv
from canon.eval.model_evaluation import evaluate_semantic_models, parse_providers
from canon.eval.qrels_review import artifact_paths, build_qrels_review_packet, qrels_from_review_csv
from canon.eval.rerank_evaluation import evaluate_rerankers, parse_list as parse_rerankers
from canon.eval.source_diversity import evaluate_source_diversity
from canon.product import readiness, report_io, service, smoke


DEFAULT_BENCHMARK_ID = "llm_judged_social_science_ir_v1_harvest10"
DEFAULT_MODEL_PROVIDERS = ["local", "openai", "cohere"]
DEFAULT_RERANKERS = ["heuristic", "cohere"]


def run_prehuman_check(
    mode: str = service.DEFAULT_MODE,
    benchmark_id: str | None = None,
    judge_provider: str = "heuristic",
    judge_model: str | None = None,
    model_providers: list[str] | None = None,
    rerankers: list[str] | None = None,
    top_k: int = 10,
    candidate_k: int = 25,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    benchmark_id = benchmark_id or default_benchmark_id(mode)
    qrels_packet = build_qrels_review_packet(mode=mode, top_k=top_k)
    _, review_csv, _ = artifact_paths(settings.reports_dir, mode)
    judged_csv = settings.reports_dir / f"qrels_review_tasks_{mode}.llm_judged.csv"
    judge_report = judge_qrels_csv(
        csv_path=review_csv,
        output_path=judged_csv,
        provider_name=judge_provider,
        model=judge_model,
        fill_relevance=True,
    )
    qrels_path = settings.root / "gold" / f"{benchmark_id}.json"
    qrels_result = qrels_from_review_csv(
        csv_path=judged_csv,
        benchmark_id=benchmark_id,
        output_path=qrels_path,
        benchmark_kind="llm_judged_qrels",
        description=(
            "LLM/heuristic-judged provisional CANON qrels. Use for automated "
            "triage only; re-run against human-reviewed qrels before release claims."
        ),
        allow_missing_relevance=True,
    )
    semantic_report = evaluate_semantic_models(
        mode=mode,
        qrels_path=qrels_path,
        providers=model_providers or DEFAULT_MODEL_PROVIDERS,
        k=top_k,
    )
    rerank_report = evaluate_rerankers(
        mode=mode,
        qrels_path=qrels_path,
        rerankers=rerankers or DEFAULT_RERANKERS,
        base_policy="rag",
        candidate_k=candidate_k,
        k=top_k,
    )
    diversity_report = evaluate_source_diversity(mode=mode, top_k=top_k)
    smoke_report = smoke.build_smoke_report(mode=mode)
    readiness_report = readiness.build_readiness_report(mode=mode)
    components = [
        component("qrels_candidate_packet", qrels_packet.get("candidate_count", 0) > 0, qrels_packet),
        component("provisional_qrels_judge", judge_report.get("status") == "judged", judge_report),
        component("provisional_qrels_export", qrels_result.get("status") == "qrels_written", qrels_result),
        component("semantic_model_evaluation", has_available_provider(semantic_report.get("providers", [])), semantic_report),
        component("rerank_evaluation", has_available_provider(rerank_report.get("rerankers", [])), rerank_report),
        component("source_diversity", diversity_report.get("status") == "pass", diversity_report),
        component("product_smoke", smoke_report.get("status") == "pass", smoke_report),
        component("product_readiness", readiness_report.get("status") == "pass", readiness_report),
        component(
            "human_review_limit",
            False,
            {
                "status": "required_for_release_claims",
                "reason": "LLM-judged qrels are provisional and must be replaced or audited by human-reviewed qrels.",
            },
        ),
    ]
    automated_components = [item for item in components if item["id"] != "human_review_limit"]
    automated_passed = all(item["passed"] for item in automated_components)
    report = {
        "report_id": "prehuman_product_check_v1",
        "mode": mode,
        "status": "automated_pass_human_review_required" if automated_passed else "automated_fail",
        "benchmark_id": benchmark_id,
        "qrels_path": relative(settings.root, qrels_path),
        "judge": {"provider": judge_provider, "model": judge_model},
        "claim": prehuman_claim(automated_passed),
        "components": [compact_component(item) for item in components],
        "model_leaderboard": semantic_report.get("leaderboard", []),
        "rerank_leaderboard": rerank_report.get("leaderboard", []),
        "human_review_boundary": {
            "required_before": [
                "declaring a production retrieval winner",
                "publishing model superiority claims",
                "claiming unsupported factual claim rates",
                "shipping an industry-pilot release",
            ],
            "next_test": "Replace this benchmark with human-reviewed qrels and re-run this command.",
        },
        "artifacts": artifacts(settings.root, settings.reports_dir, mode, benchmark_id, top_k),
    }
    if write_report:
        output = settings.reports_dir / f"prehuman_product_check_{mode}.json"
        report_io.write_json(output, report)
        report_io.write_markdown(
            settings.reports_dir / f"prehuman_product_check_{mode}.md",
            render_markdown(report),
        )
    return report


def default_benchmark_id(mode: str) -> str:
    return f"llm_judged_{mode}"


def component(identifier: str, passed: bool, report: dict[str, Any]) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "report": report}


def compact_component(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    return {
        "id": item["id"],
        "passed": item["passed"],
        "status": report.get("status", "ok" if item["passed"] else "blocked"),
        "summary": component_summary(item["id"], report),
    }


def component_summary(identifier: str, report: dict[str, Any]) -> dict[str, Any]:
    if identifier == "qrels_candidate_packet":
        return {
            "query_count": report.get("query_count", 0),
            "candidate_count": report.get("candidate_count", 0),
        }
    if identifier == "provisional_qrels_judge":
        return {
            "row_count": report.get("row_count", 0),
            "judged_count": report.get("judged_count", 0),
            "audit_summary": report.get("audit_summary", {}),
            "fill_relevance": report.get("fill_relevance"),
        }
    if identifier == "provisional_qrels_export":
        return {
            "query_count": report.get("query_count", 0),
            "judgment_count": report.get("judgment_count", 0),
            "benchmark_kind": (report.get("qrels") or {}).get("benchmark_kind"),
        }
    if identifier == "semantic_model_evaluation":
        return {
            "leaderboard": report.get("leaderboard", []),
            "recommendation": report.get("recommendation", {}),
            "unavailable": unavailable(report.get("providers", [])),
        }
    if identifier == "rerank_evaluation":
        return {
            "leaderboard": report.get("leaderboard", []),
            "unavailable": unavailable(report.get("rerankers", [])),
        }
    if identifier == "source_diversity":
        return report.get("summary", {})
    if identifier in {"product_smoke", "product_readiness"}:
        return {
            "check_count": len(report.get("checks") or []),
            "failed_checks": [
                check.get("id")
                for check in report.get("checks", [])
                if not check.get("passed")
            ],
        }
    return {"reason": report.get("reason")}


def unavailable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "provider": row.get("provider"),
            "reason": row.get("reason"),
        }
        for row in rows
        if row.get("status") != "ok"
    ]


def has_available_provider(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") == "ok" for row in rows)


def prehuman_claim(automated_passed: bool) -> str:
    if automated_passed:
        return (
            "Automated CANON gates pass with provisional LLM-judged qrels; "
            "human review remains required before production model-quality claims."
        )
    return "One or more automated pre-human gates failed; inspect component summaries before human review."


def artifacts(root: Path, reports_dir: Path, mode: str, benchmark_id: str, top_k: int) -> dict[str, str]:
    return {
        "qrels_review_json": relative(root, reports_dir / f"qrels_review_tasks_{mode}.json"),
        "qrels_review_csv": relative(root, reports_dir / f"qrels_review_tasks_{mode}.csv"),
        "llm_judged_csv": relative(root, reports_dir / f"qrels_review_tasks_{mode}.llm_judged.csv"),
        "provisional_qrels": f"gold/{benchmark_id}.json",
        "semantic_model_evaluation": relative(root, reports_dir / f"semantic_model_evaluation_{mode}.json"),
        "rerank_evaluation": relative(root, reports_dir / f"rerank_evaluation_{mode}_{benchmark_id}.json"),
        "source_diversity": relative(root, reports_dir / f"source_diversity_{mode}_rag_k{top_k}.json"),
        "prehuman_report": relative(root, reports_dir / f"prehuman_product_check_{mode}.json"),
    }


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    components = "\n".join(
        f"- {'PASS' if item['passed'] else 'BLOCKED'}: `{item['id']}` status={item['status']}"
        for item in report["components"]
    )
    artifacts_md = "\n".join(f"- `{key}`: `{value}`" for key, value in report["artifacts"].items())
    return f"""# CANON Pre-Human Product Check

Status: **{report['status']}**

Mode: `{report['mode']}`

Benchmark: `{report['benchmark_id']}`

Claim: {report['claim']}

## Components

{components}

## Human Review Boundary

These results are provisional. Human-reviewed qrels are still required before
declaring model winners, publishing unsupported-claim rates, or shipping an
industry-pilot release.

Next test: {report['human_review_boundary']['next_test']}

## Artifacts

{artifacts_md}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON automated checks up to the human-review boundary.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--judge-provider", default="heuristic")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--model-providers", default="local,openai,cohere")
    parser.add_argument("--rerankers", default="heuristic,cohere")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=25)
    args = parser.parse_args()
    report = run_prehuman_check(
        mode=args.mode,
        benchmark_id=args.benchmark_id,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        model_providers=parse_providers(args.model_providers),
        rerankers=parse_rerankers(args.rerankers),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

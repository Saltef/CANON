from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


DEFAULT_QUERY = "Do sanctions work?"


def build_smoke_report(
    mode: str = service.DEFAULT_MODE,
    query: str = DEFAULT_QUERY,
    write_report: bool = True,
) -> dict[str, Any]:
    results = {
        "health": service.health(),
        "summary": service.product_summary(mode),
        "production_status": service.production_status({"mode": mode}),
        "answer": service.answer(
            {
                "query": query,
                "mode": mode,
                "policy": service.DEFAULT_POLICY,
                "top_k": 5,
                "freedom_level": "balanced",
            }
        ),
        "compare": service.compare_retrieval(
            {
                "query": query,
                "mode": mode,
                "policies": ["lexical", "rag"],
                "top_k": 5,
            }
        ),
        "query_diagnostics": service.query_diagnostics(
            {
                "query": query,
                "mode": mode,
                "top_k": 5,
                "candidate_k": 20,
                "freedom_level": "balanced",
            }
        ),
        "evidence_packets": service.evidence_packets(
            {
                "request_id": "smoke",
                "query": query,
                "mode": mode,
                "policy": service.DEFAULT_POLICY,
                "evidence_requirements": {
                    "top_k": 5,
                    "external_expansion": {
                        "enabled": True,
                        "allowed_source_types": ["official", "academic", "news"],
                        "max_external_queries": 3,
                    },
                },
            }
        ),
        "production_workbench": service.production_evidence_workbench(
            {
                "session_id": "smoke_workbench",
                "query": query,
                "mode": mode,
                "policy": service.DEFAULT_POLICY,
                "top_k": 5,
                "freedom_level": "balanced",
                "suggest_external_expansion": True,
                "write_telemetry": False,
            }
        ),
    }
    checks = smoke_checks(results)
    report = {
        "report_id": "product_smoke_v1",
        "mode": mode,
        "query": query,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
        "endpoints": [
            "GET /health",
            "GET /v1/summary",
            "GET /v1/production/status",
            "POST /v1/production/evidence-workbench",
            "POST /v1/answer",
            "POST /v1/compare",
            "POST /v1/query-diagnostics",
            "POST /v1/evidence-packets",
        ],
        "result_summary": compact_results(results),
    }
    if write_report:
        settings = load_settings()
        write_json(settings.reports_dir / f"product_smoke_{mode}.json", report)
    return report


def smoke_checks(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check("health_ok", results["health"].get("status") == "ok"),
        check("summary_has_claim_boundaries", bool(results["summary"].get("claim_boundaries"))),
        check(
            "production_status_has_boundaries",
            bool(results["production_status"].get("shippable_without_human_review"))
            and bool(results["production_status"].get("blocked_without_human_review")),
        ),
        check("answer_has_citations", bool(results["answer"].get("citations"))),
        check("answer_has_evidence", bool(results["answer"].get("evidence"))),
        check("answer_has_query_diagnostics", bool(results["answer"].get("query_diagnostics"))),
        check("compare_has_runs", bool(results["compare"].get("runs"))),
        check("compare_has_rank_table", "rank_table" in results["compare"]),
        check("query_diagnostics_has_terms", "query_to_corpus" in results["query_diagnostics"]),
        check("query_diagnostics_has_variants", "query_variants" in results["query_diagnostics"]),
        check("evidence_packets_have_packets", bool(results["evidence_packets"].get("evidence_packets"))),
        check("evidence_packets_have_run_diagnosis", bool(results["evidence_packets"].get("run_diagnosis"))),
        check(
            "evidence_packets_contract_passed",
            (results["evidence_packets"].get("contract_validation") or {}).get("status") == "pass",
        ),
        check(
            "evidence_packets_expansion_not_executed",
            not results["evidence_packets"].get("external_expansion", {}).get("executed"),
        ),
        check(
            "production_workbench_has_evidence_cards",
            bool(results["production_workbench"].get("evidence_cards")),
        ),
        check(
            "production_workbench_has_claim_boundary",
            bool(results["production_workbench"].get("claim_boundary")),
        ),
        check(
            "production_workbench_has_run_diagnosis",
            bool(results["production_workbench"].get("run_diagnosis")),
        ),
        check(
            "production_workbench_attempts_two_model_calls",
            (results["production_workbench"].get("model_comparison") or {}).get("call_count") == 2,
        ),
    ]


def check(identifier: str, passed: bool) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed)}


def compact_results(results: dict[str, Any]) -> dict[str, Any]:
    answer = results["answer"]
    diagnostics = results["query_diagnostics"]
    packets = results["evidence_packets"]
    production = results["production_workbench"]
    return {
        "health_status": results["health"].get("status"),
        "summary_product": results["summary"].get("product"),
        "production_status": results["production_status"].get("status"),
        "answer_citation_count": len(answer.get("citations") or []),
        "answer_evidence_count": len(answer.get("evidence") or []),
        "answer_support_level": (answer.get("support_assessment") or {}).get("support_level"),
        "compare_policy_count": len(results["compare"].get("policies") or []),
        "compare_run_count": len(results["compare"].get("runs") or []),
        "query_diagnostics_stability": (diagnostics.get("stability") or {}).get("status"),
        "query_diagnostics_variant_count": len(diagnostics.get("query_variants") or []),
        "evidence_packet_count": len(packets.get("evidence_packets") or []),
        "evidence_packet_contract_status": (packets.get("contract_validation") or {}).get("status"),
        "evidence_packet_run_diagnosis_status": (packets.get("run_diagnosis") or {}).get("overall_status"),
        "external_expansion_status": (packets.get("external_expansion") or {}).get("status"),
        "production_workbench_status": production.get("status"),
        "production_workbench_evidence_count": len(production.get("evidence_cards") or []),
        "production_workbench_run_diagnosis_status": (production.get("run_diagnosis") or {}).get("overall_status"),
        "production_workbench_model_call_count": (production.get("model_comparison") or {}).get("call_count", 0),
        "production_workbench_model_success_count": (production.get("model_comparison") or {}).get("success_count", 0),
    }


def write_json(path: Path, payload: object) -> None:
    report_io.write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON product smoke checks.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_smoke_report(mode=args.mode, query=args.query)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

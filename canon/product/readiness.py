from __future__ import annotations

import argparse
import json

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


def build_readiness_report(mode: str = service.DEFAULT_MODE) -> dict:
    settings = load_settings()
    available_endpoints = endpoints()
    audit = load_json_any(
        settings.reports_dir,
        [f"scientific_audit_{mode}_baseline_methods_v1.json", "scientific_audit_dry_run_baseline_methods_v1.json"],
    )
    decision = load_json_any(
        settings.reports_dir,
        [f"claim_decision_{mode}_baseline_methods_v1.json", "claim_decision_dry_run_baseline_methods_v1.json"],
    )
    data_card = load_json_any(settings.reports_dir, [f"data_card_{mode}.json", "data_card_dry_run.json"])
    summary = {
        "product": "CANON Evidence Workbench",
        "mode": mode,
        "claim_boundaries": decision.get("global_winner_claim") or {"status": decision.get("status", "unknown")},
        "audit_status": audit.get("status"),
        "active_warnings": [warning["id"] for warning in audit.get("warnings", [])],
        "resolved_warnings": [warning["id"] for warning in audit.get("resolved_warnings", [])],
        "corpus": {
            "work_count": data_card.get("work_count"),
            "chunk_count": data_card.get("chunk_count"),
            "limitations": data_card.get("limitations") or ["data card unavailable"],
        },
    }
    smoke_report = load_json_any(
        settings.reports_dir,
        [f"product_smoke_{mode}.json", "product_smoke_dry_run.json"],
    )
    checks = [
        check("health_ok", service.health()["status"] == "ok"),
        check("summary_has_claim_boundaries", bool(summary.get("claim_boundaries"))),
        check("summary_has_audit_status", bool(summary.get("audit_status"))),
        check("small_corpus_limitation_visible", bool(summary.get("corpus", {}).get("limitations"))),
        check("query_diagnostics_endpoint_documented", "POST /v1/query-diagnostics" in available_endpoints),
        check("evidence_packets_endpoint_documented", "POST /v1/evidence-packets" in available_endpoints),
        check("product_smoke_passed", smoke_report.get("status") == "pass"),
    ]
    report = {
        "mode": mode,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "endpoints": available_endpoints,
    }
    output = settings.reports_dir / f"product_readiness_{mode}.json"
    report_io.write_json(output, report)
    return report


def load_json_any(base_dir, names: list[str]) -> dict:
    for name in names:
        payload = load_json(base_dir / name)
        if payload:
            return payload
    return {}


def load_json(path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check(identifier: str, passed: bool) -> dict:
    return {"id": identifier, "passed": bool(passed)}


def endpoints() -> list[str]:
    return [
        "GET /health",
        "GET /v1/summary",
        "GET /v1/reports/audit",
        "GET /v1/reports/claim-decision",
        "GET /v1/reports/data-card",
        "GET /v1/reports/diversity",
        "GET /v1/reports/diversity-gate",
        "GET /v1/diversity/queries",
        "GET /v1/diversity/queries/{query_id}",
        "GET /v1/reports/regression-gate",
        "POST /v1/answer",
        "POST /v1/evidence-packets",
        "POST /v1/compare",
        "POST /v1/query-diagnostics",
        "POST /v1/diversity-audit",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON product readiness report.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    args = parser.parse_args()
    print(json.dumps(build_readiness_report(args.mode), indent=2))


if __name__ == "__main__":
    main()

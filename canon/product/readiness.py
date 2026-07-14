from __future__ import annotations

import argparse
import json

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


def build_readiness_report(mode: str = service.DEFAULT_MODE) -> dict:
    settings = load_settings()
    summary = service.product_summary(mode)
    available_endpoints = endpoints()
    smoke_report = load_json(settings.reports_dir / f"product_smoke_{mode}.json")
    checks = [
        check("health_ok", service.health()["status"] == "ok"),
        check("summary_has_claim_boundaries", bool(summary.get("claim_boundaries"))),
        check("summary_has_audit_status", summary.get("audit_status") == "pass"),
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

from __future__ import annotations

import argparse
import json

from canon.config import load_settings
from canon.product import service


def build_readiness_report(mode: str = service.DEFAULT_MODE) -> dict:
    settings = load_settings()
    summary = service.product_summary(mode)
    checks = [
        check("health_ok", service.health()["status"] == "ok"),
        check("summary_has_claim_boundaries", bool(summary.get("claim_boundaries"))),
        check("summary_has_audit_status", summary.get("audit_status") == "pass"),
        check("small_corpus_limitation_visible", bool(summary.get("corpus", {}).get("limitations"))),
    ]
    report = {
        "mode": mode,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "endpoints": [
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
            "POST /v1/compare",
            "POST /v1/diversity-audit",
        ],
    }
    output = settings.reports_dir / f"product_readiness_{mode}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def check(identifier: str, passed: bool) -> dict:
    return {"id": identifier, "passed": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON product readiness report.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    args = parser.parse_args()
    print(json.dumps(build_readiness_report(args.mode), indent=2))


if __name__ == "__main__":
    main()

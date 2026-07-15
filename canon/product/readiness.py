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
    route_metadata = routes_metadata()
    smoke_report = load_json(settings.reports_dir / f"product_smoke_{mode}.json")
    checks = [
        check("health_ok", service.health()["status"] == "ok"),
        check("summary_has_claim_boundaries", bool(summary.get("claim_boundaries"))),
        check("summary_has_audit_status", summary.get("audit_status") == "pass"),
        check("small_corpus_limitation_visible", bool(summary.get("corpus", {}).get("limitations"))),
        check("routes_metadata_endpoint_documented", "GET /v1/routes" in available_endpoints),
        check("routes_metadata_has_human_review_boundary", bool(route_metadata.get("human_review_boundary"))),
        check("routes_metadata_examples_present", all_post_routes_have_examples(route_metadata)),
        check("query_diagnostics_endpoint_documented", "POST /v1/query-diagnostics" in available_endpoints),
        check("evidence_packets_endpoint_documented", "POST /v1/evidence-packets" in available_endpoints),
        check("frame_coverage_endpoint_documented", "POST /v1/frame-coverage" in available_endpoints),
        check("intelligence_brief_endpoint_documented", "POST /v1/intelligence-brief" in available_endpoints),
        check("alert_digest_endpoint_documented", "POST /v1/alert-digest" in available_endpoints),
        check("flagship_handoff_endpoint_documented", "POST /v1/flagship-handoff" in available_endpoints),
        check("intelligence_review_endpoint_documented", "POST /v1/intelligence-review/prepare" in available_endpoints),
        check("product_smoke_passed", smoke_report.get("status") == "pass"),
    ]
    report = {
        "mode": mode,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "endpoints": available_endpoints,
        "route_metadata": {
            "route_count": len(route_metadata.get("routes", [])),
            "human_review_boundary": route_metadata.get("human_review_boundary"),
        },
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
        f"{route['method']} {route['path']}"
        for route in routes_metadata().get("routes", [])
    ]


def routes_metadata() -> dict:
    from canon.product.server import api_routes

    return api_routes()


def all_post_routes_have_examples(metadata: dict) -> bool:
    return all(
        route.get("example") is not None
        for route in metadata.get("routes", [])
        if route.get("method") == "POST"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON product readiness report.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    args = parser.parse_args()
    print(json.dumps(build_readiness_report(args.mode), indent=2))


if __name__ == "__main__":
    main()

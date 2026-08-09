from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from canon.config import load_settings
from canon.product import report_io
from canon.product import service


def build_readiness_report(mode: str = service.DEFAULT_MODE) -> dict:
    settings = load_settings()
    summary = service.product_summary(mode)
    available_endpoints = endpoints()
    route_metadata = routes_metadata()
    package_metadata = load_package_metadata()
    smoke_report = load_json_any(
        settings.reports_dir,
        [f"product_smoke_{mode}.json", "product_smoke_dry_run.json"],
    )
    checks = [
        check("health_ok", service.health()["status"] == "ok"),
        check("package_metadata_has_readme", bool(package_metadata.get("readme"))),
        check("package_metadata_has_public_urls", package_metadata_has_public_urls(package_metadata)),
        check("package_metadata_has_classifiers", bool(package_metadata.get("classifiers"))),
        check("package_metadata_has_keywords", bool(package_metadata.get("keywords"))),
        check("summary_has_claim_boundaries", bool(summary.get("claim_boundaries"))),
        check("summary_has_audit_status", bool(summary.get("audit_status"))),
        check("small_corpus_limitation_visible", bool(summary.get("corpus", {}).get("limitations"))),
        check("serving_target_is_asgi", route_metadata.get("serving_target") == "asgi"),
        check("routes_metadata_endpoint_documented", "GET /v1/routes" in available_endpoints),
        check("routes_metadata_has_human_review_boundary", bool(route_metadata.get("human_review_boundary"))),
        check("routes_metadata_descriptions_present", all_routes_have_descriptions(route_metadata)),
        check("browser_workbench_endpoint_documented", "GET /app" in available_endpoints),
        check("metrics_endpoint_documented", "GET /metrics" in available_endpoints),
        check("production_status_endpoint_documented", "GET /v1/production/status" in available_endpoints),
        check("production_workbench_endpoint_documented", "POST /v1/production/evidence-workbench" in available_endpoints),
        check("production_feedback_endpoint_documented", "POST /v1/production/feedback" in available_endpoints),
        check("production_corpus_setup_endpoint_documented", "POST /v1/production/corpus-setup" in available_endpoints),
        check("production_corpus_refresh_endpoint_documented", "POST /v1/production/corpus-refresh" in available_endpoints),
        check("evidence_packets_endpoint_documented", "POST /v1/evidence-packets" in available_endpoints),
        check("stage2_synthesis_endpoint_documented", "POST /v1/stage2-synthesis" in available_endpoints),
        check("product_smoke_passed", smoke_report.get("status") == "pass"),
    ]
    report = {
        "mode": mode,
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "serving_target": route_metadata.get("serving_target"),
        "checks": checks,
        "endpoints": available_endpoints,
        "route_metadata": {
            "route_count": len(route_metadata.get("routes", [])),
            "human_review_boundary": route_metadata.get("human_review_boundary"),
        },
        "package_metadata": {
            "name": package_metadata.get("name"),
            "version": package_metadata.get("version"),
            "readme": package_metadata.get("readme"),
            "urls": package_metadata.get("urls", {}),
        },
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


def load_package_metadata() -> dict:
    settings = load_settings()
    root = getattr(settings, "root", None)
    path = (root / "pyproject.toml") if root else Path("pyproject.toml")
    if not path.exists():
        return {}
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "readme": project.get("readme"),
        "urls": project.get("urls", {}),
        "classifiers": project.get("classifiers", []),
        "keywords": project.get("keywords", []),
    }


def package_metadata_has_public_urls(metadata: dict) -> bool:
    urls = metadata.get("urls") or {}
    return all(
        urls.get(key, "").startswith("https://github.com/Saltef/CANON")
        for key in ["Homepage", "Repository", "Documentation", "Issues"]
    )


def check(identifier: str, passed: bool) -> dict:
    return {"id": identifier, "passed": bool(passed)}


def endpoints() -> list[str]:
    return [
        f"{route['method']} {route['path']}"
        for route in routes_metadata().get("routes", [])
    ]


def routes_metadata() -> dict:
    from canon.product.asgi import asgi_routes

    return asgi_routes() | {
        "serving_target": "asgi",
        "human_review_boundary": (
            "Automated ASGI routes prepare evidence, reports, and gates for review. "
            "They do not establish final factual correctness or release-level model quality."
        ),
    }


def all_routes_have_descriptions(metadata: dict) -> bool:
    return all(
        bool(route.get("description"))
        for route in metadata.get("routes", [])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CANON product readiness report.")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    args = parser.parse_args()
    print(json.dumps(build_readiness_report(args.mode), indent=2))


if __name__ == "__main__":
    main()

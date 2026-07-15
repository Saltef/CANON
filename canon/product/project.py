from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io


DEFAULT_AI_INFRA_ONTOLOGY = {
    "ontology_id": "ai_infra_geo_risk_v1",
    "entities": {
        "companies": [
            "Nvidia",
            "AMD",
            "TSMC",
            "Samsung",
            "Intel",
            "Microsoft",
            "Google",
            "Amazon",
            "Meta",
            "Oracle",
            "CoreWeave",
        ],
        "infrastructure": [
            "GPUs",
            "accelerators",
            "HBM",
            "advanced packaging",
            "fabs",
            "data centers",
            "cloud regions",
            "substations",
            "grid interconnects",
            "cooling systems",
        ],
        "jurisdictions": ["US", "EU", "China", "Taiwan", "Gulf", "India", "Latin America", "Africa"],
        "institutions": [
            "regulators",
            "energy agencies",
            "export-control bodies",
            "ministries",
            "municipal permitting bodies",
        ],
        "communities": [
            "local residents",
            "labor groups",
            "environmental groups",
            "indigenous or local-rights groups",
        ],
    },
    "issue_categories": [
        "compute supply",
        "semiconductor controls",
        "advanced packaging",
        "cloud dependency",
        "sovereign AI",
        "data localization",
        "data center buildout",
        "energy demand",
        "grid bottlenecks",
        "water and cooling",
        "environmental impact",
        "permitting and local opposition",
        "labor and construction capacity",
        "public legitimacy",
        "national security",
        "cross-border investment",
        "sanctions and export controls",
        "regional industrial policy",
    ],
}

DEFAULT_REPORT_TYPES = [
    "weekly_intelligence_brief",
    "regional_risk_report",
    "company_exposure_report",
    "public_opinion_report",
    "alert_digest",
]


def build_project_config(
    project_name: str,
    domain: str,
    regions: list[str],
    languages: list[str],
    issue_categories: list[str],
    desired_report_types: list[str] | None = None,
    review_cadence: str = "weekly",
    project_id: str | None = None,
    source_boundaries: list[str] | None = None,
    corpus_id: str | None = None,
    ontology: dict[str, Any] | None = None,
    write_report: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    project_id = project_id or slug(project_name)
    desired_report_types = desired_report_types or DEFAULT_REPORT_TYPES
    ontology = ontology or DEFAULT_AI_INFRA_ONTOLOGY
    source_boundaries = source_boundaries or []
    source_plan = build_source_plan(source_boundaries, corpus_id)
    checks = validation_checks(
        regions=regions,
        languages=languages,
        issue_categories=issue_categories,
        desired_report_types=desired_report_types,
        source_boundaries=source_boundaries,
    )
    config = {
        "report_id": "intelligence_project_config_v1",
        "status": "ready" if all(check["passed"] for check in checks) else "needs_source_boundary",
        "project_id": project_id,
        "project_name": project_name,
        "domain": domain,
        "regions": regions,
        "languages": languages,
        "issue_categories": issue_categories,
        "desired_report_types": desired_report_types,
        "review_cadence": review_cadence,
        "ontology": ontology,
        "source_plan": source_plan,
        "monitoring": {
            "enabled": bool(source_boundaries),
            "allowed_only_after_source_boundary": True,
            "cadence": review_cadence,
            "blocked_reason": "" if source_boundaries else "source_boundary_required",
        },
        "validation": {"checks": checks},
        "security_boundary": {
            "api_keys_required": False,
            "credentials_stored": False,
            "raw_private_files_committed": False,
            "notes": [
                "Connect Drive by mounting it locally or passing a local folder; this config stores paths and boundaries only.",
                "Do not put tokens, credentials, private keys, or .env files inside corpus folders.",
                "Generated data and reports remain local and gitignored by default.",
            ],
        },
        "human_review_boundary": (
            "This project config declares scope and source boundaries. It does not prove corpus quality, "
            "model quality, or report correctness without reviewed qrels and reviewed reports."
        ),
    }
    if write_report:
        target_dir = output_dir or load_settings().reports_dir / "projects" / project_id
        report_io.write_json(target_dir / "project_config.json", config)
        report_io.write_markdown(target_dir / "project_config.md", render_markdown(config))
        config["output_paths"] = [
            str((target_dir / "project_config.json")).replace("\\", "/"),
            str((target_dir / "project_config.md")).replace("\\", "/"),
        ]
    return config


def build_source_plan(source_boundaries: list[str], corpus_id: str | None) -> dict[str, Any]:
    return {
        "corpus_id": corpus_id or "",
        "source_boundaries": source_boundaries,
        "initial_source_types": [
            "company announcements and filings",
            "official policy/regulatory sources",
            "reputable news/local media",
            "public-opinion/social samples",
            "policy/think-tank reports",
            "academic or technical reports",
        ],
        "external_expansion_policy": "private_corpus_first_then_controlled_expansion",
        "monitor_ready": bool(source_boundaries),
    }


def validation_checks(
    regions: list[str],
    languages: list[str],
    issue_categories: list[str],
    desired_report_types: list[str],
    source_boundaries: list[str],
) -> list[dict[str, Any]]:
    return [
        check("regions_declared", bool(regions), regions),
        check("languages_declared", bool(languages), languages),
        check("issue_categories_declared", bool(issue_categories), issue_categories),
        check("output_types_declared", bool(desired_report_types), desired_report_types),
        check("source_boundary_declared", bool(source_boundaries), source_boundaries),
    ]


def check(identifier: str, passed: bool, value: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "value": value}


def render_markdown(config: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- {'PASS' if item['passed'] else 'MISSING'}: `{item['id']}`"
        for item in config["validation"]["checks"]
    )
    boundaries = "\n".join(f"- {item}" for item in config["source_plan"]["source_boundaries"]) or "- Not declared."
    issues = "\n".join(f"- {item}" for item in config["issue_categories"])
    reports = "\n".join(f"- {item}" for item in config["desired_report_types"])
    return f"""# CANON Intelligence Project

Status: **{config['status']}**

Project: `{config['project_id']}` - {config['project_name']}

Domain: {config['domain']}

Review cadence: {config['review_cadence']}

## Source Boundaries

{boundaries}

## Issue Categories

{issues}

## Report Types

{reports}

## Validation

{checks}

## Human Review Boundary

{config['human_review_boundary']}
"""


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return value or "project"


def csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a scoped CANON intelligence project config.")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--project-id")
    parser.add_argument("--domain", default="AI infrastructure and geopolitical risk")
    parser.add_argument("--regions", required=True, help="Comma-separated monitored regions.")
    parser.add_argument("--languages", required=True, help="Comma-separated monitored languages.")
    parser.add_argument("--issue-categories", required=True, help="Comma-separated issue categories.")
    parser.add_argument("--report-types", default=",".join(DEFAULT_REPORT_TYPES))
    parser.add_argument("--review-cadence", default="weekly")
    parser.add_argument("--source-boundaries", default="", help="Comma-separated local folders, mounted drives, or corpus boundaries.")
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    config = build_project_config(
        project_name=args.project_name,
        project_id=args.project_id,
        domain=args.domain,
        regions=csv_list(args.regions),
        languages=csv_list(args.languages),
        issue_categories=csv_list(args.issue_categories),
        desired_report_types=csv_list(args.report_types),
        review_cadence=args.review_cadence,
        source_boundaries=csv_list(args.source_boundaries),
        corpus_id=args.corpus_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

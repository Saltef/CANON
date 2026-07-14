from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report


DEFAULT_THRESHOLDS = {
    "mixed_unstructured_final_type_coverage_min": 7,
    "document_type_final_coverage_min": 2,
    "domain_corpus_coverage_min": 5,
    "domain_final_coverage_min": 5,
    "data_card_document_type_coverage_min": 5,
    "data_card_domain_coverage_min": 5,
    "unknown_document_type_share_max": 0.2,
    "low_or_review_domain_fit_share_max": 0.35,
    "web_page_final_exposure_max": 0,
    "legal_market_safety_rejected_min": 1,
}


def run_unstructured_readiness_gate(
    document_type_mode: str = "unstructured_demo_corpus",
    domain_mode: str = "mixed_domain_demo_corpus",
    thresholds: dict[str, int | float] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    artifacts = load_artifacts(settings.reports_dir, document_type_mode, domain_mode)
    checks = readiness_checks(artifacts, resolved)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "heterogeneous_unstructured_readiness_v1",
        "document_type_mode": document_type_mode,
        "domain_mode": domain_mode,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "thresholds": resolved,
        "checks": checks,
        "artifact_summary": artifact_summary(artifacts),
        "claim_boundaries": claim_boundaries(),
        "recommended_next_steps": recommended_next_steps(checks),
        "limitations": [
            "This gate verifies experiment plumbing, artifact coverage, and deterministic safety fixtures; it is not a human-validity claim.",
            "The fixture corpora are intentionally small and should be replaced with larger, labeled, domain-specific corpora.",
            "Domain-fit and document-type profiles are deterministic priors; compare them against expert labels before using them as ranking policy.",
            "Passing this gate means the heterogeneous unstructured experiment bundle is internally coherent enough for inspection, not production-ready.",
        ],
    }
    report["contract_validation"] = validate_report(report, "heterogeneous_unstructured_readiness")
    if write_report:
        path = settings.reports_dir / f"heterogeneous_unstructured_readiness_{document_type_mode}_{domain_mode}.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), render_markdown(report))
    return report


def load_artifacts(reports_dir: Path, document_type_mode: str, domain_mode: str) -> dict[str, Any]:
    filenames = {
        "mixed_unstructured": "mixed_unstructured_document_routing_v1.json",
        "document_type_slices": f"document_type_slices_{document_type_mode}.json",
        "domain_slices": f"domain_slices_{domain_mode}.json",
        "document_type_data_card": f"data_card_{document_type_mode}.json",
        "domain_data_card": f"data_card_{domain_mode}.json",
    }
    artifacts = {}
    for key, filename in filenames.items():
        path = reports_dir / filename
        artifacts[key] = load_json(path)
        artifacts[f"{key}_path"] = str(path)
    return artifacts


def readiness_checks(artifacts: dict[str, Any], thresholds: dict[str, int | float]) -> list[dict[str, Any]]:
    mixed = artifacts.get("mixed_unstructured") or {}
    doc_slices = artifacts.get("document_type_slices") or {}
    domain_slices = artifacts.get("domain_slices") or {}
    doc_card = artifacts.get("document_type_data_card") or {}
    domain_card = artifacts.get("domain_data_card") or {}
    doc_slice_summary = doc_slices.get("slice_summary") or {}
    domain_slice_summary = domain_slices.get("slice_summary") or {}
    doc_profile = doc_card.get("document_type_profile") or {}
    domain_doc_profile = domain_card.get("document_type_profile") or {}
    domain_profile = domain_card.get("domain_profile") or {}
    checks = [
        check(
            "mixed_unstructured_contract",
            (mixed.get("contract_validation") or validate_report(mixed, "mixed_unstructured")).get("status") == "pass",
            "Mixed-unstructured benchmark satisfies its report contract.",
            artifacts.get("mixed_unstructured_path"),
        ),
        check(
            "mixed_unstructured_status",
            mixed.get("summary", {}).get("status") == "pass",
            "Mixed-unstructured routing benchmark passes.",
            mixed.get("summary", {}),
        ),
        check(
            "mixed_unstructured_final_type_coverage",
            int(mixed.get("summary", {}).get("final_type_coverage_count") or 0)
            >= thresholds["mixed_unstructured_final_type_coverage_min"],
            "Mixed-unstructured fixture covers enough final document types.",
            mixed.get("summary", {}),
            thresholds["mixed_unstructured_final_type_coverage_min"],
        ),
        check(
            "mixed_unstructured_no_attack_context",
            all(run.get("attack_context_count", 0) == 0 for run in mixed.get("query_runs") or []),
            "Prompt-injected fixture text does not reach final context.",
            [run.get("attack_context_count", 0) for run in mixed.get("query_runs") or []],
        ),
        check(
            "document_type_slices_contract",
            (doc_slices.get("contract_validation") or validate_report(doc_slices, "document_type_slices")).get("status") == "pass",
            "Document-type slice report satisfies its contract.",
            artifacts.get("document_type_slices_path"),
        ),
        check(
            "document_type_final_coverage",
            int(doc_slices.get("coverage_summary", {}).get("final_document_type_coverage_count") or 0)
            >= thresholds["document_type_final_coverage_min"],
            "Document-type slices expose more than one final document type.",
            doc_slices.get("coverage_summary", {}),
            thresholds["document_type_final_coverage_min"],
        ),
        check(
            "web_page_quarantined_from_final",
            int((doc_slice_summary.get("web_page") or {}).get("final_count") or 0)
            <= thresholds["web_page_final_exposure_max"],
            "Prompt-injected web-page fixture is not exposed as final evidence.",
            doc_slice_summary.get("web_page", {}),
            thresholds["web_page_final_exposure_max"],
        ),
        check(
            "web_page_safety_rejected",
            int((doc_slice_summary.get("web_page") or {}).get("safety_rejected_count") or 0) >= 1,
            "Prompt-injected web page is safety-rejected in document-type slices.",
            doc_slice_summary.get("web_page", {}),
        ),
        check(
            "domain_slices_contract",
            (domain_slices.get("contract_validation") or validate_report(domain_slices, "domain_slices")).get("status") == "pass",
            "Domain-slice report satisfies its contract.",
            artifacts.get("domain_slices_path"),
        ),
        check(
            "domain_corpus_coverage",
            int(domain_slices.get("coverage_summary", {}).get("corpus_domain_coverage_count") or 0)
            >= thresholds["domain_corpus_coverage_min"],
            "Domain-slice corpus contains enough distinct domains.",
            domain_slices.get("coverage_summary", {}),
            thresholds["domain_corpus_coverage_min"],
        ),
        check(
            "domain_final_coverage",
            int(domain_slices.get("coverage_summary", {}).get("final_domain_coverage_count") or 0)
            >= thresholds["domain_final_coverage_min"],
            "Domain slices expose enough distinct final domains.",
            domain_slices.get("coverage_summary", {}),
            thresholds["domain_final_coverage_min"],
        ),
        check(
            "legal_market_safety_rejected",
            int((domain_slice_summary.get("legal_market") or {}).get("safety_rejected_count") or 0)
            >= thresholds["legal_market_safety_rejected_min"],
            "Legal/market domain contains safety-rejected adversarial web evidence.",
            domain_slice_summary.get("legal_market", {}),
            thresholds["legal_market_safety_rejected_min"],
        ),
        check(
            "document_type_data_card_diversity",
            len(doc_profile.get("document_type_counts") or {}) >= 3,
            "Document-type data card exposes heterogeneous source forms.",
            doc_profile.get("document_type_counts", {}),
        ),
        check(
            "domain_data_card_document_type_coverage",
            len(domain_doc_profile.get("document_type_counts") or {})
            >= thresholds["data_card_document_type_coverage_min"],
            "Domain data card covers enough document types.",
            domain_doc_profile.get("document_type_counts", {}),
            thresholds["data_card_document_type_coverage_min"],
        ),
        check(
            "domain_data_card_domain_coverage",
            len(domain_profile.get("domain_counts") or {}) >= thresholds["data_card_domain_coverage_min"],
            "Domain data card covers enough declared domains.",
            domain_profile.get("domain_counts", {}),
            thresholds["data_card_domain_coverage_min"],
        ),
        check(
            "unknown_document_type_share",
            float(domain_doc_profile.get("unknown_type_share") or 0.0)
            <= thresholds["unknown_document_type_share_max"],
            "Unknown document-type share stays below threshold.",
            domain_doc_profile,
            thresholds["unknown_document_type_share_max"],
        ),
        check(
            "low_or_review_domain_fit_share",
            float(domain_profile.get("low_or_review_fit_share") or 0.0)
            <= thresholds["low_or_review_domain_fit_share_max"],
            "Low/review-needed domain-fit share stays below threshold.",
            domain_profile,
            thresholds["low_or_review_domain_fit_share_max"],
        ),
    ]
    return checks


def artifact_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    mixed = artifacts.get("mixed_unstructured") or {}
    doc_slices = artifacts.get("document_type_slices") or {}
    domain_slices = artifacts.get("domain_slices") or {}
    domain_card = artifacts.get("domain_data_card") or {}
    return {
        "mixed_unstructured": {
            "status": mixed.get("summary", {}).get("status"),
            "final_type_coverage_count": mixed.get("summary", {}).get("final_type_coverage_count"),
            "contract_status": (mixed.get("contract_validation") or {}).get("status"),
        },
        "document_type_slices": {
            "final_document_type_coverage_count": doc_slices.get("coverage_summary", {}).get("final_document_type_coverage_count"),
            "rejected_document_type_coverage": doc_slices.get("coverage_summary", {}).get("rejected_document_type_coverage", []),
            "contract_status": (doc_slices.get("contract_validation") or {}).get("status"),
        },
        "domain_slices": {
            "corpus_domain_coverage_count": domain_slices.get("coverage_summary", {}).get("corpus_domain_coverage_count"),
            "final_domain_coverage_count": domain_slices.get("coverage_summary", {}).get("final_domain_coverage_count"),
            "contract_status": (domain_slices.get("contract_validation") or {}).get("status"),
        },
        "domain_data_card": {
            "work_count": domain_card.get("work_count"),
            "document_type_counts": (domain_card.get("document_type_profile") or {}).get("document_type_counts", {}),
            "domain_counts": (domain_card.get("domain_profile") or {}).get("domain_counts", {}),
            "limitations": domain_card.get("limitations", []),
        },
    }


def claim_boundaries() -> list[str]:
    return [
        "Allowed: the current pipeline can ingest generic JSONL, profile document type and domain fit, and run deterministic safety gates before reranking.",
        "Allowed: fixture evidence shows prompt-injected web text can be retrieved as a candidate and rejected before final context.",
        "Allowed: document-type and domain-slice reports can expose policy sensitivity across heterogeneous unstructured corpora.",
        "Not allowed: claim validated source quality, domain-quality correctness, or production prompt-injection robustness without human/external benchmarks.",
        "Not allowed: claim broad performance across economics, psychology, anthropology, history, cultural studies, legal, and market corpora from the current fixtures alone.",
    ]


def recommended_next_steps(checks: list[dict[str, Any]]) -> list[str]:
    failed = [item["id"] for item in checks if not item["passed"]]
    if failed:
        return [
            f"Fix failed readiness checks: {', '.join(failed)}.",
            "Regenerate the relevant reports and rerun this gate before interpreting heterogeneous results.",
        ]
    return [
        "Replace fixtures with larger labeled corpora for each target domain.",
        "Add human labels for document type, evidence role, source trust, safety disposition, and answer usefulness.",
        "Add external qrels or task anchors for legal/market, history, anthropology, psychology, economics, and cultural-studies use cases.",
        "Compare deterministic domain-fit priors against learned calibration and pairwise preference models.",
    ]


def check(
    identifier: str,
    passed: bool,
    description: str,
    evidence: Any = None,
    threshold: int | float | None = None,
) -> dict[str, Any]:
    row = {
        "id": identifier,
        "passed": bool(passed),
        "description": description,
    }
    if threshold is not None:
        row["threshold"] = threshold
    if evidence is not None:
        row["evidence"] = evidence
    return row


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] `{item['id']}`: {item['description']}"
        for item in report["checks"]
    )
    boundaries = "\n".join(f"- {item}" for item in report["claim_boundaries"])
    steps = "\n".join(f"- {item}" for item in report["recommended_next_steps"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Heterogeneous Unstructured Readiness

- Status: {report["status"]}
- Document-type mode: {report["document_type_mode"]}
- Domain mode: {report["domain_mode"]}

## Checks

{checks}

## Claim Boundaries

{boundaries}

## Recommended Next Steps

{steps}

## Limitations

{limitations}
"""


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heterogeneous unstructured experiment readiness gate.")
    parser.add_argument("--document-type-mode", default="unstructured_demo_corpus")
    parser.add_argument("--domain-mode", default="mixed_domain_demo_corpus")
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_unstructured_readiness_gate(
        document_type_mode=args.document_type_mode,
        domain_mode=args.domain_mode,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

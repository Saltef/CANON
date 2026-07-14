from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.anchors import build_anchor_report
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.quality.document_types import DOCUMENT_TYPES, document_type_profile
from canon.quality.domain_profiles import DOMAIN_QUALITY_PROFILES, domain_quality_profile
from canon.retrieval.corpus import work_model_from_dict


DEFAULT_MODES = [
    "unstructured_demo_corpus",
    "mixed_domain_demo_corpus",
    "social_public_opinion_demo_corpus",
]

CORE_LABEL_FAMILIES = [
    "document_type",
    "evidence_role",
    "source_trust",
    "source_independence",
    "safety_disposition",
    "answer_usefulness",
    "chunking_quality",
]

TASK_REQUIREMENTS = [
    {
        "task_id": "academic_synthesis",
        "domains": ["general_academic", "economics", "psychology"],
        "required_document_types": ["academic_article", "preprint", "policy_report"],
        "required_labels": ["evidence_role", "source_trust", "answer_usefulness", "chunking_quality"],
    },
    {
        "task_id": "interpretive_primary_source_analysis",
        "domains": ["anthropology", "history", "cultural_studies"],
        "required_document_types": ["archival_primary_source", "transcript", "news_article", "web_page"],
        "required_labels": ["evidence_role", "source_trust", "source_independence", "answer_usefulness", "chunking_quality"],
    },
    {
        "task_id": "legal_market_due_diligence",
        "domains": ["legal_market", "economics"],
        "required_document_types": ["legal_authority", "regulatory_filing", "market_report", "news_article", "transcript"],
        "required_labels": ["source_trust", "source_independence", "safety_disposition", "answer_usefulness", "chunking_quality"],
    },
    {
        "task_id": "public_opinion_aggregation",
        "domains": ["public_opinion"],
        "required_document_types": ["social_media_post", "forum_comment", "news_article", "transcript"],
        "required_labels": ["document_type", "safety_disposition", "source_independence", "answer_usefulness", "chunking_quality"],
    },
    {
        "task_id": "adversarial_rag_security",
        "domains": ["public_opinion", "legal_market", "cultural_studies"],
        "required_document_types": ["web_page", "social_media_post", "forum_comment", "transcript"],
        "required_labels": ["safety_disposition", "evidence_role", "answer_usefulness", "chunking_quality"],
    },
]


def build_unstructured_coverage_matrix(
    modes: list[str] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_modes = modes or DEFAULT_MODES
    works = load_mode_works(settings.data_dir, resolved_modes)
    anchors = load_anchor_report(settings.reports_dir)
    domain_rows = domain_coverage_rows(works, anchors)
    document_type_rows = document_type_coverage_rows(works)
    task_rows = task_coverage_rows(domain_rows, document_type_rows, anchors)
    gap_summary = summarize_gaps(domain_rows, document_type_rows, task_rows, anchors)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "unstructured_experiment_coverage_matrix_v1",
        "modes": resolved_modes,
        "status": matrix_status(gap_summary),
        "work_count": len(works),
        "domain_rows": domain_rows,
        "document_type_rows": document_type_rows,
        "task_rows": task_rows,
        "label_anchor_summary": label_anchor_summary(anchors),
        "gap_summary": gap_summary,
        "claim_boundaries": claim_boundaries(),
        "recommended_next_steps": recommended_next_steps(gap_summary),
        "limitations": limitations(),
    }
    report["contract_validation"] = validate_report(report, "unstructured_coverage_matrix")
    if write_report:
        path = settings.reports_dir / "unstructured_experiment_coverage_matrix_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), render_markdown(report))
    return report


def load_mode_works(data_dir: Path, modes: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for mode in modes:
        path = data_dir / "processed" / f"works_{mode}.json"
        for work in load_json(path, []):
            work_id = str(work.get("id") or "")
            key = f"{mode}:{work_id}"
            if key in seen:
                continue
            seen.add(key)
            row = dict(work)
            row["coverage_matrix_mode"] = mode
            rows.append(row)
    return rows


def domain_coverage_rows(works: list[dict[str, Any]], anchors: dict[str, Any]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for work in works:
        domain = normalized_domain(work)
        by_domain[domain].append(work)
    rows = []
    for domain in sorted(DOMAIN_QUALITY_PROFILES):
        profile = domain_quality_profile(domain)
        domain_works = by_domain.get(domain, [])
        observed_types = sorted({profiled_document_type(work) for work in domain_works})
        preferred = list(profile.preferred_document_types)
        covered_preferred = sorted(set(observed_types) & set(preferred))
        missing_preferred = sorted(set(preferred) - set(observed_types))
        rows.append(
            {
                "domain": domain,
                "work_count": len(domain_works),
                "observed_document_types": observed_types,
                "preferred_document_types": preferred,
                "preferred_document_type_coverage": ratio(len(covered_preferred), len(preferred)),
                "covered_preferred_document_types": covered_preferred,
                "missing_preferred_document_types": missing_preferred,
                "critical_quality_signals": list(profile.critical_quality_signals),
                "required_gates": list(profile.gate_signals),
                "label_readiness": domain_label_readiness(domain, anchors),
                "evidence_level": evidence_level(len(domain_works), len(covered_preferred), len(preferred)),
                "priority_gaps": domain_priority_gaps(domain, len(domain_works), missing_preferred, anchors),
            }
        )
    return rows


def document_type_coverage_rows(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(profiled_document_type(work) for work in works)
    examples: dict[str, list[str]] = defaultdict(list)
    for work in works:
        doc_type = profiled_document_type(work)
        if len(examples[doc_type]) < 5:
            examples[doc_type].append(str(work.get("id") or work.get("title") or "unknown"))
    rows = []
    for doc_type in sorted(DOCUMENT_TYPES):
        count = counts.get(doc_type, 0)
        rows.append(
            {
                "document_type": doc_type,
                "work_count": count,
                "evidence_level": document_type_evidence_level(count),
                "example_work_ids": examples.get(doc_type, []),
            }
        )
    return rows


def task_coverage_rows(
    domain_rows: list[dict[str, Any]],
    document_type_rows: list[dict[str, Any]],
    anchors: dict[str, Any],
) -> list[dict[str, Any]]:
    domain_by_id = {row["domain"]: row for row in domain_rows}
    doc_counts = {row["document_type"]: int(row["work_count"]) for row in document_type_rows}
    label_status = label_anchor_status(anchors)
    rows = []
    for task in TASK_REQUIREMENTS:
        covered_domains = [domain for domain in task["domains"] if (domain_by_id.get(domain) or {}).get("work_count", 0) > 0]
        covered_doc_types = [doc_type for doc_type in task["required_document_types"] if doc_counts.get(doc_type, 0) > 0]
        missing_labels = [label for label in task["required_labels"] if label_status.get(label) != "active"]
        rows.append(
            {
                "task_id": task["task_id"],
                "domains": task["domains"],
                "required_document_types": task["required_document_types"],
                "covered_domains": covered_domains,
                "covered_document_types": covered_doc_types,
                "missing_document_types": sorted(set(task["required_document_types"]) - set(covered_doc_types)),
                "required_labels": task["required_labels"],
                "missing_or_limited_labels": missing_labels,
                "coverage_status": task_status(task, covered_domains, covered_doc_types, missing_labels),
            }
        )
    return rows


def summarize_gaps(
    domain_rows: list[dict[str, Any]],
    document_type_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    domain_levels = Counter(row["evidence_level"] for row in domain_rows)
    doc_levels = Counter(row["evidence_level"] for row in document_type_rows)
    task_levels = Counter(row["coverage_status"] for row in task_rows)
    missing_domains = [row["domain"] for row in domain_rows if row["work_count"] == 0]
    fixture_only_domains = [
        row["domain"]
        for row in domain_rows
        if row["evidence_level"] in {"fixture_partial", "fixture_preferred_types_present"}
    ]
    missing_document_types = [row["document_type"] for row in document_type_rows if row["work_count"] == 0]
    return {
        "domain_evidence_levels": dict(sorted(domain_levels.items())),
        "document_type_evidence_levels": dict(sorted(doc_levels.items())),
        "task_coverage_status_counts": dict(sorted(task_levels.items())),
        "missing_domains": missing_domains,
        "fixture_only_domains": fixture_only_domains,
        "missing_document_types": missing_document_types,
        "human_label_status": label_anchor_summary(anchors),
        "ready_for_broad_validity_claims": False,
    }


def matrix_status(gap_summary: dict[str, Any]) -> str:
    if gap_summary.get("missing_domains") or gap_summary.get("missing_document_types"):
        return "needs_expansion"
    label_summary = gap_summary.get("human_label_status") or {}
    if label_summary.get("active_human_label_anchor_count", 0) < 2:
        return "needs_human_labels"
    if gap_summary.get("fixture_only_domains"):
        return "fixture_ready_needs_scale"
    return "ready_for_broad_validation"


def evidence_level(work_count: int, covered_preferred_count: int, preferred_count: int) -> str:
    if work_count <= 0:
        return "missing"
    if work_count < 50:
        return "fixture_preferred_types_present" if covered_preferred_count > 0 else "fixture_partial"
    if covered_preferred_count < max(1, preferred_count // 2):
        return "expanded_but_document_type_gaps"
    return "expanded_candidate"


def document_type_evidence_level(count: int) -> str:
    if count <= 0:
        return "missing"
    if count < 50:
        return "fixture_only"
    return "expanded_candidate"


def task_status(
    task: dict[str, Any],
    covered_domains: list[str],
    covered_doc_types: list[str],
    missing_labels: list[str],
) -> str:
    if len(covered_domains) < len(task["domains"]) or len(covered_doc_types) < len(task["required_document_types"]):
        return "needs_corpus_expansion"
    if missing_labels:
        return "needs_human_or_external_labels"
    return "ready_for_validation_run"


def domain_priority_gaps(
    domain: str,
    work_count: int,
    missing_preferred: list[str],
    anchors: dict[str, Any],
) -> list[str]:
    gaps = []
    if work_count == 0:
        gaps.append("add_domain_corpus")
    elif work_count < 50:
        gaps.append("scale_beyond_fixture")
    if missing_preferred:
        gaps.append("add_preferred_document_types:" + ",".join(missing_preferred[:5]))
    labels = domain_label_readiness(domain, anchors)
    if labels["status"] != "active":
        gaps.append("add_human_or_external_labels")
    return gaps


def domain_label_readiness(domain: str, anchors: dict[str, Any]) -> dict[str, Any]:
    statuses = label_anchor_status(anchors)
    missing = [label for label in CORE_LABEL_FAMILIES if statuses.get(label) != "active"]
    return {
        "domain": domain,
        "status": "active" if not missing else "planned_or_proxy_only",
        "missing_or_limited_label_families": missing,
    }


def label_anchor_status(anchors: dict[str, Any]) -> dict[str, str]:
    status = {label: "missing" for label in CORE_LABEL_FAMILIES}
    for anchor in anchors.get("anchors") or []:
        anchor_id = str(anchor.get("anchor_id") or "")
        anchor_status = str(anchor.get("status") or "")
        resolved = "active" if anchor_status.startswith("active") else anchor_status or "missing"
        if "evidence_role" in anchor_id:
            status["evidence_role"] = resolved
        if "source_trust" in anchor_id:
            status["source_trust"] = resolved
        if "chunking_quality" in anchor_id:
            status["chunking_quality"] = resolved
        if "rag_security" in anchor_id or "hard_negative" in anchor_id:
            status["safety_disposition"] = "active"
        if "corroboration" in anchor_id or "hard_negative" in anchor_id:
            status["source_independence"] = "active"
    task_coverage = anchors.get("label_task_coverage") or {}
    for label in task_coverage.get("covered_label_families") or []:
        label = str(label)
        if label in status and status[label] in {"missing", "planned", "task_scaffold_ready"}:
            status[label] = "task_scaffold_ready"
    return status


def label_anchor_summary(anchors: dict[str, Any]) -> dict[str, Any]:
    statuses = label_anchor_status(anchors)
    human = [
        anchor
        for anchor in anchors.get("anchors") or []
        if str(anchor.get("anchor_type") or "") == "human_labeling"
    ]
    active_human = [anchor for anchor in human if str(anchor.get("status") or "").startswith("active")]
    return {
        "label_family_status": statuses,
        "human_label_anchor_count": len(human),
        "active_human_label_anchor_count": len(active_human),
        "planned_or_limited_label_families": [label for label, status in statuses.items() if status != "active"],
    }


def load_anchor_report(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "evaluation_anchor_registry_v1.json"
    anchors = load_json(path, {}) or build_anchor_report()
    label_tasks = load_json(reports_dir / "unstructured_human_label_tasks_v1.json", {})
    coverage = label_tasks.get("coverage_summary") or {}
    if coverage:
        anchors = dict(anchors)
        anchors["label_task_coverage"] = coverage
    return anchors


def normalized_domain(work: dict[str, Any]) -> str:
    raw = work.get("raw") or {}
    return str(work.get("domain") or raw.get("domain") or "general_academic")


def profiled_document_type(work: dict[str, Any]) -> str:
    return document_type_profile(work_model_from_dict(work)).document_type


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def claim_boundaries() -> list[str]:
    return [
        "Allowed: use this matrix to decide which domains, document types, tasks, and labels are currently covered by the experiment design.",
        "Allowed: treat fixture coverage as plumbing evidence, not as field-level evidence.",
        "Not allowed: claim broad validity for a domain while its row is missing, fixture-only, or missing human/external labels.",
        "Not allowed: treat deterministic label proxies as substitutes for expert source-trust, evidence-role, or public-opinion labels.",
    ]


def recommended_next_steps(gap_summary: dict[str, Any]) -> list[str]:
    steps = []
    if gap_summary.get("missing_domains"):
        steps.append("Add corpora for missing domains: " + ", ".join(gap_summary["missing_domains"]))
    if gap_summary.get("missing_document_types"):
        steps.append("Add source forms for missing document types: " + ", ".join(gap_summary["missing_document_types"]))
    if gap_summary.get("fixture_only_domains"):
        steps.append("Scale fixture-only domains into larger sampled corpora with documented inclusion criteria.")
    steps.append("Create stratified human labels for evidence role, source trust, source independence, safety disposition, answer usefulness, and chunking quality.")
    steps.append("Attach external qrels or expert review sets to each task family before claiming broad performance.")
    return steps


def limitations() -> list[str]:
    return [
        "Coverage rows are based on processed corpus metadata and deterministic document-type profiles.",
        "A covered document type does not imply quality, representativeness, or task relevance.",
        "The matrix is an experiment-design audit; it is not an evaluation score.",
    ]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    domain_rows = "\n".join(
        "| {domain} | {work_count} | {types} | {level} | {gaps} |".format(
            domain=row["domain"],
            work_count=row["work_count"],
            types=", ".join(row["observed_document_types"]) or "none",
            level=row["evidence_level"],
            gaps=", ".join(row["priority_gaps"]) or "none",
        )
        for row in report["domain_rows"]
    )
    task_rows = "\n".join(
        f"| {row['task_id']} | {row['coverage_status']} | {', '.join(row['missing_document_types']) or 'none'} | {', '.join(row['missing_or_limited_labels']) or 'none'} |"
        for row in report["task_rows"]
    )
    steps = "\n".join(f"- {item}" for item in report["recommended_next_steps"])
    boundaries = "\n".join(f"- {item}" for item in report["claim_boundaries"])
    return f"""# Unstructured Experiment Coverage Matrix

- Status: {report["status"]}
- Works: {report["work_count"]}
- Modes: {", ".join(report["modes"])}

## Domain Coverage

| Domain | Works | Observed document types | Evidence level | Priority gaps |
| --- | ---: | --- | --- | --- |
{domain_rows}

## Task Coverage

| Task | Status | Missing document types | Missing or limited labels |
| --- | --- | --- | --- |
{task_rows}

## Label Anchors

```json
{json.dumps(report["label_anchor_summary"], indent=2)}
```

## Claim Boundaries

{boundaries}

## Recommended Next Steps

{steps}
"""


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CANON unstructured experiment coverage matrix.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    print(json.dumps(build_unstructured_coverage_matrix(modes=modes, write_report=not args.no_write_report), indent=2))


if __name__ == "__main__":
    main()

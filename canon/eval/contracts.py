from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "canon_rag_contract_v1"

DECISIONS = {"allow", "reject"}
SAFETY_DECISIONS = {"allow", "sanitize", "quarantine", "block"}
REPORT_TYPES = {
    "retrieval",
    "synthesis",
    "adversarial_rag_security",
    "adversarial_corroboration",
    "evaluation_anchors",
    "mixed_unstructured",
    "document_type_slices",
    "domain_slices",
    "heterogeneous_unstructured_readiness",
    "unstructured_coverage_matrix",
    "unstructured_experiment_portfolio",
    "phase_gate",
    "committee_gate_comparison",
    "committee_gate_usefulness",
    "syntax_retrieval_ablation",
    "unstructured_label_tasks",
    "unstructured_label_calibration",
    "chunking_strategy_eval",
    "chunking_variant_eval",
}
SIGNAL_FAMILIES = {
    "query_text_relevance",
    "syntax_profile",
    "evidence_role",
    "source_trust",
    "provenance",
}
COMPONENT_KEYS = {
    "relevance",
    "semantic_similarity",
    "syntax_relevance",
    "source_quality",
    "section_role",
    "claim_density",
    "method_signal",
    "limitation_signal",
    "evidence_specificity",
    "uncertainty_signal",
}


@dataclass
class ContractAudit:
    report_type: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_counts: dict[str, int] = field(default_factory=dict)

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(f"{path}: {message}")

    def count(self, key: str, amount: int = 1) -> None:
        self.checked_counts[key] = self.checked_counts.get(key, 0) + amount

    def to_report(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "report_type": self.report_type,
            "status": "pass" if not self.errors else "fail",
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_counts": dict(sorted(self.checked_counts.items())),
        }


def validate_report(report: dict[str, Any], report_type: str | None = None) -> dict[str, Any]:
    resolved_type = report_type or infer_report_type(report)
    audit = ContractAudit(resolved_type)
    if resolved_type not in REPORT_TYPES:
        audit.error("$", f"unknown report type '{resolved_type}'")
        return audit.to_report()
    if report.get("contract_version") != CONTRACT_VERSION:
        audit.warn("$", f"missing or unexpected contract_version; expected {CONTRACT_VERSION}")
    if resolved_type == "retrieval":
        validate_retrieval_report(report, audit)
    elif resolved_type == "synthesis":
        validate_synthesis_report(report, audit)
    elif resolved_type == "adversarial_rag_security":
        validate_security_report(report, audit)
    elif resolved_type == "adversarial_corroboration":
        validate_corroboration_report(report, audit)
    elif resolved_type == "evaluation_anchors":
        validate_anchor_report(report, audit)
    elif resolved_type == "mixed_unstructured":
        validate_mixed_unstructured_report(report, audit)
    elif resolved_type == "document_type_slices":
        validate_document_type_slices_report(report, audit)
    elif resolved_type == "domain_slices":
        validate_domain_slices_report(report, audit)
    elif resolved_type == "heterogeneous_unstructured_readiness":
        validate_unstructured_readiness_report(report, audit)
    elif resolved_type == "unstructured_coverage_matrix":
        validate_unstructured_coverage_matrix_report(report, audit)
    elif resolved_type == "unstructured_experiment_portfolio":
        validate_unstructured_portfolio_report(report, audit)
    elif resolved_type == "phase_gate":
        validate_phase_gate_report(report, audit)
    elif resolved_type == "committee_gate_comparison":
        validate_committee_gate_comparison_report(report, audit)
    elif resolved_type == "committee_gate_usefulness":
        validate_committee_gate_usefulness_report(report, audit)
    elif resolved_type == "syntax_retrieval_ablation":
        validate_syntax_retrieval_ablation_report(report, audit)
    elif resolved_type == "unstructured_label_tasks":
        validate_unstructured_label_tasks_report(report, audit)
    elif resolved_type == "unstructured_label_calibration":
        validate_unstructured_label_calibration_report(report, audit)
    elif resolved_type == "chunking_strategy_eval":
        validate_chunking_strategy_eval_report(report, audit)
    elif resolved_type == "chunking_variant_eval":
        validate_chunking_variant_eval_report(report, audit)
    return audit.to_report()


def infer_report_type(report: dict[str, Any]) -> str:
    if report.get("benchmark_id") == "adversarial_rag_security_v1":
        return "adversarial_rag_security"
    if report.get("benchmark_id") == "adversarial_corroboration_v1":
        return "adversarial_corroboration"
    if report.get("report_id") == "evaluation_anchor_registry_v1":
        return "evaluation_anchors"
    if report.get("report_id") == "importance_phase_gate_v1":
        return "phase_gate"
    if report.get("report_id") == "committee_gate_policy_comparison_v1":
        return "committee_gate_comparison"
    if report.get("report_id") == "committee_gate_usefulness_audit_v1":
        return "committee_gate_usefulness"
    if report.get("report_id") == "syntax_retrieval_ablation_v1":
        return "syntax_retrieval_ablation"
    if report.get("report_id") == "unstructured_human_label_tasks_v1":
        return "unstructured_label_tasks"
    if report.get("report_id") == "unstructured_label_calibration_v1":
        return "unstructured_label_calibration"
    if report.get("report_id") == "chunking_strategy_eval_v1":
        return "chunking_strategy_eval"
    if report.get("report_id") == "chunking_variant_eval_v1":
        return "chunking_variant_eval"
    if report.get("benchmark_id") == "mixed_unstructured_document_routing_v1":
        return "mixed_unstructured"
    if report.get("report_id") == "document_type_slices_v1":
        return "document_type_slices"
    if report.get("report_id") == "domain_slices_v1":
        return "domain_slices"
    if report.get("report_id") == "heterogeneous_unstructured_readiness_v1":
        return "heterogeneous_unstructured_readiness"
    if report.get("report_id") == "unstructured_experiment_coverage_matrix_v1":
        return "unstructured_coverage_matrix"
    if report.get("report_id") == "unstructured_experiment_portfolio_v1":
        return "unstructured_experiment_portfolio"
    if "answer" in report and "post_generation_verification" in report:
        return "synthesis"
    if "stage_summary" in report and "results" in report:
        return "retrieval"
    return "unknown"


def validate_retrieval_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(report, "$", ["query", "policy", "mode", "top_k", "stage_summary", "results"], audit)
    validate_stage_summary(report.get("stage_summary"), "$.stage_summary", audit)
    results = expect_list(report.get("results"), "$.results", audit)
    for index, result in enumerate(results):
        validate_retrieval_result(result, f"$.results[{index}]", audit)
    rejected = expect_list(report.get("rejected_candidates", []), "$.rejected_candidates", audit)
    for index, row in enumerate(rejected):
        validate_rejected_candidate(row, f"$.rejected_candidates[{index}]", audit)
    audit.count("retrieval_results", len(results))
    audit.count("rejected_candidates", len(rejected))


def validate_synthesis_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "query",
            "answer",
            "citations",
            "evidence",
            "context_packet",
            "generation_payload",
            "post_generation_verification",
            "retrieval",
        ],
        audit,
    )
    validate_retrieval_report(report.get("retrieval") or {}, nest("$.retrieval", audit))
    evidence = expect_list(report.get("evidence"), "$.evidence", audit)
    citations = expect_list(report.get("citations"), "$.citations", audit)
    citation_ids = {citation.get("citation_id") for citation in citations if isinstance(citation, dict)}
    for index, item in enumerate(evidence):
        validate_evidence_item(item, f"$.evidence[{index}]", audit)
        citation_id = item.get("citation_id") if isinstance(item, dict) else None
        if citation_id and citation_id not in citation_ids:
            audit.error(f"$.evidence[{index}].citation_id", "missing from citations")
    validate_context_packet(report.get("context_packet"), "$.context_packet", audit)
    verification = expect_dict(
        report.get("post_generation_verification"),
        "$.post_generation_verification",
        audit,
    )
    if verification and verification.get("status") not in {"pass", "fail"}:
        audit.error("$.post_generation_verification.status", "must be pass or fail")
    audit.count("evidence_items", len(evidence))
    audit.count("citations", len(citations))


def validate_security_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        ["benchmark_id", "query", "metrics", "checks", "status", "stage_summary", "verification"],
        audit,
    )
    if report.get("benchmark_id") != "adversarial_rag_security_v1":
        audit.error("$.benchmark_id", "unexpected benchmark id")
    validate_stage_summary(report.get("stage_summary"), "$.stage_summary", audit)
    metrics = expect_dict(report.get("metrics"), "$.metrics", audit)
    for key in [
        "attack_candidate_count",
        "attack_rejected_before_rerank_count",
        "attack_final_result_count",
        "attack_context_exposure_count",
        "benign_context_count",
        "benign_retention_rate",
    ]:
        if key not in metrics:
            audit.error(f"$.metrics.{key}", "missing metric")
    if metrics:
        if float(metrics.get("attack_final_result_count", 0.0) or 0.0) != 0.0:
            audit.error("$.metrics.attack_final_result_count", "attacks reached final results")
        if float(metrics.get("attack_context_exposure_count", 0.0) or 0.0) != 0.0:
            audit.error("$.metrics.attack_context_exposure_count", "attacks reached context")
        if float(metrics.get("benign_retention_rate", 0.0) or 0.0) <= 0.0:
            audit.error("$.metrics.benign_retention_rate", "no benign evidence retained")
    checks = expect_dict(report.get("checks"), "$.checks", audit)
    if report.get("status") == "pass" and checks and not all(bool(value) for value in checks.values()):
        audit.error("$.checks", "pass report contains a failed check")
    for index, row in enumerate(expect_list(report.get("rejected_candidates", []), "$.rejected_candidates", audit)):
        validate_rejected_candidate(row, f"$.rejected_candidates[{index}]", audit)
    verification = expect_dict(report.get("verification"), "$.verification", audit)
    if verification and verification.get("status") != "pass":
        audit.error("$.verification.status", "security benchmark generation verification must pass")


def validate_corroboration_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(report, "$", ["benchmark_id", "cases", "checks", "status"], audit)
    if report.get("benchmark_id") != "adversarial_corroboration_v1":
        audit.error("$.benchmark_id", "unexpected benchmark id")
    checks = expect_dict(report.get("checks"), "$.checks", audit)
    if report.get("status") == "pass" and checks and not all(bool(value) for value in checks.values()):
        audit.error("$.checks", "pass report contains a failed check")
    cases = expect_dict(report.get("cases"), "$.cases", audit)
    for case_name, case in cases.items():
        summary = expect_dict(case.get("summary") if isinstance(case, dict) else None, f"$.cases.{case_name}.summary", audit)
        if summary and "overall_verdict" not in summary:
            audit.error(f"$.cases.{case_name}.summary.overall_verdict", "missing verdict")
        audit.count("corroboration_cases")


def validate_anchor_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(report, "$", ["report_id", "anchors", "claim_limits"], audit)
    if report.get("report_id") != "evaluation_anchor_registry_v1":
        audit.error("$.report_id", "unexpected report id")
    anchors = expect_list(report.get("anchors"), "$.anchors", audit)
    claim_limits = expect_list(report.get("claim_limits"), "$.claim_limits", audit)
    if not claim_limits:
        audit.error("$.claim_limits", "must contain overclaiming limits")
    for index, anchor in enumerate(anchors):
        require_keys(
            anchor if isinstance(anchor, dict) else {},
            f"$.anchors[{index}]",
            ["anchor_id", "anchor_type", "domain", "status", "purpose", "limitation", "next_step"],
            audit,
        )
    audit.count("anchors", len(anchors))
    audit.count("claim_limits", len(claim_limits))


def validate_phase_gate_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        ["report_id", "phase_count", "phases", "artifact_contracts", "limitations", "status"],
        audit,
    )
    if report.get("report_id") != "importance_phase_gate_v1":
        audit.error("$.report_id", "unexpected report id")
    phases = expect_list(report.get("phases"), "$.phases", audit)
    phase_count = int(report.get("phase_count") or 0)
    if phase_count != len(phases):
        audit.error("$.phase_count", "must match phases length")
    if phase_count < 8:
        audit.error("$.phase_count", "must cover at least phases A-H")
    for index, phase in enumerate(phases):
        row = expect_dict(phase, f"$.phases[{index}]", audit)
        if not row:
            continue
        require_keys(row, f"$.phases[{index}]", ["phase", "name", "status", "checks"], audit)
        if row.get("status") not in {"pass", "fail"}:
            audit.error(f"$.phases[{index}].status", "must be pass or fail")
    if report.get("status") == "pass" and any(
        phase.get("status") != "pass" for phase in phases if isinstance(phase, dict)
    ):
        audit.error("$.status", "phase gate cannot pass when a phase fails")
    audit.count("phases", len(phases))


def validate_mixed_unstructured_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        ["benchmark_id", "document_count", "query_count", "corpus_profile", "query_runs", "summary", "limitations"],
        audit,
    )
    if report.get("benchmark_id") != "mixed_unstructured_document_routing_v1":
        audit.error("$.benchmark_id", "unexpected benchmark id")
    runs = expect_list(report.get("query_runs"), "$.query_runs", audit)
    if int(report.get("query_count") or 0) != len(runs):
        audit.error("$.query_count", "must match query_runs length")
    if int(report.get("document_count") or 0) <= 0:
        audit.error("$.document_count", "must include documents")
    profile = expect_dict(report.get("corpus_profile"), "$.corpus_profile", audit)
    if profile and len(profile.get("document_type_counts") or {}) < 5:
        audit.error("$.corpus_profile.document_type_counts", "must cover multiple document types")
    for index, run in enumerate(runs):
        row = expect_dict(run, f"$.query_runs[{index}]", audit)
        if not row:
            continue
        require_keys(
            row,
            f"$.query_runs[{index}]",
            ["id", "query", "expected_document_types", "checks", "status", "final_results", "rejected_candidates"],
            audit,
        )
        if row.get("status") == "pass" and not all(bool(value) for value in (row.get("checks") or {}).values()):
            audit.error(f"$.query_runs[{index}].checks", "pass run contains a failed check")
        if int(row.get("attack_context_count") or 0) != 0:
            audit.error(f"$.query_runs[{index}].attack_context_count", "attack reached final context")
    summary = expect_dict(report.get("summary"), "$.summary", audit)
    if summary:
        if summary.get("status") == "pass" and any(run.get("status") != "pass" for run in runs if isinstance(run, dict)):
            audit.error("$.summary.status", "benchmark cannot pass when a query fails")
        if int(summary.get("final_type_coverage_count") or 0) < 5:
            audit.error("$.summary.final_type_coverage_count", "must cover heterogeneous final types")
    audit.count("mixed_unstructured_queries", len(runs))


def validate_document_type_slices_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "mode",
            "document_count",
            "query_count",
            "policy_ids",
            "top_k",
            "corpus_document_type_counts",
            "runs",
            "slice_summary",
            "coverage_summary",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "document_type_slices_v1":
        audit.error("$.report_id", "unexpected report id")
    if int(report.get("document_count") or 0) <= 0:
        audit.error("$.document_count", "must include documents")
    policies = expect_list(report.get("policy_ids"), "$.policy_ids", audit)
    runs = expect_list(report.get("runs"), "$.runs", audit)
    if int(report.get("query_count") or 0) > 0 and policies:
        expected_runs = int(report.get("query_count") or 0) * len(policies)
        if expected_runs != len(runs):
            audit.error("$.runs", "must contain query_count * policy_ids runs")
    if not expect_dict(report.get("corpus_document_type_counts"), "$.corpus_document_type_counts", audit):
        audit.error("$.corpus_document_type_counts", "must include at least one document type")
    for index, run in enumerate(runs):
        row = expect_dict(run, f"$.runs[{index}]", audit)
        if not row:
            continue
        require_keys(
            row,
            f"$.runs[{index}]",
            [
                "id",
                "query",
                "intent",
                "policy",
                "candidate_document_type_counts",
                "final_document_type_counts",
                "rejected_document_type_counts",
                "candidate_safety_decision_counts",
                "candidate_slice_stats",
                "stage_summary",
                "candidate_sample",
                "final_results",
                "rejected_candidates",
            ],
            audit,
        )
        validate_stage_summary(row.get("stage_summary"), f"$.runs[{index}].stage_summary", audit)
        expect_dict(row.get("candidate_document_type_counts"), f"$.runs[{index}].candidate_document_type_counts", audit)
        expect_dict(row.get("final_document_type_counts"), f"$.runs[{index}].final_document_type_counts", audit)
        expect_dict(row.get("rejected_document_type_counts"), f"$.runs[{index}].rejected_document_type_counts", audit)
        expect_dict(row.get("candidate_slice_stats"), f"$.runs[{index}].candidate_slice_stats", audit)
        for item_index, item in enumerate(expect_list(row.get("candidate_sample"), f"$.runs[{index}].candidate_sample", audit)):
            require_keys(
                item if isinstance(item, dict) else {},
                f"$.runs[{index}].candidate_sample[{item_index}]",
                ["candidate_rank", "chunk_id", "work_id", "document_type", "safety_decision", "gate_decision"],
                audit,
            )
        for item_index, item in enumerate(expect_list(row.get("final_results"), f"$.runs[{index}].final_results", audit)):
            require_keys(
                item if isinstance(item, dict) else {},
                f"$.runs[{index}].final_results[{item_index}]",
                ["rank", "chunk_id", "work_id", "title", "final_score", "document_type", "safety_decision"],
                audit,
            )
        audit.count("document_type_slice_runs")
    summary = expect_dict(report.get("slice_summary"), "$.slice_summary", audit)
    for doc_type, row in summary.items():
        row = expect_dict(row, f"$.slice_summary.{doc_type}", audit)
        if row:
            require_keys(
                row,
                f"$.slice_summary.{doc_type}",
                [
                    "candidate_count",
                    "final_count",
                    "rejected_count",
                    "final_exposure_share",
                    "average_final_rank",
                ],
                audit,
            )
    coverage = expect_dict(report.get("coverage_summary"), "$.coverage_summary", audit)
    if coverage:
        require_keys(
            coverage,
            "$.coverage_summary",
            [
                "corpus_document_type_coverage_count",
                "final_document_type_coverage_count",
                "final_document_type_coverage",
                "corpus_types_missing_from_final_results",
            ],
            audit,
        )
    audit.count("document_type_slices", len(summary))


def validate_domain_slices_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "mode",
            "document_count",
            "query_count",
            "policy_ids",
            "top_k",
            "corpus_domain_counts",
            "runs",
            "slice_summary",
            "coverage_summary",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "domain_slices_v1":
        audit.error("$.report_id", "unexpected report id")
    if int(report.get("document_count") or 0) <= 0:
        audit.error("$.document_count", "must include documents")
    policies = expect_list(report.get("policy_ids"), "$.policy_ids", audit)
    runs = expect_list(report.get("runs"), "$.runs", audit)
    if int(report.get("query_count") or 0) > 0 and policies:
        expected_runs = int(report.get("query_count") or 0) * len(policies)
        if expected_runs != len(runs):
            audit.error("$.runs", "must contain query_count * policy_ids runs")
    if not expect_dict(report.get("corpus_domain_counts"), "$.corpus_domain_counts", audit):
        audit.error("$.corpus_domain_counts", "must include at least one domain")
    for index, run in enumerate(runs):
        row = expect_dict(run, f"$.runs[{index}]", audit)
        if not row:
            continue
        require_keys(
            row,
            f"$.runs[{index}]",
            [
                "id",
                "query",
                "intent",
                "policy",
                "candidate_domain_counts",
                "final_domain_counts",
                "rejected_domain_counts",
                "candidate_domain_fit_tier_counts",
                "candidate_safety_decision_counts",
                "candidate_slice_stats",
                "stage_summary",
                "candidate_sample",
                "final_results",
                "rejected_candidates",
            ],
            audit,
        )
        validate_stage_summary(row.get("stage_summary"), f"$.runs[{index}].stage_summary", audit)
        expect_dict(row.get("candidate_domain_counts"), f"$.runs[{index}].candidate_domain_counts", audit)
        expect_dict(row.get("final_domain_counts"), f"$.runs[{index}].final_domain_counts", audit)
        expect_dict(row.get("rejected_domain_counts"), f"$.runs[{index}].rejected_domain_counts", audit)
        for item_index, item in enumerate(expect_list(row.get("candidate_sample"), f"$.runs[{index}].candidate_sample", audit)):
            require_keys(
                item if isinstance(item, dict) else {},
                f"$.runs[{index}].candidate_sample[{item_index}]",
                ["candidate_rank", "chunk_id", "work_id", "domain_profile", "domain_fit_score", "domain_fit_tier"],
                audit,
            )
        audit.count("domain_slice_runs")
    summary = expect_dict(report.get("slice_summary"), "$.slice_summary", audit)
    for domain, row in summary.items():
        row = expect_dict(row, f"$.slice_summary.{domain}", audit)
        if row:
            require_keys(
                row,
                f"$.slice_summary.{domain}",
                [
                    "candidate_count",
                    "final_count",
                    "rejected_count",
                    "final_exposure_share",
                    "average_domain_fit_score",
                    "average_final_domain_fit_score",
                ],
                audit,
            )
    coverage = expect_dict(report.get("coverage_summary"), "$.coverage_summary", audit)
    if coverage:
        require_keys(
            coverage,
            "$.coverage_summary",
            [
                "corpus_domain_coverage_count",
                "final_domain_coverage_count",
                "final_domain_coverage",
                "corpus_domains_missing_from_final_results",
            ],
            audit,
        )
    audit.count("domain_slices", len(summary))


def validate_unstructured_readiness_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "document_type_mode",
            "domain_mode",
            "status",
            "thresholds",
            "checks",
            "artifact_summary",
            "claim_boundaries",
            "recommended_next_steps",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "heterogeneous_unstructured_readiness_v1":
        audit.error("$.report_id", "unexpected report id")
    if report.get("status") not in {"pass", "fail"}:
        audit.error("$.status", "must be pass or fail")
    checks = expect_list(report.get("checks"), "$.checks", audit)
    if not checks:
        audit.error("$.checks", "must include readiness checks")
    for index, item in enumerate(checks):
        row = expect_dict(item, f"$.checks[{index}]", audit)
        if not row:
            continue
        require_keys(row, f"$.checks[{index}]", ["id", "passed", "description"], audit)
        if not isinstance(row.get("passed"), bool):
            audit.error(f"$.checks[{index}].passed", "must be boolean")
    if report.get("status") == "pass" and any(
        item.get("passed") is not True for item in checks if isinstance(item, dict)
    ):
        audit.error("$.status", "readiness cannot pass when a check fails")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed claims")
    if not expect_dict(report.get("artifact_summary"), "$.artifact_summary", audit):
        audit.error("$.artifact_summary", "must summarize supporting artifacts")
    audit.count("unstructured_readiness_checks", len(checks))


def validate_unstructured_portfolio_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "status",
            "modes",
            "checks",
            "artifact_summary",
            "claim_boundaries",
            "recommended_next_steps",
            "limitations",
            "artifact_paths",
        ],
        audit,
    )
    if report.get("report_id") != "unstructured_experiment_portfolio_v1":
        audit.error("$.report_id", "unexpected report id")
    if report.get("status") not in {"pass", "fail"}:
        audit.error("$.status", "must be pass or fail")
    modes = expect_dict(report.get("modes"), "$.modes", audit)
    if modes:
        require_keys(modes, "$.modes", ["document_type_mode", "domain_mode", "public_opinion_mode"], audit)
    checks = expect_list(report.get("checks"), "$.checks", audit)
    if len(checks) < 8:
        audit.error("$.checks", "must include the cross-artifact portfolio checks")
    for index, item in enumerate(checks):
        row = expect_dict(item, f"$.checks[{index}]", audit)
        if not row:
            continue
        require_keys(row, f"$.checks[{index}]", ["id", "passed", "description"], audit)
        if not isinstance(row.get("passed"), bool):
            audit.error(f"$.checks[{index}].passed", "must be boolean")
    if report.get("status") == "pass" and any(
        item.get("passed") is not True for item in checks if isinstance(item, dict)
    ):
        audit.error("$.status", "portfolio cannot pass when a check fails")
    check_ids = {str(item.get("id") or "") for item in checks if isinstance(item, dict)}
    required_check_ids = {
        "boundary_chunking_isolates_unsafe_social_span",
        "large_window_chunking_isolates_mixed_unsafe_social_span",
        "committee_gate_usefulness_audit_ready_for_labels",
        "chunking_variant_comparison_tracks_tradeoffs",
        "public_opinion_collection_audit_present",
        "syntax_retrieval_ablation_tracks_form_signal",
        "unstructured_human_label_tasks_ready_for_annotation",
        "unstructured_label_calibration_honest_status",
    }
    for check_id in sorted(required_check_ids - check_ids):
        audit.error("$.checks", f"missing required portfolio check '{check_id}'")
    if not expect_dict(report.get("artifact_summary"), "$.artifact_summary", audit):
        audit.error("$.artifact_summary", "must summarize supporting artifacts")
    artifact_paths = expect_dict(report.get("artifact_paths"), "$.artifact_paths", audit)
    if not artifact_paths:
        audit.error("$.artifact_paths", "must list supporting artifacts")
    if artifact_paths and "chunking_large_window" not in artifact_paths:
        audit.error("$.artifact_paths.chunking_large_window", "must list the large-window chunking probe artifact")
    if artifact_paths and "chunking_variants" not in artifact_paths:
        audit.error("$.artifact_paths.chunking_variants", "must list the chunking variant benchmark artifact")
    if artifact_paths and "unstructured_label_tasks" not in artifact_paths:
        audit.error("$.artifact_paths.unstructured_label_tasks", "must list the unstructured human-label task artifact")
    if artifact_paths and "unstructured_label_calibration" not in artifact_paths:
        audit.error("$.artifact_paths.unstructured_label_calibration", "must list the unstructured label calibration artifact")
    if artifact_paths and "syntax_retrieval_ablation" not in artifact_paths:
        audit.error("$.artifact_paths.syntax_retrieval_ablation", "must list the syntax retrieval ablation artifact")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 5:
        audit.error("$.claim_boundaries", "must include allowed and disallowed portfolio claims")
    audit.count("unstructured_portfolio_checks", len(checks))


def validate_unstructured_label_tasks_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "modes",
            "schema",
            "label_family_to_fields",
            "task_count",
            "tasks",
            "coverage_summary",
            "claim_boundaries",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "unstructured_human_label_tasks_v1":
        audit.error("$.report_id", "unexpected report id")
    schema = expect_dict(report.get("schema"), "$.schema", audit)
    family_map = expect_dict(report.get("label_family_to_fields"), "$.label_family_to_fields", audit)
    required_families = {
        "document_type",
        "evidence_role",
        "source_trust",
        "source_independence",
        "safety_disposition",
        "answer_usefulness",
    }
    missing_families = sorted(required_families - set(family_map))
    for family in missing_families:
        audit.error("$.label_family_to_fields", f"missing label family '{family}'")
    tasks = expect_list(report.get("tasks"), "$.tasks", audit)
    if int(report.get("task_count") or 0) != len(tasks):
        audit.error("$.task_count", "must match tasks length")
    if not tasks:
        audit.error("$.tasks", "must include label tasks")
    observed_families: set[str] = set()
    for index, task in enumerate(tasks):
        row = expect_dict(task, f"$.tasks[{index}]", audit)
        if not row:
            continue
        require_keys(
            row,
            f"$.tasks[{index}]",
            [
                "id",
                "task_family",
                "label_families",
                "mode",
                "domain",
                "document_type",
                "deterministic_signals",
                "priority_score",
                "priority_reasons",
                "human_labels",
                "review_instructions",
            ],
            audit,
        )
        label_families = expect_list(row.get("label_families"), f"$.tasks[{index}].label_families", audit)
        observed_families.update(str(family) for family in label_families)
        labels = expect_dict(row.get("human_labels"), f"$.tasks[{index}].human_labels", audit)
        expected_fields: set[str] = set()
        for family in label_families:
            fields = family_map.get(str(family), []) if isinstance(family_map, dict) else []
            if not isinstance(fields, list):
                audit.error(f"$.label_family_to_fields.{family}", "must be a list")
                continue
            expected_fields.update(str(field) for field in fields)
        missing_fields = sorted(expected_fields - set(labels))
        for field in missing_fields:
            audit.error(f"$.tasks[{index}].human_labels.{field}", "missing human label slot")
        unexpected_filled = [field for field, value in labels.items() if value is not None]
        if unexpected_filled:
            audit.error(f"$.tasks[{index}].human_labels", "new label tasks should have empty human label slots")
        if not isinstance(row.get("priority_score"), int | float):
            audit.error(f"$.tasks[{index}].priority_score", "must be numeric")
        if not expect_list(row.get("review_instructions"), f"$.tasks[{index}].review_instructions", audit):
            audit.error(f"$.tasks[{index}].review_instructions", "must include annotator guidance")
    coverage = expect_dict(report.get("coverage_summary"), "$.coverage_summary", audit)
    if coverage:
        require_keys(
            coverage,
            "$.coverage_summary",
            ["label_family_counts", "domain_counts", "document_type_counts", "missing_label_families", "ready_for_annotation"],
            audit,
        )
        missing = coverage.get("missing_label_families")
        if isinstance(missing, list) and missing:
            audit.error("$.coverage_summary.missing_label_families", "must cover all required label families")
        if coverage.get("ready_for_annotation") is not True:
            audit.error("$.coverage_summary.ready_for_annotation", "must be true when task pack covers required families")
    for family in sorted(required_families - observed_families):
        audit.error("$.tasks", f"no task covers label family '{family}'")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed label-task claims")
    audit.count("unstructured_label_tasks", len(tasks))


def validate_unstructured_label_calibration_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "status",
            "task_count",
            "completed_task_count",
            "completed_label_count",
            "schema",
            "technical_label_distribution",
            "agreement",
            "label_family_summary",
            "rows",
            "claim_boundaries",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "unstructured_label_calibration_v1":
        audit.error("$.report_id", "unexpected report id")
    if report.get("status") not in {"needs_labels", "calibrated"}:
        audit.error("$.status", "must be needs_labels or calibrated")
    task_count = int(report.get("task_count") or 0)
    completed_task_count = int(report.get("completed_task_count") or 0)
    completed_label_count = int(report.get("completed_label_count") or 0)
    rows = expect_list(report.get("rows"), "$.rows", audit)
    if task_count != len(rows):
        audit.error("$.task_count", "must match rows length")
    if report.get("status") == "needs_labels" and completed_label_count != 0:
        audit.error("$.status", "needs_labels report cannot contain completed labels")
    if report.get("status") == "calibrated" and completed_label_count <= 0:
        audit.error("$.status", "calibrated report must contain completed labels")
    if completed_task_count > task_count:
        audit.error("$.completed_task_count", "cannot exceed task_count")
    agreement = expect_dict(report.get("agreement"), "$.agreement", audit)
    if agreement:
        require_keys(agreement, "$.agreement", ["overall_comparison_count", "overall_exact_agreement_rate", "by_label"], audit)
        comparison_count = int(agreement.get("overall_comparison_count") or 0)
        if completed_label_count == 0 and comparison_count != 0:
            audit.error("$.agreement.overall_comparison_count", "must be zero when no labels are completed")
        if completed_label_count > 0 and comparison_count <= 0:
            audit.error("$.agreement.overall_comparison_count", "must be positive when labels are completed")
    for index, row_value in enumerate(rows):
        row = expect_dict(row_value, f"$.rows[{index}]", audit)
        if not row:
            continue
        require_keys(
            row,
            f"$.rows[{index}]",
            [
                "task_id",
                "task_family",
                "completed_human_label_count",
                "human_labels",
                "technical_labels",
                "agreement",
                "exact_agreement_rate",
                "disagreement_labels",
            ],
            audit,
        )
        if not isinstance(row.get("completed_human_label_count"), int | float):
            audit.error(f"$.rows[{index}].completed_human_label_count", "must be numeric")
        if int(row.get("completed_human_label_count") or 0) > 0 and not expect_dict(row.get("agreement"), f"$.rows[{index}].agreement", audit):
            audit.error(f"$.rows[{index}].agreement", "completed row must include label comparisons")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed calibration claims")
    audit.count("unstructured_label_calibration_rows", len(rows))


def validate_chunking_strategy_eval_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        ["report_id", "case_count", "strategies", "chunk_tokens", "overlap_tokens", "summary", "cases", "limitations"],
        audit,
    )
    if report.get("report_id") != "chunking_strategy_eval_v1":
        audit.error("$.report_id", "unexpected chunking report id")
    strategies = set(expect_list(report.get("strategies"), "$.strategies", audit))
    if not {"fixed_window", "boundary_aware"}.issubset(strategies):
        audit.error("$.strategies", "must include fixed_window and boundary_aware")
    summary = expect_dict(report.get("summary"), "$.summary", audit)
    fixed = expect_dict(summary.get("fixed_window"), "$.summary.fixed_window", audit) if summary else {}
    boundary = expect_dict(summary.get("boundary_aware"), "$.summary.boundary_aware", audit) if summary else {}
    for label, row in [("fixed_window", fixed), ("boundary_aware", boundary)]:
        if not row:
            continue
        require_keys(
            row,
            f"$.summary.{label}",
            [
                "case_count",
                "average_boundary_aligned_share",
                "average_span_containment_rate",
                "forbidden_merge_count",
                "unsafe_isolated_count",
                "unsafe_case_count",
                "document_type_counts",
            ],
            audit,
        )
        audit.count(f"chunking_summary_{label}", 1)
    cases = expect_list(report.get("cases"), "$.cases", audit)
    case_count = int(report.get("case_count") or 0)
    if case_count <= 0:
        audit.error("$.case_count", "must be positive")
    if cases and strategies and case_count and len(cases) < case_count * min(2, len(strategies)):
        audit.error("$.cases", "must include rows for each case and strategy")
    for index, row in enumerate(cases):
        validate_chunking_case_row(row, f"$.cases[{index}]", audit)
    if fixed and boundary:
        if float(boundary.get("average_boundary_aligned_share", 0.0) or 0.0) <= float(
            fixed.get("average_boundary_aligned_share", 0.0) or 0.0
        ):
            audit.error("$.summary.boundary_aware.average_boundary_aligned_share", "must exceed fixed-window boundary alignment")
        if float(boundary.get("average_span_containment_rate", 0.0) or 0.0) < float(
            fixed.get("average_span_containment_rate", 0.0) or 0.0
        ):
            audit.error("$.summary.boundary_aware.average_span_containment_rate", "must not underperform fixed-window span containment")
        if int(report.get("chunk_tokens") or 0) >= 1000:
            if int(boundary.get("forbidden_merge_count") or 0) != 0:
                audit.error("$.summary.boundary_aware.forbidden_merge_count", "large-window boundary-aware run must avoid forbidden merges")
            if int(fixed.get("forbidden_merge_count") or 0) <= 0:
                audit.error("$.summary.fixed_window.forbidden_merge_count", "large-window fixed-window baseline must expose a forbidden-merge regression")
            if int(boundary.get("unsafe_isolated_count") or 0) <= int(fixed.get("unsafe_isolated_count") or 0):
                audit.error("$.summary.boundary_aware.unsafe_isolated_count", "large-window boundary-aware run must isolate more unsafe spans than fixed windows")
    audit.count("chunking_cases", len(cases))


def validate_chunking_case_row(row: Any, path: str, audit: ContractAudit) -> None:
    row = expect_dict(row, path, audit)
    if not row:
        return
    require_keys(
        row,
        path,
        [
            "case_id",
            "strategy",
            "document_type",
            "domain",
            "chunking_profile",
            "must_keep_span_count",
            "contained_span_count",
            "span_containment_rate",
            "forbidden_merge",
            "unsafe_isolation",
            "chunks",
        ],
        audit,
    )
    profile = expect_dict(row.get("chunking_profile"), f"{path}.chunking_profile", audit)
    if profile:
        require_keys(
            profile,
            f"{path}.chunking_profile",
            [
                "chunk_count",
                "boundary_aligned_share",
                "strategy_counts",
                "chunk_resolution_counts",
                "context_expansion_policy_counts",
                "chunk_quality_tier_counts",
            ],
            audit,
        )
    forbidden = expect_dict(row.get("forbidden_merge"), f"{path}.forbidden_merge", audit)
    if forbidden:
        require_keys(
            forbidden,
            f"{path}.forbidden_merge",
            ["forbidden_pair_count", "forbidden_merge_count", "forbidden_merges_avoided"],
            audit,
        )


def validate_chunking_variant_eval_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "case_count",
            "variants",
            "variant_descriptions",
            "chunk_tokens",
            "overlap_tokens",
            "summary",
            "cases",
            "best_variant_by_objective",
            "claim_boundaries",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "chunking_variant_eval_v1":
        audit.error("$.report_id", "unexpected chunking variant report id")
    variants = expect_list(report.get("variants"), "$.variants", audit)
    variant_set = set(str(variant) for variant in variants)
    required = {"fixed_window", "boundary_aware", "atomic_sentence", "semantic_cohesion", "embedding_semantic"}
    for variant in sorted(required - variant_set):
        audit.error("$.variants", f"missing required variant '{variant}'")
    summary = expect_dict(report.get("summary"), "$.summary", audit)
    for variant in variants:
        row = expect_dict(summary.get(variant), f"$.summary.{variant}", audit) if summary else {}
        if not row:
            continue
        validate_chunking_variant_summary(row, f"$.summary.{variant}", audit)
    cases = expect_list(report.get("cases"), "$.cases", audit)
    case_count = int(report.get("case_count") or 0)
    if case_count <= 0:
        audit.error("$.case_count", "must be positive")
    if cases and variants and case_count and len(cases) < case_count * len(variants):
        audit.error("$.cases", "must include a row for each case and variant")
    for index, row in enumerate(cases):
        validate_chunking_case_row(row, f"$.cases[{index}]", audit)
    fixed = summary.get("fixed_window") if isinstance(summary, dict) else None
    boundary = summary.get("boundary_aware") if isinstance(summary, dict) else None
    semantic = summary.get("semantic_cohesion") if isinstance(summary, dict) else None
    embedding = summary.get("embedding_semantic") if isinstance(summary, dict) else None
    atomic = summary.get("atomic_sentence") if isinstance(summary, dict) else None
    if int(report.get("chunk_tokens") or 0) >= 1000 and all(isinstance(row, dict) for row in [fixed, boundary, semantic, embedding, atomic]):
        if int(fixed.get("forbidden_merge_count") or 0) <= 0:
            audit.error("$.summary.fixed_window.forbidden_merge_count", "large-window baseline must expose forbidden merges")
        for variant, row in [("boundary_aware", boundary), ("semantic_cohesion", semantic), ("embedding_semantic", embedding), ("atomic_sentence", atomic)]:
            if int(row.get("forbidden_merge_count") or 0) != 0:
                audit.error(f"$.summary.{variant}.forbidden_merge_count", "safer technical variants must avoid forbidden merges")
        fixed_utility = ((fixed.get("objective_scores") or {}).get("balanced_utility") or 0.0)
        boundary_utility = ((boundary.get("objective_scores") or {}).get("balanced_utility") or 0.0)
        if float(boundary_utility) < float(fixed_utility):
            audit.error("$.summary.boundary_aware.objective_scores.balanced_utility", "boundary-aware utility should not underperform fixed windows")
    winners = expect_dict(report.get("best_variant_by_objective"), "$.best_variant_by_objective", audit)
    if winners:
        for objective in ["span_preservation", "forbidden_merge_avoidance", "unsafe_isolation", "balanced_utility", "safety_first"]:
            if objective not in winners:
                audit.error(f"$.best_variant_by_objective.{objective}", "missing objective winner")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed variant claims")
    audit.count("chunking_variant_cases", len(cases))
    audit.count("chunking_variant_count", len(variants))


def validate_chunking_variant_summary(row: dict[str, Any], path: str, audit: ContractAudit) -> None:
    require_keys(
        row,
        path,
        [
            "variant_family",
            "case_count",
            "average_chunk_count",
            "average_span_containment_rate",
            "forbidden_merge_count",
            "unsafe_isolated_count",
            "unsafe_case_count",
            "objective_scores",
        ],
        audit,
    )
    scores = expect_dict(row.get("objective_scores"), f"{path}.objective_scores", audit)
    if scores:
        for key in ["span_preservation", "forbidden_merge_avoidance", "unsafe_isolation", "fragmentation_penalty", "balanced_utility", "safety_first"]:
            if key not in scores:
                audit.error(f"{path}.objective_scores.{key}", "missing objective score")
            elif not isinstance(scores.get(key), int | float):
                audit.error(f"{path}.objective_scores.{key}", "must be numeric")


def validate_committee_gate_comparison_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "baseline_policy",
            "gated_policy",
            "top_k",
            "modes",
            "summary",
            "claim_boundaries",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "committee_gate_policy_comparison_v1":
        audit.error("$.report_id", "unexpected report id")
    modes = expect_list(report.get("modes"), "$.modes", audit)
    if not modes:
        audit.error("$.modes", "must include at least one mode")
    for mode_index, mode in enumerate(modes):
        row = expect_dict(mode, f"$.modes[{mode_index}]", audit)
        if not row:
            continue
        require_keys(row, f"$.modes[{mode_index}]", ["mode", "document_count", "query_count", "pairs", "summary"], audit)
        pairs = expect_list(row.get("pairs"), f"$.modes[{mode_index}].pairs", audit)
        if int(row.get("query_count") or 0) != len(pairs):
            audit.error(f"$.modes[{mode_index}].query_count", "must match pairs length")
        validate_committee_comparison_summary(row.get("summary"), f"$.modes[{mode_index}].summary", audit)
        for pair_index, pair in enumerate(pairs):
            validate_committee_comparison_pair(pair, f"$.modes[{mode_index}].pairs[{pair_index}]", audit)
        audit.count("committee_gate_modes")
        audit.count("committee_gate_pairs", len(pairs))
    validate_committee_comparison_summary(report.get("summary"), "$.summary", audit)
    summary = expect_dict(report.get("summary"), "$.summary", audit)
    if summary:
        if int(summary.get("gated_selected_context_policy_violation_count") or 0) > int(
            summary.get("baseline_selected_context_policy_violation_count") or 0
        ):
            audit.error("$.summary.gated_selected_context_policy_violation_count", "gated policy should not increase policy violations")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed comparison claims")


def validate_committee_gate_usefulness_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "baseline_policy",
            "gated_policy",
            "top_k",
            "label_schema",
            "modes",
            "summary",
            "claim_boundaries",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "committee_gate_usefulness_audit_v1":
        audit.error("$.report_id", "unexpected report id")
    schema = expect_dict(report.get("label_schema"), "$.label_schema", audit)
    if schema:
        for label in ["human_evidence_usefulness", "human_exclusion_correctness", "human_abstention_correctness"]:
            if label not in schema:
                audit.error(f"$.label_schema.{label}", "missing human label dimension")
    modes = expect_list(report.get("modes"), "$.modes", audit)
    if not modes:
        audit.error("$.modes", "must include at least one mode")
    for mode_index, mode in enumerate(modes):
        row = expect_dict(mode, f"$.modes[{mode_index}]", audit)
        if not row:
            continue
        require_keys(row, f"$.modes[{mode_index}]", ["mode", "document_count", "query_count", "queries", "summary"], audit)
        queries = expect_list(row.get("queries"), f"$.modes[{mode_index}].queries", audit)
        if int(row.get("query_count") or 0) != len(queries):
            audit.error(f"$.modes[{mode_index}].query_count", "must match queries length")
        validate_usefulness_summary(row.get("summary"), f"$.modes[{mode_index}].summary", audit)
        for query_index, query_row in enumerate(queries):
            validate_usefulness_query(query_row, f"$.modes[{mode_index}].queries[{query_index}]", audit)
        audit.count("committee_gate_usefulness_modes")
        audit.count("committee_gate_usefulness_queries", len(queries))
    validate_usefulness_summary(report.get("summary"), "$.summary", audit)
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed usefulness claims")


def validate_syntax_retrieval_ablation_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "modes",
            "document_count",
            "case_count",
            "top_k",
            "variants",
            "rows",
            "summary",
            "checks",
            "status",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "syntax_retrieval_ablation_v1":
        audit.error("$.report_id", "unexpected report id")
    variants = expect_list(report.get("variants"), "$.variants", audit)
    variant_ids = {str(row.get("id")) for row in variants if isinstance(row, dict)}
    required_variants = {"lexical_semantic_no_syntax", "syntax_candidate_only", "syntax_candidate_and_rerank"}
    for variant in sorted(required_variants - variant_ids):
        audit.error(f"$.variants.{variant}", "missing required ablation variant")
    rows = expect_list(report.get("rows"), "$.rows", audit)
    for index, row in enumerate(rows):
        validate_syntax_ablation_row(row, f"$.rows[{index}]", audit)
    if int(report.get("case_count") or 0) <= 0:
        audit.error("$.case_count", "must be positive")
    if int(report.get("top_k") or 0) <= 0:
        audit.error("$.top_k", "must be positive")
    checks = expect_list(report.get("checks"), "$.checks", audit)
    check_ids = {str(row.get("id")) for row in checks if isinstance(row, dict)}
    for required_check in ["syntax_not_worse_than_no_syntax_mrr", "no_unsafe_final_exposure"]:
        if required_check not in check_ids:
            audit.error(f"$.checks.{required_check}", "missing required syntax ablation check")
    if report.get("status") == "pass":
        for index, row in enumerate(checks):
            if isinstance(row, dict) and row.get("passed") is not True:
                audit.error(f"$.checks[{index}].passed", "pass report contains failed check")
    validate_syntax_ablation_summary(report.get("summary"), "$.summary", audit)
    audit.count("syntax_ablation_rows", len(rows))


def validate_syntax_ablation_row(row_value: Any, path: str, audit: ContractAudit) -> None:
    row = expect_dict(row_value, path, audit)
    if not row:
        return
    require_keys(
        row,
        path,
        [
            "case_id",
            "query",
            "variant",
            "expected_work_ids",
            "expected_syntax_categories",
            "candidate_hit",
            "final_hit",
            "final_reciprocal_rank",
            "top_syntax_category",
            "syntax_category_match",
            "unsafe_final_count",
            "candidate_sample",
        ],
        audit,
    )
    expect_list(row.get("expected_work_ids"), f"{path}.expected_work_ids", audit)
    expect_list(row.get("expected_syntax_categories"), f"{path}.expected_syntax_categories", audit)
    sample = expect_list(row.get("candidate_sample"), f"{path}.candidate_sample", audit)
    for sample_index, sample_row in enumerate(sample):
        sample_dict = expect_dict(sample_row, f"{path}.candidate_sample[{sample_index}]", audit)
        if sample_dict:
            require_keys(
                sample_dict,
                f"{path}.candidate_sample[{sample_index}]",
                ["rank", "work_id", "candidate_score", "syntax_relevance", "syntax_category"],
                audit,
            )
    for key in ["final_reciprocal_rank", "unsafe_final_count"]:
        if key in row and not isinstance(row.get(key), int | float):
            audit.error(f"{path}.{key}", "must be numeric")


def validate_syntax_ablation_summary(summary_value: Any, path: str, audit: ContractAudit) -> None:
    summary = expect_dict(summary_value, path, audit)
    if not summary:
        return
    require_keys(summary, path, ["by_variant", "syntax_vs_no_syntax", "case_deltas"], audit)
    by_variant = expect_dict(summary.get("by_variant"), f"{path}.by_variant", audit)
    for variant in ["lexical_semantic_no_syntax", "syntax_candidate_only", "syntax_candidate_and_rerank"]:
        if variant not in by_variant:
            audit.error(f"{path}.by_variant.{variant}", "missing variant summary")
            continue
        variant_summary = expect_dict(by_variant.get(variant), f"{path}.by_variant.{variant}", audit)
        if variant_summary:
            require_keys(
                variant_summary,
                f"{path}.by_variant.{variant}",
                [
                    "case_count",
                    "candidate_hit_rate",
                    "final_hit_rate",
                    "mean_reciprocal_final_rank",
                    "syntax_category_match_rate",
                    "unsafe_final_count",
                ],
                audit,
            )
    deltas = expect_dict(summary.get("syntax_vs_no_syntax"), f"{path}.syntax_vs_no_syntax", audit)
    if deltas:
        for key in ["final_hit_rate_delta", "mean_reciprocal_final_rank_delta", "syntax_category_match_rate_delta"]:
            if key not in deltas:
                audit.error(f"{path}.syntax_vs_no_syntax.{key}", "missing syntax delta")
    expect_list(summary.get("case_deltas"), f"{path}.case_deltas", audit)


def validate_usefulness_summary(summary: Any, path: str, audit: ContractAudit) -> None:
    summary = expect_dict(summary, path, audit)
    if not summary:
        return
    require_keys(
        summary,
        path,
        [
            "lost_baseline_selected_count",
            "potentially_useful_loss_count",
            "review_needed_loss_count",
            "proxy_acceptable_exclusion_count",
            "gated_abstention_count",
            "human_label_task_count",
        ],
        audit,
    )
    for key in [
        "lost_baseline_selected_count",
        "potentially_useful_loss_count",
        "review_needed_loss_count",
        "proxy_acceptable_exclusion_count",
        "gated_abstention_count",
        "human_label_task_count",
    ]:
        if key in summary and not isinstance(summary.get(key), int | float):
            audit.error(f"{path}.{key}", "must be numeric")


def validate_usefulness_query(row_value: Any, path: str, audit: ContractAudit) -> None:
    row = expect_dict(row_value, path, audit)
    if not row:
        return
    require_keys(
        row,
        path,
        [
            "id",
            "query",
            "intent",
            "baseline_final_context_count",
            "gated_final_context_count",
            "lost_baseline_selected_count",
            "potentially_useful_loss_count",
            "review_needed_loss_count",
            "proxy_acceptable_exclusion_count",
            "gated_abstained",
            "lost_candidates",
        ],
        audit,
    )
    lost = expect_list(row.get("lost_candidates"), f"{path}.lost_candidates", audit)
    for index, item in enumerate(lost):
        candidate = expect_dict(item, f"{path}.lost_candidates[{index}]", audit)
        if not candidate:
            continue
        require_keys(
            candidate,
            f"{path}.lost_candidates[{index}]",
            [
                "chunk_id",
                "work_id",
                "title",
                "committee_decision",
                "allowed_uses",
                "proxy_usefulness_risk",
                "human_labels",
            ],
            audit,
        )
        labels = expect_dict(candidate.get("human_labels"), f"{path}.lost_candidates[{index}].human_labels", audit)
        if labels and not all(key.startswith("human_") for key in labels):
            audit.error(f"{path}.lost_candidates[{index}].human_labels", "human label keys must be explicit")
    if row.get("gated_abstained"):
        abstention = expect_dict(row.get("abstention_review"), f"{path}.abstention_review", audit)
        if abstention:
            require_keys(
                abstention,
                f"{path}.abstention_review",
                ["query_id", "baseline_final_context_count", "gated_final_context_count", "human_labels"],
                audit,
            )


def validate_committee_comparison_summary(summary: Any, path: str, audit: ContractAudit) -> None:
    summary = expect_dict(summary, path, audit)
    if not summary:
        return
    require_keys(
        summary,
        path,
        [
            "pair_count",
            "baseline_selected_context_policy_violation_count",
            "gated_selected_context_policy_violation_count",
            "policy_violation_reduction",
            "baseline_final_context_count",
            "gated_final_context_count",
            "final_context_retention_rate",
            "gated_abstention_count",
        ],
        audit,
    )
    for key in [
        "pair_count",
        "baseline_selected_context_policy_violation_count",
        "gated_selected_context_policy_violation_count",
        "policy_violation_reduction",
        "baseline_final_context_count",
        "gated_final_context_count",
        "final_context_retention_rate",
        "gated_abstention_count",
    ]:
        if key in summary and not isinstance(summary.get(key), int | float):
            audit.error(f"{path}.{key}", "must be numeric")


def validate_committee_comparison_pair(pair: Any, path: str, audit: ContractAudit) -> None:
    row = expect_dict(pair, path, audit)
    if not row:
        return
    require_keys(row, path, ["id", "query", "intent", "baseline", "gated", "delta"], audit)
    for side in ["baseline", "gated"]:
        side_row = expect_dict(row.get(side), f"{path}.{side}", audit)
        if side_row:
            require_keys(
                side_row,
                f"{path}.{side}",
                [
                    "policy",
                    "candidate_count",
                    "final_context_count",
                    "selected_context_policy_violation_count",
                    "committee_decision_counts",
                    "selected_decision_counts",
                    "stage_summary",
                ],
                audit,
            )
    delta = expect_dict(row.get("delta"), f"{path}.delta", audit)
    if delta:
        require_keys(
            delta,
            f"{path}.delta",
            [
                "selected_context_policy_violation_delta",
                "final_context_delta",
                "final_context_retention_rate",
                "gated_abstained",
                "baseline_had_policy_violation",
                "gated_has_policy_violation",
            ],
            audit,
        )


def validate_unstructured_coverage_matrix_report(report: dict[str, Any], audit: ContractAudit) -> None:
    require_keys(
        report,
        "$",
        [
            "report_id",
            "status",
            "modes",
            "work_count",
            "domain_rows",
            "document_type_rows",
            "task_rows",
            "label_anchor_summary",
            "gap_summary",
            "claim_boundaries",
            "recommended_next_steps",
            "limitations",
        ],
        audit,
    )
    if report.get("report_id") != "unstructured_experiment_coverage_matrix_v1":
        audit.error("$.report_id", "unexpected report id")
    if report.get("status") not in {
        "needs_expansion",
        "needs_human_labels",
        "fixture_ready_needs_scale",
        "ready_for_broad_validation",
    }:
        audit.error("$.status", "unexpected matrix status")
    domain_rows = expect_list(report.get("domain_rows"), "$.domain_rows", audit)
    if len(domain_rows) < 7:
        audit.error("$.domain_rows", "must cover configured target domains")
    for index, row_value in enumerate(domain_rows):
        row = expect_dict(row_value, f"$.domain_rows[{index}]", audit)
        if row:
            require_keys(
                row,
                f"$.domain_rows[{index}]",
                ["domain", "work_count", "observed_document_types", "preferred_document_types", "evidence_level", "priority_gaps"],
                audit,
            )
    document_type_rows = expect_list(report.get("document_type_rows"), "$.document_type_rows", audit)
    if len(document_type_rows) < 10:
        audit.error("$.document_type_rows", "must cover configured document types")
    task_rows = expect_list(report.get("task_rows"), "$.task_rows", audit)
    if not task_rows:
        audit.error("$.task_rows", "must include task coverage rows")
    for index, row_value in enumerate(task_rows):
        row = expect_dict(row_value, f"$.task_rows[{index}]", audit)
        if row:
            require_keys(
                row,
                f"$.task_rows[{index}]",
                ["task_id", "coverage_status", "required_document_types", "missing_or_limited_labels"],
                audit,
            )
    if not expect_dict(report.get("label_anchor_summary"), "$.label_anchor_summary", audit):
        audit.error("$.label_anchor_summary", "must summarize label anchors")
    if not expect_dict(report.get("gap_summary"), "$.gap_summary", audit):
        audit.error("$.gap_summary", "must summarize coverage gaps")
    boundaries = expect_list(report.get("claim_boundaries"), "$.claim_boundaries", audit)
    if len(boundaries) < 4:
        audit.error("$.claim_boundaries", "must include allowed and disallowed matrix claims")
    audit.count("unstructured_matrix_domains", len(domain_rows))
    audit.count("unstructured_matrix_document_types", len(document_type_rows))
    audit.count("unstructured_matrix_tasks", len(task_rows))


def validate_retrieval_result(result: Any, path: str, audit: ContractAudit) -> None:
    result = expect_dict(result, path, audit)
    if not result:
        return
    require_keys(
        result,
        path,
        [
            "rank",
            "chunk_id",
            "work_id",
            "title",
            "preview",
            "final_score",
            "base_score",
            "components",
            "decision",
            "explanation",
        ],
        audit,
    )
    validate_components(result.get("components"), f"{path}.components", audit)
    validate_decision(result.get("decision"), f"{path}.decision", audit)


def validate_evidence_item(item: Any, path: str, audit: ContractAudit) -> None:
    item = expect_dict(item, path, audit)
    if not item:
        return
    require_keys(item, path, ["citation_id", "chunk_id", "work_id", "title", "preview", "decision"], audit)
    if "components" in item:
        validate_components(item.get("components"), f"{path}.components", audit)
    if "decision" in item:
        validate_decision(item.get("decision"), f"{path}.decision", audit, require_stage=False)


def validate_rejected_candidate(row: Any, path: str, audit: ContractAudit) -> None:
    row = expect_dict(row, path, audit)
    if not row:
        return
    require_keys(row, path, ["chunk_id", "work_id", "title", "stage"], audit)
    stage = expect_dict(row.get("stage"), f"{path}.stage", audit)
    if stage:
        require_keys(stage, f"{path}.stage", ["candidate", "signals", "safety_gate"], audit)
        validate_stage_signals(stage.get("signals"), f"{path}.stage.signals", audit)
        safety_gate = expect_dict(stage.get("safety_gate"), f"{path}.stage.safety_gate", audit)
        committee_gate = stage.get("committee_gate")
        committee_exclusion = bool(committee_gate) and "committee_decision" in row
        if safety_gate and safety_gate.get("decision") != "reject" and not committee_exclusion:
            audit.error(f"{path}.stage.safety_gate.decision", "rejected candidate must carry reject gate or committee exclusion")
        if committee_exclusion:
            require_keys(row, path, ["committee_decision", "allowed_uses", "conflict_tags"], audit)
            allowed_uses = expect_list(row.get("allowed_uses"), f"{path}.allowed_uses", audit)
            if "generator_context" in allowed_uses:
                audit.error(f"{path}.allowed_uses", "committee-excluded candidate cannot allow generator_context")


def validate_components(components: Any, path: str, audit: ContractAudit) -> None:
    components = expect_dict(components, path, audit)
    if not components:
        return
    missing = sorted(COMPONENT_KEYS - set(components))
    for key in missing:
        audit.error(f"{path}.{key}", "missing required score component")
    for key, value in components.items():
        if not isinstance(value, int | float):
            audit.error(f"{path}.{key}", "must be numeric")


def validate_decision(
    decision: Any,
    path: str,
    audit: ContractAudit,
    require_stage: bool = True,
) -> None:
    decision = expect_dict(decision, path, audit)
    if not decision:
        return
    require_keys(decision, path, ["decision", "score", "safety_decision", "safety_risk", "reasons"], audit)
    if decision.get("decision") not in DECISIONS:
        audit.error(f"{path}.decision", "must be allow or reject")
    safety_decision = decision.get("safety_decision")
    if safety_decision not in SAFETY_DECISIONS:
        audit.error(f"{path}.safety_decision", "unexpected safety decision")
    if decision.get("decision") == "allow" and safety_decision in {"block", "quarantine"}:
        audit.error(f"{path}.decision", "blocked or quarantined evidence cannot be allowed")
    if not isinstance(decision.get("score", 0.0), int | float):
        audit.error(f"{path}.score", "must be numeric")
    if require_stage:
        stage = expect_dict(decision.get("stage"), f"{path}.stage", audit)
        if stage:
            require_keys(stage, f"{path}.stage", ["candidate", "signals", "safety_gate"], audit)
            validate_stage_signals(stage.get("signals"), f"{path}.stage.signals", audit)


def validate_stage_signals(signals: Any, path: str, audit: ContractAudit) -> None:
    signals = expect_dict(signals, path, audit)
    if not signals:
        return
    missing = sorted(SIGNAL_FAMILIES - set(signals))
    for family in missing:
        audit.error(f"{path}.{family}", "missing signal family")
    source_trust = signals.get("source_trust")
    if isinstance(source_trust, dict) and "source_trust" not in source_trust:
        audit.error(f"{path}.source_trust.source_trust", "missing source trust value")


def validate_stage_summary(summary: Any, path: str, audit: ContractAudit) -> None:
    summary = expect_dict(summary, path, audit)
    if not summary:
        return
    for key in [
        "candidate_count",
        "enriched_count",
        "allowed_after_safety_gate",
        "rejected_after_safety_gate",
        "safety_decision_counts",
    ]:
        if key not in summary:
            audit.error(f"{path}.{key}", "missing stage summary field")


def validate_context_packet(packet: Any, path: str, audit: ContractAudit) -> None:
    packet = expect_dict(packet, path, audit)
    if not packet:
        return
    items = expect_list(packet.get("items"), f"{path}.items", audit)
    for index, context_item in enumerate(items):
        row = expect_dict(context_item, f"{path}.items[{index}]", audit)
        if not row:
            continue
        require_keys(row, f"{path}.items[{index}]", ["citation_id", "role", "allowed_use", "evidence"], audit)
        evidence = row.get("evidence") or {}
        decision = evidence.get("decision") if isinstance(evidence, dict) else {}
        safety_decision = decision.get("safety_decision") if isinstance(decision, dict) else None
        if safety_decision in {"block", "quarantine"}:
            audit.error(f"{path}.items[{index}].evidence.decision", "unsafe evidence reached context")


def require_keys(row: dict[str, Any], path: str, keys: list[str], audit: ContractAudit) -> None:
    for key in keys:
        if key not in row:
            audit.error(f"{path}.{key}", "missing required field")


def expect_dict(value: Any, path: str, audit: ContractAudit) -> dict[str, Any]:
    if not isinstance(value, dict):
        audit.error(path, "must be an object")
        return {}
    return value


def expect_list(value: Any, path: str, audit: ContractAudit) -> list[Any]:
    if not isinstance(value, list):
        audit.error(path, "must be a list")
        return []
    return value


def nest(path: str, parent: ContractAudit) -> ContractAudit:
    class NestedAudit(ContractAudit):
        def error(self, nested_path: str, message: str) -> None:
            parent.error(f"{path}{nested_path[1:] if nested_path.startswith('$') else nested_path}", message)

        def warn(self, nested_path: str, message: str) -> None:
            parent.warn(f"{path}{nested_path[1:] if nested_path.startswith('$') else nested_path}", message)

        def count(self, key: str, amount: int = 1) -> None:
            parent.count(key, amount)

    return NestedAudit(parent.report_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CANON report contract.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--type", choices=sorted(REPORT_TYPES), default=None)
    args = parser.parse_args()
    report = json.loads(args.path.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(report, args.type), indent=2))


if __name__ == "__main__":
    main()

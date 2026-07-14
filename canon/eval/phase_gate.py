from __future__ import annotations

import argparse
import json
from typing import Any

from canon.config import load_settings
from canon.eval.anchors import build_anchor_report
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.corroboration import evaluate_adversarial_corroboration
from canon.eval.security import evaluate_adversarial_rag_security
from canon.eval.unstructured import evaluate_mixed_unstructured
from canon.evidence.domain import DOMAIN_DECISION_PROFILES
from canon.labeling.tasks import LABEL_SCHEMA
from canon.safety.sanitizer import retrieved_instruction_exception_policy


PHASES = {
    "A": "typed_decisions",
    "B": "text_level_importance",
    "C": "source_trust_layer",
    "D": "corroboration_conflict_importance",
    "E": "safety_before_generation",
    "F": "adversarial_rag_security_benchmark",
    "G": "human_external_evaluation_anchors",
    "H": "heterogeneous_unstructured_data_routing",
}


def run_phase_gate(write_report: bool = False) -> dict[str, Any]:
    security = evaluate_adversarial_rag_security()
    corroboration = evaluate_adversarial_corroboration()
    anchors = build_anchor_report()
    unstructured = evaluate_mixed_unstructured()
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "importance_phase_gate_v1",
        "phase_count": len(PHASES),
        "phases": phase_rows(security, corroboration, anchors, unstructured),
        "domain_profile_count": len(DOMAIN_DECISION_PROFILES),
        "label_schema_count": len(LABEL_SCHEMA),
        "retrieved_instruction_exception_policy": retrieved_instruction_exception_policy(
            source_authorized=False,
            bounded_workflow=True,
            external_action_requested=False,
        ),
        "artifact_contracts": {
            "security": validate_report(security, "adversarial_rag_security"),
            "corroboration": validate_report(corroboration, "adversarial_corroboration"),
            "anchors": validate_report(anchors, "evaluation_anchors"),
            "mixed_unstructured": validate_report(unstructured, "mixed_unstructured"),
        },
        "limitations": [
            "Deterministic phase gates prove pipeline coverage, not scientific validity.",
            "Human labels are still required for source-trust and evidence-role calibration.",
            "Prompt-injection robustness is limited to current synthetic fixtures.",
            "Domain profiles are starting priors and must be validated separately.",
            "Heterogeneous document routing is tested on fixtures and needs real mixed-corpus labels.",
        ],
    }
    report["status"] = "pass" if all(phase["status"] == "pass" for phase in report["phases"]) else "fail"
    report["contract_validation"] = validate_report(report, "phase_gate")
    if write_report:
        settings = load_settings()
        path = settings.reports_dir / "importance_phase_gate_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def phase_rows(
    security: dict[str, Any],
    corroboration: dict[str, Any],
    anchors: dict[str, Any],
    unstructured: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = {
        "A": [
            ("contract_version_present", security.get("contract_version") == CONTRACT_VERSION),
            ("artifact_contracts_pass", security.get("contract_validation", {}).get("status") == "pass"),
        ],
        "B": [
            ("label_schema_has_evidence_role", "evidence_role_fit" in LABEL_SCHEMA),
            ("domain_profiles_available", len(DOMAIN_DECISION_PROFILES) >= 7),
        ],
        "C": [
            ("source_trust_anchor_declared", has_anchor(anchors, "human_source_trust_labels_v1")),
            ("domain_profiles_available", len(DOMAIN_DECISION_PROFILES) >= 7),
        ],
        "D": [
            ("corroboration_benchmark_passes", corroboration.get("status") == "pass"),
            ("echo_attack_flagged", corroboration.get("checks", {}).get("echo_attack_flagged") is True),
        ],
        "E": [
            ("security_benchmark_passes", security.get("status") == "pass"),
            ("no_attack_context_exposure", security.get("metrics", {}).get("attack_context_exposure_count") == 0),
        ],
        "F": [
            ("attacks_retrieved_then_rejected", security.get("checks", {}).get("attacks_rejected_before_rerank") is True),
            ("hidden_text_attack_included", security.get("metrics", {}).get("attack_candidate_count", 0) >= 3),
        ],
        "G": [
            ("anchor_registry_contract_passes", anchors.get("contract_validation", {}).get("status") == "pass"),
            ("claim_limits_present", len(anchors.get("claim_limits") or []) >= 4),
        ],
        "H": [
            ("mixed_unstructured_benchmark_passes", unstructured.get("summary", {}).get("status") == "pass"),
            ("document_type_coverage", unstructured.get("summary", {}).get("final_type_coverage_count", 0) >= 7),
            (
                "web_attack_not_exposed",
                all(run.get("attack_context_count", 0) == 0 for run in unstructured.get("query_runs") or []),
            ),
        ],
    }
    rows = []
    for phase_id, name in PHASES.items():
        phase_checks = [
            {"id": check_id, "passed": bool(passed)}
            for check_id, passed in checks[phase_id]
        ]
        rows.append(
            {
                "phase": phase_id,
                "name": name,
                "status": "pass" if all(check["passed"] for check in phase_checks) else "fail",
                "checks": phase_checks,
            }
        )
    return rows


def has_anchor(report: dict[str, Any], anchor_id: str) -> bool:
    return any(anchor.get("anchor_id") == anchor_id for anchor in report.get("anchors") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic CANON importance phase gate.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_phase_gate(write_report=args.write_report), indent=2))


if __name__ == "__main__":
    main()

import unittest

from canon.eval.contracts import validate_report
from canon.reports.unstructured_portfolio import portfolio_checks, slug


class UnstructuredPortfolioTests(unittest.TestCase):
    def test_portfolio_checks_pass_complete_artifact_bundle(self):
        checks = portfolio_checks(complete_artifacts())

        self.assertTrue(all(check["passed"] for check in checks))

    def test_portfolio_checks_fail_missing_artifacts(self):
        checks = portfolio_checks({})
        failed_ids = {check["id"] for check in checks if not check["passed"]}

        self.assertIn("heterogeneous_readiness_passes", failed_ids)
        self.assertIn("public_opinion_synthesis_excludes_generator_context", failed_ids)

    def test_portfolio_contract_requires_claim_boundaries(self):
        report = {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "unstructured_experiment_portfolio_v1",
            "status": "pass",
            "modes": {
                "document_type_mode": "unstructured_demo_corpus",
                "domain_mode": "mixed_domain_demo_corpus",
                "public_opinion_mode": "social_public_opinion_demo_corpus",
            },
            "checks": [
                {"id": "boundary_chunking_isolates_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "large_window_chunking_isolates_mixed_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "chunking_variant_comparison_tracks_tradeoffs", "passed": True, "description": "ok"},
                {"id": "committee_gate_usefulness_audit_ready_for_labels", "passed": True, "description": "ok"},
                {"id": "public_opinion_collection_audit_present", "passed": True, "description": "ok"},
                {"id": "syntax_retrieval_ablation_tracks_form_signal", "passed": True, "description": "ok"},
                {"id": "unstructured_human_label_tasks_ready_for_annotation", "passed": True, "description": "ok"},
                {"id": "unstructured_label_calibration_honest_status", "passed": True, "description": "ok"},
                *[{"id": f"check_{index}", "passed": True, "description": "ok"} for index in range(3)],
            ],
            "artifact_summary": {"phase_gate_status": "pass"},
            "claim_boundaries": ["Allowed"] * 5,
            "recommended_next_steps": [],
            "limitations": [],
            "artifact_paths": {
                "phase_gate": "reports/importance_phase_gate_v1.json",
                "chunking_large_window": "reports/chunking_strategy_eval_ct1000_ov100.json",
                "chunking_variants": "reports/chunking_variant_eval_ct1000_ov100.json",
                "syntax_retrieval_ablation": "reports/syntax_retrieval_ablation_v1.json",
                "unstructured_label_tasks": "reports/unstructured_human_label_tasks_v1.json",
                "unstructured_label_calibration": "reports/unstructured_label_calibration_v1.json",
            },
        }

        audit = validate_report(report, "unstructured_experiment_portfolio")

        self.assertEqual(audit["status"], "pass")

    def test_portfolio_contract_requires_large_window_chunking_probe(self):
        report = {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "unstructured_experiment_portfolio_v1",
            "status": "pass",
            "modes": {
                "document_type_mode": "unstructured_demo_corpus",
                "domain_mode": "mixed_domain_demo_corpus",
                "public_opinion_mode": "social_public_opinion_demo_corpus",
            },
            "checks": [
                {"id": "boundary_chunking_isolates_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "committee_gate_usefulness_audit_ready_for_labels", "passed": True, "description": "ok"},
                *[{"id": f"check_{index}", "passed": True, "description": "ok"} for index in range(6)],
            ],
            "artifact_summary": {"phase_gate_status": "pass"},
            "claim_boundaries": ["Allowed"] * 5,
            "recommended_next_steps": [],
            "limitations": [],
            "artifact_paths": {"phase_gate": "reports/importance_phase_gate_v1.json"},
        }

        audit = validate_report(report, "unstructured_experiment_portfolio")

        self.assertEqual(audit["status"], "fail")
        self.assertTrue(any("large_window_chunking" in error for error in audit["errors"]))

    def test_portfolio_contract_requires_unstructured_label_task_artifact(self):
        report = {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "unstructured_experiment_portfolio_v1",
            "status": "pass",
            "modes": {
                "document_type_mode": "unstructured_demo_corpus",
                "domain_mode": "mixed_domain_demo_corpus",
                "public_opinion_mode": "social_public_opinion_demo_corpus",
            },
            "checks": [
                {"id": "boundary_chunking_isolates_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "large_window_chunking_isolates_mixed_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "chunking_variant_comparison_tracks_tradeoffs", "passed": True, "description": "ok"},
                {"id": "committee_gate_usefulness_audit_ready_for_labels", "passed": True, "description": "ok"},
                {"id": "public_opinion_collection_audit_present", "passed": True, "description": "ok"},
                {"id": "syntax_retrieval_ablation_tracks_form_signal", "passed": True, "description": "ok"},
                *[{"id": f"check_{index}", "passed": True, "description": "ok"} for index in range(5)],
            ],
            "artifact_summary": {"phase_gate_status": "pass"},
            "claim_boundaries": ["Allowed"] * 5,
            "recommended_next_steps": [],
            "limitations": [],
            "artifact_paths": {
                "phase_gate": "reports/importance_phase_gate_v1.json",
                "chunking_large_window": "reports/chunking_strategy_eval_ct1000_ov100.json",
                "chunking_variants": "reports/chunking_variant_eval_ct1000_ov100.json",
                "syntax_retrieval_ablation": "reports/syntax_retrieval_ablation_v1.json",
            },
        }

        audit = validate_report(report, "unstructured_experiment_portfolio")

        self.assertEqual(audit["status"], "fail")
        self.assertTrue(any("unstructured_label_tasks" in error for error in audit["errors"]))

    def test_portfolio_contract_requires_unstructured_label_calibration_artifact(self):
        report = {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "unstructured_experiment_portfolio_v1",
            "status": "pass",
            "modes": {
                "document_type_mode": "unstructured_demo_corpus",
                "domain_mode": "mixed_domain_demo_corpus",
                "public_opinion_mode": "social_public_opinion_demo_corpus",
            },
            "checks": [
                {"id": "boundary_chunking_isolates_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "large_window_chunking_isolates_mixed_unsafe_social_span", "passed": True, "description": "ok"},
                {"id": "chunking_variant_comparison_tracks_tradeoffs", "passed": True, "description": "ok"},
                {"id": "committee_gate_usefulness_audit_ready_for_labels", "passed": True, "description": "ok"},
                {"id": "public_opinion_collection_audit_present", "passed": True, "description": "ok"},
                {"id": "syntax_retrieval_ablation_tracks_form_signal", "passed": True, "description": "ok"},
                {"id": "unstructured_human_label_tasks_ready_for_annotation", "passed": True, "description": "ok"},
                *[{"id": f"check_{index}", "passed": True, "description": "ok"} for index in range(5)],
            ],
            "artifact_summary": {"phase_gate_status": "pass"},
            "claim_boundaries": ["Allowed"] * 5,
            "recommended_next_steps": [],
            "limitations": [],
            "artifact_paths": {
                "phase_gate": "reports/importance_phase_gate_v1.json",
                "chunking_large_window": "reports/chunking_strategy_eval_ct1000_ov100.json",
                "chunking_variants": "reports/chunking_variant_eval_ct1000_ov100.json",
                "syntax_retrieval_ablation": "reports/syntax_retrieval_ablation_v1.json",
                "unstructured_label_tasks": "reports/unstructured_human_label_tasks_v1.json",
            },
        }

        audit = validate_report(report, "unstructured_experiment_portfolio")

        self.assertEqual(audit["status"], "fail")
        self.assertTrue(any("unstructured_label_calibration" in error for error in audit["errors"]))

    def test_slug_matches_synthesis_report_naming(self):
        self.assertEqual(
            slug("public opinion battery storage safety concerns"),
            "public-opinion-battery-storage-safety-concerns",
        )


def complete_artifacts():
    return {
        "unstructured_readiness": {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "heterogeneous_unstructured_readiness_v1",
            "status": "pass",
            "thresholds": {},
            "checks": [{"id": "ready", "passed": True, "description": "ready"}],
            "artifact_summary": {"mixed_unstructured": {"status": "pass"}},
            "claim_boundaries": ["Allowed", "Allowed", "Not allowed", "Not allowed"],
            "recommended_next_steps": [],
            "limitations": [],
            "contract_validation": {"status": "pass"},
        },
        "phase_gate": {
            "report_id": "importance_phase_gate_v1",
            "status": "pass",
            "contract_validation": {"status": "pass"},
            "phases": [],
        },
        "adversarial_security": {
            "status": "pass",
            "metrics": {"attack_context_exposure_count": 0},
        },
        "chunking": {
            "summary": {
                "boundary_aware": {"average_span_containment_rate": 1.0, "unsafe_isolated_count": 1},
                "fixed_window": {"average_span_containment_rate": 0.5, "unsafe_isolated_count": 0},
            },
        },
        "chunking_large_window": {
            "report_id": "chunking_strategy_eval_v1",
            "chunk_tokens": 1000,
            "overlap_tokens": 100,
            "summary": {
                "boundary_aware": {"average_span_containment_rate": 1.0, "unsafe_isolated_count": 1},
                "fixed_window": {"average_span_containment_rate": 1.0, "unsafe_isolated_count": 0},
            },
        },
        "chunking_variants": {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "chunking_variant_eval_v1",
            "chunk_tokens": 1000,
            "overlap_tokens": 100,
            "best_variant_by_objective": {"balanced_utility": "boundary_aware"},
            "summary": {
                "fixed_window": {
                    "forbidden_merge_count": 2,
                    "objective_scores": {"balanced_utility": 0.5},
                },
                "boundary_aware": {
                    "forbidden_merge_count": 0,
                    "objective_scores": {"balanced_utility": 1.0},
                },
                "semantic_cohesion": {
                    "forbidden_merge_count": 0,
                    "objective_scores": {"balanced_utility": 0.9},
                },
                "embedding_semantic": {
                    "forbidden_merge_count": 0,
                    "objective_scores": {"balanced_utility": 0.95},
                },
                "atomic_sentence": {
                    "forbidden_merge_count": 0,
                    "objective_scores": {"balanced_utility": 0.85},
                },
            },
            "contract_validation": {"status": "pass"},
        },
        "coverage_matrix": {
            "report_id": "unstructured_experiment_coverage_matrix_v1",
            "status": "needs_expansion",
            "contract_validation": {"status": "pass"},
            "gap_summary": {"ready_for_broad_validity_claims": False},
        },
        "public_opinion": {
            "report_id": "public_opinion_analysis_v1",
            "aggregation_readiness": {"status": "fixture_or_insufficient_for_inference"},
            "collection_audit": {
                "status": "fixture_only_or_needs_collection_metadata",
                "metadata_coverage": {"terms_review_share": 0.0},
                "privacy_risk": {"direct_identifier_record_count": 0},
            },
        },
        "evidence_committee": {
            "summary": {"committee_decision_counts": {"aggregate_only": 2}},
            "runs": [
                {
                    "policy": "committee_rag",
                    "selected_context_policy_violation_count": 0,
                    "final_context_count": 1,
                }
            ],
        },
        "committee_gate_comparison": {
            "report_id": "committee_gate_policy_comparison_v1",
            "summary": {
                "baseline_selected_context_policy_violation_count": 3,
                "gated_selected_context_policy_violation_count": 0,
                "policy_violation_reduction": 3,
                "final_context_retention_rate": 0.5,
            },
            "contract_validation": {"status": "pass"},
        },
        "committee_gate_usefulness": {
            "report_id": "committee_gate_usefulness_audit_v1",
            "summary": {
                "lost_baseline_selected_count": 3,
                "potentially_useful_loss_count": 0,
                "review_needed_loss_count": 1,
                "proxy_acceptable_exclusion_count": 3,
                "gated_abstention_count": 1,
                "human_label_task_count": 4,
            },
            "contract_validation": {"status": "pass"},
        },
        "syntax_retrieval_ablation": {
            "contract_version": "canon_rag_contract_v1",
            "report_id": "syntax_retrieval_ablation_v1",
            "status": "pass",
            "summary": {
                "by_variant": {
                    "syntax_candidate_and_rerank": {
                        "unsafe_final_count": 0,
                        "mean_reciprocal_final_rank": 1.0,
                    }
                },
                "syntax_vs_no_syntax": {"mean_reciprocal_final_rank_delta": 0.0},
            },
            "checks": [{"id": "no_unsafe_final_exposure", "passed": True, "description": "ok"}],
            "contract_validation": {"status": "pass"},
        },
        "unstructured_label_tasks": {
            "report_id": "unstructured_human_label_tasks_v1",
            "task_count": 12,
            "coverage_summary": {
                "ready_for_annotation": True,
                "missing_label_families": [],
                "label_family_counts": {
                    "answer_usefulness": 2,
                    "document_type": 4,
                    "evidence_role": 4,
                    "safety_disposition": 4,
                    "source_independence": 4,
                    "source_trust": 4,
                },
            },
            "contract_validation": {"status": "pass"},
        },
        "unstructured_label_calibration": {
            "report_id": "unstructured_label_calibration_v1",
            "status": "needs_labels",
            "task_count": 12,
            "completed_task_count": 0,
            "completed_label_count": 0,
            "agreement": {"overall_comparison_count": 0},
            "contract_validation": {"status": "pass"},
        },
        "public_opinion_synthesis": {
            "contract_validation": {"status": "pass"},
            "context_packet": {
                "items": [],
                "assembly_summary": {"aggregate_only_count": 2},
            },
        },
        "evaluation_anchors": {
            "report_id": "evaluation_anchor_registry_v1",
            "claim_limits": ["a", "b", "c", "d"],
        },
    }


if __name__ == "__main__":
    unittest.main()

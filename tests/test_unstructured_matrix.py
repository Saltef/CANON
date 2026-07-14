import unittest

from canon.eval.contracts import validate_report
from canon.eval.unstructured_matrix import (
    build_unstructured_coverage_matrix,
    document_type_evidence_level,
    evidence_level,
    label_anchor_status,
    task_coverage_rows,
)


class UnstructuredMatrixTests(unittest.TestCase):
    def test_evidence_level_separates_missing_fixture_and_expanded(self):
        self.assertEqual(evidence_level(0, 0, 3), "missing")
        self.assertEqual(evidence_level(3, 1, 3), "fixture_preferred_types_present")
        self.assertEqual(evidence_level(100, 0, 4), "expanded_but_document_type_gaps")
        self.assertEqual(evidence_level(100, 3, 4), "expanded_candidate")

    def test_document_type_evidence_level_tracks_fixture_scale(self):
        self.assertEqual(document_type_evidence_level(0), "missing")
        self.assertEqual(document_type_evidence_level(4), "fixture_only")
        self.assertEqual(document_type_evidence_level(60), "expanded_candidate")

    def test_label_anchor_status_uses_active_deterministic_safety_but_not_human_labels(self):
        status = label_anchor_status(
            {
                "anchors": [
                    {"anchor_id": "human_evidence_role_labels_v1", "status": "planned"},
                    {"anchor_id": "human_source_trust_labels_v1", "status": "planned"},
                    {"anchor_id": "human_chunking_quality_labels_v1", "status": "task_scaffold_ready"},
                    {"anchor_id": "adversarial_rag_security_v1", "status": "active_deterministic"},
                    {"anchor_id": "adversarial_corroboration_v1", "status": "active_deterministic"},
                ]
            }
        )

        self.assertEqual(status["evidence_role"], "planned")
        self.assertEqual(status["source_trust"], "planned")
        self.assertEqual(status["chunking_quality"], "task_scaffold_ready")
        self.assertEqual(status["safety_disposition"], "active")
        self.assertEqual(status["source_independence"], "active")

    def test_label_anchor_status_uses_task_scaffold_without_marking_active(self):
        status = label_anchor_status(
            {
                "anchors": [],
                "label_task_coverage": {
                    "covered_label_families": [
                        "document_type",
                        "evidence_role",
                        "source_trust",
                        "answer_usefulness",
                        "chunking_quality",
                    ]
                },
            }
        )

        self.assertEqual(status["document_type"], "task_scaffold_ready")
        self.assertEqual(status["answer_usefulness"], "task_scaffold_ready")
        self.assertEqual(status["chunking_quality"], "task_scaffold_ready")
        self.assertNotEqual(status["source_trust"], "active")

    def test_task_rows_require_labels_after_domain_and_doc_coverage(self):
        rows = task_coverage_rows(
            domain_rows=[
                {"domain": "public_opinion", "work_count": 4},
                {"domain": "legal_market", "work_count": 2},
                {"domain": "cultural_studies", "work_count": 1},
            ],
            document_type_rows=[
                {"document_type": "web_page", "work_count": 1},
                {"document_type": "social_media_post", "work_count": 1},
                {"document_type": "forum_comment", "work_count": 1},
                {"document_type": "transcript", "work_count": 1},
            ],
            anchors={"anchors": []},
        )
        security = next(row for row in rows if row["task_id"] == "adversarial_rag_security")

        self.assertEqual(security["coverage_status"], "needs_human_or_external_labels")
        self.assertIn("chunking_quality", security["missing_or_limited_labels"])

    def test_matrix_builds_current_fixture_report_with_contract(self):
        report = build_unstructured_coverage_matrix(write_report=False)

        self.assertEqual(report["report_id"], "unstructured_experiment_coverage_matrix_v1")
        self.assertIn(report["status"], {"needs_expansion", "needs_human_labels", "fixture_ready_needs_scale"})
        self.assertFalse(report["gap_summary"]["ready_for_broad_validity_claims"])
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertGreaterEqual(len(report["domain_rows"]), 7)
        self.assertTrue(any(row["domain"] == "public_opinion" for row in report["domain_rows"]))

    def test_matrix_contract_accepts_required_shape(self):
        report = build_unstructured_coverage_matrix(write_report=False)
        audit = validate_report(report, "unstructured_coverage_matrix")

        self.assertEqual(audit["status"], "pass")


if __name__ == "__main__":
    unittest.main()

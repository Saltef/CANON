import unittest

from canon.product.run_diagnosis import build_run_diagnosis


class RunDiagnosisTests(unittest.TestCase):
    def test_ready_run_still_marks_human_review_boundary(self):
        diagnosis = build_run_diagnosis(
            query="grid risk",
            packet_response={
                "status": "complete",
                "answer_report": {
                    "support_assessment": {
                        "support_level": "moderate",
                        "support_confidence": 0.72,
                        "query_term_coverage": 1.0,
                        "covered_focus_terms": ["grid", "risk"],
                    }
                },
            },
            evidence=[{"evidence_id": "C1", "title": "Grid evidence"}],
            relevance_gate={
                "status": "pass",
                "draft_allowed": True,
                "focus_terms": ["grid", "risk"],
                "matched_terms": ["grid", "risk"],
                "weak_terms": [],
                "query_term_coverage": 1.0,
                "usable_evidence_count": 1,
                "diagnostic_candidate_count": 0,
            },
            draft_brief={"status": "model_draft_ready", "citation_count": 1},
            retrieval={
                "engine": "model_candidate_pool",
                "candidate_count": 12,
                "reranker_provider": "cohere",
                "reranker_model": "rerank-v4.0-pro",
            },
            coverage_gaps=[],
            external_cards=[],
            frame_coverage={"status": "complete"},
        )

        self.assertEqual(diagnosis["overall_status"], "ready_for_user_inspection")
        self.assertEqual(diagnosis["failure_class"], "human_review_required")
        self.assertIn("human_review_required", diagnosis["issue_categories"])
        self.assertEqual(diagnosis["scorecard"]["candidate_retrieval"], "pass")
        self.assertEqual(diagnosis["scorecard"]["generation_grounding"], "pass")
        self.assertTrue(diagnosis["human_review_required"])

    def test_corpus_mismatch_explains_blocked_draft(self):
        diagnosis = build_run_diagnosis(
            query="AI risks",
            packet_response={
                "status": "complete",
                "answer_report": {
                    "support_assessment": {
                        "support_level": "weak",
                        "query_term_coverage": 0.0,
                        "covered_focus_terms": [],
                        "missing_focus_terms": ["ai", "risks"],
                    }
                },
            },
            evidence=[{"evidence_id": "C1", "title": "Civil war dataset"}],
            relevance_gate={
                "status": "corpus_mismatch",
                "draft_allowed": False,
                "focus_terms": ["ai", "risks"],
                "matched_terms": [],
                "weak_terms": ["ai", "risks"],
                "query_term_coverage": 0.0,
                "usable_evidence_count": 0,
                "diagnostic_candidate_count": 1,
            },
            draft_brief={"status": "blocked_by_relevance_gate", "citation_count": 0},
            retrieval={"engine": "canon_synthesis", "candidate_count": 1},
            coverage_gaps=[],
            external_cards=[],
            frame_coverage={"status": "partial"},
        )

        self.assertEqual(diagnosis["overall_status"], "failed_no_grounded_answer")
        self.assertEqual(diagnosis["failure_class"], "corpus_mismatch")
        self.assertIn("draft_blocked_by_relevance_gate", diagnosis["issue_categories"])
        self.assertEqual(diagnosis["scorecard"]["corpus_fit"], "fail")
        self.assertTrue(any("Switch corpus" in action for action in diagnosis["recommended_actions"]))


if __name__ == "__main__":
    unittest.main()

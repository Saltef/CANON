import unittest

from scripts.run_product_model_matrix import KIMI_GENERATION_MODELS, judge_product_session, public_qrels_for_mode, public_qrels_summary


class ProductModelMatrixScriptTests(unittest.TestCase):
    def test_public_qrels_for_scifact_full(self):
        self.assertEqual(public_qrels_for_mode("beir_scifact_full"), "gold/beir_scifact_full_qrels.json")
        self.assertEqual(public_qrels_for_mode("missing"), None)

    def test_judge_product_session_uses_usable_evidence_only(self):
        session = {
            "session_id": "s1",
            "query": "claim",
            "draft_brief": {
                "text": "Answer [C1].",
                "citations": [{"citation_id": "C1"}],
                "support_assessment": {"support_level": "mixed"},
            },
            "evidence_cards": [
                {
                    "evidence_id": "C1",
                    "title": "Relevant",
                    "text": "Relevant evidence",
                    "relevance": {"usage_status": "usable_candidate_evidence"},
                },
                {
                    "evidence_id": "C2",
                    "title": "Diagnostic",
                    "text": "Diagnostic evidence",
                    "relevance": {"usage_status": "diagnostic_candidate_only"},
                },
            ],
        }

        judgment = judge_product_session(
            session,
            provider=None,
            provider_name="heuristic",
            model=None,
            category="unit",
        )

        self.assertEqual(judgment["status"], "judged")
        self.assertEqual(judgment["judge_source"], "heuristic_judge")
        self.assertIn(judgment["judge_answer_usefulness"], {"usable", "partially_useful", "not_useful"})

    def test_public_qrels_summary_reports_hit_rate_and_judged_count(self):
        summary = public_qrels_summary(
            [
                {"hit_at_k": True, "recall_at_k": 0.5, "llm_judge": {"status": "judged"}},
                {"hit_at_k": False, "recall_at_k": 0.0, "llm_judge": {"status": "skipped"}},
            ]
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["hit_rate"], 0.5)
        self.assertEqual(summary["mean_recall_at_k"], 0.25)
        self.assertEqual(summary["judged_count"], 1)

    def test_kimi_generation_models_include_k3(self):
        self.assertIn("moonshotai/kimi-k3", KIMI_GENERATION_MODELS)
        self.assertIn("moonshotai/kimi-k2.7-code", KIMI_GENERATION_MODELS)


if __name__ == "__main__":
    unittest.main()

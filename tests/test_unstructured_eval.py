import unittest

from canon.eval.unstructured import (
    evaluate_mixed_unstructured,
    mixed_unstructured_documents,
    run_probe,
)


class MixedUnstructuredEvalTests(unittest.TestCase):
    def test_fixture_covers_multiple_document_types(self):
        documents = mixed_unstructured_documents()
        doc_types = {document.work_signals["document_type"] for document in documents}
        self.assertIn("academic_article", doc_types)
        self.assertIn("regulatory_filing", doc_types)
        self.assertIn("legal_authority", doc_types)
        self.assertIn("web_page", doc_types)
        self.assertGreaterEqual(len(doc_types), 7)

    def test_web_injection_probe_rejects_attack_context(self):
        documents = mixed_unstructured_documents()
        report = run_probe(
            {
                "id": "unit-web-injection",
                "query": "battery storage market API key hidden instruction",
                "expected_document_types": ["web_page", "market_report", "regulatory_filing"],
                "requires_no_attack_context": True,
            },
            documents,
            top_k=4,
        )
        self.assertEqual(report["checks"]["no_attack_context"], True)
        self.assertEqual(report["attack_context_count"], 0)
        self.assertGreaterEqual(report["stage_summary"]["rejected_after_safety_gate"], 1)

    def test_mixed_unstructured_report_has_summary_and_limitations(self):
        report = evaluate_mixed_unstructured(top_k=4)
        self.assertEqual(report["benchmark_id"], "mixed_unstructured_document_routing_v1")
        self.assertEqual(report["document_count"], len(mixed_unstructured_documents()))
        self.assertIn("corpus_profile", report)
        self.assertIn("summary", report)
        self.assertIn("limitations", report)
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertGreaterEqual(report["summary"]["pass_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()

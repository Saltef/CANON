import unittest
from unittest.mock import patch

from canon.product import service


class ProductServiceTests(unittest.TestCase):
    def test_optional_int_validates_positive_values(self):
        self.assertEqual(service.optional_int("5"), 5)
        with self.assertRaises(service.ProductError):
            service.optional_int("0")

    def test_require_text_rejects_empty_queries(self):
        with self.assertRaises(service.ProductError):
            service.require_text({"query": " "}, "query")

    def test_product_summary_has_claim_boundaries(self):
        summary = service.product_summary("social_science_ir_v1_harvest10")
        self.assertEqual(summary["product"], "CANON Evidence Workbench")
        self.assertIn("claim_boundaries", summary)

    def test_unknown_report_raises_404(self):
        with self.assertRaises(service.ProductError) as context:
            service.report("missing", "social_science_ir_v1_harvest10")
        self.assertEqual(context.exception.status_code, 404)

    def test_diversity_gate_report_is_allowed(self):
        with patch("canon.product.service.load_report", return_value={"status": "pass"}):
            report = service.report("diversity-gate", "social_science_ir_10k")
        self.assertEqual(report["status"], "pass")

    def test_compare_retrieval_preserves_result_explanations(self):
        with patch("canon.product.service.compare") as compare:
            compare.return_value = {
                "query": "democratic peace",
                "mode": "m",
                "policies": ["rag"],
                "rank_table": [],
                "runs": [
                    {
                        "policy": "rag",
                        "top_k": 1,
                        "results": [
                            {
                                "chunk_id": "c1",
                                "explanation": {"reasons": ["high_lexical_relevance"]},
                            }
                        ],
                    }
                ],
            }
            report = service.compare_retrieval({"query": "democratic peace", "mode": "m", "policies": ["rag"]})
        self.assertEqual(report["runs"][0]["results"][0]["explanation"]["reasons"], ["high_lexical_relevance"])

    def test_compact_diversity_report_keeps_product_fields(self):
        compact = service.compact_diversity_report(
            {
                "mode": "m",
                "method_set_id": "methods",
                "diverse_method_id": "diverse",
                "baseline_method_id": "lexical",
                "query_count": 25,
                "aggregate": {"verdict_counts": {"useful_breadth": 10}},
                "queries": [{"large": "payload"}],
            }
        )
        self.assertNotIn("queries", compact)
        self.assertEqual(compact["aggregate"]["verdict_counts"]["useful_breadth"], 10)

    def test_diversity_audit_uses_defaults_and_returns_compact_report(self):
        with patch("canon.product.service.run_diversity_audit") as audit:
            audit.return_value = {
                "mode": service.DEFAULT_DIVERSITY_MODE,
                "method_set_id": "baseline_methods_v1",
                "diverse_method_id": service.DEFAULT_DIVERSE_METHOD_ID,
                "baseline_method_id": service.DEFAULT_DIVERSITY_BASELINE_METHOD_ID,
                "query_count": 25,
                "aggregate": {"verdict_counts": {"mixed_breadth": 1}},
                "queries": [{"large": "payload"}],
            }
            report = service.diversity_audit({})
        audit.assert_called_once_with(
            mode=service.DEFAULT_DIVERSITY_MODE,
            diverse_method_id=service.DEFAULT_DIVERSE_METHOD_ID,
            baseline_method_id=service.DEFAULT_DIVERSITY_BASELINE_METHOD_ID,
        )
        self.assertNotIn("queries", report)
        self.assertEqual(report["query_count"], 25)

    def test_compact_diversity_query_omits_added_and_removed_titles(self):
        compact = service.compact_diversity_query(
            {
                "query_id": "q1",
                "query_type": "synthesis",
                "query": "What changed?",
                "verdict": "useful_breadth",
                "cluster_delta": 2,
                "rank_overlap": 0.25,
                "added_titles": [{"title": "A"}],
                "removed_titles": [{"title": "B"}],
            }
        )
        self.assertEqual(compact["query_id"], "q1")
        self.assertNotIn("added_titles", compact)
        self.assertNotIn("removed_titles", compact)

    def test_diversity_queries_filters_loaded_report(self):
        payload = {
            "mode": "m",
            "method_set_id": "methods",
            "diverse_method_id": "diverse",
            "baseline_method_id": "lexical",
            "query_count": 2,
            "aggregate": {},
            "queries": [
                {"query_id": "q1", "query_type": "synthesis", "verdict": "useful_breadth"},
                {"query_id": "q2", "query_type": "off_topic", "verdict": "off_topic_breadth_risk"},
            ],
        }
        with patch("canon.product.service.load_diversity_report_from_params", return_value=payload):
            report = service.diversity_queries({"verdict": "useful_breadth", "limit": "1"})
        self.assertEqual(report["result_count"], 1)
        self.assertEqual(report["queries"][0]["query_id"], "q1")

    def test_diversity_query_detail_returns_full_query_payload(self):
        payload = {
            "mode": "m",
            "method_set_id": "methods",
            "diverse_method_id": "diverse",
            "baseline_method_id": "lexical",
            "query_count": 1,
            "aggregate": {},
            "queries": [
                {"query_id": "q1", "added_titles": [{"title": "A"}]},
            ],
        }
        with patch("canon.product.service.load_diversity_report_from_params", return_value=payload):
            report = service.diversity_query_detail("q1", {})
        self.assertIn("added_titles", report["query"])

    def test_diversity_query_detail_unknown_query_raises_404(self):
        with patch("canon.product.service.load_diversity_report_from_params", return_value={"queries": []}):
            with self.assertRaises(service.ProductError) as context:
                service.diversity_query_detail("missing", {})
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

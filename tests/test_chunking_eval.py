import unittest

from canon.eval.contracts import validate_report
from canon.eval.chunking import evaluate_chunking_strategies


class ChunkingStrategyEvalTests(unittest.TestCase):
    def test_boundary_aware_strategy_improves_span_preservation(self):
        report = evaluate_chunking_strategies(chunk_tokens=14, overlap_tokens=0)
        summary = report["summary"]

        self.assertGreaterEqual(report["case_count"], 10)
        self.assertIn("fixed_window", summary)
        self.assertIn("boundary_aware", summary)
        self.assertGreaterEqual(
            summary["boundary_aware"]["average_span_containment_rate"],
            summary["fixed_window"]["average_span_containment_rate"],
        )
        self.assertGreater(
            summary["boundary_aware"]["average_boundary_aligned_share"],
            summary["fixed_window"]["average_boundary_aligned_share"],
        )
        self.assertIn("web_page", summary["boundary_aware"]["document_type_counts"])
        self.assertIn("regulatory_filing", summary["boundary_aware"]["document_type_counts"])

    def test_social_injection_case_tracks_unsafe_isolation(self):
        report = evaluate_chunking_strategies(chunk_tokens=14, overlap_tokens=0)
        rows = {
            (row["case_id"], row["strategy"]): row
            for row in report["cases"]
        }

        boundary_social = rows[("social_injection_isolation", "boundary_aware")]
        fixed_social = rows[("social_injection_isolation", "fixed_window")]
        self.assertTrue(boundary_social["unsafe_isolation"]["unsafe_isolated"])
        self.assertGreaterEqual(
            fixed_social["unsafe_isolation"]["scanner_rejected_chunk_count"],
            1,
        )

    def test_boundary_aware_social_chunking_isolates_unsafe_text_with_large_windows(self):
        report = evaluate_chunking_strategies(chunk_tokens=1000, overlap_tokens=100)
        rows = {
            (row["case_id"], row["strategy"]): row
            for row in report["cases"]
        }

        boundary_social = rows[("social_injection_isolation", "boundary_aware")]
        fixed_social = rows[("social_injection_isolation", "fixed_window")]
        self.assertTrue(boundary_social["unsafe_isolation"]["unsafe_isolated"])
        self.assertFalse(fixed_social["unsafe_isolation"]["unsafe_isolated"])
        self.assertGreater(
            fixed_social["unsafe_isolation"]["unsafe_contamination_count"],
            boundary_social["unsafe_isolation"]["unsafe_contamination_count"],
        )

    def test_boundary_aware_strategy_tracks_forbidden_merges(self):
        report = evaluate_chunking_strategies(chunk_tokens=1000, overlap_tokens=100)
        summary = report["summary"]

        self.assertGreater(summary["fixed_window"]["forbidden_merge_count"], 0)
        self.assertEqual(summary["boundary_aware"]["forbidden_merge_count"], 0)

    def test_chunking_report_satisfies_contract(self):
        report = evaluate_chunking_strategies(chunk_tokens=1000, overlap_tokens=100)
        validation = validate_report(report, "chunking_strategy_eval")

        self.assertEqual(validation["status"], "pass")

    def test_chunking_contract_rejects_large_window_boundary_forbidden_merges(self):
        report = evaluate_chunking_strategies(chunk_tokens=1000, overlap_tokens=100)
        report["summary"]["boundary_aware"]["forbidden_merge_count"] = 1
        validation = validate_report(report, "chunking_strategy_eval")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("forbidden_merge_count" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()

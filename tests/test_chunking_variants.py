import unittest

from canon.eval.contracts import validate_report
from canon.eval.chunking_variants import evaluate_chunking_variants


class ChunkingVariantEvalTests(unittest.TestCase):
    def test_variant_eval_compares_multiple_technical_strategies(self):
        report = evaluate_chunking_variants(chunk_tokens=1000, overlap_tokens=100)

        self.assertEqual(report["report_id"], "chunking_variant_eval_v1")
        self.assertIn("fixed_window", report["summary"])
        self.assertIn("boundary_aware", report["summary"])
        self.assertIn("semantic_cohesion", report["summary"])
        self.assertIn("embedding_semantic", report["summary"])
        self.assertIn("atomic_sentence", report["summary"])
        self.assertGreaterEqual(report["case_count"], 10)

    def test_semantic_and_boundary_variants_avoid_large_window_forbidden_merges(self):
        report = evaluate_chunking_variants(chunk_tokens=1000, overlap_tokens=100)
        summary = report["summary"]

        self.assertGreater(summary["fixed_window"]["forbidden_merge_count"], 0)
        self.assertEqual(summary["boundary_aware"]["forbidden_merge_count"], 0)
        self.assertEqual(summary["semantic_cohesion"]["forbidden_merge_count"], 0)
        self.assertEqual(summary["embedding_semantic"]["forbidden_merge_count"], 0)
        self.assertEqual(summary["atomic_sentence"]["forbidden_merge_count"], 0)

    def test_variant_objectives_track_tradeoffs(self):
        report = evaluate_chunking_variants(chunk_tokens=1000, overlap_tokens=100)
        summary = report["summary"]

        self.assertGreaterEqual(
            summary["boundary_aware"]["objective_scores"]["balanced_utility"],
            summary["fixed_window"]["objective_scores"]["balanced_utility"],
        )
        self.assertGreater(
            summary["atomic_sentence"]["average_chunk_count"],
            summary["fixed_window"]["average_chunk_count"],
        )
        self.assertIn("balanced_utility", report["best_variant_by_objective"])

    def test_variant_report_satisfies_contract(self):
        report = evaluate_chunking_variants(chunk_tokens=1000, overlap_tokens=100)
        validation = validate_report(report, "chunking_variant_eval")

        self.assertEqual(validation["status"], "pass")

    def test_variant_contract_rejects_missing_semantic_comparison(self):
        report = evaluate_chunking_variants(chunk_tokens=1000, overlap_tokens=100)
        report["variants"].remove("embedding_semantic")
        validation = validate_report(report, "chunking_variant_eval")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("embedding_semantic" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()

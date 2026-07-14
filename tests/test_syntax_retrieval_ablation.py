import unittest

from canon.eval.contracts import validate_report
from canon.eval.syntax_retrieval_ablation import evaluate_syntax_retrieval_ablation


class SyntaxRetrievalAblationTests(unittest.TestCase):
    def test_syntax_ablation_report_satisfies_contract(self):
        report = evaluate_syntax_retrieval_ablation(write_report=False)

        self.assertEqual(report["report_id"], "syntax_retrieval_ablation_v1")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertEqual(report["status"], "pass")
        self.assertIn("syntax_candidate_and_rerank", report["summary"]["by_variant"])
        self.assertEqual(report["summary"]["by_variant"]["syntax_candidate_and_rerank"]["unsafe_final_count"], 0)

    def test_syntax_ablation_contract_requires_core_variants(self):
        report = evaluate_syntax_retrieval_ablation(write_report=False)
        report["variants"] = [variant for variant in report["variants"] if variant["id"] != "syntax_candidate_only"]

        audit = validate_report(report, "syntax_retrieval_ablation")

        self.assertEqual(audit["status"], "fail")
        self.assertTrue(any("syntax_candidate_only" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()

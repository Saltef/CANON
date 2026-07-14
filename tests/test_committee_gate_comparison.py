import unittest

from canon.eval.committee_gate_comparison import evaluate_committee_gate_comparison
from canon.eval.contracts import validate_report


class CommitteeGateComparisonTests(unittest.TestCase):
    def test_comparison_report_quantifies_policy_violation_reduction(self):
        report = evaluate_committee_gate_comparison(
            modes=["social_public_opinion_demo_corpus"],
            top_k=3,
            write_report=False,
        )

        self.assertEqual(report["report_id"], "committee_gate_policy_comparison_v1")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertEqual(report["summary"]["gated_selected_context_policy_violation_count"], 0)
        self.assertGreater(report["summary"]["policy_violation_reduction"], 0)
        self.assertGreater(report["summary"]["gated_abstention_count"], 0)

    def test_contract_rejects_gated_policy_increasing_violations(self):
        report = evaluate_committee_gate_comparison(
            modes=["social_public_opinion_demo_corpus"],
            top_k=3,
            write_report=False,
        )
        report["summary"]["gated_selected_context_policy_violation_count"] = (
            report["summary"]["baseline_selected_context_policy_violation_count"] + 1
        )

        validation = validate_report(report, "committee_gate_comparison")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("gated policy should not increase" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()

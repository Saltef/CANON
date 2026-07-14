import unittest

from canon.eval.committee_gate_usefulness import evaluate_committee_gate_usefulness
from canon.eval.contracts import validate_report


class CommitteeGateUsefulnessTests(unittest.TestCase):
    def test_usefulness_audit_creates_human_label_ready_loss_tasks(self):
        report = evaluate_committee_gate_usefulness(
            modes=["mixed_domain_demo_corpus", "social_public_opinion_demo_corpus"],
            top_k=3,
            write_report=False,
        )

        self.assertEqual(report["report_id"], "committee_gate_usefulness_audit_v1")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertGreater(report["summary"]["lost_baseline_selected_count"], 0)
        self.assertGreater(report["summary"]["human_label_task_count"], 0)
        self.assertGreater(report["summary"]["review_needed_loss_count"], 0)
        first_lost = next(
            candidate
            for mode in report["modes"]
            for query in mode["queries"]
            for candidate in query["lost_candidates"]
        )
        self.assertIn("human_evidence_usefulness", first_lost["human_labels"])
        self.assertIn(first_lost["proxy_usefulness_risk"], {"acceptable_exclusion", "review_needed_loss"})

    def test_contract_rejects_missing_human_labels_on_lost_candidate(self):
        report = evaluate_committee_gate_usefulness(
            modes=["social_public_opinion_demo_corpus"],
            top_k=3,
            write_report=False,
        )
        del report["modes"][0]["queries"][0]["lost_candidates"][0]["human_labels"]

        validation = validate_report(report, "committee_gate_usefulness")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("human_labels" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()

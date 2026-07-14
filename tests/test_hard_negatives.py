import unittest

from canon.eval.hard_negatives import (
    anchor_preference_cases,
    anchor_preference_examples,
    build_hard_negative_report,
)
from canon.modeling.preference_model import build_preference_report_from_anchors


class HardNegativeAnchorTests(unittest.TestCase):
    def test_anchor_cases_cover_core_failure_modes(self):
        cases = anchor_preference_cases()
        case_ids = {case.case_id for case in cases}
        self.assertIn("anchor_prompt_injection_gate_001", case_ids)
        self.assertIn("anchor_independent_corroboration_001", case_ids)
        self.assertIn("anchor_source_quality_not_access_001", case_ids)
        self.assertGreaterEqual(len(cases), 5)

    def test_anchor_examples_create_preferred_and_dispreferred_rows(self):
        cases = anchor_preference_cases()
        examples = anchor_preference_examples()
        self.assertEqual(len(examples), len(cases) * 2)
        self.assertEqual(sum(1 for example in examples if example.target == 1.0), len(cases))
        self.assertEqual(sum(1 for example in examples if example.target == 0.0), len(cases))
        self.assertTrue(all(example.target_source == "hard_negative_anchor" for example in examples))

    def test_hard_negative_report_contains_limitations(self):
        report = build_hard_negative_report()
        self.assertEqual(report["target_source"], "hard_negative_anchor")
        self.assertEqual(report["case_count"], len(anchor_preference_cases()))
        self.assertIn("limitations", report)

    def test_preference_model_can_train_from_anchor_source(self):
        report = build_preference_report_from_anchors(mode="unit_anchor_preferences")
        self.assertEqual(report["source_report"], "hard_negative_anchor_preferences")
        self.assertEqual(report["target_source_counts"], {"hard_negative_anchor": 10})
        self.assertGreaterEqual(report["leave_one_query_out_metrics"]["accuracy"], 0.5)
        self.assertTrue(any("hard-negative anchors" in item for item in report["limitations"]))


if __name__ == "__main__":
    unittest.main()
